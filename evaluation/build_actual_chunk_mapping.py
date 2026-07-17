"""将 50 题参考证据映射到 Milvus 中实际存在的 chunk。

本脚本不调用 Embedding 或 LLM，只执行以下确定性步骤：

1. 使用当前项目相同的 Markdown 切块规则重建 chunk 顺序；
2. 从 Milvus collection ``biz`` 读取真实主键、正文和元数据；
3. 校验重建 chunk 的文本哈希与 Milvus 中的实际内容完全一致；
4. 将每题参考证据映射到一个或多个实际 chunk；
5. 输出可直接用于 Ragas ID-based 指标的数据集和人工复核表。

逻辑 chunk ID 格式：

    {document_id}:{sha256(chunk_content)[:16]}

Milvus 随机 UUID 主键会另外保存在 ``milvus_pks`` 中。逻辑 ID 用于评测，
因为它在相同语料和切块规则下可重复生成，并能自动合并重复写入的相同 chunk。
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pymilvus import Collection, connections

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.generate_ragas_50 import EVIDENCE_CHECKS


ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = ROOT / "evaluation"
SOURCE_DATASET = EVALUATION_DIR / "ragas_50_dataset.jsonl"
SOURCE_MANIFEST = EVALUATION_DIR / "ragas_50_manifest.jsonl"

MAPPING_PATH = EVALUATION_DIR / "ragas_50_actual_chunk_mapping.jsonl"
ACTUAL_DATASET_PATH = EVALUATION_DIR / "ragas_50_actual_chunks.jsonl"
REVIEW_PATH = EVALUATION_DIR / "ragas_50_actual_chunk_review.csv"
SUMMARY_PATH = EVALUATION_DIR / "ragas_50_actual_chunk_summary.json"

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "biz"

# 必须与 app/config.py 和 DocumentSplitterService 当前配置一致。
CHUNK_MAX_SIZE = 1600
SECONDARY_CHUNK_SIZE = CHUNK_MAX_SIZE * 2
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 300
MAX_SELECTED_CHUNKS = 1


@dataclass
class IndexedChunk:
    document_id: str
    chunk_index: int
    content: str
    content_hash: str
    actual_chunk_id: str
    milvus_pks: list[str]
    metadata: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _logical_chunk_id(document_id: str, content_hash: str) -> str:
    return f"{document_id}:{content_hash}"


def _split_markdown(content: str) -> list[str]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SECONDARY_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    markdown_docs = markdown_splitter.split_text(content)
    split_docs = text_splitter.split_documents(markdown_docs)

    merged: list[str] = []
    current: str | None = None
    for doc in split_docs:
        text = doc.page_content
        if current is None:
            current = text
        elif len(text) < MIN_CHUNK_SIZE and len(current) < SECONDARY_CHUNK_SIZE:
            current += "\n\n" + text
        else:
            merged.append(current)
            current = text
    if current is not None:
        merged.append(current)
    return merged


def _normalize(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"</?[A-Za-z][^<>]*>", " ", text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = text.casefold()
    text = re.sub(r"[^\wα-ωΑ-Ω]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _term_present(text: str, term: str) -> bool:
    normalized_text = _normalize(text)
    normalized_term = _normalize(term)
    if not normalized_term:
        return False
    return normalized_term in normalized_text


def _token_coverage(reference: str, candidate: str) -> float:
    ref_tokens = Counter(_normalize(reference).split())
    candidate_tokens = Counter(_normalize(candidate).split())
    total = sum(ref_tokens.values())
    if total == 0:
        return 0.0
    overlap = sum((ref_tokens & candidate_tokens).values())
    return overlap / total


def _match_score(reference: str, candidate: str, evidence_terms: tuple[str, ...]) -> dict:
    normalized_reference = _normalize(reference)
    normalized_candidate = _normalize(candidate)
    sequence_ratio = SequenceMatcher(
        None,
        normalized_reference,
        normalized_candidate,
        autojunk=False,
    ).ratio()
    token_coverage = _token_coverage(reference, candidate)
    term_hits = [term for term in evidence_terms if _term_present(candidate, term)]
    term_coverage = len(term_hits) / len(evidence_terms) if evidence_terms else 1.0
    score = 0.45 * term_coverage + 0.35 * token_coverage + 0.20 * sequence_ratio
    return {
        "score": score,
        "sequence_ratio": sequence_ratio,
        "token_coverage": token_coverage,
        "term_hits": term_hits,
        "term_coverage": term_coverage,
    }


def _load_milvus_rows(document_ids: set[str]) -> dict[str, list[dict]]:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        timeout=10,
    )
    collection = Collection(COLLECTION_NAME)
    collection.load()
    rows = collection.query(
        expr="",
        output_fields=["id", "content", "metadata"],
        limit=10000,
    )

    by_document: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        document_id = metadata.get("_document_id", "")
        if document_id in document_ids:
            by_document[document_id].append(row)
    return by_document


def _build_indexed_chunks(
    document_id: str,
    markdown_path: Path,
    milvus_rows: list[dict],
) -> tuple[list[IndexedChunk], dict]:
    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    split_contents = _split_markdown(markdown)

    milvus_by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in milvus_rows:
        milvus_by_hash[_content_hash(row.get("content", ""))].append(row)

    split_hashes = {_content_hash(content) for content in split_contents}
    milvus_hashes = set(milvus_by_hash)
    if split_hashes != milvus_hashes:
        raise ValueError(
            f"{document_id} 重建切块与 Milvus 不一致："
            f"missing={sorted(split_hashes - milvus_hashes)[:5]}, "
            f"extra={sorted(milvus_hashes - split_hashes)[:5]}"
        )

    chunks: list[IndexedChunk] = []
    duplicate_row_count = 0
    for chunk_index, content in enumerate(split_contents):
        content_hash = _content_hash(content)
        matched_rows = milvus_by_hash[content_hash]
        duplicate_row_count += max(0, len(matched_rows) - 1)
        chunks.append(
            IndexedChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                content_hash=content_hash,
                actual_chunk_id=_logical_chunk_id(document_id, content_hash),
                milvus_pks=sorted(str(row["id"]) for row in matched_rows),
                metadata=dict(matched_rows[0].get("metadata") or {}),
            )
        )

    validation = {
        "reconstructed_chunk_count": len(split_contents),
        "milvus_unique_chunk_count": len(milvus_hashes),
        "milvus_row_count": len(milvus_rows),
        "duplicate_milvus_row_count": duplicate_row_count,
        "exact_hash_set_match": True,
    }
    return chunks, validation


def _select_chunks(
    reference_context: str,
    chunks: list[IndexedChunk],
    evidence_terms: tuple[str, ...],
) -> tuple[list[IndexedChunk], list[dict], dict]:
    scored: list[tuple[IndexedChunk, dict]] = [
        (chunk, _match_score(reference_context, chunk.content, evidence_terms))
        for chunk in chunks
    ]
    scored.sort(key=lambda item: item[1]["score"], reverse=True)

    # 题目均由单一摘要/总结证据生成。金标准应保留“最小充分 chunk”，
    # 不能因为参考证据还包含引言，就把无关引言 chunk 一并标为相关。
    selected: list[tuple[IndexedChunk, dict]] = [scored[0]]

    selected.sort(key=lambda item: item[0].chunk_index)
    selected_chunks = [item[0] for item in selected]
    selected_scores = [item[1] for item in selected]

    covered_terms = sorted(
        {
            term
            for score in selected_scores
            for term in score["term_hits"]
        }
    )
    combined_content = "\n\n".join(chunk.content for chunk in selected_chunks)
    combined_token_coverage = _token_coverage(reference_context, combined_content)
    status = {
        "covered_terms": covered_terms,
        "missing_terms": sorted(set(evidence_terms) - set(covered_terms)),
        "term_coverage": len(covered_terms) / len(evidence_terms),
        "reference_token_coverage": combined_token_coverage,
        "max_match_score": max(score["score"] for score in selected_scores),
        "auto_mapping_pass": (
            len(covered_terms) >= max(1, len(evidence_terms) - 1)
            and combined_token_coverage >= 0.25
            and max(score["score"] for score in selected_scores) >= 0.55
        ),
    }
    return selected_chunks, selected_scores, status


def main() -> None:
    dataset_rows = _read_jsonl(SOURCE_DATASET)
    manifest_rows = _read_jsonl(SOURCE_MANIFEST)
    if len(dataset_rows) != 50 or len(manifest_rows) != 50:
        raise ValueError("输入数据集和 manifest 必须各包含 50 条")

    document_ids = {
        row["reference_context_metadata"][0]["document_id"]
        for row in manifest_rows
    }
    milvus_by_document = _load_milvus_rows(document_ids)

    mapping_rows: list[dict] = []
    actual_dataset_rows: list[dict] = []
    review_rows: list[dict] = []
    total_duplicate_rows = 0

    for dataset_row, manifest_row in zip(dataset_rows, manifest_rows, strict=True):
        question_id = manifest_row["question_id"]
        question = manifest_row["user_input"]
        source_meta = manifest_row["reference_context_metadata"][0]
        document_id = source_meta["document_id"]
        evidence_context_id = manifest_row["reference_context_ids"][0]
        evidence_context = dataset_row["reference_contexts"][0]
        markdown_path = ROOT / source_meta["markdown_path"]
        evidence_terms = EVIDENCE_CHECKS[document_id]

        if document_id not in milvus_by_document:
            raise ValueError(f"Milvus 中没有找到 {document_id}")

        indexed_chunks, index_validation = _build_indexed_chunks(
            document_id=document_id,
            markdown_path=markdown_path,
            milvus_rows=milvus_by_document[document_id],
        )
        total_duplicate_rows += index_validation["duplicate_milvus_row_count"]

        selected_chunks, selected_scores, mapping_status = _select_chunks(
            reference_context=evidence_context,
            chunks=indexed_chunks,
            evidence_terms=evidence_terms,
        )

        actual_chunk_ids = [chunk.actual_chunk_id for chunk in selected_chunks]
        actual_contexts = [chunk.content for chunk in selected_chunks]

        mapping_rows.append(
            {
                "question_id": question_id,
                "user_input": question,
                "document_id": document_id,
                "evidence_context_id": evidence_context_id,
                "actual_chunk_ids": actual_chunk_ids,
                "actual_chunks": [
                    {
                        "actual_chunk_id": chunk.actual_chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "content_hash": chunk.content_hash,
                        "milvus_pks": chunk.milvus_pks,
                        "source_id": chunk.metadata.get("_source", ""),
                        "parsed_source": chunk.metadata.get("_parsed_source", ""),
                        "content": chunk.content,
                        "match": {
                            "score": round(score["score"], 6),
                            "sequence_ratio": round(score["sequence_ratio"], 6),
                            "token_coverage": round(score["token_coverage"], 6),
                            "term_hits": score["term_hits"],
                        },
                    }
                    for chunk, score in zip(
                        selected_chunks,
                        selected_scores,
                        strict=True,
                    )
                ],
                "mapping_status": {
                    **mapping_status,
                    "reference_token_coverage": round(
                        mapping_status["reference_token_coverage"], 6
                    ),
                    "max_match_score": round(
                        mapping_status["max_match_score"], 6
                    ),
                },
                "index_validation": index_validation,
            }
        )

        actual_dataset_rows.append(
            {
                "user_input": question,
                "response": None,
                "retrieved_contexts": [],
                "retrieved_context_ids": [],
                "reference": None,
                "reference_contexts": actual_contexts,
                "reference_context_ids": actual_chunk_ids,
            }
        )

        review_rows.append(
            {
                "question_id": question_id,
                "question": question,
                "document_id": document_id,
                "evidence_context_id": evidence_context_id,
                "actual_chunk_ids": ";".join(actual_chunk_ids),
                "actual_chunk_indexes": ";".join(
                    str(chunk.chunk_index) for chunk in selected_chunks
                ),
                "milvus_pks": ";".join(
                    ",".join(chunk.milvus_pks) for chunk in selected_chunks
                ),
                "selected_chunk_count": len(selected_chunks),
                "evidence_terms_covered": len(mapping_status["covered_terms"]),
                "evidence_terms_total": len(evidence_terms),
                "missing_terms": ";".join(mapping_status["missing_terms"]),
                "reference_token_coverage": round(
                    mapping_status["reference_token_coverage"], 4
                ),
                "max_match_score": round(mapping_status["max_match_score"], 4),
                "auto_mapping_pass": mapping_status["auto_mapping_pass"],
                "duplicate_milvus_rows_for_document": index_validation[
                    "duplicate_milvus_row_count"
                ],
                "chunk_preview": " || ".join(
                    chunk.content[:350].replace("\n", " ")
                    for chunk in selected_chunks
                ),
                "mapping_accept": "",
                "reviewer_notes": "",
            }
        )

    _write_jsonl(MAPPING_PATH, mapping_rows)
    _write_jsonl(ACTUAL_DATASET_PATH, actual_dataset_rows)

    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    passed = sum(
        1 for row in mapping_rows if row["mapping_status"]["auto_mapping_pass"]
    )
    selected_counts = [
        len(row["actual_chunk_ids"]) for row in mapping_rows
    ]
    summary = {
        "question_count": len(mapping_rows),
        "mapped_question_count": sum(bool(row["actual_chunk_ids"]) for row in mapping_rows),
        "auto_mapping_pass_count": passed,
        "manual_review_required_count": len(mapping_rows) - passed,
        "unique_actual_chunk_id_count": len(
            {
                chunk_id
                for row in mapping_rows
                for chunk_id in row["actual_chunk_ids"]
            }
        ),
        "selected_chunk_count": {
            "min": min(selected_counts),
            "max": max(selected_counts),
            "mean": round(sum(selected_counts) / len(selected_counts), 2),
        },
        "all_50_document_chunk_sets_exactly_match_milvus": all(
            row["index_validation"]["exact_hash_set_match"]
            for row in mapping_rows
        ),
        "duplicate_milvus_row_count_across_question_documents": total_duplicate_rows,
        "logical_chunk_id_format": "{document_id}:{sha256(chunk_content)[:16]}",
        "files": {
            "mapping": MAPPING_PATH.relative_to(ROOT).as_posix(),
            "actual_dataset": ACTUAL_DATASET_PATH.relative_to(ROOT).as_posix(),
            "review": REVIEW_PATH.relative_to(ROOT).as_posix(),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
