"""Build a versioned, claim-level candidate evaluation set from the current index.

This script reads Milvus but never writes to it.  It keeps the historical strict
chunk target for diagnosis and adds claim-level alternative chunks so that an
abstract and a more detailed body chunk can both be valid evidence.

The generated references and alternative mappings are extractive candidates.
They are deliberately marked as pending review and must not be presented as a
human-reviewed gold set until the review CSV is completed.
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


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.milvus_client import milvus_manager
from evaluation.generate_ragas_50 import EVIDENCE_CHECKS


EVALUATION_DIR = ROOT / "evaluation"
SOURCE_DATASET = EVALUATION_DIR / "ragas_50_dataset.jsonl"
SOURCE_MANIFEST = EVALUATION_DIR / "ragas_50_manifest.jsonl"
DATASET_PATH = EVALUATION_DIR / "ragas_50_v2_dataset.jsonl"
MAPPING_PATH = EVALUATION_DIR / "ragas_50_v2_mapping.jsonl"
REVIEW_PATH = EVALUATION_DIR / "ragas_50_v2_review.csv"
SUMMARY_PATH = EVALUATION_DIR / "ragas_50_v2_summary.json"
REVIEW_OVERRIDES_DIR = EVALUATION_DIR / "review_overrides"

GENERATION_VERSION = "ragas_avf_50_v2_claim_alternatives_candidate"
CORPUS_VERSION = "milvus_biz_66docs_20260722"
DOCUMENT_ID_ALIASES = {
    "doc_26b3f0": "doc_2954af",
    "doc_49476e": "doc_edffea",
}
MAX_ALTERNATIVES_PER_CLAIM = 4
MAX_REFERENCE_CLAIMS = 4


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    source_id: str
    chunk_index: int
    content_hash: str
    content: str
    milvus_id: str
    metadata: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_review_overrides() -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    if not REVIEW_OVERRIDES_DIR.exists():
        return overrides
    for path in sorted(REVIEW_OVERRIDES_DIR.glob("rq*_rq*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("review_type") != "model_assisted_candidate_cleanup":
            raise ValueError(f"Unsupported review type in {path.name}")
        for question_id, override in payload.get("questions", {}).items():
            if question_id in overrides:
                raise ValueError(f"Duplicate review override for {question_id}")
            if not str(override.get("reference", "")).strip():
                raise ValueError(f"Empty reviewed reference for {question_id}")
            chunk_ids = override.get("acceptable_chunk_ids")
            if not isinstance(chunk_ids, list) or not chunk_ids:
                raise ValueError(f"Empty reviewed chunks for {question_id}")
            overrides[question_id] = dict(override)
    return overrides


def _normalize(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"</?[A-Za-z][^<>]*>", " ", text)
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = text.casefold()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _term_present(text: str, term: str) -> bool:
    normalized_term = _normalize(term)
    return bool(normalized_term) and normalized_term in _normalize(text)


def _token_coverage(reference: str, candidate: str) -> float:
    reference_tokens = Counter(_normalize(reference).split())
    candidate_tokens = Counter(_normalize(candidate).split())
    total = sum(reference_tokens.values())
    if not total:
        return 0.0
    return sum((reference_tokens & candidate_tokens).values()) / total


def _score_text(
    reference: str,
    candidate: str,
    evidence_terms: tuple[str, ...],
) -> dict[str, Any]:
    term_hits = [term for term in evidence_terms if _term_present(candidate, term)]
    term_coverage = len(term_hits) / len(evidence_terms) if evidence_terms else 1.0
    token_coverage = _token_coverage(reference, candidate)
    normalized_reference = _normalize(reference)
    normalized_candidate = _normalize(candidate)
    sequence_ratio = float(
        bool(normalized_reference) and normalized_reference in normalized_candidate
    )
    return {
        "score": 0.55 * term_coverage + 0.40 * token_coverage + 0.05 * sequence_ratio,
        "term_hits": term_hits,
        "term_coverage": term_coverage,
        "token_coverage": token_coverage,
        "sequence_ratio": sequence_ratio,
    }


def _load_current_chunks() -> dict[str, list[IndexedChunk]]:
    milvus_manager.connect()
    rows = milvus_manager.get_collection().query(
        expr='id != ""',
        output_fields=["id", "content", "metadata"],
        limit=10000,
    )
    chunks_by_source: dict[str, list[IndexedChunk]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source_id = str(metadata.get("source_id", ""))
        content = str(row.get("content", ""))
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        chunk_id = str(metadata.get("chunk_id", ""))
        chunk_index = metadata.get("chunk_index")
        if not source_id or not isinstance(chunk_index, int):
            raise ValueError("Current Milvus rows must have source_id and integer chunk_index")
        if metadata.get("content_hash") != content_hash:
            raise ValueError(f"Invalid content_hash for Milvus row {row.get('id')}")
        if chunk_id != f"{source_id}:{content_hash}":
            raise ValueError(f"Invalid chunk_id for Milvus row {row.get('id')}")
        chunks_by_source[source_id].append(
            IndexedChunk(
                chunk_id=chunk_id,
                source_id=source_id,
                chunk_index=chunk_index,
                content_hash=content_hash,
                content=content,
                milvus_id=str(row.get("id", "")),
                metadata=dict(metadata),
            )
        )
    for chunks in chunks_by_source.values():
        chunks.sort(key=lambda item: item.chunk_index)
    return chunks_by_source


def _sentence_candidates(reference_context: str) -> list[str]:
    cleaned = re.sub(r"^#{1,6}\s+", "", reference_context, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    boilerplate = (
        "copyright",
        "all rights reserved",
        "keywords:",
        "department of",
        "published by",
    )
    candidates = []
    for sentence in sentences:
        sentence = sentence.strip()
        letters = [character for character in sentence if character.isalpha()]
        uppercase_ratio = (
            sum(character.isupper() for character in letters) / len(letters)
            if letters
            else 0.0
        )
        folded = sentence.casefold()
        if not 35 <= len(sentence) <= 700:
            continue
        if any(token in folded for token in boilerplate):
            continue
        if uppercase_ratio > 0.55 or "member, ieee" in folded or "@" in sentence:
            continue
        candidates.append(sentence)
    return candidates


def _extract_reference_claims(
    reference_context: str,
    evidence_terms: tuple[str, ...],
    question_type: str,
) -> list[str]:
    sentences = _sentence_candidates(reference_context)
    if not sentences:
        return [reference_context[:700].strip()]

    scored = []
    for position, sentence in enumerate(sentences):
        hits = {term for term in evidence_terms if _term_present(sentence, term)}
        folded = sentence.casefold()
        intention_only = any(
            phrase in folded
            for phrase in (
                "aimed at investigating",
                "aim of this",
                "objective of this",
                "purpose of this",
                "we sought to",
                "we investigate whether",
            )
        )
        if intention_only:
            continue
        result_cues = sum(
            cue in folded
            for cue in (
                "result",
                "found",
                "showed",
                "demonstrate",
                "suggest",
                "associated",
                "correlat",
                "higher",
                "lower",
                "increase",
                "decrease",
                "compared",
                "sensitivity",
                "specificity",
                "achieved",
            )
        )
        method_cues = sum(
            cue in folded
            for cue in (
                "we used",
                "we performed",
                "was used",
                "were used",
                "model",
                "classifier",
                "feature",
                "recorded",
                "measured",
                "simulation",
            )
        )
        preferred_cues = (
            method_cues
            if question_type in {"method", "extractive"}
            else result_cues
        )
        scored.append((len(hits), preferred_cues, -position, sentence, hits))
    scored.sort(reverse=True)

    selected: list[str] = []
    covered: set[str] = set()
    for _, preferred_cues, _, sentence, hits in scored:
        if not selected and hits:
            selected.append(sentence)
            covered.update(hits)
        elif hits - covered and preferred_cues:
            selected.append(sentence)
            covered.update(hits)
        if (
            len(covered) >= max(1, len(evidence_terms) - 1)
            or len(selected) >= MAX_REFERENCE_CLAIMS
        ):
            break
    if not selected and scored:
        selected.append(scored[0][3])
    return selected[:MAX_REFERENCE_CLAIMS]


def _claim_alternatives(
    claim: str,
    ranked_chunks: list[tuple[IndexedChunk, dict[str, Any]]],
    evidence_terms: tuple[str, ...],
) -> list[tuple[IndexedChunk, float]]:
    scores: list[tuple[IndexedChunk, float]] = []
    normalized_claim = _normalize(claim)
    for chunk, _ in ranked_chunks[:10]:
        normalized_chunk = _normalize(chunk.content)
        exact = bool(normalized_claim) and normalized_claim in normalized_chunk
        coverage = _token_coverage(claim, chunk.content)
        sequence = SequenceMatcher(
            None, normalized_claim, normalized_chunk, autojunk=False
        ).ratio()
        score = max(1.0 if exact else 0.0, 0.75 * coverage + 0.25 * sequence)
        scores.append((chunk, score))
    scores.sort(key=lambda item: (-item[1], item[0].chunk_index))
    best = scores[0][1]
    threshold = max(0.28, best * 0.72)
    alternatives = [item for item in scores if item[1] >= threshold]
    alternative_ids = {item[0].chunk_id for item in alternatives}
    for chunk, match in ranked_chunks[:10]:
        if (
            chunk.chunk_id not in alternative_ids
            and match["term_coverage"] >= (2 / max(2, len(evidence_terms)))
            and match["score"] >= 0.45
        ):
            alternatives.append((chunk, match["score"]))
            alternative_ids.add(chunk.chunk_id)
    alternatives.sort(key=lambda item: (-item[1], item[0].chunk_index))
    return alternatives[:MAX_ALTERNATIVES_PER_CLAIM] or scores[:1]


def _round_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": round(score["score"], 6),
        "term_hits": score["term_hits"],
        "term_coverage": round(score["term_coverage"], 6),
        "token_coverage": round(score["token_coverage"], 6),
        "sequence_ratio": round(score["sequence_ratio"], 6),
    }


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset_rows = _read_jsonl(SOURCE_DATASET)
    manifest_rows = _read_jsonl(SOURCE_MANIFEST)
    if len(dataset_rows) != 50 or len(manifest_rows) != 50:
        raise ValueError("The source dataset and manifest must each contain 50 rows")

    chunks_by_source = _load_current_chunks()
    review_overrides = _load_review_overrides()
    expected_question_ids = {row["question_id"] for row in manifest_rows}
    if review_overrides and set(review_overrides) != expected_question_ids:
        raise ValueError(
            "Review overrides must cover all 50 questions exactly: "
            f"missing={sorted(expected_question_ids - set(review_overrides))}, "
            f"extra={sorted(set(review_overrides) - expected_question_ids)}"
        )
    dataset_v2: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    automatic_candidate_count = 0

    for dataset_row, manifest_row in zip(dataset_rows, manifest_rows, strict=True):
        question_id = manifest_row["question_id"]
        question = manifest_row["user_input"]
        old_source_id = manifest_row["reference_context_metadata"][0]["document_id"]
        source_id = DOCUMENT_ID_ALIASES.get(old_source_id, old_source_id)
        evidence_terms = EVIDENCE_CHECKS[old_source_id]
        reference_context = dataset_row["reference_contexts"][0]
        chunks = chunks_by_source.get(source_id, [])
        if not chunks:
            raise ValueError(f"Current index has no chunks for {source_id} ({question_id})")

        ranked_chunks = [
            (chunk, _score_text(reference_context, chunk.content, evidence_terms))
            for chunk in chunks
        ]
        ranked_chunks.sort(
            key=lambda item: (-item[1]["score"], item[0].chunk_index)
        )
        strict_chunk = ranked_chunks[0][0]
        claim_texts = _extract_reference_claims(
            reference_context,
            evidence_terms,
            manifest_row["question_type"],
        )

        reference_claims = []
        acceptable_by_id: dict[str, IndexedChunk] = {}
        for claim_number, claim_text in enumerate(claim_texts, start=1):
            alternatives = _claim_alternatives(
                claim_text, ranked_chunks, evidence_terms
            )
            alternative_ids = [chunk.chunk_id for chunk, _ in alternatives]
            for chunk, _ in alternatives:
                acceptable_by_id[chunk.chunk_id] = chunk
            reference_claims.append(
                {
                    "claim_id": f"{question_id}-c{claim_number}",
                    "claim_text": claim_text,
                    "acceptable_chunk_ids": alternative_ids,
                    "support_scores": {
                        chunk.chunk_id: round(score, 6)
                        for chunk, score in alternatives
                    },
                }
            )

        acceptable_by_id.setdefault(strict_chunk.chunk_id, strict_chunk)
        acceptable_chunks = sorted(
            acceptable_by_id.values(), key=lambda item: item.chunk_index
        )
        acceptable_ids = [chunk.chunk_id for chunk in acceptable_chunks]
        automatic_candidate_count += len(acceptable_ids)
        reference = " ".join(claim_texts)
        cleanup_override = review_overrides.get(question_id)
        if cleanup_override:
            chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            acceptable_ids = list(cleanup_override["acceptable_chunk_ids"])
            unknown_ids = sorted(set(acceptable_ids) - set(chunks_by_id))
            if unknown_ids:
                raise ValueError(
                    f"{question_id} cleanup references unknown chunks: {unknown_ids}"
                )
            if len(acceptable_ids) != len(set(acceptable_ids)):
                raise ValueError(f"{question_id} cleanup contains duplicate chunks")
            acceptable_chunks = sorted(
                (chunks_by_id[chunk_id] for chunk_id in acceptable_ids),
                key=lambda item: item.chunk_index,
            )
            acceptable_ids = [chunk.chunk_id for chunk in acceptable_chunks]
            reference = str(cleanup_override["reference"]).strip()
            strict_chunk = (
                strict_chunk
                if strict_chunk.chunk_id in acceptable_ids
                else acceptable_chunks[0]
            )
            reference_claims = [
                {
                    "claim_id": f"{question_id}-c1",
                    "claim_text": reference,
                    "acceptable_chunk_ids": acceptable_ids,
                    "review_stage": "model_assisted_candidate_cleanup",
                }
            ]

        dataset_v2.append(
            {
                "question_id": question_id,
                "user_input": question,
                "response": None,
                "retrieved_contexts": [],
                "retrieved_context_ids": [],
                "reference": reference,
                "reference_claims": reference_claims,
                "reference_contexts": [chunk.content for chunk in acceptable_chunks],
                "reference_context_ids": acceptable_ids,
                "strict_chunk_id": strict_chunk.chunk_id,
                "acceptable_chunk_ids": acceptable_ids,
                "document_id": source_id,
                "source_document_id": old_source_id,
                "reference_reviewed": False,
                "model_cleanup_applied": cleanup_override is not None,
                "review_status": "pending_human_review",
                "generation_version": GENERATION_VERSION,
                "corpus_version": CORPUS_VERSION,
            }
        )

        mapping_rows.append(
            {
                "question_id": question_id,
                "document_id": source_id,
                "source_document_id": old_source_id,
                "document_id_migrated": source_id != old_source_id,
                "evidence_terms": list(evidence_terms),
                "strict_chunk_id": strict_chunk.chunk_id,
                "acceptable_chunk_ids": acceptable_ids,
                "reference_claims": reference_claims,
                "model_cleanup_applied": cleanup_override is not None,
                "model_cleanup_notes": (
                    cleanup_override.get("review_notes", "")
                    if cleanup_override
                    else ""
                ),
                "ranked_candidates": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "match": _round_score(score),
                        "preview": chunk.content[:500].replace("\n", " "),
                    }
                    for chunk, score in ranked_chunks[:10]
                ],
            }
        )

        review_rows.append(
            {
                "question_id": question_id,
                "question": question,
                "document_id": source_id,
                "source_document_id": old_source_id,
                "document_id_migrated": source_id != old_source_id,
                "strict_chunk_id": strict_chunk.chunk_id,
                "acceptable_chunk_ids": ";".join(acceptable_ids),
                "acceptable_chunk_count": len(acceptable_ids),
                "reference_candidate": reference,
                "reference_claims_json": json.dumps(
                    reference_claims, ensure_ascii=False
                ),
                "evidence_preview": " || ".join(
                    f"[{chunk.chunk_index}] {chunk.content[:600].replace(chr(10), ' ')}"
                    for chunk in acceptable_chunks
                ),
                "question_supported": "",
                "reference_answer_reviewed": "",
                "acceptable_chunks_reviewed": "",
                "model_cleanup_applied": cleanup_override is not None,
                "model_cleanup_notes": (
                    cleanup_override.get("review_notes", "")
                    if cleanup_override
                    else ""
                ),
                "reviewer_notes": "",
            }
        )

    acceptable_counts = [len(row["acceptable_chunk_ids"]) for row in dataset_v2]
    claim_counts = [len(row["reference_claims"]) for row in dataset_v2]
    cleaned_candidate_count = sum(acceptable_counts)
    summary = {
        "generation_version": GENERATION_VERSION,
        "corpus_version": CORPUS_VERSION,
        "question_count": len(dataset_v2),
        "unique_question_count": len({row["question_id"] for row in dataset_v2}),
        "unique_current_document_count": len(
            {row["document_id"] for row in dataset_v2}
        ),
        "migrated_document_count": sum(
            row["document_id"] != row["source_document_id"] for row in dataset_v2
        ),
        "reference_reviewed_count": 0,
        "model_cleanup_applied_count": sum(
            bool(row["model_cleanup_applied"]) for row in dataset_v2
        ),
        "automatic_candidate_chunk_count": automatic_candidate_count,
        "cleaned_candidate_chunk_count": cleaned_candidate_count,
        "removed_false_gold_chunk_count": (
            automatic_candidate_count - cleaned_candidate_count
        ),
        "pending_human_review_count": len(dataset_v2),
        "ready_for_formal_ragas_context_recall": False,
        "planned_retrieval_metrics": {
            "Recall@3": "1 if any reviewed acceptable chunk is in ranks 1-3, else 0",
            "Recall@5": "1 if any reviewed acceptable chunk is in ranks 1-5, else 0",
            "MRR": "mean reciprocal rank; per question score is 1 / first acceptable rank, or 0",
        },
        "acceptable_chunk_count": {
            "min": min(acceptable_counts),
            "max": max(acceptable_counts),
            "mean": round(sum(acceptable_counts) / len(acceptable_counts), 2),
            "questions_with_multiple": sum(count > 1 for count in acceptable_counts),
        },
        "reference_claim_count": {
            "min": min(claim_counts),
            "max": max(claim_counts),
            "mean": round(sum(claim_counts) / len(claim_counts), 2),
        },
        "document_id_aliases": DOCUMENT_ID_ALIASES,
        "review_requirements": [
            "Confirm every reference claim is directly supported by the paper.",
            "Confirm every acceptable chunk directly supports its assigned claim.",
            "Remove boilerplate, OCR errors, and claims that do not answer the question.",
            "Set all three review flags before promoting the candidate set.",
        ],
        "files": {
            "dataset": DATASET_PATH.relative_to(ROOT).as_posix(),
            "mapping": MAPPING_PATH.relative_to(ROOT).as_posix(),
            "review": REVIEW_PATH.relative_to(ROOT).as_posix(),
        },
    }
    return dataset_v2, mapping_rows, {"summary": summary, "review_rows": review_rows}


def main() -> None:
    dataset_rows, mapping_rows, result = build()
    _write_jsonl(DATASET_PATH, dataset_rows)
    _write_jsonl(MAPPING_PATH, mapping_rows)
    review_rows = result["review_rows"]
    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    SUMMARY_PATH.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
