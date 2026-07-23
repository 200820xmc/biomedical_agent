"""只读审计本地 PDF 语料与 Milvus ``biz`` collection 的一致性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.milvus_client import milvus_manager


ORIGINALS_DIR = PROJECT_ROOT / "uploads" / "originals"
PARSED_DIR = PROJECT_ROOT / "uploads" / "parsed"
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "KB_MAINTENANCE_20260722"
    / "milvus_corpus_audit.json"
)
REQUIRED_METADATA = ("source_id", "chunk_index", "content_hash", "chunk_id")


def _local_documents() -> dict[str, dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    for directory in sorted(ORIGINALS_DIR.glob("doc_*")):
        if not directory.is_dir():
            continue
        pdfs = sorted(directory.glob("*.pdf"))
        markdowns = sorted((PARSED_DIR / directory.name).glob("*.md"))
        if len(pdfs) == 1 and len(markdowns) == 1 and markdowns[0].stat().st_size:
            documents[directory.name] = {
                "pdf": str(pdfs[0].relative_to(PROJECT_ROOT)),
                "markdown": str(markdowns[0].relative_to(PROJECT_ROOT)),
            }
    return documents


def _all_rows() -> list[dict[str, Any]]:
    collection = milvus_manager.get_collection()
    collection.flush()
    return collection.query(
        expr='id != ""',
        output_fields=["id", "content", "metadata"],
        limit=10000,
    )


def audit(expected_count: int) -> dict[str, Any]:
    local = _local_documents()
    rows = _all_rows()
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_metadata = Counter()
    invalid_chunk_id: list[str] = []
    invalid_content_hash: list[str] = []
    invalid_chunk_index_type: list[str] = []
    logical_chunk_keys: list[tuple[str, str]] = []
    content_hash_sources: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row_id = str(row.get("id", ""))
        for field in REQUIRED_METADATA:
            if field not in metadata or metadata[field] in (None, ""):
                missing_metadata[field] += 1

        source_id = str(metadata.get("source_id", ""))
        content = str(row.get("content", ""))
        content_hash = str(metadata.get("content_hash", ""))
        chunk_id = str(metadata.get("chunk_id", ""))
        source_rows[source_id].append(row)
        logical_chunk_keys.append((source_id, chunk_id))
        content_hash_sources[hashlib.sha256(content.encode("utf-8")).hexdigest()].add(source_id)

        if source_id and content_hash and chunk_id != f"{source_id}:{content_hash}":
            invalid_chunk_id.append(row_id)
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        if content_hash != expected_hash:
            invalid_content_hash.append(row_id)
        if not isinstance(metadata.get("chunk_index"), int):
            invalid_chunk_index_type.append(row_id)

    index_issues: dict[str, dict[str, Any]] = {}
    for source_id, source_items in source_rows.items():
        indices = [
            item.get("metadata", {}).get("chunk_index")
            for item in source_items
            if isinstance(item.get("metadata", {}).get("chunk_index"), int)
        ]
        expected = list(range(len(source_items)))
        if sorted(indices) != expected:
            index_issues[source_id] = {
                "row_count": len(source_items),
                "indices": sorted(indices),
                "expected": expected,
            }

    duplicate_logical_chunks = {
        f"{source_id}|{chunk_id}": count
        for (source_id, chunk_id), count in Counter(logical_chunk_keys).items()
        if count > 1
    }
    duplicate_bodies = {
        body_hash: sorted(sources)
        for body_hash, sources in content_hash_sources.items()
        if len(sources) > 1
    }
    milvus_sources = {source for source in source_rows if source}
    local_sources = set(local)

    checks = {
        "local_document_count_matches_expected": len(local) == expected_count,
        "milvus_source_count_matches_expected": len(milvus_sources) == expected_count,
        "local_and_milvus_sources_match": local_sources == milvus_sources,
        "required_metadata_complete": not missing_metadata,
        "chunk_id_formula_valid": not invalid_chunk_id,
        "content_hash_valid": not invalid_content_hash,
        "chunk_index_integer": not invalid_chunk_index_type,
        "chunk_index_contiguous_per_source": not index_issues,
        "no_duplicate_logical_chunks": not duplicate_logical_chunks,
        "no_duplicate_bodies_across_sources": not duplicate_bodies,
        "no_empty_source_id": "" not in source_rows,
    }
    return {
        "audited_at": datetime.now().astimezone().isoformat(),
        "collection": "biz",
        "expected_document_count": expected_count,
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "milvus_rows": len(rows),
            "milvus_sources": len(milvus_sources),
            "local_documents": len(local),
            "missing_metadata": dict(missing_metadata),
            "invalid_chunk_ids": len(invalid_chunk_id),
            "invalid_content_hashes": len(invalid_content_hash),
            "invalid_chunk_index_types": len(invalid_chunk_index_type),
            "sources_with_index_issues": len(index_issues),
            "duplicate_logical_chunks": len(duplicate_logical_chunks),
            "duplicate_bodies_across_sources": len(duplicate_bodies),
        },
        "differences": {
            "local_only_sources": sorted(local_sources - milvus_sources),
            "milvus_only_sources": sorted(milvus_sources - local_sources),
            "index_issues": index_issues,
            "duplicate_logical_chunks": duplicate_logical_chunks,
            "duplicate_bodies_across_sources": duplicate_bodies,
            "invalid_chunk_id_rows": invalid_chunk_id,
            "invalid_content_hash_rows": invalid_content_hash,
            "invalid_chunk_index_type_rows": invalid_chunk_index_type,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--expected-count", type=int, default=66)
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    report_path.relative_to(PROJECT_ROOT.resolve())
    _ = milvus_manager.connect()
    report = audit(args.expected_count)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), **report}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
