"""Formal ablation: run the reviewed questions directly through Full retrieval.

This bypasses Agent-generated tool queries but keeps the production retrieval
pipeline unchanged: Top-20 recall, one-shot rerank, threshold, Top-5 anchors,
chunk-index neighbor expansion, and context budgeting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.milvus_client import milvus_manager
from app.services.retrieval import retrieval_service
from app.services.vector_store_manager import vector_store_manager
from evaluation.common import write_csv, write_json
from evaluation.formal_eval_contract import (
    DEFAULT_REVIEW_CSV,
    load_formal_review_rows,
)


CST = timezone(timedelta(hours=8))


def _rank_documents(
    documents: list[dict[str, Any]],
    target_doc_id: str,
    target_chunk_ids: set[str],
) -> tuple[int | None, int | None]:
    doc_rank = None
    chunk_rank = None
    for rank, document in enumerate(documents, start=1):
        if doc_rank is None and document.get("source_id") == target_doc_id:
            doc_rank = rank
        if chunk_rank is None and document.get("chunk_id") in target_chunk_ids:
            chunk_rank = rank
    return doc_rank, chunk_rank


async def run_ablation(
    review_csv: str,
    output_dir: str,
    limit: int = 0,
) -> dict[str, Any]:
    rows, review_contract = load_formal_review_rows(review_csv, limit=limit)
    details: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        qid = row["question_id"]
        question = row["question"]
        target_doc_id = row["document_id"].strip()
        target_chunk_ids = {
            value.strip()
            for value in (row.get("acceptable_chunk_ids") or "").split(";")
            if value.strip()
        }
        print(f"[DIRECT {index}/{len(rows)}] {qid}: {question[:60]}")
        detail: dict[str, Any] = {
            "question_id": qid,
            "question": question,
            "target_doc_id": target_doc_id,
            "target_chunk_ids": ";".join(sorted(target_chunk_ids)),
        }
        try:
            _context, artifact = await retrieval_service.retrieve(
                query=question,
                search_mode="auto",
            )
            documents = artifact.get("documents", [])
            doc_rank, chunk_rank = _rank_documents(
                documents,
                target_doc_id,
                target_chunk_ids,
            )
            detail.update(
                {
                    "doc_hit_rank": doc_rank,
                    "context_rank": chunk_rank,
                    "doc_in_results": doc_rank is not None,
                    "context_hit": chunk_rank is not None,
                    "acceptable_chunk_recall_at_3": int(
                        bool(chunk_rank and chunk_rank <= 3)
                    ),
                    "acceptable_chunk_recall_at_5": int(
                        bool(chunk_rank and chunk_rank <= 5)
                    ),
                    "acceptable_chunk_reciprocal_rank": (
                        1.0 / chunk_rank if chunk_rank else 0.0
                    ),
                    "retrieved_count": len(documents),
                    "retrieved_chunk_ids": ";".join(
                        document.get("chunk_id", "") for document in documents
                    ),
                    "retrieved_contexts_json": json.dumps(
                        [document.get("content", "") for document in documents],
                        ensure_ascii=False,
                    ),
                    "artifact_json": json.dumps(artifact, ensure_ascii=False),
                    "rerank_status": artifact.get("rerank_status"),
                    "rerank_reason": artifact.get("rerank_reason"),
                    "threshold_fallback": artifact.get("threshold_fallback"),
                }
            )
        except Exception as exc:
            detail["error"] = str(exc)
        details.append(detail)
        await asyncio.sleep(0.2)

    question_count = len(rows)
    errors = sum(bool(row.get("error")) for row in details)
    empty_results = sum(not row.get("retrieved_count") for row in details)
    doc_ranks = [int(row["doc_hit_rank"]) for row in details if row.get("doc_hit_rank")]
    chunk_ranks = [int(row["context_rank"]) for row in details if row.get("context_rank")]
    recall_at_3 = sum(rank <= 3 for rank in chunk_ranks) / question_count
    recall_at_5 = sum(rank <= 5 for rank in chunk_ranks) / question_count
    mrr = sum(1.0 / rank for rank in chunk_ranks) / question_count
    completion = {
        "status": (
            "valid"
            if len(details) == question_count and errors == 0 and empty_results == 0
            else "invalid"
        ),
        "expected_question_count": question_count,
        "detail_count": len(details),
        "error_count": errors,
        "empty_result_count": empty_results,
    }
    summary = {
        "run_id": datetime.now(CST).strftime("DIRECT_%Y%m%d_%H%M%S"),
        "ran_at": datetime.now(CST).isoformat(),
        "evaluation_variant": "Raw reviewed question -> Full retrieval (Agent bypassed)",
        "review_contract": review_contract,
        "completion": completion,
        "id_based_metrics": {
            "Doc-Hit": f"{len(doc_ranks)}/{question_count} = {len(doc_ranks)/question_count:.1%}",
            "Doc-Hit@1": round(sum(rank <= 1 for rank in doc_ranks) / question_count, 4),
            "Doc-Hit@3": round(sum(rank <= 3 for rank in doc_ranks) / question_count, 4),
            "Doc-Hit@5": round(sum(rank <= 5 for rank in doc_ranks) / question_count, 4),
            "Doc_mean_rank": round(statistics.mean(doc_ranks), 2) if doc_ranks else None,
            "Chunk-Hit": f"{len(chunk_ranks)}/{question_count}",
            "Acceptable-Chunk-Recall@3": round(recall_at_3, 4),
            "Acceptable-Chunk-Recall@5": round(recall_at_5, 4),
            "Acceptable-Chunk-MRR": round(mrr, 4),
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(summary, str(destination / "direct_retrieval_summary.json"))
    write_csv(details, str(destination / "direct_retrieval_details.csv"))
    return summary


async def _run_with_read_only_runtime(args: argparse.Namespace) -> dict[str, Any]:
    try:
        milvus_manager.connect(allow_collection_mutation=False)
        vector_store_manager.initialize()
        return await run_ablation(args.csv, args.output, args.limit)
    finally:
        vector_store_manager.shutdown()
        milvus_manager.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    summary = asyncio.run(_run_with_read_only_runtime(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["completion"]["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
