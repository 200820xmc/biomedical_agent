"""Ablation: raw question + saved Agent query, fused by RRF before rerank."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from http import HTTPStatus
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.retrieval.context_builder import ContextBuilder
from app.services.retrieval.recall_service import RecallService
from app.services.retrieval.rerank_service import RerankService
from app.services.retrieval.retrieval_models import RetrievalItem
from app.services.retrieval.retrieval_models import RerankResult
from app.services.vector_store_manager import vector_store_manager
from evaluation.common import write_csv, write_json
from evaluation.evaluate_direct_retrieval_ablation import _rank_documents
from evaluation.formal_eval_contract import DEFAULT_REVIEW_CSV, load_formal_review_rows


CST = timezone(timedelta(hours=8))
RRF_K = 60
PER_QUERY_K = 20
FUSED_K = 20
RERANK_K = 10
FINAL_K = 5
DASHSCOPE_RERANK_TIMEOUT_SECONDS = 30


class DashScopeRerankService:
    """Evaluation-only adapter for DashScope's dedicated text reranker."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalItem],
        top_k: int | None = None,
    ) -> RerankResult:
        if not candidates:
            return RerankResult(items=[], status="skipped", reason="no_candidates")

        k = top_k or RERANK_K
        if len(candidates) <= k:
            return self._fallback(
                candidates,
                k,
                status="skipped",
                reason="candidate_count_not_above_rerank_k",
            )

        try:
            from dashscope import AioTextReRank

            response = await asyncio.wait_for(
                AioTextReRank.call(
                    model=self.model,
                    query=query,
                    documents=[item.content for item in candidates],
                    top_n=k,
                    return_documents=False,
                    api_key=config.dashscope_api_key,
                ),
                timeout=DASHSCOPE_RERANK_TIMEOUT_SECONDS,
            )
            if response.status_code != HTTPStatus.OK:
                reason = f"dashscope_http_{response.status_code}:{response.code or 'unknown'}"
                return self._fallback(
                    candidates,
                    k,
                    status="degraded",
                    reason=reason,
                    degraded=True,
                )

            results = list(response.output.results or [])
            if not results:
                return self._fallback(
                    candidates,
                    k,
                    status="degraded",
                    reason="dashscope_empty_results",
                    degraded=True,
                )

            ranked: list[RetrievalItem] = []
            seen: set[int] = set()
            for result in results:
                index = int(result.index)
                if index in seen or not 0 <= index < len(candidates):
                    continue
                seen.add(index)
                item = candidates[index]
                item.rerank_score = float(result.relevance_score)
                item.metadata["rerank_status"] = "dashscope_dedicated"
                item.metadata["rerank_model"] = self.model
                ranked.append(item)
            if len(ranked) != min(k, len(candidates)):
                return self._fallback(
                    candidates,
                    k,
                    status="degraded",
                    reason="dashscope_partial_results",
                    degraded=True,
                )
            ranked.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
            return RerankResult(items=ranked[:k], status="applied", applied=True)
        except asyncio.TimeoutError:
            return self._fallback(
                candidates,
                k,
                status="degraded",
                reason="dashscope_timeout",
                degraded=True,
            )
        except Exception as exc:
            return self._fallback(
                candidates,
                k,
                status="degraded",
                reason=f"dashscope_exception:{type(exc).__name__}",
                degraded=True,
            )

    @staticmethod
    def _fallback(
        candidates: list[RetrievalItem],
        k: int,
        *,
        status: str,
        reason: str,
        degraded: bool = False,
    ) -> RerankResult:
        for item in candidates:
            item.rerank_score = None
            item.metadata["rerank_status"] = "rrf_fallback"
        return RerankResult(
            items=sorted(
                candidates,
                key=lambda item: item.vector_score or 0.0,
                reverse=True,
            )[:k],
            status=status,
            applied=False,
            degraded=degraded,
            reason=reason,
        )


def _candidate_key(item: RetrievalItem) -> str:
    return item.chunk_id or f"content:{item.content}"


def _rrf_fuse(
    raw_items: list[RetrievalItem],
    agent_items: list[RetrievalItem],
    *,
    rrf_k: int = RRF_K,
    fused_k: int = FUSED_K,
) -> list[RetrievalItem]:
    """Fuse exact logical chunks and retain provenance for audit/fallback."""
    entries: dict[str, dict[str, Any]] = {}
    for route, items in (("raw", raw_items), ("agent", agent_items)):
        seen_route: set[str] = set()
        for rank, item in enumerate(items, start=1):
            key = _candidate_key(item)
            if key in seen_route:
                continue
            seen_route.add(key)
            entry = entries.setdefault(
                key,
                {
                    "item": item,
                    "rrf_score": 0.0,
                    "best_rank": rank,
                    "raw_rank": None,
                    "agent_rank": None,
                    "raw_vector_score": None,
                    "agent_vector_score": None,
                },
            )
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
            entry[f"{route}_rank"] = rank
            entry[f"{route}_vector_score"] = item.vector_score

    ordered = sorted(
        entries.values(),
        key=lambda value: (
            value["rrf_score"],
            -value["best_rank"],
            value["item"].vector_score or 0.0,
        ),
        reverse=True,
    )[:fused_k]
    fused: list[RetrievalItem] = []
    for value in ordered:
        item = value["item"]
        item.metadata = dict(item.metadata)
        item.metadata.update(
            {
                "fusion": "rrf",
                "rrf_k": rrf_k,
                "rrf_score": value["rrf_score"],
                "raw_rank": value["raw_rank"],
                "agent_rank": value["agent_rank"],
                "raw_vector_score": value["raw_vector_score"],
                "agent_vector_score": value["agent_vector_score"],
            }
        )
        item.vector_score = value["rrf_score"]
        item.rerank_score = None
        fused.append(item)
    return fused


def _load_agent_queries(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, str] = {}
    for row in rows:
        artifact = json.loads(row["artifact_json"])
        query = str(artifact.get("original_query", "")).strip()
        if query:
            result[row["question_id"]] = query
    return result


async def _retrieve_dual(
    raw_query: str,
    agent_query: str,
    rerank_query: str,
    recall: RecallService,
    rerank: RerankService,
    builder: ContextBuilder,
    recall_mode: str = "dual",
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    raw_items = (
        recall.recall(raw_query, candidate_count=PER_QUERY_K)
        if recall_mode in {"dual", "raw"}
        else []
    )
    if recall_mode not in {"dual", "agent"}:
        agent_items = []
    elif recall_mode == "dual" and agent_query.strip() == raw_query.strip():
        agent_items = raw_items
    else:
        agent_items = recall.recall(agent_query, candidate_count=PER_QUERY_K)
    fused = _rrf_fuse(raw_items, agent_items)
    rerank_result = await rerank.rerank(rerank_query, fused, top_k=RERANK_K)
    reranked = rerank_result.items

    threshold = getattr(config, "rag_rerank_threshold", 0.65)
    threshold_applied = rerank_result.applied
    threshold_fallback = False
    if threshold_applied:
        filtered = [
            item
            for item in reranked
            if item.rerank_score is not None and item.rerank_score >= threshold
        ]
        if not filtered and reranked:
            fallback_k = min(
                max(1, getattr(config, "rag_threshold_fallback_k", 3)),
                FINAL_K,
                len(reranked),
            )
            filtered = reranked[:fallback_k]
            threshold_fallback = True
    else:
        filtered = reranked

    anchors = filtered[:FINAL_K]
    expanded = recall.expand_neighbors(anchors, window=1, top_n=3)
    _context, artifact = builder.build(
        items=expanded,
        original_query=raw_query,
        search_mode="dual_query_rrf",
        candidate_count=len(fused),
        reranked_count=len(reranked),
        rerank_applied=rerank_result.applied,
        rerank_status=rerank_result.status,
        rerank_degraded=rerank_result.degraded,
        rerank_reason=rerank_result.reason,
        threshold_applied=threshold_applied,
        threshold_fallback=threshold_fallback,
    )
    artifact.duration_ms = {"total": (time.time() - started) * 1000}
    trace = {
        "raw_candidate_count": len(raw_items),
        "agent_candidate_count": len(agent_items),
        "union_count": len({_candidate_key(item) for item in raw_items + agent_items}),
        "fused_candidate_count": len(fused),
        "fused_overlap_count": sum(
            item.metadata.get("raw_rank") is not None
            and item.metadata.get("agent_rank") is not None
            for item in fused
        ),
        "anchor_count": len(anchors),
        "expanded_count": len(expanded),
    }
    return asdict(artifact), trace


async def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    rows, review_contract = load_formal_review_rows(args.csv, limit=args.limit)
    if args.question_ids:
        requested = {
            value.strip() for value in args.question_ids.split(",") if value.strip()
        }
        rows = [row for row in rows if row["question_id"] in requested]
        found = {row["question_id"] for row in rows}
        if found != requested:
            raise ValueError(f"unknown or unavailable question_ids: {sorted(requested - found)}")
        review_contract["selected_question_count"] = len(rows)
    agent_queries = _load_agent_queries(Path(args.agent_details))
    missing_queries = [row["question_id"] for row in rows if row["question_id"] not in agent_queries]
    if missing_queries:
        raise ValueError(f"missing saved Agent queries: {missing_queries}")

    recall = RecallService()
    rerank = (
        DashScopeRerankService(args.dashscope_model)
        if args.reranker_backend == "dashscope"
        else RerankService()
    )
    builder = ContextBuilder()
    details: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        qid = row["question_id"]
        raw_query = row["question"]
        agent_query = agent_queries[qid]
        if args.rerank_query_mode == "raw":
            rerank_query = raw_query
        elif args.rerank_query_mode == "agent":
            rerank_query = agent_query
        else:
            rerank_query = f"{raw_query}\n核心检索词：{agent_query}"
        target_doc_id = row["document_id"].strip()
        target_chunk_ids = {
            value.strip()
            for value in (row.get("acceptable_chunk_ids") or "").split(";")
            if value.strip()
        }
        print(f"[DUAL-RRF {index}/{len(rows)}] {qid}: {raw_query[:55]}")
        detail: dict[str, Any] = {
            "question_id": qid,
            "question": raw_query,
            "agent_query": agent_query,
            "target_doc_id": target_doc_id,
            "target_chunk_ids": ";".join(sorted(target_chunk_ids)),
        }
        try:
            artifact, fusion_trace = await _retrieve_dual(
                raw_query,
                agent_query,
                rerank_query,
                recall,
                rerank,
                builder,
                args.recall_mode,
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
                    "acceptable_chunk_recall_at_3": int(bool(chunk_rank and chunk_rank <= 3)),
                    "acceptable_chunk_recall_at_5": int(bool(chunk_rank and chunk_rank <= 5)),
                    "acceptable_chunk_reciprocal_rank": 1.0 / chunk_rank if chunk_rank else 0.0,
                    "retrieved_count": len(documents),
                    "retrieved_chunk_ids": ";".join(
                        document.get("chunk_id", "") for document in documents
                    ),
                    "artifact_json": json.dumps(artifact, ensure_ascii=False),
                    "fusion_trace_json": json.dumps(fusion_trace, ensure_ascii=False),
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
    completion = {
        "status": "valid" if errors == 0 and empty_results == 0 and len(details) == question_count else "invalid",
        "expected_question_count": question_count,
        "detail_count": len(details),
        "error_count": errors,
        "empty_result_count": empty_results,
    }
    summary = {
        "run_id": datetime.now(CST).strftime("DUAL_RRF_%Y%m%d_%H%M%S"),
        "ran_at": datetime.now(CST).isoformat(),
        "evaluation_variant": (
            f"{args.recall_mode} recall -> RRF@20 -> "
            f"{args.reranker_backend} rerank"
        ),
        "parameters": {
            "rrf_k": RRF_K,
            "per_query_k": PER_QUERY_K,
            "fused_k": FUSED_K,
            "rerank_k": RERANK_K,
            "final_k": FINAL_K,
            "rerank_query_mode": args.rerank_query_mode,
            "recall_mode": args.recall_mode,
            "reranker_backend": args.reranker_backend,
            "dashscope_model": (
                args.dashscope_model if args.reranker_backend == "dashscope" else None
            ),
        },
        "review_contract": review_contract,
        "agent_query_source": str(Path(args.agent_details).resolve()),
        "completion": completion,
        "id_based_metrics": {
            "Doc-Hit": f"{len(doc_ranks)}/{question_count} = {len(doc_ranks)/question_count:.1%}",
            "Doc-Hit@1": round(sum(rank <= 1 for rank in doc_ranks) / question_count, 4),
            "Doc-Hit@3": round(sum(rank <= 3 for rank in doc_ranks) / question_count, 4),
            "Doc-Hit@5": round(sum(rank <= 5 for rank in doc_ranks) / question_count, 4),
            "Doc_mean_rank": round(statistics.mean(doc_ranks), 2) if doc_ranks else None,
            "Chunk-Hit": f"{len(chunk_ranks)}/{question_count}",
            "Acceptable-Chunk-Recall@3": round(sum(rank <= 3 for rank in chunk_ranks) / question_count, 4),
            "Acceptable-Chunk-Recall@5": round(sum(rank <= 5 for rank in chunk_ranks) / question_count, 4),
            "Acceptable-Chunk-MRR": round(sum(1.0 / rank for rank in chunk_ranks) / question_count, 4),
        },
    }
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(summary, str(destination / "dual_rrf_summary.json"))
    write_csv(details, str(destination / "dual_rrf_details.csv"))
    return summary


async def _run_with_read_only_runtime(args: argparse.Namespace) -> dict[str, Any]:
    try:
        milvus_manager.connect(allow_collection_mutation=False)
        vector_store_manager.initialize()
        return await run_ablation(args)
    finally:
        vector_store_manager.shutdown()
        milvus_manager.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument(
        "--agent-details",
        default=str(
            ROOT
            / "evaluation"
            / "results"
            / "FULL_V2_20260723_FORMAL_RAGAS_RETRY"
            / "review_eval_details.csv"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-ids", default="")
    parser.add_argument(
        "--rerank-query-mode",
        choices=("raw", "agent", "combined"),
        default="raw",
    )
    parser.add_argument(
        "--recall-mode",
        choices=("dual", "raw", "agent"),
        default="dual",
    )
    parser.add_argument(
        "--reranker-backend",
        choices=("llm", "dashscope"),
        default="llm",
    )
    parser.add_argument("--dashscope-model", default="gte-rerank-v2")
    args = parser.parse_args()
    summary = asyncio.run(_run_with_read_only_runtime(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["completion"]["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
