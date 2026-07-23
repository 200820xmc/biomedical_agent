"""Full全链路正式评测：v2多Gold、Recall/MRR与Ragas。

指标：
- ID-based Hit@K / MRR：标注 chunk 在检索结果中的命中率和排名
- Faithfulness（RAGAS）：答案是否忠实于检索上下文
- Context Recall（RAGAS）：参考答案事实是否被实际上下文覆盖

用法：
    python evaluation/evaluate_review.py
"""

import asyncio
import csv
import json
import math
import sys
import time
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Monkey-patch: RAGAS 0.4.3 缺 langchain_community.chat_models.vertexai ──
import types as _types
_fake_vertexai = _types.ModuleType("langchain_community.chat_models.vertexai")
class _FakeVertexAI: pass
_fake_vertexai.ChatVertexAI = _FakeVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _fake_vertexai

# ── 项目路径 ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os as _os
_os.environ.setdefault("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

from loguru import logger
from app.config import config
from app.services.vector_embedding_service import get_vector_embedding_service
from evaluation.common import write_json, write_csv
from evaluation.formal_eval_contract import (
    DEFAULT_REVIEW_CSV,
    build_completion_status,
    load_formal_review_rows,
)

CST = timezone(timedelta(hours=8))

# RAGAS
from ragas import evaluate as ragas_evaluate
from ragas.metrics import context_recall, faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from datasets import Dataset as HFDataset


async def run_review_eval(
    review_csv: str,
    output_dir: str = "",
    limit: int = 0,
):
    # ── 加载评测集 ──────────────────────────────────
    rows, review_contract = load_formal_review_rows(review_csv, limit=limit)
    references_reviewed = True

    logger.info(f"加载 {len(rows)} 道 review 题目")

    # ── 评测 LLM ────────────────────────────────────
    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        model=config.dashscope_model,
        temperature=0.0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=config.dashscope_api_key,
    ))

    # ── 逐题评测 ────────────────────────────────────
    details = []
    id_precision_hits = 0
    id_precision_ranks = []
    ragas_data = []
    ragas_qids = []

    for i, row in enumerate(rows):
        qid = row["question_id"]
        question = row["question"]
        target_doc_id = row["document_id"].strip()
        raw_target_ids = (
            row.get("acceptable_chunk_ids")
            or row.get("actual_chunk_ids")
            or row.get("context_id")
            or ""
        )
        target_chunk_ids = {
            value.strip()
            for value in raw_target_ids.split(";")
            if value.strip()
        }

        logger.info(f"[{i+1}/{len(rows)}] {qid}")

        result = {
            "question_id": qid,
            "question": question[:120],
            "target_doc_id": target_doc_id,
            "target_chunk_ids": ";".join(sorted(target_chunk_ids)),
        }

        try:
            # ── 1. 执行一次 Agent，并读取同一次调用的检索轨迹 ──
            from app.services.rag_agent_service import rag_agent_service
            trace = await rag_agent_service.query_with_trace(
                question=question,
                session_id=f"review-{qid}",
            )
            answer = trace.get("answer", "")
            retrieval_artifacts = trace.get("retrieval_artifacts", [])
            artifact = retrieval_artifacts[-1] if retrieval_artifacts else {}
            result["tool_call_count"] = trace.get("tool_call_count", 0)
            if result["tool_call_count"] != 1:
                raise ValueError(
                    f"正式Full要求每题恰好一次检索，实际为{result['tool_call_count']}次"
                )

            # ── 2. ID-based Hit@K ─────────────────
            # 检查标注 chunk 是否在召回候选 + 最终选择中
            artifact_documents = (
                artifact.get("documents", [])
                if isinstance(artifact, dict)
                else getattr(artifact, "documents", [])
            )
            retrieved_chunks = []
            for idx, document in enumerate(artifact_documents):
                if isinstance(document, dict):
                    retrieved_chunks.append({
                        "chunk_id": document.get("chunk_id", ""),
                        "source_id": (
                            document.get("source_id")
                            or document.get("source", "")
                        ),
                        "rank": idx + 1,
                        "rerank_score": document.get("rerank_score"),
                        "content": document.get("content", ""),
                        "selected": True,
                    })
                else:
                    retrieved_chunks.append({
                        "chunk_id": document.chunk_id,
                        "source_id": document.source_id or document.source,
                        "rank": idx + 1,
                        "rerank_score": document.rerank_score,
                        "content": document.content,
                        "selected": True,
                    })

            # ID-based 匹配：
            # 1. Document级别：目标 document_id 是否出现在检索结果中
            # 2. Chunk级别：目标 chunk 的 content_hash 是否匹配
            hit_in_selected = False
            hit_rank = None
            hit_doc = False

            for ch in retrieved_chunks:
                # Document 级别匹配
                ch_source = ch.get("source_id", "")
                if target_doc_id and target_doc_id == ch_source:
                    if not hit_doc:
                        hit_doc = True
                        result["doc_hit_rank"] = ch["rank"]
                # Chunk 级别：使用与运行时一致的完整稳定逻辑 ID 精确匹配。
                if ch["chunk_id"] in target_chunk_ids:
                    hit_in_selected = True
                    hit_rank = ch["rank"]
                    break

            result["doc_in_results"] = hit_doc

            result["context_hit"] = hit_in_selected
            result["context_rank"] = hit_rank
            result["retrieved_count"] = len(retrieved_chunks)
            result["retrieved_chunk_ids"] = ";".join(
                chunk["chunk_id"] for chunk in retrieved_chunks
            )
            result["retrieved_sources"] = ";".join(
                chunk["source_id"] for chunk in retrieved_chunks
            )
            result["retrieved_previews"] = " || ".join(
                chunk["content"][:180].replace("\n", " ")
                for chunk in retrieved_chunks
            )
            result["artifact_confidence"] = (
                artifact.get("confidence", "?")
                if isinstance(artifact, dict)
                else getattr(artifact, "confidence", "?")
            )
            result["artifact_json"] = json.dumps(artifact, ensure_ascii=False)

            if hit_in_selected:
                id_precision_hits += 1
                id_precision_ranks.append(hit_rank)

            # ── 3. 保存同一次 Agent 调用生成的答案 ──
            result["answer"] = answer if answer else "（无回答）"
            result["answer_chars"] = len(answer) if answer else 0

            # ── 4. RAGAS Faithfulness + Answer Relevancy ──
            contexts_list = [
                chunk["content"]
                for chunk in retrieved_chunks
                if chunk.get("content")
            ]
            result["retrieved_contexts_json"] = json.dumps(
                contexts_list, ensure_ascii=False
            )

            if answer and contexts_list:
                reference = (
                    row.get("reference")
                    or row.get("reference_candidate")
                    or ""
                ).strip()
                ragas_data.append({
                    "question": question,
                    "answer": answer,
                    "contexts": contexts_list,
                    "ground_truth": reference if references_reviewed else "",
                })
                ragas_qids.append(qid)
                result["ragas_queued"] = True
            else:
                result["ragas_queued"] = False

        except Exception as e:
            result["error"] = str(e)[:200]
            logger.error(f"  {qid} 失败: {e}")

        details.append(result)
        await asyncio.sleep(0.5)

    # ── RAGAS 批量计算 ────────────────────────────
    ragas_scores = {}
    if ragas_data:
        logger.info(f"RAGAS 计算: {len(ragas_data)} 条...")
        hf_ds = HFDataset.from_list(ragas_data)
        try:
            metrics = [faithfulness]
            if references_reviewed:
                metrics.append(context_recall)
            ragas_result = ragas_evaluate(
                dataset=hf_ds,
                metrics=metrics,
                llm=eval_llm,
                embeddings=get_vector_embedding_service(),
            )
            detail_by_qid = {row["question_id"]: row for row in details}
            for metric in [item.name for item in metrics]:
                raw_values = list(ragas_result[metric])
                for qid, value in zip(ragas_qids, raw_values, strict=True):
                    numeric = float(value) if value is not None else None
                    detail_by_qid[qid][f"ragas_{metric}"] = (
                        numeric
                        if numeric is not None and math.isfinite(numeric)
                        else None
                    )
                values = [
                    float(value)
                    for value in raw_values
                    if value is not None and math.isfinite(float(value))
                ]
                if values:
                    ragas_scores[metric] = {
                        "mean": round(sum(values) / len(values), 4),
                        "min": round(min(values), 4),
                        "max": round(max(values), 4),
                        "count": len(values),
                    }
        except Exception as e:
            ragas_scores["error"] = str(e)

    # ── 汇总 ─────────────────────────────────────
    # Document-level precision
    doc_hits = sum(1 for r in details if r.get("doc_in_results"))
    doc_hit_ranks = [r["doc_hit_rank"] for r in details if r.get("doc_hit_rank")]
    doc_precision_at_k = {}
    for k in [1, 3, 5, 10]:
        hits = sum(1 for r in details if r.get("doc_hit_rank") and r["doc_hit_rank"] <= k)
        doc_precision_at_k[f"Doc-Hit@{k}"] = round(hits / len(details), 4) if details else 0

    # Chunk-level precision
    chunk_hits = sum(1 for r in details if r.get("context_hit"))
    chunk_hit_ranks = [r["context_rank"] for r in details if r.get("context_rank")]
    chunk_recall_at_3 = sum(rank <= 3 for rank in chunk_hit_ranks) / len(rows)
    chunk_recall_at_5 = sum(rank <= 5 for rank in chunk_hit_ranks) / len(rows)
    chunk_mrr = sum(1.0 / rank for rank in chunk_hit_ranks) / len(rows)
    for row in details:
        rank = row.get("context_rank")
        row["acceptable_chunk_recall_at_3"] = int(bool(rank and rank <= 3))
        row["acceptable_chunk_recall_at_5"] = int(bool(rank and rank <= 5))
        row["acceptable_chunk_reciprocal_rank"] = 1.0 / rank if rank else 0.0

    completion = build_completion_status(
        details,
        expected_count=len(rows),
        required_ragas_metrics=("faithfulness", "context_recall"),
    )

    run_id = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
    summary = {
        "run_id": run_id,
        "ran_at": datetime.now(CST).isoformat(),
        "review_csv": review_csv,
        "question_count": len(rows),
        "evaluation_variant": "Full Top-20 / Rerank Top-10 / final Top-5",
        "review_contract": review_contract,
        "completion": completion,
        "id_based_metrics": {
            "Doc-Hit": f"{doc_hits}/{len(rows)} = {doc_hits/len(rows):.1%}",
            **doc_precision_at_k,
            "Doc_mean_rank": round(sum(doc_hit_ranks) / len(doc_hit_ranks), 1) if doc_hit_ranks else None,
            "Chunk-Hit": f"{chunk_hits}/{len(rows)}",
            "Acceptable-Chunk-Recall@3": round(chunk_recall_at_3, 4),
            "Acceptable-Chunk-Recall@5": round(chunk_recall_at_5, 4),
            "Acceptable-Chunk-MRR": round(chunk_mrr, 4),
            "Chunk_mean_rank": round(sum(chunk_hit_ranks) / len(chunk_hit_ranks), 1) if chunk_hit_ranks else None,
        },
        "ragas_metrics": ragas_scores,
        "notes": [
            "ID-based Hit@K: 标注的真实 Milvus 逻辑 chunk ID 是否在检索结果中",
            "Recall@K: 任一可接受Chunk出现在前K条即记1，否则记0",
            "MRR: 每题按1/首个可接受Chunk名次计分，未命中为0，再取平均",
            "Faithfulness: RAGAS LLM评判",
            "Context Recall: 仅在全部参考答案通过人工审核后启用",
        ],
    }

    if not output_dir:
        output_dir = str(Path(__file__).resolve().parent / "results" / run_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_json(summary, str(Path(output_dir) / "review_eval_summary.json"))
    write_csv(details, str(Path(output_dir) / "review_eval_details.csv"))

    print("\n" + "=" * 60)
    print("ID-based RAG 评测结果")
    print("=" * 60)
    print(f"  Doc-Hit (文档命中):      {doc_hits}/{len(rows)} = {doc_hits/len(rows):.1%}")
    if doc_hit_ranks:
        print(f"  Doc 平均排名:             {sum(doc_hit_ranks)/len(doc_hit_ranks):.1f}")
    print(f"  Chunk-Hit (chunk命中):    {chunk_hits}/{len(rows)}")
    if chunk_hit_ranks:
        print(f"  Chunk 平均排名:           {sum(chunk_hit_ranks)/len(chunk_hit_ranks):.1f}")
    for metric, info in ragas_scores.items():
        if isinstance(info, dict) and "mean" in info:
            print(f"  {metric:25s}: {info['mean']:.4f}")
        elif isinstance(info, str):
            print(f"  {metric:25s}: {info}")
    print(f"\n结果: {output_dir}")
    print("=" * 60)

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_REVIEW_CSV),
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    from app.core.milvus_client import milvus_manager
    from app.services.rag_agent_service import rag_agent_service
    from app.services.vector_store_manager import vector_store_manager

    async def _run_with_runtime():
        try:
            milvus_manager.connect(allow_collection_mutation=False)
            vector_store_manager.initialize()
            rag_agent_service.initialize()
            return await run_review_eval(args.csv, args.output, args.limit)
        finally:
            await rag_agent_service.cleanup()
            vector_store_manager.shutdown()
            milvus_manager.close()

    summary = asyncio.run(_run_with_runtime())
    if summary["completion"]["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
