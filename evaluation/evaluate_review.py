"""基于真实 Milvus chunk ID 映射的 RAG 评测

指标：
- ID-based Hit@K / MRR：标注 chunk 在检索结果中的命中率和排名
- Faithfulness（RAGAS）：答案是否忠实于检索上下文
- Response Relevancy（RAGAS）：答案与问题的相关度

暂不启用（等人工审核后）：
- Context Recall
- 答案正确性

用法：
    python evaluation/evaluate_review.py
"""

import asyncio
import csv
import json
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
from app.services.vector_embedding_service import vector_embedding_service
from evaluation.common import write_json, write_csv

CST = timezone(timedelta(hours=8))

# RAGAS
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from datasets import Dataset as HFDataset


async def run_review_eval(
    review_csv: str,
    output_dir: str = "",
    limit: int = 0,
):
    # ── 加载评测集 ──────────────────────────────────
    rows = []
    with open(review_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if limit > 0:
        rows = rows[:limit]

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

    for i, row in enumerate(rows):
        qid = row["question_id"]
        question = row["question"]
        target_doc_id = row["document_id"].strip()
        raw_target_ids = (
            row.get("actual_chunk_ids")
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
                if target_doc_id and target_doc_id in ch_source:
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

            if hit_in_selected:
                id_precision_hits += 1
                id_precision_ranks.append(hit_rank)

            # ── 3. 保存同一次 Agent 调用生成的答案 ──
            result["answer"] = answer[:500] if answer else "（无回答）"
            result["answer_chars"] = len(answer) if answer else 0

            # ── 4. RAGAS Faithfulness + Answer Relevancy ──
            contexts_list = [
                chunk["content"]
                for chunk in retrieved_chunks
                if chunk.get("content")
            ]

            if answer and contexts_list:
                ragas_data.append({
                    "question": question,
                    "answer": answer,
                    "contexts": contexts_list,
                    "ground_truth": "",
                })
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
            ragas_result = ragas_evaluate(
                dataset=hf_ds,
                metrics=[faithfulness, answer_relevancy],
                llm=eval_llm,
                embeddings=vector_embedding_service,
            )
            for metric in ["faithfulness", "answer_relevancy"]:
                values = [v for v in ragas_result[metric] if v is not None]
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

    run_id = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
    summary = {
        "run_id": run_id,
        "ran_at": datetime.now(CST).isoformat(),
        "review_csv": review_csv,
        "question_count": len(rows),
        "id_based_metrics": {
            "Doc-Hit": f"{doc_hits}/{len(rows)} = {doc_hits/len(rows):.1%}",
            **doc_precision_at_k,
            "Doc_mean_rank": round(sum(doc_hit_ranks) / len(doc_hit_ranks), 1) if doc_hit_ranks else None,
            "Chunk-Hit": f"{chunk_hits}/{len(rows)}",
            "Chunk_mean_rank": round(sum(chunk_hit_ranks) / len(chunk_hit_ranks), 1) if chunk_hit_ranks else None,
        },
        "ragas_metrics": ragas_scores,
        "notes": [
            "ID-based Hit@K: 标注的真实 Milvus 逻辑 chunk ID 是否在检索结果中",
            "Faithfulness: RAGAS LLM评判",
            "Response Relevancy: RAGAS LLM评判",
            "Context Recall: 待人工审核参考答案后启用",
            "答案正确性: 待人工审核后启用",
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
        default="evaluation/ragas_50_actual_chunk_review.csv",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run_review_eval(args.csv, args.output, args.limit))


if __name__ == "__main__":
    main()
