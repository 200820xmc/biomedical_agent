"""BL-1 基线评测：Dense Top-5，不使用全链路增强模块。

BL-1 的项目内定义：

    中文问题
      → Milvus Dense 向量检索
      → 按向量距离排序并删除完全重复 chunk
      → 保留前 5 个唯一 chunk
      → 使用与全链路相同的 ContextBuilder、Agent 模型和系统提示词
      → Ragas Faithfulness + Context Recall

明确关闭：

- LLM Rerank
- Rerank 阈值过滤
- 来源多样性选择
- 相邻 chunk 扩展
- Query Rewrite / Multi Query

说明：

Milvus 当前存在完全相同 chunk 的重复 UUID 行。为获得“前 5 个唯一 Dense
结果”，底层最多读取前 15 个原始结果，但只删除正文完全相同的重复项，不进行
语义扩召、重排或来源干预。最终进入模型的仍是 Dense 排序前 5 个唯一 chunk。

运行：

    # 先跑 3 题冒烟测试
    python evaluation/evaluate_bl1.py --limit 3

    # 全链路 50 题结束后再运行完整 BL-1
    python evaluation/evaluate_bl1.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# Ragas 0.4.3 在当前环境导入旧 VertexAI 兼容模块时会失败。
_fake_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")


class _FakeVertexAI:
    pass


_fake_vertexai.ChatVertexAI = _FakeVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _fake_vertexai


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "DASHSCOPE_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

from datasets import Dataset as HFDataset
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from loguru import logger
from ragas import evaluate as ragas_evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_recall, faithfulness

from app.config import config
from app.services.rag_agent_service import rag_agent_service
from app.services.retrieval.context_builder import ContextBuilder
from app.services.retrieval.recall_service import RecallService
from app.services.vector_embedding_service import get_vector_embedding_service
from evaluation.common import write_csv, write_json
from evaluation.formal_eval_contract import (
    DEFAULT_REVIEW_CSV,
    build_completion_status,
    load_formal_review_rows,
)


CST = timezone(timedelta(hours=8))
BL1_FINAL_K = 5
BL1_RAW_K = 15


_recall_service = RecallService()
_context_builder = ContextBuilder()


def _dedupe_preserve_dense_order(items: list) -> list:
    """只去除完全相同的逻辑 chunk ID，保持原始 Dense 排名。"""
    seen_ids: set[str] = set()
    seen_contents: set[str] = set()
    unique = []
    for item in items:
        if item.chunk_id in seen_ids or item.content in seen_contents:
            continue
        if item.chunk_id:
            seen_ids.add(item.chunk_id)
        seen_contents.add(item.content)
        unique.append(item)
    return unique


def _artifact_to_dict(
    *,
    query: str,
    raw_count: int,
    unique_count: int,
    documents: list,
    duration_ms: float,
) -> dict[str, Any]:
    return {
        "pipeline": "BL-1",
        "definition": "Milvus Dense Top-5 unique chunks",
        "original_query": query,
        "search_mode": "bl1_dense_top5",
        "candidate_count": raw_count,
        "unique_candidate_count": unique_count,
        "reranked_count": 0,
        "selected_count": len(documents),
        "confidence": "baseline",
        "rerank_applied": False,
        "diversity_applied": False,
        "neighbor_expansion_applied": False,
        "duration_ms": {
            "recall": round(duration_ms, 3),
            "total": round(duration_ms, 3),
        },
        "documents": [
            {
                "chunk_id": item.chunk_id,
                "source_id": item.source_id,
                "source": item.source,
                "chunk_index": item.chunk_index,
                "vector_score": item.vector_score,
                "rerank_score": None,
                "content": item.content,
                "metadata": item.metadata,
            }
            for item in documents
        ],
    }


@tool("retrieve_knowledge", response_format="content_and_artifact")
async def retrieve_knowledge_bl1(
    query: str,
) -> tuple[str, dict[str, Any]]:
    """使用 BL-1 Dense Top-5 基线从知识库检索论文证据。

    该工具仅执行向量相似度检索和完全重复 chunk 删除，不执行 Rerank、
    阈值过滤、来源多样性选择或相邻 chunk 扩展。
    """
    start = time.perf_counter()

    raw_items = _recall_service.recall(
        query=query,
        candidate_count=BL1_RAW_K,
    )
    unique_items = _dedupe_preserve_dense_order(raw_items)
    selected = unique_items[:BL1_FINAL_K]

    context, built_artifact = _context_builder.build(
        items=selected,
        original_query=query,
        search_mode="bl1_dense_top5",
        candidate_count=len(raw_items),
        reranked_count=0,
        rerank_applied=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # ContextBuilder 可能因上下文预算减少最终证据数，artifact 必须只记录
    # 实际交给生成模型的 chunk。
    kept_documents = built_artifact.documents
    artifact = _artifact_to_dict(
        query=query,
        raw_count=len(raw_items),
        unique_count=len(unique_items),
        documents=kept_documents,
        duration_ms=elapsed_ms,
    )
    return context, artifact


def _build_bl1_agent():
    """创建只拥有 BL-1 检索工具的 Agent。"""
    model = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        temperature=0.7,
        streaming=False,
    )
    return create_agent(
        model,
        tools=[retrieve_knowledge_bl1],
    )


def _build_eval_llm() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=config.dashscope_model,
            temperature=0.0,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=config.dashscope_api_key,
        )
    )


def _extract_answer_and_artifact(result: dict) -> tuple[str, dict, int]:
    messages = result.get("messages", [])
    answer = ""
    if messages:
        last_message = messages[-1]
        answer = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

    artifacts: list[dict] = []
    for message in messages:
        artifact = getattr(message, "artifact", None)
        if (
            isinstance(artifact, dict)
            and artifact.get("pipeline") == "BL-1"
        ):
            artifacts.append(artifact)

    return answer, artifacts[-1] if artifacts else {}, len(artifacts)


async def run_bl1_eval(
    review_csv: str,
    output_dir: str = "",
    limit: int = 0,
    skip_ragas: bool = False,
) -> dict[str, Any]:
    rows, review_contract = load_formal_review_rows(review_csv, limit=limit)
    references_reviewed = True

    logger.info(f"BL-1 加载题目: {len(rows)}")
    agent = _build_bl1_agent()
    eval_llm = _build_eval_llm()

    details: list[dict] = []
    ragas_data: list[dict] = []
    ragas_qids: list[str] = []

    for index, row in enumerate(rows, start=1):
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

        print(f"[BL-1 {index}/{len(rows)}] {qid}: {question[:60]}")
        result_row: dict[str, Any] = {
            "question_id": qid,
            "question": question[:120],
            "target_doc_id": target_doc_id,
            "target_chunk_ids": ";".join(sorted(target_chunk_ids)),
        }

        try:
            result = await agent.ainvoke(
                input={
                    "messages": [
                        SystemMessage(content=rag_agent_service.system_prompt),
                        HumanMessage(content=question),
                    ]
                }
            )
            answer, artifact, tool_call_count = _extract_answer_and_artifact(result)
            if tool_call_count != 1:
                raise ValueError(
                    f"正式BL-1要求每题恰好一次检索，实际为{tool_call_count}次"
                )

            documents = artifact.get("documents", [])

            doc_hit_rank = None
            chunk_hit_rank = None
            for rank, document in enumerate(documents, start=1):
                source_id = (
                    document.get("source_id")
                    or document.get("source", "")
                )
                if doc_hit_rank is None and target_doc_id == source_id:
                    doc_hit_rank = rank
                if (
                    chunk_hit_rank is None
                    and document.get("chunk_id") in target_chunk_ids
                ):
                    chunk_hit_rank = rank

            contexts = [
                document.get("content", "")
                for document in documents
                if document.get("content")
            ]

            result_row["tool_call_count"] = tool_call_count
            if doc_hit_rank is not None:
                result_row["doc_hit_rank"] = doc_hit_rank
            result_row["doc_in_results"] = doc_hit_rank is not None
            result_row["context_hit"] = chunk_hit_rank is not None
            result_row["context_rank"] = chunk_hit_rank
            result_row["retrieved_count"] = len(documents)
            result_row["retrieved_chunk_ids"] = ";".join(
                document.get("chunk_id", "")
                for document in documents
            )
            result_row["retrieved_sources"] = ";".join(
                document.get("source_id")
                or document.get("source", "")
                for document in documents
            )
            result_row["retrieved_previews"] = " || ".join(
                document.get("content", "")[:180].replace("\n", " ")
                for document in documents
            )
            result_row["retrieved_contexts_json"] = json.dumps(
                contexts, ensure_ascii=False
            )
            result_row["artifact_confidence"] = artifact.get(
                "confidence", "?"
            )
            result_row["artifact_json"] = json.dumps(
                artifact, ensure_ascii=False
            )
            result_row["answer"] = answer if answer else "（无回答）"
            result_row["answer_chars"] = len(answer) if answer else 0

            if answer and contexts and not skip_ragas:
                reference = (
                    row.get("reference")
                    or row.get("reference_candidate")
                    or ""
                ).strip()
                ragas_data.append(
                    {
                        "question": question,
                        "answer": answer,
                        "contexts": contexts,
                        "ground_truth": reference if references_reviewed else "",
                    }
                )
                ragas_qids.append(qid)
                result_row["ragas_queued"] = True
            else:
                result_row["ragas_queued"] = False

        except Exception as exc:
            result_row["error"] = str(exc)[:200]
            logger.error(f"BL-1 {qid} 失败: {exc}")

        details.append(result_row)
        await asyncio.sleep(0.5)

    ragas_scores: dict[str, Any] = {}
    if ragas_data and not skip_ragas:
        logger.info(f"BL-1 Ragas 评分: {len(ragas_data)} 条")
        try:
            metrics = [faithfulness]
            if references_reviewed:
                metrics.append(context_recall)
            ragas_result = ragas_evaluate(
                dataset=HFDataset.from_list(ragas_data),
                metrics=metrics,
                llm=eval_llm,
                embeddings=get_vector_embedding_service(),
            )
            detail_by_qid = {row["question_id"]: row for row in details}
            for metric_name in [item.name for item in metrics]:
                raw_values = list(ragas_result[metric_name])
                for qid, value in zip(ragas_qids, raw_values, strict=True):
                    numeric = float(value) if value is not None else None
                    detail_by_qid[qid][f"ragas_{metric_name}"] = (
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
                    ragas_scores[metric_name] = {
                        "mean": round(statistics.mean(values), 4),
                        "min": round(min(values), 4),
                        "max": round(max(values), 4),
                        "count": len(values),
                    }
        except Exception as exc:
            ragas_scores["error"] = str(exc)
            logger.error(f"BL-1 Ragas 评分失败: {exc}")

    question_count = len(rows)
    doc_hit_ranks = [
        int(row["doc_hit_rank"])
        for row in details
        if row.get("doc_hit_rank")
    ]
    chunk_hit_ranks = [
        int(row["context_rank"])
        for row in details
        if row.get("context_rank")
    ]
    chunk_recall_at_3 = sum(rank <= 3 for rank in chunk_hit_ranks) / question_count
    chunk_recall_at_5 = sum(rank <= 5 for rank in chunk_hit_ranks) / question_count
    chunk_mrr = sum(1.0 / rank for rank in chunk_hit_ranks) / question_count
    for row in details:
        rank = row.get("context_rank")
        row["acceptable_chunk_recall_at_3"] = int(bool(rank and rank <= 3))
        row["acceptable_chunk_recall_at_5"] = int(bool(rank and rank <= 5))
        row["acceptable_chunk_reciprocal_rank"] = 1.0 / rank if rank else 0.0

    required_ragas = () if skip_ragas else ("faithfulness", "context_recall")
    completion = build_completion_status(
        details,
        expected_count=question_count,
        required_ragas_metrics=required_ragas,
    )
    id_based_metrics: dict[str, Any] = {
        "Doc-Hit": (
            f"{len(doc_hit_ranks)}/{question_count} = "
            f"{len(doc_hit_ranks) / question_count:.1%}"
        ),
        "Doc-Hit@1": round(
            sum(rank <= 1 for rank in doc_hit_ranks) / question_count,
            4,
        ),
        "Doc-Hit@3": round(
            sum(rank <= 3 for rank in doc_hit_ranks) / question_count,
            4,
        ),
        "Doc-Hit@5": round(
            sum(rank <= 5 for rank in doc_hit_ranks) / question_count,
            4,
        ),
        "Doc-Hit@10": round(
            sum(rank <= 10 for rank in doc_hit_ranks) / question_count,
            4,
        ),
        "Doc_mean_rank": (
            round(sum(doc_hit_ranks) / len(doc_hit_ranks), 1)
            if doc_hit_ranks
            else None
        ),
        "Chunk-Hit": f"{len(chunk_hit_ranks)}/{question_count}",
        "Acceptable-Chunk-Recall@3": round(chunk_recall_at_3, 4),
        "Acceptable-Chunk-Recall@5": round(chunk_recall_at_5, 4),
        "Acceptable-Chunk-MRR": round(chunk_mrr, 4),
        "Chunk_mean_rank": (
            round(sum(chunk_hit_ranks) / len(chunk_hit_ranks), 1)
            if chunk_hit_ranks
            else None
        ),
    }

    run_id = datetime.now(CST).strftime("BL1_%Y%m%d_%H%M%S")
    summary = {
        "run_id": run_id,
        "ran_at": datetime.now(CST).isoformat(),
        "review_csv": review_csv,
        "question_count": question_count,
        "evaluation_variant": "BL-1 Dense Top-5 unique chunks",
        "review_contract": review_contract,
        "completion": completion,
        "id_based_metrics": id_based_metrics,
        "ragas_metrics": ragas_scores,
        "notes": [
            "Evaluation variant: BL-1 Dense Top-5 unique chunks",
            "ID-based Hit@K: 标注的真实 Milvus 逻辑 chunk ID 是否在检索结果中",
            "Recall@K: 任一可接受Chunk出现在前K条即记1，否则记0",
            "MRR: 每题按1/首个可接受Chunk名次计分，未命中为0，再取平均",
            "Faithfulness: RAGAS LLM评判",
            "Context Recall: 仅在全部参考答案通过人工审核后启用",
        ],
    }

    if not output_dir:
        output_path = ROOT / "evaluation" / "results" / run_id
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_json(summary, str(output_path / "review_eval_summary.json"))
    write_csv(details, str(output_path / "review_eval_details.csv"))

    print("\n" + "=" * 72)
    print("BL-1 Dense Top-5 评测完成")
    print("=" * 72)
    print(f"题目:              {question_count}")
    print(
        f"Doc-Hit:           {len(doc_hit_ranks)}/{question_count} = "
        f"{len(doc_hit_ranks) / question_count:.1%}"
    )
    if doc_hit_ranks:
        print(
            "Doc 平均排名:      "
            f"{sum(doc_hit_ranks) / len(doc_hit_ranks):.1f}"
        )
    print(f"Chunk-Hit:         {len(chunk_hit_ranks)}/{question_count}")
    if chunk_hit_ranks:
        print(
            "Chunk 平均排名:    "
            f"{sum(chunk_hit_ranks) / len(chunk_hit_ranks):.1f}"
        )
    if "faithfulness" in ragas_scores:
        print(
            "Faithfulness:      "
            f"{ragas_scores['faithfulness']['mean']:.4f}"
        )
    if "answer_relevancy" in ragas_scores:
        print(
            "Answer Relevancy:  "
            f"{ragas_scores['answer_relevancy']['mean']:.4f}"
        )
    if ragas_scores.get("error"):
        print(f"Ragas Error:       {ragas_scores['error']}")
    print(f"结果目录:          {output_path}")
    print("=" * 72)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 BL-1 Dense Top-5 基线评测"
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_REVIEW_CSV),
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="仅调试检索和生成，不执行 Ragas 评分",
    )
    args = parser.parse_args()

    from app.core.milvus_client import milvus_manager
    from app.services.vector_store_manager import vector_store_manager

    try:
        milvus_manager.connect(allow_collection_mutation=False)
        vector_store_manager.initialize()
        summary = asyncio.run(
            run_bl1_eval(
                review_csv=args.csv,
                output_dir=args.output,
                limit=args.limit,
                skip_ragas=args.skip_ragas,
            )
        )
    finally:
        vector_store_manager.shutdown()
        milvus_manager.close()
    if summary["completion"]["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
