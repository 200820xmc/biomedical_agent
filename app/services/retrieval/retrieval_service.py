"""检索编排服务 — 协调固定Top-20召回、单次Rerank和上下文构建

完整检索流程：
    query
      → RecallService.recall()          # Milvus超额召回Top-20
      → RerankService.rerank()          # 一次性LLM精排并保留Top-10
      → RecallService.expand_neighbors()# 按chunk_index直接读取邻居
      → ContextBuilder.build()          # 格式化证据文本 + Artifact

该服务是 retrieval 子包对外的唯一入口。Agent 工具只需调用
RetrievalService.retrieve()，无需了解内部编排细节。
"""

import time
from typing import Any

from loguru import logger

from app.config import config
from app.services.retrieval.context_builder import ContextBuilder
from app.services.retrieval.recall_service import RecallService
from app.services.retrieval.rerank_service import RerankService
from app.utils.logger import describe_text


class RetrievalService:
    """两阶段检索编排服务

    职责：
    - 按顺序编排 recall → rerank → threshold → neighbor → context_builder
    - 记录各阶段耗时
    - 统一返回 (格式化上下文, 结构化 artifact) 元组
    """

    def __init__(
        self,
        recall: RecallService | None = None,
        rerank: RerankService | None = None,
        builder: ContextBuilder | None = None,
    ) -> None:
        self.recall = recall or RecallService()
        self.rerank = rerank or RerankService()
        self.builder = builder or ContextBuilder()

        logger.info("检索编排服务初始化完成")

    async def retrieve(
        self,
        query: str,
        search_mode: str = "auto",
    ) -> tuple[str, dict[str, Any]]:
        """执行完整的两阶段检索流程

        Args:
            query: 用户问题或经过整理的检索 query
            search_mode: 检索模式（auto / focused / comparison / broad）

        Returns:
            tuple[str, dict]: (格式化上下文文本, artifact 字典)

        Raises:
            ValueError: query 为空时抛出
        """
        if not query or not query.strip():
            raise ValueError("检索 query 不能为空")

        total_start = time.time()
        duration_ms: dict[str, float] = {}

        # 根据检索模式调整参数
        mode_params = self._get_mode_params(search_mode)
        final_k = mode_params["final_chunks"]

        logger.info(
            f"开始两阶段检索: {describe_text(query, 'query')}, "
            f"mode={search_mode}, target={final_k}"
        )

        # ── 阶段 1：超额召回 ──────────────────────────────────
        recall_start = time.time()
        candidates = self.recall.recall(
            query=query,
            candidate_count=mode_params["candidate_k"],
        )
        duration_ms["recall"] = (time.time() - recall_start) * 1000

        if not candidates:
            logger.warning("超额召回无结果")
            return (
                "当前知识库中未检索到相关资料。",
                {
                    "original_query": query,
                    "search_mode": search_mode,
                    "candidate_count": 0,
                    "reranked_count": 0,
                    "selected_count": 0,
                    "confidence": "low",
                    "rerank_applied": False,
                    "rerank_status": "skipped",
                    "rerank_degraded": False,
                    "rerank_reason": "no_candidates",
                    "threshold_applied": False,
                    "threshold_fallback": False,
                    "documents": [],
                },
            )

        # ── 阶段 2：Rerank 精排 ────────────────────────────────
        rerank_start = time.time()

        # P1-3: Rerank 前精确去重（chunk_id > content_hash > 正文）
        deduped_candidates = _dedup_candidates(candidates)
        if len(deduped_candidates) < len(candidates):
            logger.info(
                f"候选去重: {len(candidates)} → {len(deduped_candidates)}"
            )
            candidates = deduped_candidates

        rerank_result = await self.rerank.rerank(
            query=query,
            candidates=candidates,
            top_k=mode_params["rerank_k"],
        )
        reranked = rerank_result.items
        duration_ms["rerank"] = (time.time() - rerank_start) * 1000

        # ── 阶段 3：相关性阈值过滤 ───────────────────────────
        threshold = getattr(config, "rag_rerank_threshold", 0.65)
        threshold_applied = rerank_result.applied
        threshold_fallback = False
        if threshold_applied:
            filtered = [
                item
                for item in reranked
                if item.rerank_score is not None and item.rerank_score >= threshold
            ]
            if len(filtered) < len(reranked):
                logger.info(
                    f"阈值过滤: {len(reranked)} → {len(filtered)} "
                    f"(threshold={threshold})"
                )
            if not filtered and reranked:
                fallback_k = min(
                    max(1, getattr(config, "rag_threshold_fallback_k", 3)),
                    final_k,
                    len(reranked),
                )
                filtered = reranked[:fallback_k]
                threshold_fallback = True
                logger.warning(
                    f"LLM Rerank结果全部低于阈值，启用Top-{fallback_k}保底"
                )
        else:
            # 向量排序与LLM分数不是同一分布，禁止套用Rerank阈值。
            filtered = reranked

        # ── 阶段 4：直接按排序取Top-K，不再人为扩大来源覆盖 ────
        selected = filtered[:final_k]

        # ── 阶段 5：按source_id + chunk_index直接查询相邻Chunk ─
        expand_start = time.time()
        expanded = self.recall.expand_neighbors(selected, window=1, top_n=3)
        duration_ms["expand"] = (time.time() - expand_start) * 1000
        if len(expanded) > len(selected):
            logger.info(f"邻居扩展: {len(selected)} → {len(expanded)} 个chunk")
            selected = expanded

        # ── 阶段 6：上下文格式化 ───────────────────────────────
        context_start = time.time()
        context, artifact = self.builder.build(
            items=selected,
            original_query=query,
            search_mode=search_mode,
            candidate_count=len(candidates),
            reranked_count=len(reranked),
            rerank_applied=rerank_result.applied,
            rerank_status=rerank_result.status,
            rerank_degraded=rerank_result.degraded,
            rerank_reason=rerank_result.reason,
            threshold_applied=threshold_applied,
            threshold_fallback=threshold_fallback,
        )
        duration_ms["context"] = (time.time() - context_start) * 1000

        duration_ms["total"] = (time.time() - total_start) * 1000
        artifact.duration_ms = duration_ms
        logger.info(
            f"两阶段检索完成: 召回={len(candidates)} → "
            f"Rerank={len(reranked)} → 最终={len(selected)}, "
            f"总耗时={duration_ms['total']:.0f}ms, "
            f"置信度={artifact.confidence}"
        )

        # P2-5: 结构化检索链路日志
        _log_retrieval_trace(
            query=query,
            search_mode=search_mode,
            candidate_count=len(candidates),
            reranked_count=len(reranked),
            selected_count=len(selected),
            selected_sources=list(set(
                item.source_id or item.source for item in selected
            )),
            best_rerank_score=max(
                (item.rerank_score or 0) for item in selected
            ) if selected else 0,
            confidence=artifact.confidence,
            rerank_applied=rerank_result.applied,
            rerank_degraded=rerank_result.degraded,
            rerank_status=rerank_result.status,
            rerank_reason=rerank_result.reason,
            threshold_applied=threshold_applied,
            threshold_fallback=threshold_fallback,
            duration_ms=duration_ms,
        )

        return context, _artifact_to_dict(artifact)

    def _get_mode_params(self, search_mode: str) -> dict:
        """根据检索模式返回参数组合

        不同模式对应不同的候选数量、Rerank 保留数和来源限制。
        参考设计文档第 7.2 节。

        Args:
            search_mode: 检索模式标识

        Returns:
            dict: 包含 candidate_k、rerank_k、final_chunks 的参数字典
        """
        fixed = {
            "candidate_k": getattr(config, "rag_candidate_k", 20),
            "rerank_k": getattr(config, "rag_rerank_k", 10),
            "final_chunks": getattr(config, "rag_final_chunks", 5),
        }
        return fixed


def _log_retrieval_trace(
    query: str,
    search_mode: str,
    candidate_count: int,
    reranked_count: int,
    selected_count: int,
    selected_sources: list,
    best_rerank_score: float,
    confidence: str,
    rerank_applied: bool,
    rerank_degraded: bool,
    rerank_status: str,
    rerank_reason: str,
    threshold_applied: bool,
    threshold_fallback: bool,
    duration_ms: dict,
) -> None:
    """P2-5: 结构化检索链路日志（不含API Key）"""
    from loguru import logger as _logger

    trace = {
        "event": "retrieval_trace",
        "query_length": len(query),
        "query_sha256": describe_text(query, "query").split("query_sha256=", 1)[1],
        "search_mode": search_mode,
        "candidate_count": candidate_count,
        "reranked_count": reranked_count,
        "selected_count": selected_count,
        "source_count": len(selected_sources),
        "sources_preview": selected_sources[:5],
        "best_rerank_score": round(best_rerank_score, 4),
        "confidence": confidence,
        "rerank_applied": rerank_applied,
        "rerank_degraded": rerank_degraded,
        "rerank_status": rerank_status,
        "rerank_reason": rerank_reason,
        "threshold_applied": threshold_applied,
        "threshold_fallback": threshold_fallback,
        "recall_ms": round(duration_ms.get("recall", 0), 0),
        "rerank_ms": round(duration_ms.get("rerank", 0), 0),
        "total_ms": round(duration_ms.get("total", 0), 0),
    }
    _logger.info(f"RETRIEVAL_TRACE: {trace}")


def _dedup_candidates(candidates: list) -> list:
    """P1-3: 精确去重 chunk_id > content_hash > 正文完全相同"""
    kept: list = []
    for item in candidates:
        duplicate_index = None
        for index, existing in enumerate(kept):
            same_chunk_id = bool(
                item.chunk_id
                and existing.chunk_id
                and item.chunk_id == existing.chunk_id
            )
            item_hash = item.metadata.get("content_hash", "")
            existing_hash = existing.metadata.get("content_hash", "")
            same_content_hash = bool(
                item_hash and existing_hash and item_hash == existing_hash
            )
            same_full_content = item.content == existing.content
            if same_chunk_id or same_content_hash or same_full_content:
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(item)
        elif (item.vector_score or 0) > (kept[duplicate_index].vector_score or 0):
            kept[duplicate_index] = item
    return kept


def _artifact_to_dict(artifact) -> dict:
    """将 RetrievalArtifact 转换为可序列化的字典

    Args:
        artifact: RetrievalArtifact 实例

    Returns:
        dict: 可 JSON 序列化的字典
    """
    return {
        "original_query": artifact.original_query,
        "search_mode": artifact.search_mode,
        "candidate_count": artifact.candidate_count,
        "reranked_count": artifact.reranked_count,
        "selected_count": artifact.selected_count,
        "confidence": artifact.confidence,
        "rerank_applied": artifact.rerank_applied,
        "rerank_status": artifact.rerank_status,
        "rerank_degraded": artifact.rerank_degraded,
        "rerank_reason": artifact.rerank_reason,
        "threshold_applied": artifact.threshold_applied,
        "threshold_fallback": artifact.threshold_fallback,
        "documents": [
            {
                "chunk_id": doc.chunk_id,
                "source_id": doc.source_id,
                "source": doc.source,
                "chunk_index": doc.chunk_index,
                "vector_score": doc.vector_score,
                "rerank_score": doc.rerank_score,
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in artifact.documents
        ],
        "duration_ms": getattr(artifact, "duration_ms", {}),
    }


# 全局单例
retrieval_service = RetrievalService()
