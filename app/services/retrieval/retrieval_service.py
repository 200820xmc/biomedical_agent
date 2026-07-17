"""检索编排服务 — 协调超额召回、Rerank、来源多样性和上下文构建

完整检索流程：
    query
      → RecallService.recall()          # Milvus 超额召回 Top 50
      → RerankService.rerank()          # LLM 语义精排 Top 20
      → DiversityService.select()       # 来源多样性选择 Top 8
      → ContextBuilder.build()          # 格式化证据文本 + Artifact

该服务是 retrieval 子包对外的唯一入口。Agent 工具只需调用
RetrievalService.retrieve()，无需了解内部编排细节。
"""

import time
from typing import Any

from loguru import logger

from app.config import config
from app.services.retrieval.context_builder import ContextBuilder
from app.services.retrieval.diversity_service import DiversityService
from app.services.retrieval.recall_service import RecallService
from app.services.retrieval.rerank_service import RerankService


class RetrievalService:
    """两阶段检索编排服务

    职责：
    - 按顺序编排 recall → rerank → diversity → context_builder
    - 记录各阶段耗时
    - 统一返回 (格式化上下文, 结构化 artifact) 元组
    """

    def __init__(self) -> None:
        self.recall = RecallService()
        self.rerank = RerankService()
        self.diversity = DiversityService()
        self.builder = ContextBuilder()

        logger.info("检索编排服务初始化完成")

    async def retrieve(
        self,
        query: str,
        search_mode: str = "auto",
        top_k: int | None = None,
        source_filter: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """执行完整的两阶段检索流程

        Args:
            query: 用户问题或经过整理的检索 query
            search_mode: 检索模式（auto / focused / comparison / broad）
            top_k: 最终返回的 chunk 数量，不传则使用配置默认值（8）
            source_filter: 可选的论文来源过滤列表（第一阶段预留，暂不实现过滤逻辑）

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
        final_k = top_k or mode_params["final_chunks"]

        logger.info(
            f"开始两阶段检索: query='{query[:80]}...', "
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

        reranked = await self.rerank.rerank(
            query=query,
            candidates=candidates,
            top_k=mode_params["rerank_k"],
        )
        duration_ms["rerank"] = (time.time() - rerank_start) * 1000
        rerank_applied = self.rerank.enabled and len(candidates) > mode_params["rerank_k"]
        # P1-7: 记录 Rerank 降级状态
        rerank_degraded = not rerank_applied and self.rerank.enabled

        # ── 阶段 3：相关性阈值过滤 ───────────────────────────
        threshold = getattr(config, "rag_rerank_threshold", 0.65)
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

        # ── 阶段 4：来源多样性选择 ─────────────────────────────
        diversity_start = time.time()
        selected = self.diversity.select(
            candidates=filtered,  # 使用阈值过滤后的候选
            max_chunks_per_source=mode_params["max_per_source"],
            final_count=final_k,
        )
        duration_ms["diversity"] = (time.time() - diversity_start) * 1000

        # ── 阶段 5：P2-2 相邻chunk扩展（Top3高分证据的±1邻居）──
        expand_start = time.time()
        expanded = _expand_neighbors(selected, candidates, window=1, top_n=3)
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
            rerank_applied=rerank_applied,
        )
        duration_ms["context"] = (time.time() - context_start) * 1000

        duration_ms["total"] = (time.time() - total_start) * 1000
        artifact.duration_ms = duration_ms
        # P1-7: 标记 Rerank 降级状态
        object.__setattr__(artifact, 'rerank_degraded', rerank_degraded)

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
            rerank_applied=rerank_applied,
            rerank_degraded=rerank_degraded,
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
            dict: 包含 candidate_k, rerank_k, final_chunks, max_per_source 的参数字典
        """
        modes = {
            "focused": {
                "candidate_k": 30,
                "rerank_k": 12,
                "final_chunks": 6,
                "max_per_source": 4,
            },
            "comparison": {
                "candidate_k": 60,
                "rerank_k": 24,
                "final_chunks": 10,
                "max_per_source": 2,
            },
            "broad": {
                "candidate_k": 80,
                "rerank_k": 30,
                "final_chunks": 12,
                "max_per_source": 2,
            },
            "auto": {
                "candidate_k": getattr(config, "rag_candidate_k", 50),
                "rerank_k": getattr(config, "rag_rerank_k", 20),
                "final_chunks": getattr(config, "rag_final_chunks", 8),
                "max_per_source": getattr(config, "rag_max_chunks_per_source", 2),
            },
        }
        return modes.get(search_mode, modes["auto"])


def _expand_neighbors(
    selected: list,
    candidates: list,
    window: int = 1,
    top_n: int = 3,
) -> list:
    """P2-2: 为Top-N高分chunk扩展相邻chunk

    从候选池中找到同源、相邻chunk_index的候选项，计入上下文。
    """
    if not selected or not candidates:
        return selected

    # 取Top-N高分项
    top_items = sorted(
        selected,
        key=lambda x: x.rerank_score or x.vector_score or 0,
        reverse=True,
    )[:top_n]

    # 构建候选索引（source -> [chunks sorted by chunk_index]）
    source_index: dict[str, list] = {}
    for c in candidates:
        sid = c.source_id or c.source
        if sid not in source_index:
            source_index[sid] = []
        source_index[sid].append(c)

    existing_ids = {item.chunk_id for item in selected}
    neighbors = []

    for item in top_items:
        sid = item.source_id or item.source
        pool = source_index.get(sid, [])
        pool_sorted = sorted(pool, key=lambda x: x.chunk_index)
        current_idx = item.chunk_index

        for offset in range(1, window + 1):
            for neighbor_idx in (current_idx - offset, current_idx + offset):
                for nb in pool_sorted:
                    if nb.chunk_index == neighbor_idx and nb.chunk_id not in existing_ids:
                        neighbors.append(nb)
                        existing_ids.add(nb.chunk_id)
                        break

    return selected + neighbors


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
    duration_ms: dict,
) -> None:
    """P2-5: 结构化检索链路日志（不含API Key）"""
    from loguru import logger as _logger

    trace = {
        "event": "retrieval_trace",
        "query_preview": query[:200],
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
        "recall_ms": round(duration_ms.get("recall", 0), 0),
        "rerank_ms": round(duration_ms.get("rerank", 0), 0),
        "total_ms": round(duration_ms.get("total", 0), 0),
    }
    _logger.info(f"RETRIEVAL_TRACE: {trace}")


def _dedup_candidates(candidates: list) -> list:
    """P1-3: 精确去重 chunk_id > content_hash > 正文完全相同"""
    seen: dict[str, object] = {}
    for item in candidates:
        key = item.chunk_id if item.chunk_id else ""
        if not key:
            key = item.metadata.get("content_hash", "")
        if not key:
            key = item.content[:200]  # fallback: 前200字符

        if key not in seen:
            seen[key] = item
        else:
            # 保留 vector_score 更高的
            existing = seen[key]
            if (item.vector_score or 0) > (existing.vector_score or 0):  # type: ignore[union-attr]
                seen[key] = item
    return list(seen.values())


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
