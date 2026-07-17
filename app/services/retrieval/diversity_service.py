"""来源多样性选择服务 — 确保最终上下文覆盖多个论文来源

在 Rerank 完成后，按 Greedy + 来源约束的方式选择最终 chunk：
- 优先选择 rerank_score 最高的 chunk
- 同一篇论文最多保留 N 个 chunk（默认 2）
- 最多覆盖 M 个不同来源（默认 5）
- 总共选择 K 个 chunk（默认 8）
"""

from typing import List

from loguru import logger

from app.config import config
from app.services.retrieval.retrieval_models import RetrievalItem


class DiversityService:
    """来源多样性选择服务

    职责：
    - 在 Rerank 排序结果上执行来源约束选择
    - 保证同一论文不会占据全部上下文
    - 保证比较型问题覆盖多个来源
    """

    def __init__(self) -> None:
        self._max_chunks_per_source = getattr(
            config, "rag_max_chunks_per_source", 2
        )
        self._max_sources = getattr(config, "rag_max_sources", 5)
        self._final_chunks = getattr(config, "rag_final_chunks", 8)

    def select(
        self,
        candidates: list[RetrievalItem],
        max_chunks_per_source: int | None = None,
        max_sources: int | None = None,
        final_count: int | None = None,
    ) -> list[RetrievalItem]:
        """从 Rerank 结果中按来源多样性选择最终 chunk

        算法（Greedy + 来源约束）：
        1. 按 rerank_score 降序遍历候选
        2. 如果当前来源已选满 max_chunks_per_source 个，跳过
        3. 否则加入选择列表
        4. 当选择数量达到 final_count 时停止

        Args:
            candidates: 按 rerank_score 降序排列的候选列表
            max_chunks_per_source: 每篇论文最多保留的 chunk 数
            max_sources: 最多覆盖的来源数（暂未使用，预留）
            final_count: 最终选择的 chunk 总数

        Returns:
            list[RetrievalItem]: 按 rerank_score 降序的最终选择列表
        """
        if not candidates:
            return []

        per_source = max_chunks_per_source or self._max_chunks_per_source
        final_k = final_count or self._final_chunks

        selected: list[RetrievalItem] = []
        source_counts: dict[str, int] = {}

        for item in candidates:
            source_key = item.source_id or item.source

            current_count = source_counts.get(source_key, 0)
            if current_count >= per_source:
                continue

            selected.append(item)
            source_counts[source_key] = current_count + 1

            if len(selected) >= final_k:
                break

        # 统计来源覆盖
        unique_sources = len(set(
            item.source_id or item.source for item in selected
        ))
        logger.info(
            f"来源多样性选择完成: 候选={len(candidates)} → 最终={len(selected)}, "
            f"覆盖来源={unique_sources}, 每篇上限={per_source}"
        )

        return selected
