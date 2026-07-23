"""超额召回服务 — 从 Milvus 向量数据库召回候选 chunk

召回阶段追求"不遗漏"：适当扩大候选数量，为后续 Rerank 提供足够多的候选。
"""

import hashlib
import time
from typing import List

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.retrieval.retrieval_models import RetrievalItem
from app.services.vector_store_manager import vector_store_manager


class RecallService:
    """Milvus 超额召回服务

    职责：
    - 调用 Milvus 向量检索，获取带分数的候选 chunk
    - 将 L2 距离转换为 0~1 的相似度得分
    - 为每个 chunk 分配稳定标识（source_id、chunk_index）
    """

    def __init__(self, store_manager=None) -> None:
        self._store_manager = store_manager or vector_store_manager
        self._default_candidate_count = getattr(
            config, "rag_candidate_k", 20
        )

    def recall(
        self,
        query: str,
        candidate_count: int | None = None,
    ) -> list[RetrievalItem]:
        """从 Milvus 超额召回候选 chunk

        Args:
            query: 用户问题或改写后的检索 query
            candidate_count: 候选数量，不传则使用配置默认值（50）

        Returns:
            list[RetrievalItem]: 候选 chunk 列表，按向量相似度降序排列

        Raises:
            ValueError: query 为空时抛出
        """
        if not query or not query.strip():
            raise ValueError("检索 query 不能为空")

        k = min(candidate_count or self._default_candidate_count, self._default_candidate_count)
        start_time = time.time()

        try:
            vector_store = self._store_manager.get_vector_store()

            # 使用 similarity_search_with_score 获取带 L2 距离的结果
            # Milvus 返回 (Document, score) 元组，score 是 L2 距离（越小越相关）
            results_with_scores = vector_store.similarity_search_with_score(
                query, k=k
            )

            items: list[RetrievalItem] = []
            for idx, (doc, l2_distance) in enumerate(results_with_scores):
                # 将 L2 距离转换为相似度得分（0~1，越高越相关）
                vector_score = self._l2_to_similarity(l2_distance)

                # 提取 metadata 中的信息
                metadata = doc.metadata or {}

                # source_id：优先使用显式的 source_id，否则用 _source 路径
                source_id = metadata.get("source_id") or metadata.get("_source", "")

                # source：显示用的文件名
                source = metadata.get("_file_name", source_id)

                # chunk_index：chunk 在文档中的顺序
                chunk_index = metadata.get("chunk_index", idx)

                # content_hash：用于去重和构造稳定逻辑 chunk_id
                content_hash = metadata.get(
                    "content_hash",
                    hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16],
                )
                document_id = (
                    metadata.get("_document_id")
                    or metadata.get("source_id")
                    or source_id
                )
                stable_chunk_id = metadata.get(
                    "chunk_id",
                    f"{document_id}:{content_hash}",
                )

                item = RetrievalItem(
                    chunk_id=stable_chunk_id,
                    source_id=source_id,
                    source=source,
                    chunk_index=chunk_index if isinstance(chunk_index, int) else idx,
                    content=doc.page_content,
                    vector_score=vector_score,
                    metadata=dict(metadata),
                )
                items.append(item)

            elapsed = (time.time() - start_time) * 1000
            logger.info(
                f"超额召回完成: 候选数={len(items)}, "
                f"最高分={items[0].vector_score:.4f}" if items else "超额召回: 无结果",
            )
            logger.debug(f"召回耗时: {elapsed:.0f}ms")

            return items

        except Exception as e:
            logger.error(f"超额召回失败: {e}")
            raise

    def expand_neighbors(
        self,
        selected: list[RetrievalItem],
        window: int = 1,
        top_n: int = 3,
    ) -> list[RetrievalItem]:
        """按source_id和真实chunk_index直接查询相邻Chunk。

        邻居不再依赖Dense召回池是否碰巧包含相邻行。
        """
        if not selected:
            return selected

        top_items = sorted(
            selected,
            key=lambda item: item.rerank_score or item.vector_score or 0,
            reverse=True,
        )[:top_n]
        existing_ids = {item.chunk_id for item in selected}
        neighbors: list[RetrievalItem] = []

        for parent in top_items:
            source_id = parent.source_id
            if not source_id or not isinstance(parent.chunk_index, int):
                continue
            target_indices = {
                parent.chunk_index + offset
                for offset in range(-window, window + 1)
                if offset and parent.chunk_index + offset >= 0
            }
            rows = self._store_manager.get_document_rows(source_id)
            for row in rows:
                metadata = row.get("metadata", {})
                chunk_index = metadata.get("chunk_index")
                if chunk_index not in target_indices:
                    continue
                content = str(row.get("content", ""))
                content_hash = metadata.get(
                    "content_hash",
                    hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
                )
                chunk_id = metadata.get(
                    "chunk_id",
                    f"{source_id}:{content_hash}",
                )
                if chunk_id in existing_ids:
                    continue
                neighbor_metadata = dict(metadata)
                neighbor_metadata["neighbor_of"] = parent.chunk_id
                neighbors.append(
                    RetrievalItem(
                        chunk_id=chunk_id,
                        source_id=source_id,
                        source=metadata.get("_file_name", parent.source),
                        chunk_index=chunk_index,
                        content=content,
                        vector_score=None,
                        rerank_score=None,
                        metadata=neighbor_metadata,
                    )
                )
                existing_ids.add(chunk_id)

        return selected + neighbors

    @staticmethod
    def _l2_to_similarity(l2_distance: float) -> float:
        """将 Milvus L2 距离转换为 0~1 的相似度得分

        转换公式：1 / (1 + l2_distance)

        - L2 距离为 0（完全相同）时，得分为 1.0
        - L2 距离越大（越不相似），得分越接近 0
        - Embedding 维度为 1024，归一化后 L2 距离范围约为 0~2

        Args:
            l2_distance: Milvus 返回的 L2 距离

        Returns:
            float: 0~1 的相似度得分
        """
        return 1.0 / (1.0 + l2_distance)
