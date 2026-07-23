"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

from collections.abc import Callable
import json
import time
from typing import Any, List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger
from app.utils.logger import describe_text

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import (
    get_vector_embedding_service,
    reset_vector_embedding_service,
)


# 统一使用 biz collection
COLLECTION_NAME = "biz"


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(
        self,
        milvus_client_manager: Any = milvus_manager,
        embedding_factory: Callable[[], Any] = get_vector_embedding_service,
        vector_store_factory: Callable[..., Milvus] = Milvus,
    ):
        """创建惰性管理器；构造阶段不连接Milvus或创建Embedding客户端。"""
        self.vector_store: Milvus | None = None
        self.collection_name = COLLECTION_NAME
        self._milvus_manager = milvus_client_manager
        self._embedding_factory = embedding_factory
        self._vector_store_factory = vector_store_factory

    @property
    def is_initialized(self) -> bool:
        return self.vector_store is not None

    def initialize(self) -> Milvus:
        """在应用生命周期启动阶段显式初始化Milvus VectorStore。"""
        if self.vector_store is not None:
            return self.vector_store
        try:
            if not self._milvus_manager.health_check():
                raise RuntimeError("Milvus未连接，请先在应用lifespan中调用connect()")

            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 创建 LangChain Milvus VectorStore
            # 使用 biz collection，字段映射：text_field -> content, vector_field -> vector
            self.vector_store = self._vector_store_factory(
                embedding_function=self._embedding_factory(),
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,  # 使用自定义 id
                drop_old=False,
                text_field="content",  # 文本内容存储到 content 字段
                vector_field="vector",  # 向量存储到 vector 字段
                primary_field="id",  # 主键字段
                metadata_field="metadata",  # 元数据字段
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )
            return self.vector_store

        except Exception as e:
            self.vector_store = None
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def shutdown(self) -> None:
        """释放当前进程持有的VectorStore和Embedding客户端引用。"""
        self.vector_store = None
        reset_vector_embedding_service()

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动分批，每批最多20个以适配API限制）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            import time
            import uuid
            start_time = time.time()
            vector_store = self.get_vector_store()

            all_ids = []
            batch_size = 20  # DashScope embedding API 限制

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                ids = [str(uuid.uuid4()) for _ in batch]
                vector_store.add_documents(batch, ids=ids)
                all_ids.extend(ids)
                logger.debug(f"  批次 {i // batch_size + 1}: {len(batch)} 个文档")

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return all_ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    @staticmethod
    def _escape_expr_string(value: str) -> str:
        """转义 Milvus 表达式中的字符串字面量。"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def get_document_chunk_ids(self, document_id: str) -> list[str]:
        """按 document_id 查询实际存在于 Milvus 的 Chunk 主键。

        同时兼容新版 ``source_id`` 与旧版 ``_document_id`` 元数据。
        该方法只读，用于入库幂等判断和状态校验。
        """
        if not document_id or not document_id.strip():
            raise ValueError("document_id 不能为空")

        return sorted(
            str(row["id"])
            for row in self.get_document_rows(document_id, output_fields=["id"])
            if row.get("id")
        )

    def get_document_rows(
        self,
        document_id: str,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """按新旧document_id字段查询并按Milvus主键去重。"""
        if not document_id or not document_id.strip():
            raise ValueError("document_id不能为空")
        collection = self._milvus_manager.get_collection()
        escaped = self._escape_expr_string(document_id.strip())
        rows_by_id: dict[str, dict[str, Any]] = {}
        fields = output_fields or ["id", "content", "metadata"]
        for field in ("source_id", "_document_id"):
            rows = collection.query(
                expr=f'metadata["{field}"] == "{escaped}"',
                output_fields=fields,
                limit=10000,
            )
            for row in rows:
                if row.get("id"):
                    rows_by_id[str(row["id"])] = row
        return list(rows_by_id.values())

    def has_document(self, document_id: str) -> bool:
        """以 Milvus 实际数据判断文献是否已经入库。"""
        return bool(self.get_document_chunk_ids(document_id))

    def get_source_rows(self, file_path: str) -> list[dict[str, Any]]:
        """查询一个逻辑来源的全部版本，仅用于写后校验和版本切换。"""
        collection = self._milvus_manager.get_collection()
        escaped_path = self._escape_expr_string(file_path)
        return collection.query(
            expr=f'metadata["_source"] == "{escaped_path}"',
            output_fields=["id", "content", "metadata"],
            limit=10000,
        )

    def verify_source_version(
        self,
        file_path: str,
        version_id: str,
        expected_chunk_ids: list[str],
        attempts: int = 5,
        delay_seconds: float = 0.2,
    ) -> bool:
        """确认新版本行数、逻辑Chunk集合和必需元数据完整。"""
        collection = self._milvus_manager.get_collection()
        if hasattr(collection, "flush"):
            collection.flush()
        expected = sorted(expected_chunk_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("待写入文档包含重复逻辑chunk_id")

        for attempt in range(max(1, attempts)):
            rows = [
                row
                for row in self.get_source_rows(file_path)
                if row.get("metadata", {}).get("_version_id") == version_id
            ]
            chunk_ids = [
                str(row.get("metadata", {}).get("chunk_id", ""))
                for row in rows
            ]
            metadata_complete = all(
                all(
                    key in row.get("metadata", {})
                    for key in ("source_id", "chunk_index", "content_hash", "chunk_id")
                )
                for row in rows
            )
            if (
                len(rows) == len(expected)
                and sorted(chunk_ids) == expected
                and metadata_complete
            ):
                return True
            if attempt + 1 < attempts and delay_seconds > 0:
                time.sleep(delay_seconds)
        return False

    def delete_source_version(self, file_path: str, version_id: str) -> int:
        """删除指定来源的一个版本，用于失败补偿。"""
        rows = [
            row
            for row in self.get_source_rows(file_path)
            if row.get("metadata", {}).get("_version_id") == version_id
        ]
        return self._delete_ids([str(row["id"]) for row in rows])

    def _delete_ids(self, ids_to_delete: list[str]) -> int:
        if not ids_to_delete:
            return 0
        collection = self._milvus_manager.get_collection()
        deleted = 0
        for index in range(0, len(ids_to_delete), 100):
            batch = ids_to_delete[index : index + 100]
            ids_str = ", ".join(json.dumps(value) for value in batch)
            result = collection.delete(expr=f"id in [{ids_str}]")
            deleted += result.delete_count if hasattr(result, "delete_count") else 0
        if hasattr(collection, "flush"):
            collection.flush()
        return deleted

    def delete_by_source(self, file_path: str, exclude_version: str = "") -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径
            exclude_version: 排除的版本ID（P1-8: 保护新写入的版本不被删除）

        Returns:
            int: 删除的文档数量
        """
        try:
            if exclude_version:
                logger.info(f"删除旧版本: {file_path} (保留 v={exclude_version})")
            else:
                logger.debug(f"删除文件数据: {file_path}")

            # 按 _source 匹配
            collection = self._milvus_manager.get_collection()

            # 获取该 source 下所有 chunk，按版本过滤
            escaped_path = self._escape_expr_string(file_path)
            expr = f'metadata["_source"] == "{escaped_path}"'
            # Milvus 表达式不支持 != ""，用 query + Python 过滤
            old_chunks = collection.query(expr=expr, output_fields=["id", "metadata"], limit=10000)
            ids_to_delete = []
            for chunk in old_chunks:
                meta = chunk.get("metadata", {})
                ver = meta.get("_version_id", "")
                if exclude_version and ver == exclude_version:
                    continue  # 保护新版本
                ids_to_delete.append(chunk["id"])

            if not ids_to_delete:
                return 0

            deleted = self._delete_ids([str(value) for value in ids_to_delete])

            logger.info(f"删除旧版本: {file_path}, 数量: {deleted}")
            return deleted

        except Exception as e:
            logger.error(f"删除旧数据失败: {e}")
            raise RuntimeError(f"删除旧数据失败: {e}") from e

    def get_vector_store(self) -> Milvus:
        """
        获取 VectorStore 实例

        Returns:
            Milvus: VectorStore 实例
        """
        if self.vector_store is None:
            raise RuntimeError("向量存储尚未初始化或Milvus当前不可用")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        try:
            vector_store = self.get_vector_store()
            docs = vector_store.similarity_search(query, k=k)
            logger.debug(
                f"相似度搜索完成: {describe_text(query, 'query')}, 结果数={len(docs)}"
            )
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 惰性全局管理器：对象本身不连接外部服务，连接由FastAPI lifespan触发。
vector_store_manager = VectorStoreManager()
