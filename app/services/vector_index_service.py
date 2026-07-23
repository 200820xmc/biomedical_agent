"""向量索引服务模块"""

import hashlib
import uuid
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


def _assign_stable_chunk_metadata(
    documents: list,
    fallback_source_id: str,
) -> None:
    """为每个 chunk 写入可重复生成的逻辑 ID 和文档内序号。"""
    for chunk_index, doc in enumerate(documents):
        metadata = doc.metadata
        document_id = (
            metadata.get("_document_id")
            or metadata.get("source_id")
            or fallback_source_id
        )
        content_hash = hashlib.sha256(
            doc.page_content.encode("utf-8")
        ).hexdigest()[:16]
        metadata["source_id"] = document_id
        metadata["chunk_index"] = chunk_index
        metadata["content_hash"] = content_hash
        metadata["chunk_id"] = f"{document_id}:{content_hash}"


class VectorIndexService:
    """向量索引服务 - 负责读取文件、生成向量、存储到 Milvus"""

    def __init__(
        self,
        store_manager=None,
        splitter_service=None,
        version_factory: Callable[[], str] | None = None,
    ):
        """初始化向量索引服务"""
        self._store_manager = store_manager or vector_store_manager
        self._splitter = splitter_service or document_splitter_service
        self._version_factory = version_factory or (lambda: uuid.uuid4().hex[:12])
        logger.info("向量索引服务初始化完成")

    def index_single_file(self, file_path: str):
        """
        索引单个文件 (使用新的 LangChain 分割器)

        Args:
            file_path: 文件路径

        Raises:
            ValueError: 文件不存在时抛出
            RuntimeError: 索引失败时抛出
        """
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始索引文件: {path}")

        try:
            # 1. 读取文件内容
            content = path.read_text(encoding="utf-8")
            logger.info(f"读取文件: {path}, 内容长度: {len(content)} 字符")

            # 2. 先分块和准备新版本，旧索引在验证完成前保持不变。
            normalized_path = path.as_posix()
            documents = self._splitter.split_document(content, normalized_path)
            logger.info(f"文档分割完成: {file_path} -> {len(documents)} 个分片")

            if not documents:
                raise RuntimeError("文件内容为空或无法分块")

            chunk_count = self._write_and_switch_version(
                documents=documents,
                logical_source=normalized_path,
                display_filename=path.name,
            )
            logger.info(f"文件索引完成: {file_path}, 共 {chunk_count} 个分片")
            return chunk_count

        except Exception as e:
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            raise RuntimeError(f"索引文件失败: {e}") from e

    def index_content(
        self,
        content: str,
        logical_source: str,
        display_filename: str,
        parsed_source: str = "",
        extra_metadata: dict | None = None,
    ) -> int:
        """将文本内容直接分块并写入向量库（不读取文件）

        用于 PDF 解析后场景：xParse 已将 PDF 转为 Markdown 文本，
        此方法跳过"读取文件"步骤，直接对内存中的文本进行分块和索引。

        Args:
            content: 要索引的文本内容（Markdown 或纯文本）
            logical_source: 逻辑来源路径（如原始 PDF 路径），用于 _source 元数据
            display_filename: 显示用的文件名（如 'Zhou_2023.pdf'）
            parsed_source: 解析后的 Markdown 文件路径（可选）
            extra_metadata: 额外的元数据字典（如 _parser, _document_id）

        Returns:
            int: 生成的分片数量

        Raises:
            RuntimeError: 索引失败时抛出
        """
        if not content or not content.strip():
            raise ValueError("索引内容不能为空")

        logger.info(
            f"开始索引内容: source={logical_source}, "
            f"display={display_filename}, length={len(content)} 字符"
        )

        try:
            # ── 1. 分块（复用现有的 DocumentSplitterService）─
            ext = Path(display_filename).suffix.lower()
            if ext in (".md", ".pdf"):
                documents = self._splitter.split_markdown(content, logical_source)
            else:
                documents = self._splitter.split_text(content, logical_source)

            if not documents:
                logger.warning(f"内容为空或无法分块: {display_filename}")
                return 0

            # ── 2. 修正元数据 + 版本标记 ────────────────────
            base_metadata = extra_metadata or {}
            for doc in documents:
                doc.metadata["_file_name"] = display_filename
                doc.metadata["_source"] = logical_source
                if parsed_source:
                    doc.metadata["_parsed_source"] = parsed_source
                doc.metadata.update(base_metadata)

            logger.info(f"文档分块完成: {display_filename} -> {len(documents)} 个分片")
            return self._write_and_switch_version(
                documents=documents,
                logical_source=logical_source,
                display_filename=display_filename,
            )

        except Exception as e:
            logger.error(f"索引内容失败: {display_filename}, 错误: {e}")
            raise RuntimeError(f"索引内容失败: {e}") from e

    def _write_and_switch_version(
        self,
        documents: list,
        logical_source: str,
        display_filename: str,
    ) -> int:
        """先写并校验新版本，再删除旧版本；失败时补偿删除新版本。"""
        version_id = self._version_factory()
        if not version_id:
            raise RuntimeError("无法生成索引版本ID")

        for doc in documents:
            doc.metadata["_source"] = logical_source
            doc.metadata["_file_name"] = display_filename
            doc.metadata["_version_id"] = version_id
        _assign_stable_chunk_metadata(documents, logical_source)
        expected_chunk_ids = [str(doc.metadata["chunk_id"]) for doc in documents]
        write_attempted = False

        try:
            write_attempted = True
            self._store_manager.add_documents(documents)
            if not self._store_manager.verify_source_version(
                logical_source,
                version_id,
                expected_chunk_ids,
            ):
                raise RuntimeError("新版本写后校验失败")

            self._store_manager.delete_by_source(
                logical_source,
                exclude_version=version_id,
            )
            logger.info(
                f"安全索引完成: {display_filename}, v={version_id}, "
                f"共 {len(documents)} 个分片"
            )
            return len(documents)
        except Exception as exc:
            rollback_error = None
            if write_attempted:
                try:
                    self._store_manager.delete_source_version(
                        logical_source,
                        version_id,
                    )
                except Exception as rollback_exc:
                    rollback_error = rollback_exc
            if rollback_error is not None:
                raise RuntimeError(
                    f"索引切换失败且新版本回滚失败: {exc}; rollback={rollback_error}"
                ) from exc
            raise


# 全局单例
vector_index_service = VectorIndexService()
