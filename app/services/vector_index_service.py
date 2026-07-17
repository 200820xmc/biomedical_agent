"""向量索引服务模块"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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


class IndexingResult:
    """索引结果类"""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def increment_success_count(self):
        """增加成功计数"""
        self.success_count += 1

    def increment_fail_count(self):
        """增加失败计数"""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        """添加失败文件"""
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """获取耗时（毫秒）"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """向量索引服务 - 负责读取文件、生成向量、存储到 Milvus"""

    def __init__(self):
        """初始化向量索引服务"""
        self.upload_path = "./uploads"
        logger.info("向量索引服务初始化完成")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """
        索引指定目录下的所有文件

        Args:
            directory_path: 目录路径（可选，默认使用配置的上传目录）

        Returns:
            IndexingResult: 索引结果
        """
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # 使用指定目录或默认上传目录
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在或不是有效目录: {target_path}")

            result.directory_path = str(dir_path)

            # 获取所有支持的文件
            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始索引目录: {target_path}, 找到 {len(files)} 个文件")

            # 遍历并索引每个文件
            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"✓ 文件索引成功: {file_path.name}")
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"✗ 文件索引失败: {file_path.name}, 错误: {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"目录索引完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 失败={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

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

            # 2. 删除该文件的旧数据（如果存在）
            normalized_path = path.as_posix()
            vector_store_manager.delete_by_source(normalized_path)

            # 3. 使用新的文档分割器
            documents = document_splitter_service.split_document(content, normalized_path)
            logger.info(f"文档分割完成: {file_path} -> {len(documents)} 个分片")

            # 4. 添加文档到向量存储
            if documents:
                _assign_stable_chunk_metadata(documents, normalized_path)
                vector_store_manager.add_documents(documents)
                logger.info(f"文件索引完成: {file_path}, 共 {len(documents)} 个分片")
            else:
                logger.warning(f"文件内容为空或无法分割: {file_path}")

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
                documents = document_splitter_service.split_markdown(content, logical_source)
            else:
                documents = document_splitter_service.split_text(content, logical_source)

            if not documents:
                logger.warning(f"内容为空或无法分块: {display_filename}")
                return 0

            # ── 2. 修正元数据 + 版本标记 ────────────────────
            import uuid
            version_id = uuid.uuid4().hex[:8]
            base_metadata = extra_metadata or {}
            for doc in documents:
                doc.metadata["_file_name"] = display_filename
                doc.metadata["_source"] = logical_source
                doc.metadata["_version_id"] = version_id
                if parsed_source:
                    doc.metadata["_parsed_source"] = parsed_source
                doc.metadata.update(base_metadata)
            _assign_stable_chunk_metadata(documents, logical_source)

            logger.info(f"文档分块完成: {display_filename} -> {len(documents)} 个分片")

            # ── 3. P1-8: 先写入新版本 → 验证成功 → 再删旧版本 ─
            vector_store_manager.add_documents(documents)
            logger.info(f"新版本写入完成: {display_filename}, {len(documents)} 个分片, v={version_id}")

            # ── 4. 验证写入成功后再删除旧版本 ────────────────
            vector_store_manager.delete_by_source(logical_source, exclude_version=version_id)

            logger.info(f"安全索引完成: {display_filename}, v={version_id}, 共 {len(documents)} 个分片")
            return len(documents)

        except Exception as e:
            logger.error(f"索引内容失败: {display_filename}, 错误: {e}")
            raise RuntimeError(f"索引内容失败: {e}") from e


# 全局单例
vector_index_service = VectorIndexService()
