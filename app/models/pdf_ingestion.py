"""PDF 入库相关数据模型

定义 PDF 上传登记、入库任务和会话附件的 Pydantic 模型。
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UploadedDocument(BaseModel):
    """上传文档登记记录"""

    document_id: str = Field(..., description="文档唯一标识")
    original_filename: str = Field(..., description="用户上传的原始文件名")
    stored_path: str = Field(..., description="服务器存储路径")
    content_type: str = Field(..., description="MIME 类型")
    file_size: int = Field(..., description="文件大小（字节）")
    sha256: str = Field(..., description="文件 SHA-256 哈希值")
    status: str = Field(default="uploaded", description="文档状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class IngestionJob(BaseModel):
    """入库任务记录"""

    job_id: str = Field(..., description="任务唯一标识")
    document_id: str = Field(..., description="关联的文档 ID")
    original_filename: str = Field(default="", description="原始文件名")
    parser: str = Field(default="xparse", description="解析器名称")
    parser_mode: str = Field(default="free", description="解析器模式")
    status: str = Field(default="queued", description="任务状态")
    progress: int = Field(default=5, description="进度百分比 0-100")
    markdown_path: Optional[str] = Field(default=None, description="解析后 Markdown 路径")
    chunk_count: Optional[int] = Field(default=None, description="最终入库分片数")
    parser_exit_code: Optional[int] = Field(default=None, description="CLI 退出码")
    parser_suggestion_tag: Optional[str] = Field(default=None, description="CLI 建议标签")
    parser_request_id: Optional[str] = Field(default=None, description="TextIn API request_id")
    error_code: Optional[str] = Field(default=None, description="统一错误码")
    error_message: Optional[str] = Field(default=None, description="错误详情")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后更新时间")


class AttachmentRef(BaseModel):
    """会话附件引用"""

    document_id: str = Field(..., description="文档 ID")
    filename: str = Field(..., description="文件名")
    content_type: str = Field(default="application/pdf", description="MIME 类型")


class PDFIngestionInput(BaseModel):
    """PDF 入库工具输入参数"""

    document_id: str = Field(
        ...,
        description="上传接口返回的 PDF 文档 ID",
    )
    force_reindex: bool = Field(
        default=False,
        description="是否强制重新解析并索引",
    )


class IngestionStatusInput(BaseModel):
    """入库状态查询输入参数"""

    job_id: str = Field(
        ...,
        description="PDF 入库任务 ID",
    )
