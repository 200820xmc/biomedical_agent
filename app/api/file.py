"""文件上传接口模块 — 支持 Markdown、TXT 和 PDF

上传流程：
- MD/TXT：保持现有流程，上传后直接索引到 Milvus
- PDF：保存原始文件 → 生成 document_id → 返回 uploaded 状态
        Agent 显式调用 ingest_pdf_to_knowledge_base 后才开始解析和入库
"""

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import config
from app.services.vector_index_service import vector_index_service
from loguru import logger

router = APIRouter()

# 文件上传后存储的路径
UPLOAD_DIR = Path("./uploads")
# 原始 PDF 存储子目录
ORIGINALS_DIR = UPLOAD_DIR / "originals"
# 支持的文件类型
ALLOWED_EXTENSIONS = ["txt", "md", "pdf"]
# 单个文件支持最大大小（默认 10MB，和免费 API 一致）
MAX_FILE_SIZE = getattr(config, "pdf_max_file_size", 10 * 1024 * 1024)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件并自动创建向量索引（MD/TXT）或登记 PDF 待入库

    Args:
        file: 上传的文件

    Returns:
        JSONResponse: 上传结果
    """
    try:
        # ── 1. 验证文件 ────────────────────────────────────
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        safe_filename = _sanitize_filename(file.filename)

        file_extension = _get_file_extension(safe_filename)
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # ── 2. 读取内容 ────────────────────────────────────
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件大小超过限制（最大 {MAX_FILE_SIZE // 1024 // 1024}MB）",
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="文件内容为空")

        # ── 3. 分支：PDF 走新流程，MD/TXT 走现有流程 ──────
        if file_extension == "pdf":
            return await _handle_pdf_upload(safe_filename, content)
        else:
            return await _handle_md_txt_upload(safe_filename, content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")


async def _handle_md_txt_upload(filename: str, content: bytes) -> JSONResponse:
    """处理 MD/TXT 文件上传（保持现有流程：直接索引）"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / filename

    if file_path.exists():
        logger.info(f"文件已存在，将覆盖: {file_path}")
        file_path.unlink()

    file_path.write_bytes(content)
    logger.info(f"文件上传成功: {file_path}")

    # 自动创建向量索引（P1-9: 分离上传和索引状态）
    index_success = True
    index_error = None
    try:
        vector_index_service.index_single_file(str(file_path))
        logger.info(f"向量索引创建成功: {file_path}")
    except Exception as e:
        index_success = False
        index_error = str(e)
        logger.error(f"向量索引创建失败: {file_path}, 错误: {e}")

    status_code = 200 if index_success else 207  # 207 Multi-Status
    return JSONResponse(
        status_code=status_code,
        content={
            "code": 200,
            "message": "success" if index_success else "partial_success",
            "data": {
                "filename": filename,
                "file_path": str(file_path),
                "size": len(content),
                "upload_success": True,
                "index_success": index_success,
                "index_error": index_error,
            },
        },
    )


async def _handle_pdf_upload(filename: str, content: bytes) -> JSONResponse:
    """处理 PDF 文件上传（新流程：保存原文 + 生成 document_id）"""
    # ── 验证 PDF 文件头 ──────────────────────────────────
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件不是有效的 PDF 格式（文件头校验失败）")

    # ── 计算 SHA-256 ─────────────────────────────────────
    sha256_hash = hashlib.sha256(content).hexdigest()

    # ── 生成 document_id ─────────────────────────────────
    document_id = f"doc_{sha256_hash[:6]}"

    # ── 保存原始 PDF ─────────────────────────────────────
    doc_dir = ORIGINALS_DIR / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = doc_dir / filename
    pdf_path.write_bytes(content)

    logger.info(
        f"PDF 上传成功: {filename}, document_id={document_id}, "
        f"size={len(content)} bytes, sha256={sha256_hash[:16]}..."
    )

    # ── 返回上传响应（不自动解析，不自动写 Milvus）──────
    return JSONResponse(
        status_code=201,
        content={
            "code": 201,
            "message": "PDF 上传成功，等待解析入库",
            "data": {
                "document_id": document_id,
                "filename": filename,
                "content_type": "application/pdf",
                "size": len(content),
                "sha256": sha256_hash,
                "status": "uploaded",
                "indexed": False,
                "suggested_action": "请在对话中要求 Agent 解析并加入知识库",
            },
        },
    )


@router.post("/index_directory")
async def index_directory(directory_path: str = None):
    """
    索引指定目录下的所有文件

    Args:
        directory_path: 目录路径（可选，默认使用 uploads 目录）

    Returns:
        JSONResponse: 索引结果
    """
    try:
        logger.info(f"开始索引目录: {directory_path or 'uploads'}")

        result = vector_index_service.index_directory(directory_path)

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": result.to_dict(),
            },
        )

    except Exception as e:
        logger.error(f"索引目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引目录失败: {e}")


def _get_file_extension(filename: str) -> str:
    """获取文件扩展名（小写，不含点）"""
    parts = filename.rsplit(".", 1)
    if len(parts) == 2:
        return parts[1].lower()
    return ""


def _sanitize_filename(filename: str) -> str:
    """规范化文件名，去除空格和特殊字符"""
    sanitized = filename.replace(" ", "_")
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        sanitized = sanitized.replace(char, "_")
    return sanitized
