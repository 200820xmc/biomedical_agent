"""PDF 入库 Agent 工具 — 让 Agent 可以触发 PDF 解析和知识库入库

三个工具：
- list_pending_pdf_documents: 列出尚未入库的 PDF
- ingest_pdf_to_knowledge_base: 提交解析和入库任务
- get_pdf_ingestion_status: 查询任务进度

安全要求：
- 只接受 document_id，不接受文件路径
- 解析器和 API 模式由后端配置控制
- 不返回凭证信息
"""

from typing import Any

from langchain_core.tools import tool
from loguru import logger

from app.models.pdf_ingestion import PDFIngestionInput, IngestionStatusInput
from app.services.pdf_ingestion_service import pdf_ingestion_service


@tool(response_format="content_and_artifact")
async def list_pending_pdf_documents() -> tuple[str, list[dict[str, Any]]]:
    """列出 uploads 目录中尚未完成知识库入库的 PDF 文件

    返回所有状态不是 indexed 的 PDF，包括 uploaded（仅上传）和 failed（入库失败）的。

    Returns:
        Tuple[str, list[dict]]: (可读文本摘要, 文档列表)
    """
    try:
        documents = await pdf_ingestion_service.list_pending()

        if not documents:
            return "当前没有等待入库的 PDF 文件。", []

        # 构建可读文本
        lines = ["以下 PDF 尚未入库：", ""]
        for doc in documents:
            lines.append(
                f"- {doc.original_filename}"
                f"  document_id={doc.document_id}"
                f"  状态={doc.status}"
            )

        text = "\n".join(lines)
        data = [doc.model_dump() for doc in documents]

        logger.info(f"list_pending: 找到 {len(documents)} 个待处理 PDF")
        return text, data

    except Exception as e:
        logger.error(f"list_pending_pdf_documents 失败: {e}")
        return f"查询待处理 PDF 时出错: {e}", []


@tool(args_schema=PDFIngestionInput, response_format="content_and_artifact")
async def ingest_pdf_to_knowledge_base(
    document_id: str,
    force_reindex: bool = False,
) -> tuple[str, dict[str, Any]]:
    """提交 PDF 解析和知识库入库任务

    **仅当用户明确要求解析、导入或索引某个 PDF 时使用此工具。**

    工具会：
    1. 验证 document_id 对应的 PDF 文件是否存在
    2. 创建后台入库任务（xParse 解析 → 分块 → Embedding → Milvus）
    3. 立即返回 job_id，不等待入库完成

    注意：
    - document_id 必须来自上传接口的返回值或待处理文件列表
    - 不得编造 document_id
    - 入库需要时间，请稍后通过 get_pdf_ingestion_status 查询进度
    - 返回 queued 状态不代表入库已完成

    Args:
        document_id: PDF 文档唯一标识（来自上传接口）
        force_reindex: 是否强制重新解析和索引（默认 False）

    Returns:
        Tuple[str, dict]: (任务状态描述, 任务详情)
    """
    try:
        logger.info(f"ingest_pdf 被调用: document_id={document_id}, force={force_reindex}")

        job = await pdf_ingestion_service.submit(
            document_id=document_id,
            force_reindex=force_reindex,
        )

        message = (
            f"PDF 入库任务已提交。\n"
            f"文件：{job.original_filename}\n"
            f"任务 ID：{job.job_id}\n"
            f"解析器：TextIn xParse（{job.parser_mode} 模式）\n"
            f"当前状态：{job.status}\n"
            f"⚠️ 任务尚未完成，请稍后通过 get_pdf_ingestion_status 查询进度。"
        )

        return message, job.model_dump()

    except FileNotFoundError as e:
        logger.warning(f"ingest_pdf: PDF 不存在 - {e}")
        return (
            f"找不到 document_id={document_id} 对应的 PDF 文件。"
            f"请确认文件已上传且 document_id 正确。",
            {"error": "PDF_NOT_FOUND", "document_id": document_id},
        )
    except ValueError as e:
        logger.warning(f"ingest_pdf: 参数错误 - {e}")
        return (
            f"无法提交入库任务：{e}",
            {"error": "INVALID_REQUEST", "detail": str(e)},
        )
    except Exception as e:
        logger.error(f"ingest_pdf 失败: {e}")
        return (
            f"提交入库任务时出错：{e}",
            {"error": "INTERNAL_ERROR", "detail": str(e)},
        )


@tool(args_schema=IngestionStatusInput, response_format="content_and_artifact")
async def get_pdf_ingestion_status(
    job_id: str,
) -> tuple[str, dict[str, Any]]:
    """查询 PDF 解析和知识库入库任务的当前状态

    用户询问"入库完成了吗"、"任务进度如何"时使用此工具。

    状态说明：
    - queued: 任务已提交，等待执行
    - parsing: 正在调用 xParse 解析 PDF
    - parsed: Markdown 已生成，准备分块
    - splitting: 正在对 Markdown 进行分块
    - embedding: 正在生成向量并写入 Milvus
    - indexed: ✅ 入库完成，PDF 内容已进入知识库
    - failed: ❌ 入库失败
    - interrupted: ⚠️ 任务被中断（服务重启等原因）

    Args:
        job_id: PDF 入库任务 ID（来自 ingest_pdf_to_knowledge_base 的返回值）

    Returns:
        Tuple[str, dict]: (状态描述, 任务详情)
    """
    try:
        job = await pdf_ingestion_service.get_status(job_id)

        if job.status == "indexed":
            message = (
                f"✅ PDF 已完成入库！\n"
                f"文件：{job.original_filename}\n"
                f"解析器：{job.parser}\n"
                f"生成分片：{job.chunk_count} 个\n"
                f"状态：已进入知识库，可以开始提问。"
            )
        elif job.status == "failed":
            message = (
                f"❌ PDF 入库失败。\n"
                f"文件：{job.original_filename}\n"
                f"错误码：{job.error_code}\n"
                f"错误信息：{job.error_message}\n"
                f"请尝试重新提交或检查 PDF 文件是否正常。"
            )
        elif job.status == "interrupted":
            message = (
                f"⚠️ 任务曾被中断（可能是服务重启）。\n"
                f"文件：{job.original_filename}\n"
                f"请重新提交入库任务。"
            )
        else:
            message = (
                f"⏳ 任务仍在处理中。\n"
                f"文件：{job.original_filename}\n"
                f"任务 ID：{job.job_id}\n"
                f"当前状态：{job.status}\n"
                f"进度：{job.progress}%\n"
                f"请稍后再查询。"
            )

        logger.info(
            f"get_status: job_id={job_id}, status={job.status}, progress={job.progress}%"
        )

        return message, job.model_dump()

    except FileNotFoundError:
        return (
            f"找不到任务 {job_id}。请确认任务 ID 是否正确。",
            {"error": "JOB_NOT_FOUND", "job_id": job_id},
        )
    except Exception as e:
        logger.error(f"get_pdf_ingestion_status 失败: {e}")
        return (
            f"查询任务状态时出错：{e}",
            {"error": "INTERNAL_ERROR", "detail": str(e)},
        )
