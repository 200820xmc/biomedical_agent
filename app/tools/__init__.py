"""工具模块 — 供 Agent 调用的各种工具"""

from app.tools.knowledge_tool import retrieve_knowledge
from app.tools.time_tool import get_current_time
from app.tools.pdf_ingestion_tool import (
    get_pdf_ingestion_status,
    ingest_pdf_to_knowledge_base,
    list_pending_pdf_documents,
)

# 默认本地工具集：知识检索 + 时间 + PDF 入库
DEFAULT_LOCAL_AGENT_TOOLS = (
    retrieve_knowledge,
    get_current_time,
    list_pending_pdf_documents,
    ingest_pdf_to_knowledge_base,
    get_pdf_ingestion_status,
)

__all__ = [
    "DEFAULT_LOCAL_AGENT_TOOLS",
    "retrieve_knowledge",
    "get_current_time",
    "list_pending_pdf_documents",
    "ingest_pdf_to_knowledge_base",
    "get_pdf_ingestion_status",
]
