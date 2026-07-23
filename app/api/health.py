"""服务与外部依赖健康检查。"""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.xparse_parser_service import XParseParserService

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """报告核心依赖与可选外部能力状态，不发起外部模型请求。"""
    manager = getattr(request.app.state, "milvus_manager", milvus_manager)
    try:
        milvus_healthy = bool(manager.health_check())
        milvus_state = "connected" if milvus_healthy else "disconnected"
    except Exception as exc:
        logger.warning(f"Milvus健康检查失败: {type(exc).__name__}")
        milvus_state = "error"

    try:
        xparse_available = XParseParserService().health_check()
    except Exception as exc:
        logger.warning(f"xParse本地CLI检查失败: {type(exc).__name__}")
        xparse_available = False

    vector_ready = bool(
        getattr(request.app.state, "vector_store_ready", False)
    )
    agent_ready = bool(getattr(request.app.state, "agent_ready", False))
    core_ready = milvus_state == "connected" and vector_ready and agent_ready
    status_code = 200 if core_ready else 503

    external_dependencies: dict[str, Any] = {
        "milvus": {"status": milvus_state, "required": True},
        "dashscope": {
            "status": "configured" if config.dashscope_api_key else "not_configured",
            "required": True,
            "model": config.rag_model,
            "embedding_model": config.dashscope_embedding_model,
            "note": "configuration_only; no network probe",
        },
        "xparse": {
            "status": "available" if xparse_available else "unavailable",
            "required": False,
            "mode": config.xparse_api_mode,
            "required_for": "pdf_ingestion",
        },
        "mcp": {
            "status": "disabled",
            "required": False,
            "mode": "experimental_not_loaded",
        },
    }

    health_data: dict[str, Any] = {
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy" if core_ready else "unhealthy",
        "vector_store": {"status": "ready" if vector_ready else "unavailable"},
        "agent": {"status": "ready" if agent_ready else "unavailable"},
        # 保留旧字段，避免已有监控调用方立即失效。
        "milvus": external_dependencies["milvus"],
        "external_dependencies": external_dependencies,
    }
    if not core_ready:
        health_data["error"] = "核心依赖不可用"
        health_data["startup_errors"] = getattr(
            request.app.state, "startup_errors", {}
        )

    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": "服务运行正常" if core_ready else "服务不可用",
            "data": health_data,
        },
    )
