"""FastAPI服务依赖：资源就绪检查与轻量并发保护。"""

import asyncio
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request, status

from app.config import config
from app.services.rag_agent_service import RagAgentService


_chat_semaphore = asyncio.Semaphore(max(1, config.chat_max_concurrency))
_upload_semaphore = asyncio.Semaphore(max(1, config.upload_max_concurrency))


async def _acquire_slot(
    semaphore: asyncio.Semaphore,
    service_name: str,
) -> AsyncIterator[None]:
    acquired = False
    try:
        await asyncio.wait_for(
            semaphore.acquire(),
            timeout=max(0.01, config.request_queue_timeout_seconds),
        )
        acquired = True
        yield
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"{service_name}请求过多，请稍后重试",
        ) from exc
    finally:
        if acquired:
            semaphore.release()


async def chat_concurrency_slot() -> AsyncIterator[None]:
    async for item in _acquire_slot(_chat_semaphore, "问答"):
        yield item


async def upload_concurrency_slot() -> AsyncIterator[None]:
    async for item in _acquire_slot(_upload_semaphore, "上传"):
        yield item


def get_agent_service(request: Request) -> RagAgentService:
    """返回lifespan初始化的Agent；初始化失败时明确返回503。"""
    service = getattr(request.app.state, "rag_agent_service", None)
    if service is None or not getattr(request.app.state, "agent_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Agent服务尚未就绪",
        )
    return service


def get_ready_rag_agent_service(request: Request) -> RagAgentService:
    """对问答入口同时要求Agent和知识库后端就绪。"""
    service = get_agent_service(request)
    if not getattr(request.app.state, "knowledge_backend_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="知识库后端不可用，请确认Milvus连接状态",
        )
    return service
