"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import re
import uuid

from app.config import STATIC_DIR, config
from loguru import logger
from app.api import chat, health, file
from app.core.milvus_client import milvus_manager
from app.services.rag_agent_service import rag_agent_service
from app.services.vector_store_manager import vector_store_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("=" * 60)
    logger.info(f"{config.app_name} v{config.app_version} 启动中...")
    logger.info(f"环境: {'开发' if config.debug else '生产'}")
    logger.info(f"监听地址: http://{config.host}:{config.port}")
    logger.info(f"API 文档: http://{config.host}:{config.port}/docs")
    
    app.state.milvus_manager = milvus_manager
    app.state.vector_store_manager = vector_store_manager
    app.state.rag_agent_service = rag_agent_service
    app.state.milvus_ready = False
    app.state.vector_store_ready = False
    app.state.knowledge_backend_ready = False
    app.state.agent_ready = False
    app.state.startup_errors = {}

    # 外部资源只在lifespan中初始化。失败时API仍可启动并通过503/health暴露降级状态。
    try:
        logger.info("正在连接 Milvus...")
        milvus_manager.connect()
        app.state.milvus_ready = True
        vector_store_manager.initialize()
        app.state.vector_store_ready = True
        app.state.knowledge_backend_ready = True
        logger.info("Milvus 与 VectorStore 初始化成功")
    except Exception as exc:
        app.state.startup_errors["knowledge_backend"] = str(exc)
        logger.error(f"知识库后端初始化失败，API将以降级模式启动: {exc}")

    try:
        rag_agent_service.initialize()
        app.state.agent_ready = True
        logger.info("RAG Agent 初始化成功")
    except Exception as exc:
        app.state.startup_errors["agent"] = str(exc)
        logger.error(f"RAG Agent初始化失败，相关接口将返回503: {exc}")
    
    logger.info("=" * 60)
    
    yield
    
    # 关闭时按依赖逆序释放资源
    await rag_agent_service.cleanup()
    vector_store_manager.shutdown()
    logger.info("正在关闭 Milvus 连接...")
    milvus_manager.close()
    logger.info(f"{config.app_name} 关闭")


CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "img-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
        "form-action 'self'",
    ]
)

async def root():
    """返回首页"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


def create_app() -> FastAPI:
    """应用工厂；导入和构造应用时不连接任何外部服务。"""
    application = FastAPI(
        title=config.app_name,
        version=config.app_version,
        description="AVF 动静脉瘘狭窄深度学习科研助手 — 基于 RAG 知识库的论文检索与科研分析",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    @application.middleware("http")
    async def add_request_context(request, call_next):  # type: ignore[no-untyped-def]
        supplied = request.headers.get("X-Request-ID", "").strip()
        request_id = (
            supplied
            if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", supplied)
            else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        with logger.contextualize(request_id=request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    if config.security_headers_enabled:
        @application.middleware("http")
        async def add_security_headers(request, call_next):  # type: ignore[no-untyped-def]
            response = await call_next(request)
            response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Permissions-Policy"] = (
                "camera=(), microphone=(), geolocation=()"
            )
            if request.url.scheme == "https":
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )
            return response
    application.include_router(health.router, tags=["健康检查"])
    application.include_router(chat.router, prefix="/api", tags=["对话"])
    application.include_router(file.router, prefix="/api", tags=["文件管理"])
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    application.add_api_route("/", root, methods=["GET"])
    return application


app = create_app()
