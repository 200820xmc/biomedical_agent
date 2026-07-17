"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "AVF Research Assistant"
    app_version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 5
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # ── 超额召回与 Rerank 配置 ──────────────────────────────
    # 最终结果
    rag_final_chunks: int = 8        # 最终返回给模型的 chunk 数量
    rag_max_sources: int = 5         # 最多覆盖的来源（论文）数量
    rag_max_chunks_per_source: int = 2  # 每篇论文最多保留的 chunk 数量

    # 超额召回
    rag_candidate_k: int = 50        # Milvus 超额召回的候选数量
    rag_rerank_k: int = 20           # Rerank 后保留的候选数量

    # Rerank
    rag_rerank_enabled: bool = True  # 是否启用 LLM Rerank
    rag_rerank_model: str = ""       # Rerank 专用模型（空字符串表示复用 rag_model）
    rag_rerank_threshold: float = 0.65  # 最低相关性阈值（低于此分数的候选不进入上下文）

    # 上下文预算
    rag_max_context_tokens: int = 6000   # 上下文最大 token 数（估算用）
    rag_max_context_chars: int = 12000   # 上下文最大字符数（实际截断依据）
    rag_max_chars_per_evidence: int = 1600  # 单个证据最大字符数

    # Query 处理（第二阶段启用）
    rag_query_rewrite_enabled: bool = False
    rag_multi_query_enabled: bool = False
    rag_multi_query_count: int = 3

    # 文档分块配置
    chunk_max_size: int = 1600
    chunk_overlap: int = 200
    min_chunk_size: int = 300

    # MCP 服务配置（预留：可接入 PubMed 检索等科研工具）
    # transport 类型：stdio | sse | streamable-http
    # 示例：mcp_pubmed_url: str = "http://localhost:8003/mcp"

    # ── TextIn xParse PDF 解析配置 ─────────────────────────
    xparse_cli_path: str = r"C:\Users\Ming\.xparse-cli\bin\xparse-cli.exe"
    xparse_api_mode: str = "free"
    xparse_app_id: str = ""            # 付费 API 的 App ID
    xparse_secret_code: str = ""       # 付费 API 的 Secret Code
    xparse_base_url: str = ""
    xparse_timeout_seconds: int = 600
    xparse_max_retries: int = 1
    xparse_max_concurrency: int = 1
    xparse_include_image_data: bool = False
    pdf_max_file_size: int = 10 * 1024 * 1024  # 10MB（免费模式限制），付费模式最大 500MB


# 全局配置实例
config = Settings()
