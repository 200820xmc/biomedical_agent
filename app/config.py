"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
LOGS_DIR = PROJECT_ROOT / "logs"
STATIC_DIR = PROJECT_ROOT / "static"


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
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

    # Web安全与资源限制
    cors_allowed_origins: str = "http://localhost:9900,http://127.0.0.1:9900"
    security_headers_enabled: bool = True
    max_question_length: int = 1000
    max_session_id_length: int = 128
    max_filename_length: int = 255
    chat_max_concurrency: int = 8
    upload_max_concurrency: int = 2
    request_queue_timeout_seconds: float = 1.0

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # ── 超额召回与 Rerank 配置 ──────────────────────────────
    # 最终结果
    rag_final_chunks: int = 5        # Rerank/阈值后选入上下文的 Top-K 数量（邻居另行扩展）
    # 超额召回
    rag_candidate_k: int = 20        # Milvus超额召回候选数量（单次Rerank输入）
    rag_rerank_k: int = 10           # 单次Rerank后保留数量

    # Rerank
    rag_rerank_enabled: bool = True  # 是否启用专用 Rerank
    rag_rerank_model: str = "qwen3-rerank"
    rag_rerank_timeout_seconds: float = 30.0
    rag_rerank_threshold: float = 0.65  # 最低相关性阈值（低于此分数的候选不进入上下文）
    rag_threshold_fallback_k: int = 3   # LLM精排全部低于阈值时的Top-N保底

    # 上下文预算
    rag_max_context_chars: int = 12000   # 上下文最大字符数（实际截断依据）
    rag_max_chars_per_evidence: int = 1600  # 单个证据最大字符数

    # 文档分块配置
    chunk_max_size: int = 1600
    chunk_overlap: int = 200
    min_chunk_size: int = 300

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

    @property
    def cors_origins(self) -> list[str]:
        """将逗号分隔的允许来源转换为CORS中间件列表。"""
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]


# 全局配置实例
config = Settings()
