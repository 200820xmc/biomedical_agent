"""PDF 入库编排服务 — 编排 xParse 解析 → 分块 → Embedding → Milvus 全流程

职责：
- 管理入库任务（提交、状态查询、待处理列表）
- 通过 asyncio.create_task 后台执行入库
- 任务状态持久化到 uploads/jobs/{job_id}.json
- 控制并发（免费模式串行）
- 幂等性（相同 document_id 不重复创建任务）

不引入 Celery 等重量级任务系统，使用进程内 asyncio + JSON 文件持久化。
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import config
from app.models.pdf_ingestion import IngestionJob, UploadedDocument
from app.services.vector_index_service import vector_index_service
from app.services.xparse_parser_service import XParseParserService, XParseExecutionError


# ── 目录常量 ──────────────────────────────────────────────
UPLOAD_DIR = Path("./uploads")
ORIGINALS_DIR = UPLOAD_DIR / "originals"
PARSED_DIR = UPLOAD_DIR / "parsed"
JOBS_DIR = UPLOAD_DIR / "jobs"


class PDFIngestionService:
    """PDF 入库编排服务

    使用示例:
        service = PDFIngestionService()
        job = await service.submit(document_id="doc_abc123")
        status = await service.get_status(job.job_id)
        pending = await service.list_pending()
    """

    def __init__(self) -> None:
        self._parser = XParseParserService()
        self._semaphore = asyncio.Semaphore(
            getattr(config, "xparse_max_concurrency", 1)
        )
        self._active_jobs: dict[str, asyncio.Task] = {}

        # 确保目录存在
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        PARSED_DIR.mkdir(parents=True, exist_ok=True)

        # 启动时扫描：将 interrupted 的任务标记出来
        self._scan_interrupted_jobs()

        logger.info(
            f"PDFIngestionService 初始化完成, "
            f"max_concurrency={self._semaphore._value}"
        )

    # ── 公共接口 ──────────────────────────────────────────

    async def submit(
        self,
        document_id: str,
        force_reindex: bool = False,
    ) -> IngestionJob:
        """提交 PDF 入库任务

        创建后台任务并立即返回 job_id。后台任务异步执行解析和索引。

        Args:
            document_id: PDF 的 document_id（来自上传接口）
            force_reindex: 是否强制重新索引（即使已 indexed）

        Returns:
            IngestionJob: 任务记录

        Raises:
            FileNotFoundError: document_id 对应的 PDF 不存在
            ValueError: 任务正在进行中
        """
        # ── 验证 PDF 存在 ──────────────────────────────────
        pdf_path = self._find_pdf(document_id)
        if pdf_path is None:
            raise FileNotFoundError(f"找不到 document_id={document_id} 对应的 PDF")

        # ── 幂等性：如果已有活跃任务，返回现有 job ─────────
        existing = self._find_active_job(document_id)
        if existing and not force_reindex:
            logger.info(f"document_id={document_id} 已有进行中的任务，返回现有 job")
            return existing

        # ── 创建任务记录 ────────────────────────────────────
        job_id = f"job_{uuid.uuid4().hex[:6]}"
        job = IngestionJob(
            job_id=job_id,
            document_id=document_id,
            original_filename=pdf_path.name,
            parser="xparse",
            parser_mode=self._parser.api_mode,
            status="queued",
            progress=5,
        )
        self._save_job(job)

        # ── 启动后台任务 ────────────────────────────────────
        task = asyncio.create_task(self._run_job(job_id))
        self._active_jobs[job_id] = task

        logger.info(
            f"入库任务已提交: job_id={job_id}, "
            f"document_id={document_id}, file={pdf_path.name}"
        )

        return job

    async def get_status(self, job_id: str) -> IngestionJob:
        """查询任务状态

        Args:
            job_id: 任务 ID

        Returns:
            IngestionJob: 任务记录

        Raises:
            FileNotFoundError: 任务不存在
        """
        job = self._load_job(job_id)
        if job is None:
            raise FileNotFoundError(f"任务不存在: {job_id}")
        return job

    async def list_pending(self) -> list[UploadedDocument]:
        """列出所有尚未完成入库的 PDF

        Returns:
            list[UploadedDocument]: 待处理文档列表
        """
        pending: list[UploadedDocument] = []

        if not ORIGINALS_DIR.exists():
            return pending

        for doc_dir in ORIGINALS_DIR.iterdir():
            if not doc_dir.is_dir():
                continue

            document_id = doc_dir.name

            # 查找该文档的原始 PDF
            pdf_files = list(doc_dir.glob("*.pdf"))
            if not pdf_files:
                continue

            pdf_path = pdf_files[0]

            # 检查是否已有 indexed 任务
            existing_job = self._find_active_job(document_id)
            if existing_job and existing_job.status == "indexed":
                continue

            pending.append(UploadedDocument(
                document_id=document_id,
                original_filename=pdf_path.name,
                stored_path=str(pdf_path),
                content_type="application/pdf",
                file_size=pdf_path.stat().st_size,
                sha256="",
                status=existing_job.status if existing_job else "uploaded",
            ))

        return pending

    # ── 后台任务 ──────────────────────────────────────────

    async def _run_job(self, job_id: str) -> None:
        """后台执行完整入库流程

        流程：解析 → 分块 → Embedding → Milvus 写入
        各阶段状态更新和进度推进。

        Args:
            job_id: 任务 ID
        """
        job = self._load_job(job_id)
        if job is None:
            return

        async with self._semaphore:
            try:
                # ── 步骤 1: 解析 PDF ────────────────────────
                job.status = "parsing"
                job.progress = 10
                job.updated_at = datetime.now()
                self._save_job(job)

                pdf_path = self._find_pdf(job.document_id)
                if pdf_path is None:
                    raise FileNotFoundError(f"PDF 丢失: {job.document_id}")

                output_dir = PARSED_DIR / job.document_id
                output_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"[{job_id}] 开始 xParse 解析: {pdf_path.name}")

                parse_result = await self._parser.parse_to_markdown(
                    source_path=pdf_path,
                    output_dir=output_dir,
                )

                job.parser_exit_code = parse_result.exit_code
                job.parser_suggestion_tag = parse_result.suggestion_tag
                job.parser_request_id = parse_result.request_id

                logger.info(
                    f"[{job_id}] xParse 解析完成: {parse_result.markdown_path}"
                )

                # ── 步骤 2: 读取 Markdown ────────────────────
                job.status = "parsed"
                job.progress = 65
                job.markdown_path = parse_result.markdown_path
                job.updated_at = datetime.now()
                self._save_job(job)

                md_content = Path(parse_result.markdown_path).read_text(encoding="utf-8")

                if not md_content.strip():
                    raise RuntimeError("xParse 生成的 Markdown 内容为空")

                # ── 步骤 3: 分块 ─────────────────────────────
                job.status = "splitting"
                job.progress = 75
                job.updated_at = datetime.now()
                self._save_job(job)

                # ── 步骤 4: Embedding + Milvus 写入 ──────────
                job.status = "embedding"
                job.progress = 80
                job.updated_at = datetime.now()
                self._save_job(job)

                chunk_count = vector_index_service.index_content(
                    content=md_content,
                    logical_source=str(pdf_path),
                    display_filename=pdf_path.name,
                    parsed_source=parse_result.markdown_path,
                    extra_metadata={
                        "_parser": "TextIn xParse",
                        "_parser_mode": self._parser.api_mode,
                        "_document_id": job.document_id,
                    },
                )

                # ── 完成！ ────────────────────────────────────
                job.status = "indexed"
                job.progress = 100
                job.chunk_count = chunk_count
                job.updated_at = datetime.now()
                self._save_job(job)

                logger.info(
                    f"[{job_id}] 入库完成: {pdf_path.name} → "
                    f"{chunk_count} 个分片已写入 Milvus"
                )

            except XParseExecutionError as e:
                self._mark_failed(job, f"XPARSE_EXECUTION_ERROR", str(e), e.exit_code)
            except asyncio.TimeoutError:
                self._mark_failed(job, "XPARSE_TIMEOUT", "xParse 解析超时")
            except FileNotFoundError as e:
                self._mark_failed(job, "PDF_NOT_FOUND", str(e))
            except RuntimeError as e:
                self._mark_failed(job, "EMBEDDING_FAILED", str(e))
            except Exception as e:
                self._mark_failed(job, "UNKNOWN_ERROR", str(e))
            finally:
                # 清理活跃任务引用
                self._active_jobs.pop(job_id, None)

    def _mark_failed(
        self,
        job: IngestionJob,
        error_code: str,
        message: str,
        exit_code: Optional[int] = None,
    ) -> None:
        """标记任务失败"""
        job.status = "failed"
        job.error_code = error_code
        job.error_message = message
        if exit_code is not None:
            job.parser_exit_code = exit_code
        job.updated_at = datetime.now()
        self._save_job(job)
        logger.error(f"[{job.job_id}] 入库失败 [{error_code}]: {message[:200]}")

    # ── 辅助方法 ──────────────────────────────────────────

    def _find_pdf(self, document_id: str) -> Optional[Path]:
        """查找 document_id 对应的 PDF 文件"""
        doc_dir = ORIGINALS_DIR / document_id
        if not doc_dir.exists():
            return None
        pdf_files = list(doc_dir.glob("*.pdf"))
        return pdf_files[0] if pdf_files else None

    def _find_active_job(self, document_id: str) -> Optional[IngestionJob]:
        """查找 document_id 对应的活跃任务"""
        if not JOBS_DIR.exists():
            return None
        for job_file in JOBS_DIR.glob("*.json"):
            job = self._load_job_from_file(job_file)
            if job and job.document_id == document_id:
                if job.status in ("queued", "parsing", "parsed", "splitting", "embedding"):
                    return job
        return None

    def _save_job(self, job: IngestionJob) -> None:
        """将任务状态保存到 JSON 文件"""
        job_path = JOBS_DIR / f"{job.job_id}.json"
        data = job.model_dump(mode="json")
        job_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_job(self, job_id: str) -> Optional[IngestionJob]:
        """从 JSON 文件加载任务"""
        return self._load_job_from_file(JOBS_DIR / f"{job_id}.json")

    @staticmethod
    def _load_job_from_file(file_path: Path) -> Optional[IngestionJob]:
        """从指定 JSON 文件加载任务"""
        if not file_path.exists():
            return None
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return IngestionJob(**data)
        except Exception as e:
            logger.warning(f"加载任务文件失败: {file_path}, 错误: {e}")
            return None

    def _scan_interrupted_jobs(self) -> None:
        """扫描并标记异常中断的任务"""
        if not JOBS_DIR.exists():
            return
        interrupted = 0
        for job_file in JOBS_DIR.glob("*.json"):
            job = self._load_job_from_file(job_file)
            if job and job.status in ("parsing", "splitting", "embedding"):
                job.status = "interrupted"
                job.updated_at = datetime.now()
                self._save_job(job)
                interrupted += 1
        if interrupted:
            logger.warning(f"发现 {interrupted} 个异常中断的任务，已标记为 interrupted")


# ── 全局单例 ──────────────────────────────────────────────
pdf_ingestion_service = PDFIngestionService()
