"""TextIn xParse PDF 解析服务 — 安全封装 xparse-cli 命令行工具

通过异步子进程调用 xparse-cli，将 PDF 解析为 Markdown。
包含完整的错误处理、超时控制、建议标签解析和输出验证。

安全要求：
- 必须使用参数数组（asyncio.create_subprocess_exec），禁止 shell=True
- 凭证通过环境变量传递，不出现在命令行参数中
- 路径由后端生成，不接受用户输入
"""

import asyncio
import re
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from app.config import config


class ParseResult(BaseModel):
    """xParse 解析结果"""

    markdown_path: str
    """生成的 Markdown 文件路径"""

    exit_code: int
    """CLI 退出码（0=成功）"""

    api_mode: str
    """使用的 API 模式（free/paid）"""

    suggestion_tag: Optional[str] = None
    """stderr 中的建议标签：fix / retry / fallback / ask human"""

    request_id: Optional[str] = None
    """TextIn API 的 request_id，用于排查问题"""


class XParseExecutionError(Exception):
    """xparse-cli 执行异常"""

    def __init__(self, exit_code: int, message: str, suggestion_tag: Optional[str] = None):
        self.exit_code = exit_code
        self.message = message
        self.suggestion_tag = suggestion_tag
        super().__init__(f"xparse-cli 退出码 {exit_code}: {message[:200]}")


class XParseParserService:
    """TextIn xParse PDF 解析服务

    职责：
    - 检查 CLI 是否已安装
    - 构建受控参数列表
    - 通过异步子进程执行 xparse-cli
    - 设置超时和错误处理
    - 解析 stderr 中的建议标签
    - 验证输出 Markdown 文件
    """

    def __init__(self) -> None:
        self.cli_path = getattr(config, "xparse_cli_path", "xparse-cli")
        self.timeout = getattr(config, "xparse_timeout_seconds", 600)
        self.api_mode = getattr(config, "xparse_api_mode", "free")
        self.include_image_data = getattr(config, "xparse_include_image_data", False)
        self._max_retries = getattr(config, "xparse_max_retries", 1)

        logger.info(
            f"XParseParserService 初始化: cli={self.cli_path}, "
            f"mode={self.api_mode}, timeout={self.timeout}s"
        )

    def health_check(self) -> bool:
        """检查 xparse-cli 是否可用

        Returns:
            bool: CLI 已安装且可执行
        """
        return shutil.which(self.cli_path) is not None

    async def parse_to_markdown(
        self,
        source_path: Path,
        output_dir: Path,
        page_range: Optional[str] = None,
        password: Optional[str] = None,
    ) -> ParseResult:
        """将 PDF 解析为 Markdown

        Args:
            source_path: PDF 源文件路径
            output_dir: Markdown 输出目录
            page_range: 可选，页面范围（如 "1-10"）
            password: 可选，加密 PDF 的密码

        Returns:
            ParseResult: 解析结果

        Raises:
            FileNotFoundError: PDF 文件不存在
            ValueError: 文件格式不支持
            XParseExecutionError: CLI 执行失败
            RuntimeError: 超时或输出为空
        """
        # ── 输入校验 ──────────────────────────────────────
        if not source_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {source_path}")

        if source_path.suffix.lower() != ".pdf":
            raise ValueError(f"当前只支持 PDF 文件，收到: {source_path.suffix}")

        # ── 确保输出目录存在 ──────────────────────────────
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── 构建安全参数列表 ──────────────────────────────
        args = [
            self.cli_path,
            "parse",
            str(source_path),
            "--view", "markdown",
            "--api", self.api_mode,
            "--output", str(output_dir),
        ]

        if page_range:
            args.extend(["--page-range", page_range])

        if password:
            args.extend(["--password", password])

        if not self.include_image_data:
            args.append("--include-image-data=false")

        # ── 构建子进程环境（注入凭证，不记录完整环境） ────
        env = _build_subprocess_env()

        logger.info(
            f"启动 xparse-cli: source={source_path.name}, "
            f"mode={self.api_mode}, output_dir={output_dir}"
        )

        # ── 重试逻辑 ──────────────────────────────────────
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._run_parse(args, env, source_path, output_dir)
            except XParseExecutionError as e:
                # 只有 [retry] 标签才重试
                if e.suggestion_tag == "retry" and attempt < self._max_retries:
                    wait = 2 ** attempt  # 指数退避：1s, 2s
                    logger.warning(
                        f"xparse-cli 返回 [retry]，{wait}s 后重试 "
                        f"({attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(wait)
                    last_error = e
                else:
                    raise
            except (TimeoutError, RuntimeError) as e:
                if attempt < self._max_retries:
                    logger.warning(f"xparse-cli 超时，重试 ({attempt + 1}/{self._max_retries})")
                    last_error = e
                else:
                    raise

        # 不应到达这里
        raise last_error  # type: ignore[misc]

    async def _run_parse(
        self,
        args: list[str],
        env: Optional[dict],
        source_path: Path,
        output_dir: Path,
    ) -> ParseResult:
        """执行单次解析

        Args:
            args: CLI 参数列表
            env: 子进程环境变量
            source_path: PDF 源文件
            output_dir: 输出目录

        Returns:
            ParseResult: 解析结果
        """
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"xparse-cli 解析超时（{self.timeout}s）: {source_path.name}"
            )

        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""

        # 解析建议标签
        suggestion_tag = _parse_suggestion_tag(stderr_text)

        # 解析 request_id（通常出现在 stderr 中）
        request_id = _parse_request_id(stderr_text)

        # ── 检查退出码 ────────────────────────────────────
        if process.returncode != 0:
            # 检查是否为免费限制
            if "free" in self.api_mode.lower() and _is_free_limit(stderr_text):
                raise XParseExecutionError(
                    exit_code=process.returncode,
                    message="免费 API 限制：文件可能超过 10MB 或达到调用上限",
                    suggestion_tag="ask human",
                )

            # 检查是否需要密码
            if _is_password_required(stderr_text):
                raise XParseExecutionError(
                    exit_code=process.returncode,
                    message="PDF 已加密，需要提供密码",
                    suggestion_tag="ask human",
                )

            raise XParseExecutionError(
                exit_code=process.returncode,
                message=stderr_text or "未知错误",
                suggestion_tag=suggestion_tag,
            )

        # ── 查找生成的 Markdown 文件 ──────────────────────
        # xparse-cli 在 output_dir 下生成一个与 PDF 同名的 .md 文件
        md_name = source_path.stem + ".md"
        md_path = output_dir / md_name

        if not md_path.exists():
            # 可能文件名被 CLI 改变了，尝试搜索
            md_files = list(output_dir.glob("*.md"))
            if md_files:
                md_path = md_files[0]
                logger.info(f"Markdown 文件不匹配预期名称，使用: {md_path.name}")
            else:
                raise RuntimeError(
                    f"xparse-cli 执行成功但未生成 Markdown 文件: {output_dir}"
                )

        if md_path.stat().st_size == 0:
            raise RuntimeError(f"xparse-cli 生成的 Markdown 文件为空: {md_path.name}")

        file_size_kb = md_path.stat().st_size / 1024
        logger.info(
            f"xparse-cli 解析成功: {source_path.name} → {md_path.name} "
            f"({file_size_kb:.1f} KB), request_id={request_id}"
        )

        return ParseResult(
            markdown_path=str(md_path),
            exit_code=process.returncode,
            api_mode=self.api_mode,
            suggestion_tag=suggestion_tag,
            request_id=request_id,
        )


# ── 子进程环境构建 ─────────────────────────────────────


def _build_subprocess_env() -> dict | None:
    """构建 xparse-cli 子进程环境变量

    继承当前进程环境，注入 XPARSE_APP_ID 和 XPARSE_SECRET_CODE。
    不在日志中输出凭证内容。

    Returns:
        dict | None: 环境变量字典（None 表示完全继承）
    """
    import os

    app_id = getattr(config, "xparse_app_id", "")
    secret = getattr(config, "xparse_secret_code", "")

    if not app_id or not secret:
        # 免费模式，不需要额外凭证
        return None

    env = os.environ.copy()
    env["XPARSE_APP_ID"] = app_id
    env["XPARSE_SECRET_CODE"] = secret

    logger.debug("已注入 XPARSE_APP_ID 和 XPARSE_SECRET_CODE 到子进程环境")
    return env


# ── stderr 解析工具函数 ────────────────────────────────────


def _parse_suggestion_tag(stderr_text: str) -> Optional[str]:
    """从 stderr 中解析建议标签

    支持的标签：fix / retry / fallback / ask human

    Args:
        stderr_text: CLI stderr 输出

    Returns:
        Optional[str]: 标签名称（小写），未找到返回 None
    """
    if not stderr_text:
        return None
    match = re.search(
        r"\[(fix|retry|fallback|ask\s*human)\]",
        stderr_text,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower().replace(" ", "_") if match else None


def _parse_request_id(stderr_text: str) -> Optional[str]:
    """从 stderr 中提取 TextIn API request_id

    Args:
        stderr_text: CLI stderr 输出

    Returns:
        Optional[str]: request_id 或 None
    """
    if not stderr_text:
        return None
    # 尝试匹配常见的 request_id 格式
    match = re.search(r"request[_-]?id[:\s]+([a-zA-Z0-9\-]+)", stderr_text, re.IGNORECASE)
    return match.group(1) if match else None


def _is_free_limit(stderr_text: str) -> bool:
    """判断是否为免费 API 限制错误"""
    keywords = ["free", "limit", "exceed", "quota", "restriction", "too large"]
    text_lower = stderr_text.lower()
    return any(kw in text_lower for kw in keywords)


def _is_password_required(stderr_text: str) -> bool:
    """判断是否因 PDF 加密需要密码"""
    keywords = ["password", "encrypted", "decrypt", "protected"]
    text_lower = stderr_text.lower()
    return any(kw in text_lower for kw in keywords)
