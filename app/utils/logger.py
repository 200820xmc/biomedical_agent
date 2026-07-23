"""日志配置模块

使用 Loguru 配置应用日志
"""

import hashlib
import sys

from loguru import logger

from app.config import LOGS_DIR, config


def describe_text(value: str, label: str = "text") -> str:
    """返回文本长度和不可逆短哈希，避免日志保存科研问题正文。"""
    encoded = value.encode("utf-8", errors="replace")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{label}_len={len(value)}, {label}_sha256={digest}"


def format_exception_chain(exc: BaseException) -> str:
    """展开ExceptionGroup/TaskGroup，便于定位子异常。"""
    sub_exceptions = getattr(exc, "exceptions", None)
    if sub_exceptions is not None:
        lines = [f"{type(exc).__name__}: {exc}"]
        for index, sub_exception in enumerate(sub_exceptions):
            lines.append(
                f"  [{index}] {format_exception_chain(sub_exception)}"
            )
        return "\n".join(lines)
    message = f"{type(exc).__name__}: {exc}"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return f"{message}\n  caused by: {format_exception_chain(cause)}"
    return message


def setup_logger():
    """配置日志系统

    按照 Loguru 最佳实践配置全局 logger：
    1. 移除默认处理器
    2. 添加控制台输出（带颜色）
    3. 添加文件输出（按天轮转，自动压缩，异步写入）
    """
    # 移除默认处理器
    logger.remove()
    logger.configure(
        patcher=lambda record: record["extra"].setdefault("request_id", "-")
    )
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "request_id={extra[request_id]} | {module}.{function}:{line} | {message}"
    )

    # 添加控制台输出（带颜色格式）
    logger.add(
        sys.stdout,
        format=log_format,
        level="DEBUG" if config.debug else "INFO",
        colorize=True,
        backtrace=True,  # 显示完整异常栈信息
        diagnose=config.debug,  # Debug 模式下显示变量值
    )

    # 添加文件输出（按天轮转，自动压缩）
    file_sink_options = {
        "rotation": "00:00",
        "retention": "7 days",
        "compression": "zip",
        "encoding": "utf-8",
        "backtrace": True,
        "diagnose": config.debug,
        "level": "INFO",
        "format": log_format,
    }
    try:
        logger.add(
            LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
            enqueue=True,
            **file_sink_options,
        )
    except (OSError, PermissionError):
        # 受限Windows账户可能禁止Loguru创建多进程管道；同步文件日志仍可用。
        logger.add(
            LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
            enqueue=False,
            **file_sink_options,
        )

setup_logger()
