"""AVF Research Assistant 的统一 Python 服务入口。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

from app.config import config


def main() -> None:
    """从项目配置启动Uvicorn，不覆写代理或外部API地址。"""
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info",
    )


if __name__ == "__main__":
    main()
