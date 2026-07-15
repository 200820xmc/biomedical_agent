"""工具模块 — 供 Agent 调用的各种工具"""

from app.tools.knowledge_tool import retrieve_knowledge
from app.tools.time_tool import get_current_time

# 默认本地工具集：知识检索 + 时间
DEFAULT_LOCAL_AGENT_TOOLS = (
    retrieve_knowledge,
    get_current_time,
)

__all__ = [
    "DEFAULT_LOCAL_AGENT_TOOLS",
    "retrieve_knowledge",
    "get_current_time",
]
