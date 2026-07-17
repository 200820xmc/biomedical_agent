"""检索数据模型 — 超额召回、Rerank 和上下文格式化使用的数据结构"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalItem:
    """单个检索候选项

    表示一个候选 chunk，包含向量检索得分和（可选的）Rerank 得分。
    """

    chunk_id: str
    """chunk 唯一标识，来自 Milvus 主键"""

    source_id: str
    """文档稳定标识，来自 metadata['_source'] 或 metadata.get('source_id')"""

    source: str
    """显示用的文件名，例如 'Zhou_2023.pdf'"""

    chunk_index: int
    """chunk 在文档内的顺序编号，0-based"""

    content: str
    """chunk 正文内容"""

    vector_score: float | None = None
    """向量相似度得分（0~1，越高越相关），由 L2 距离转换而来"""

    rerank_score: float | None = None
    """Rerank 语义相关性得分（0~1，越高越相关），仅在 Rerank 阶段后填充"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """原始 metadata 字典，包含 h1、h2、_file_name 等字段"""


@dataclass
class RetrievalArtifact:
    """检索过程的完整结构化输出

    用于：
    - 检索效果评测（对比原始召回 vs Rerank vs 最终选择）
    - 问题排查和日志记录
    - 前端展示引用卡片
    - 保存真实 Agent 检索轨迹
    """

    original_query: str
    """原始用户问题"""

    search_mode: str
    """检索模式：auto / focused / comparison / broad"""

    candidate_count: int
    """Milvus 超额召回的候选数量"""

    reranked_count: int
    """Rerank 后保留的数量（Rerank 未启用时与 candidate_count 相同）"""

    selected_count: int
    """来源多样性选择后的最终 chunk 数量"""

    confidence: str = "medium"
    """检索置信度：high / medium / low"""

    rerank_applied: bool = False
    """是否实际执行了 Rerank（降级时为 False）"""

    documents: list[RetrievalItem] = field(default_factory=list)
    """最终返回给模型的文档列表"""

    duration_ms: dict[str, float] = field(default_factory=dict)
    """各阶段耗时（毫秒），例如 {'recall': 120, 'rerank': 800, 'total': 950}"""
