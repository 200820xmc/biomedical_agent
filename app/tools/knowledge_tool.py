"""知识检索工具 — 两阶段检索（超额召回 + Rerank）的 Agent 工具封装

Agent 通过此工具从知识库中检索相关论文内容。
工具内部编排：超额召回 → Rerank 精排 → 来源多样性选择 → 上下文格式化。

和旧版的主要区别：
- 候选召回从 15 个提升到 50 个（超额召回）
- 新增 LLM 语义 Rerank，替代简单的向量相似度排序
- Rerank 之后才进行来源去重（不再提前丢弃同论文的 chunk）
- 每篇论文最多保留 2 个 chunk（原来只保留 1 个）
- 返回结构化 artifact 字典，包含各阶段得分和耗时
"""

from typing import Any

from langchain_core.tools import tool
from loguru import logger

from app.services.retrieval import retrieval_service


@tool(response_format="content_and_artifact")
async def retrieve_knowledge(
    query: str,
    top_k: int | None = None,
    search_mode: str = "auto",
    source_filter: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """从知识库中检索相关论文内容来回答问题

    当用户的问题涉及专业知识、文献内容或需要参考资料时，使用此工具。
    工具会执行超额召回、语义精排和来源多样性选择。

    Args:
        query: 用户的问题或查询。应尽量具体、完整，包含关键词。
        top_k: 最终返回的论文 chunk 数量。留空则根据检索模式自动选择。
        search_mode: 检索模式。
            - "auto"（默认）：根据问题自动选择最佳参数
            - "focused"：适用于单一概念或单篇论文的深入分析
            - "comparison"：适用于多篇论文、多模型之间的对比
            - "broad"：适用于文献综述、研究进展等广泛问题
        source_filter: 可选的论文来源过滤列表（暂未启用）。

    Returns:
        Tuple[str, dict]: (格式化的上下文文本, 检索详情 artifact)
            - 上下文文本: 带 [证据 N] 标签和引用格式的证据内容
            - artifact 字典: 包含 candidate_count, reranked_count, confidence, documents 等
    """
    try:
        logger.info(
            f"知识检索工具被调用: query='{query[:100]}...', "
            f"mode={search_mode}, top_k={top_k}"
        )

        context, artifact = await retrieval_service.retrieve(
            query=query,
            search_mode=search_mode,
            top_k=top_k,
            source_filter=source_filter,
        )

        # 低置信度时在上下文前添加提示
        if artifact.get("confidence") == "low":
            context = (
                "（注意：以下检索结果的置信度较低，知识库中可能缺少直接相关的论文。"
                "请在回答中如实告知用户。）\n\n"
                + context
            )

        logger.info(
            f"知识检索完成: 选中{artifact.get('selected_count', 0)}个chunk, "
            f"置信度={artifact.get('confidence', 'unknown')}, "
            f"总耗时={artifact.get('duration_ms', {}).get('total', 0):.0f}ms"
        )

        return context, artifact

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return (
            f"检索知识时发生错误: {str(e)}",
            {
                "error": str(e),
                "confidence": "low",
                "documents": [],
            },
        )
