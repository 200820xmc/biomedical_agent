"""知识检索工具 — 两阶段检索（超额召回 + Rerank）的 Agent 工具封装

Agent 通过此工具从知识库中检索相关论文内容。
工具内部编排：Top-20超额召回 → 单次Rerank Top-10 → 按Chunk索引扩展邻居 → 上下文格式化。

和旧版的主要区别：
- 固定召回20个候选，一次性交给LLM Rerank并保留Top-10
- Rerank失败、超时或解析失败时回退到向量Top-10
- 相邻Chunk通过source_id和chunk_index直接查询，不依赖召回池
- 返回结构化 artifact 字典，包含各阶段得分和耗时
"""

from typing import Any

from langchain_core.tools import tool
from loguru import logger

from app.services.retrieval import retrieval_service
from app.utils.logger import describe_text


@tool(response_format="content_and_artifact")
async def retrieve_knowledge(
    query: str,
) -> tuple[str, dict[str, Any]]:
    """从知识库中检索相关论文内容来回答问题

    当用户的问题涉及专业知识、文献内容或需要参考资料时，使用此工具。
    工具会执行Top-20超额召回、单次语义精排和按索引的相邻Chunk扩展。

    Args:
        query: 用户的问题或查询。应尽量具体、完整，包含关键词。

    Returns:
        Tuple[str, dict]: (格式化的上下文文本, 检索详情 artifact)
            - 上下文文本: 带 [证据 N] 标签和引用格式的证据内容
            - artifact 字典: 包含 candidate_count, reranked_count, confidence, documents 等
    """
    try:
        logger.info(f"知识检索工具被调用: {describe_text(query, 'query')}")

        context, artifact = await retrieval_service.retrieve(
            query=query,
            search_mode="auto",
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
