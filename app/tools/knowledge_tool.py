"""知识检索工具 - 从向量数据库中检索相关信息"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题
    
    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    
    Args:
        query: 用户的问题或查询
        
    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 从向量存储中检索相关文档
        # 先拉取 top_k * 3 个候选分片（同一个论文可能被切成多片，需要超额拉取）
        fetch_count = config.rag_top_k * 3
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": fetch_count}
        )

        candidate_docs = retriever.invoke(query)

        if not candidate_docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        # 按论文来源去重：同一篇论文只保留最相关的那个分片
        seen_files = set()
        deduped_docs = []
        for doc in candidate_docs:
            file_name = doc.metadata.get("_file_name", "")
            if file_name not in seen_files:
                seen_files.add(file_name)
                deduped_docs.append(doc)
            if len(deduped_docs) >= config.rag_top_k:
                break

        docs = deduped_docs

        # 格式化文档为上下文
        context = format_docs(docs)

        logger.info(f"检索到 {len(candidate_docs)} 个候选分片 → 去重后 {len(docs)} 篇论文（目标 {config.rag_top_k} 篇）")
        return context, docs
        
    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def _parse_citation(source: str) -> str:
    """
    从文件名解析引用格式：(Author et al. Year)

    文件名示例:
      "Zhou 等 - 2023 - Deep learning analysis of blood flow sounds..."
      "Seo和Mittal - 2012 - A coupled flow-acoustic computational study..."
      "Grochowina和Leniowska - 2016 - The new method of the selection..."

    Returns:
        str: "(Author et al. Year)" 或 "(Author Year)"
    """
    import re

    # 清理后缀
    name = source.replace(".md", "").replace(".txt", "").strip()

    # 尝试匹配 "作者 - 年份" 或 "作者和作者 - 年份" 格式
    # 模式: 英文名 (可能含空格/连字符) + 等/和 + 年份
    match = re.match(r'^([A-Za-zÀ-ɏ\-]+)', name)
    if not match:
        return ""

    first_author = match.group(1)

    # 提取年份 (4位数字，1900-2099)
    year_match = re.search(r'(19|20)\d{2}', name)
    year = year_match.group(0) if year_match else ""

    # 判断是单个作者还是多个作者（文件名含 "等" 或 "和" 或 "and" 或 "&"）
    rest_after_author = name[match.end():]
    has_coauthors = bool(re.match(r'\s*(等|和|&|and)', rest_after_author, re.IGNORECASE))

    if has_coauthors:
        if year:
            return f"({first_author} et al. {year})"
        else:
            return f"({first_author} et al.)"
    else:
        if year:
            return f"({first_author} {year})"
        else:
            return f"({first_author})"


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本，带学术引用格式

    Args:
        docs: 文档列表

    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    ref_list = []  # 参考文献列表

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")

        # 解析引用标签: (Author et al. Year)
        citation = _parse_citation(source)
        ref_label = citation if citation else f"[{i}]"

        # 提取标题信息
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        header_str = " > ".join(headers) if headers else ""

        # 构建格式化文本 —— 引用标签嵌入内容前
        formatted = f"[{ref_label}] 开始\n"
        if header_str:
            formatted += f"标题: {header_str}\n"
        formatted += f"内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

        # 参考文献条目
        ref_list.append(f"{ref_label}: {source}")

    # 拼接：内容 + 参考文献列表
    result = "\n".join(formatted_parts)
    if ref_list:
        result += "\n---\n参考文献列表:\n"
        result += "\n".join(ref_list)

    return result
