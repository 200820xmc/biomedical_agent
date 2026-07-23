"""上下文构建器 — 将最终选择的 chunk 格式化为模型可读的证据文本

输出格式：
    [证据 1]
    引用：(Zhou et al. 2023)
    章节：Experiments > Results
    内容：
    ……

    ---
    参考文献：
    (Zhou et al. 2023): Zhou 等 - 2023 - Deep learning analysis...
"""

import re
from dataclasses import replace
from typing import List, Tuple

from loguru import logger

from app.config import config
from app.services.retrieval.retrieval_models import RetrievalArtifact, RetrievalItem


class ContextBuilder:
    """上下文格式化与预算控制服务

    职责：
    - 将 RetrievalItem 列表格式化为 [证据 N] 格式的文本
    - 生成参考文献列表
    - 控制上下文长度不超过预算
    - 构建 RetrievalArtifact 供评测和前端使用
    """

    def __init__(self) -> None:
        self._max_chars = getattr(config, "rag_max_context_chars", 12000)
        self._max_per_evidence = getattr(config, "rag_max_chars_per_evidence", 1600)

    def build(
        self,
        items: list[RetrievalItem],
        original_query: str = "",
        search_mode: str = "auto",
        candidate_count: int = 0,
        reranked_count: int = 0,
        rerank_applied: bool = False,
        rerank_status: str = "skipped",
        rerank_degraded: bool = False,
        rerank_reason: str = "",
        threshold_applied: bool = False,
        threshold_fallback: bool = False,
        max_chars: int | None = None,
    ) -> tuple[str, RetrievalArtifact]:
        """构建格式化的上下文文本和结构化 artifact

        Args:
            items: 最终选择的 RetrievalItem 列表
            original_query: 原始用户问题
            search_mode: 检索模式
            candidate_count: Milvus 召回候选数
            reranked_count: Rerank 后保留数
            rerank_applied: 是否实际执行了 Rerank
            max_chars: 上下文最大字符数，不传则使用配置默认值

        Returns:
            tuple[str, RetrievalArtifact]: (格式化上下文文本, 结构化 artifact)
        """
        budget = max_chars or self._max_chars

        # 按 rerank_score 降序排列（如果已排序则保持不变）
        sorted_items = sorted(
            items,
            key=lambda x: x.rerank_score or x.vector_score or 0,
            reverse=True,
        )

        # 构建证据段落
        evidence_parts: list[str] = []
        ref_map: dict[str, str] = {}  # 引用标签 → 完整来源
        kept_items: list[RetrievalItem] = []
        context = ""

        for item in sorted_items:
            evidence_number = len(kept_items) + 1
            citation = self._parse_citation(item.source, item.metadata)
            ref_label = citation if citation else f"[{evidence_number}]"

            # 章节信息
            headers = []
            for key in ("h1", "h2"):
                val = item.metadata.get(key, "")
                if val:
                    headers.append(val)
            header_str = " > ".join(headers) if headers else ""

            # 构建证据文本
            evidence_header = f"[证据 {evidence_number}]\n引用：{ref_label}\n"
            if header_str:
                evidence_header += f"章节：{header_str}\n"

            # ── 预算感知选择（P0-4）──────────────────────────
            candidate_refs = dict(ref_map)
            candidate_refs[ref_label] = item.source

            # 1. 完整内容适配，同时计算证据间换行和参考文献。
            full_evidence = evidence_header + f"内容：\n{item.content}\n"
            full_context = self._compose_context(
                evidence_parts + [full_evidence], candidate_refs
            )
            if len(full_context) <= budget:
                evidence_parts.append(full_evidence)
                kept_items.append(item)
                ref_map = candidate_refs
                context = full_context
                continue

            # 2. 在完整最终上下文预算内截断正文。
            empty_evidence = evidence_header + "内容：\n\n"
            fixed_context = self._compose_context(
                evidence_parts + [empty_evidence], candidate_refs
            )
            max_content = min(
                self._max_per_evidence,
                budget - len(fixed_context) - 3,
            )
            if max_content > 200:
                truncated = item.content[:max_content] + "..."
                short_evidence = evidence_header + f"内容：\n{truncated}\n"
                short_context = self._compose_context(
                    evidence_parts + [short_evidence], candidate_refs
                )
                if len(short_context) <= budget:
                    evidence_parts.append(short_evidence)
                    kept_items.append(replace(item, content=truncated))
                    ref_map = candidate_refs
                    context = short_context
                    continue

            # 3. 当前候选过长，跳过并继续检查后续候选
            logger.debug(f"跳过过长候选: {ref_label} ({len(item.content)} 字符)")

        context = self._compose_context(evidence_parts, ref_map)
        if len(context) > budget:
            raise RuntimeError(
                f"上下文预算控制失败: actual={len(context)}, budget={budget}"
            )

        # 计算置信度
        confidence = self._compute_confidence(kept_items)

        # 构建 Artifact
        artifact = RetrievalArtifact(
            original_query=original_query,
            search_mode=search_mode,
            candidate_count=candidate_count,
            reranked_count=reranked_count,
            selected_count=len(kept_items),
            confidence=confidence,
            rerank_applied=rerank_applied,
            rerank_status=rerank_status,
            rerank_degraded=rerank_degraded,
            rerank_reason=rerank_reason,
            threshold_applied=threshold_applied,
            threshold_fallback=threshold_fallback,
            documents=kept_items,
        )

        logger.info(
            f"上下文构建完成: {len(kept_items)} 个证据, "
            f"{len(context)} 字符, 置信度={confidence}"
        )

        return context, artifact

    @staticmethod
    def _compose_context(
        evidence_parts: list[str],
        ref_map: dict[str, str],
    ) -> str:
        """组合证据与参考文献，供预算检查和最终输出共同使用。"""
        evidence_text = "\n".join(evidence_parts)
        if not ref_map:
            return evidence_text
        ref_lines = [f"{label}: {source}" for label, source in ref_map.items()]
        return evidence_text + "\n---\n参考文献：\n" + "\n".join(ref_lines)

    @staticmethod
    def _parse_citation(source: str, metadata: dict | None = None) -> str:
        """从文件名或metadata解析引用标签

        优先使用metadata中的parsed_title/authors/year，
        否则回退到文件名解析（P2-4）。
        """
        # P2-4: 优先使用正式元数据
        if metadata:
            title = metadata.get("title", "")
            authors = metadata.get("authors", "")
            year = metadata.get("year", "")

            if authors and year:
                # 提取第一作者姓氏
                first_author = authors.split(",")[0].strip().split()[-1] if "," in authors else authors.split()[0] if authors.split() else ""
                if first_author:
                    coauthors = len(authors.split(",")) > 1 if "," in authors else len(authors.split()) > 1
                    if coauthors:
                        return f"({first_author} et al. {year})"
                    else:
                        return f"({first_author} {year})"
            elif title and year:
                # 有标题和年份但没有明确的作者
                short_title = title.split(":")[0].strip()[:50]
                return f"({short_title} {year})"

        # 回退：从文件名解析
        return ContextBuilder._parse_citation_from_name(source)

    @staticmethod
    def _parse_citation_from_name(source: str) -> str:
        """从文件名解析引用标签：(Author et al. Year)

        复用知识检索工具中相同的引用解析逻辑。

        文件名示例:
          "Zhou 等 - 2023 - Deep learning analysis..."
          "Seo和Mittal - 2012 - A coupled flow-acoustic..."
          "Grochowina和Leniowska - 2016 - The new method..."

        Returns:
            str: "(Author et al. Year)" 或 "(Author Year)" 或 ""
        """
        # 清理后缀
        name = source.replace(".md", "").replace(".txt", "").replace(".pdf", "").strip()

        # 匹配开头的英文作者名
        match = re.match(r'^([A-Za-zÀ-ɏ\-]+)', name)
        if not match:
            return ""

        first_author = match.group(1)

        # 提取年份（4 位数字，1900~2099）
        year_match = re.search(r'(19|20)\d{2}', name)
        year = year_match.group(0) if year_match else ""

        # 判断是否有合作者
        rest = name[match.end():]
        has_coauthors = bool(re.match(r'\s*(等|和|&|and)', rest, re.IGNORECASE))

        if has_coauthors:
            return f"({first_author} et al. {year})" if year else f"({first_author} et al.)"
        else:
            return f"({first_author} {year})" if year else f"({first_author})"

    @staticmethod
    def _compute_confidence(items: list[RetrievalItem]) -> str:
        """根据 Rerank 分数和来源覆盖度计算置信度

        规则：
        - high：至少 2 个来源，最高分 ≥ 0.7
        - medium：有结果但未达到 high 标准
        - low：无结果或最高分 < 0.3

        Args:
            items: 最终选择的 chunk 列表

        Returns:
            str: "high" | "medium" | "low"
        """
        if not items:
            return "low"

        sources = set(item.source_id or item.source for item in items)
        best_score = max(
            (item.rerank_score or item.vector_score or 0) for item in items
        )

        if len(sources) >= 2 and best_score >= 0.7:
            return "high"
        elif best_score < 0.3:
            return "low"
        else:
            return "medium"
