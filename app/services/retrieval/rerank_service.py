"""Rerank 语义精排服务 — 对超额召回的候选 chunk 进行语义相关性重排序

使用 LLM 进行列表式批量精排（Listwise Rerank），将候选分批交给模型，
由模型根据语义相关性打分，最终按 rerank_score 降序排列。

Rerank 失败时自动降级为原始向量排序，不影响整体检索可用性。
"""

import asyncio
import re
import time
from typing import List

from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import config
from app.services.retrieval.retrieval_models import RetrievalItem


# Rerank 批次大小：每批最多多少个候选一起打分
_RERANK_BATCH_SIZE = 15

# 每个候选在 prompt 中展示的最大字符数（截断过长内容）
_RERANK_CONTENT_MAX_CHARS = 800

# Rerank 超时时间（秒）
_RERANK_TIMEOUT_SECONDS = 30


class RerankService:
    """LLM 列表式语义精排服务

    职责：
    - 将候选分批次交给 LLM 进行语义相关性打分
    - 解析 LLM 返回的分数
    - Rerank 失败时降级为原始向量排序

    设计说明：
    第一阶段使用 LLM 进行 Rerank（不需要额外部署专用 Reranker 模型）。
    后续可以替换为 DashScope Rerank API 或本地 BGE-Reranker，
    只需实现相同的 rerank() 接口即可。
    """

    def __init__(self) -> None:
        self._enabled = getattr(config, "rag_rerank_enabled", True)
        self._rerank_k = getattr(config, "rag_rerank_k", 20)
        # DashScope OpenAI 兼容模式 URL（和 llm_factory.py 保持一致）
        self._base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # 用于 LLM 调用的模型实例（temperature=0 保证打分稳定）
        self._model: ChatOpenAI | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_model(self) -> ChatOpenAI:
        """延迟创建 LLM 实例（避免在模块导入时初始化）

        使用 ChatOpenAI（OpenAI 兼容模式）调用 DashScope，
        和 llm_factory.py 保持一致的配置方式。
        """
        if self._model is None:
            self._model = ChatOpenAI(
                model=config.rag_model,
                temperature=0.0,  # 打分任务需要稳定输出
                streaming=False,
                base_url=self._base_url,
                api_key=config.dashscope_api_key,
            )
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalItem],
        top_k: int | None = None,
    ) -> list[RetrievalItem]:
        """对候选 chunk 列表进行语义精排

        Args:
            query: 用户原始问题
            candidates: 超额召回的候选 chunk 列表
            top_k: Rerank 后保留的数量，不传则使用配置默认值（20）

        Returns:
            list[RetrievalItem]: 按 rerank_score 降序排列的候选列表，
                               每个 item 都已填充 rerank_score 字段。
                               失败时返回按 vector_score 降序排列的列表。
        """
        if not candidates:
            return []

        k = top_k or self._rerank_k

        # 如果候选数量很少，不需要 Rerank
        if len(candidates) <= k:
            for item in candidates:
                item.rerank_score = item.vector_score
            return candidates

        if not self._enabled:
            logger.info("Rerank 未启用，使用向量排序")
            for item in candidates:
                item.rerank_score = item.vector_score
            return sorted(
                candidates,
                key=lambda x: x.rerank_score or 0,
                reverse=True,
            )[:k]

        try:
            start_time = time.time()

            # 分批次进行 LLM 打分
            all_scored: list[RetrievalItem] = []

            for batch_start in range(0, len(candidates), _RERANK_BATCH_SIZE):
                batch = candidates[batch_start : batch_start + _RERANK_BATCH_SIZE]
                scored_batch = await self._rerank_batch(query, batch)
                all_scored.extend(scored_batch)

            # 按 rerank_score 降序排列
            all_scored.sort(key=lambda x: x.rerank_score or 0, reverse=True)

            elapsed = (time.time() - start_time) * 1000
            top_score = all_scored[0].rerank_score if all_scored else 0
            logger.info(
                f"Rerank 完成: 输入={len(candidates)} → 输出={len(all_scored)}, "
                f"最高分={top_score:.4f}, 耗时={elapsed:.0f}ms"
            )

            return all_scored[:k]

        except Exception as e:
            logger.warning(f"Rerank 失败，降级为向量排序: {e}")
            # 降级：按 vector_score 排序
            for item in candidates:
                item.rerank_score = item.vector_score
            return sorted(
                candidates,
                key=lambda x: x.rerank_score or 0,
                reverse=True,
            )[:k]

    async def _rerank_batch(
        self,
        query: str,
        batch: list[RetrievalItem],
    ) -> list[RetrievalItem]:
        """对一批候选进行 LLM 打分

        Args:
            query: 用户问题
            batch: 一批候选 chunk

        Returns:
            list[RetrievalItem]: 已填充 rerank_score 的候选项
        """
        # 构建 prompt
        prompt = self._build_rerank_prompt(query, batch)

        try:
            model = self._get_model()
            response = await asyncio.wait_for(
                model.ainvoke(
                    prompt,
                    config={"tags": ["internal_rerank"]},
                ),
                timeout=_RERANK_TIMEOUT_SECONDS,
            )
            response_text = response.content if hasattr(response, "content") else str(response)

            # 解析分数
            scores = self._parse_scores(response_text, len(batch))

            # 如果解析失败，使用 vector_score
            if scores is None:
                logger.warning("Rerank 分数解析失败，该批次使用向量分数")
                for item in batch:
                    item.rerank_score = item.vector_score
                return batch

            # 填充 rerank_score（-1 表示解析缺失，回退到 vector_score）
            for idx, item in enumerate(batch):
                if idx < len(scores) and scores[idx] >= 0:
                    item.rerank_score = scores[idx]
                else:
                    item.rerank_score = item.vector_score
                    item.metadata["rerank_status"] = "fallback"

            return batch

        except asyncio.TimeoutError:
            logger.warning("Rerank LLM 调用超时，该批次使用向量分数")
            for item in batch:
                item.rerank_score = item.vector_score
            return batch

    def _build_rerank_prompt(
        self,
        query: str,
        batch: list[RetrievalItem],
    ) -> str:
        """构建 Rerank 评分 prompt

        使用列表式（Listwise）评分方式：一次性呈现所有候选，
        要求模型为每个候选打出 0~10 的整数分数。

        Args:
            query: 用户问题
            batch: 一批候选 chunk

        Returns:
            str: 完整的评分 prompt
        """
        # 格式化候选列表
        candidates_text_parts: list[str] = []
        for idx, item in enumerate(batch):
            # 截断过长内容
            content = item.content
            if len(content) > _RERANK_CONTENT_MAX_CHARS:
                content = content[:_RERANK_CONTENT_MAX_CHARS] + "..."

            # 提取章节信息
            h1 = item.metadata.get("h1", "")
            h2 = item.metadata.get("h2", "")
            header = f" > {h1}" if h1 else ""
            header += f" > {h2}" if h2 else ""

            candidates_text_parts.append(
                f"[{idx}] 来源: {item.source}{header}\n内容: {content}\n"
            )

        candidates_text = "\n".join(candidates_text_parts)

        prompt = f"""请评估以下每篇文献片段与问题的语义相关性，为每篇打出 0~10 的整数分数。

问题：{query}

评分标准：
- 10 分：内容直接回答问题，包含关键方法、数据或结论
- 7~9 分：内容高度相关，涉及问题的某个方面
- 4~6 分：内容部分相关，可作为背景参考
- 1~3 分：内容仅有微弱关联
- 0 分：内容完全无关

重要约束：
- 如果片段仅讨论检测/诊断/分类技术（如深度学习模型、信号处理方法），但未涉及问题的核心主题（如病因、机制、影响因素），应评为 0~2 分
- 不要因为片段来自同一领域就自动给高分，必须语义匹配

文献片段：
{candidates_text}

请严格按以下格式输出分数（每行一个，不要输出其他内容）：
0:分数
1:分数
...
"""

        return prompt

    def _parse_scores(
        self,
        response_text: str,
        expected_count: int,
    ) -> list[float] | None:
        """从 LLM 响应中解析分数

        支持的格式：
            0:8
            1:6
            ...

        容错：如果解析到的分数数量不足 expected_count，
        缺失的分数将设为对应位置的 vector_score（由调用方处理）。

        Args:
            response_text: LLM 返回的文本
            expected_count: 期望的分数数量

        Returns:
            list[float] | None: 解析成功返回 0~1 的分数列表，失败返回 None
        """
        # 按行匹配 "数字:数字" 格式
        pattern = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$", re.MULTILINE)
        matches = pattern.findall(response_text)

        if not matches:
            # 尝试更宽松的匹配：只提取 "数字:数字"
            pattern_loose = re.compile(r"(\d+)\s*:\s*(\d+)")
            matches = pattern_loose.findall(response_text)

        if not matches:
            logger.warning(f"无法从 LLM 响应中解析分数。响应前 200 字符: {response_text[:200]}")
            return None

        # 构建 index → score 的映射
        score_map: dict[int, int] = {}
        for idx_str, score_str in matches:
            try:
                idx = int(idx_str)
                score = int(score_str)
                if 0 <= score <= 10:
                    score_map[idx] = score
            except ValueError:
                continue

        if not score_map:
            return None

        # 按顺序输出分数，归一化到 0~1
        scores: list[float] = []
        for i in range(expected_count):
            if i in score_map:
                scores.append(score_map[i] / 10.0)
            else:
                scores.append(-1.0)  # 标记为缺失，后续由调用方用 vector_score 回退

        logger.debug(f"Rerank 分数解析成功: {len(scores)} 个分数, 范围 {min(scores):.1f}~{max(scores):.1f}")
        return scores
