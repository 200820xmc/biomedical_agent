"""Dedicated semantic reranking with safe vector-order fallback."""

from __future__ import annotations

import asyncio
import time
from http import HTTPStatus

from dashscope import AioTextReRank
from loguru import logger

from app.config import config
from app.services.retrieval.retrieval_models import RerankResult, RetrievalItem


class RerankService:
    """Rerank one candidate list through DashScope's dedicated text API."""

    def __init__(self) -> None:
        self._enabled = getattr(config, "rag_rerank_enabled", True)
        self._rerank_k = getattr(config, "rag_rerank_k", 10)
        self._model = getattr(config, "rag_rerank_model", "qwen3-rerank")
        self._timeout_seconds = getattr(
            config,
            "rag_rerank_timeout_seconds",
            30.0,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalItem],
        top_k: int | None = None,
    ) -> RerankResult:
        """Return dedicated rerank results or a non-empty vector fallback."""
        if not candidates:
            return RerankResult(items=[], status="skipped", reason="no_candidates")

        k = top_k or self._rerank_k
        if len(candidates) <= k:
            return self._vector_result(
                candidates,
                k,
                status="skipped",
                reason="candidate_count_not_above_rerank_k",
            )
        if not self._enabled:
            return self._vector_result(
                candidates,
                k,
                status="disabled",
                reason="rerank_disabled",
            )

        started = time.time()
        try:
            response = await asyncio.wait_for(
                AioTextReRank.call(
                    model=self._model,
                    query=query,
                    documents=[item.content for item in candidates],
                    top_n=k,
                    return_documents=False,
                    api_key=config.dashscope_api_key,
                ),
                timeout=self._timeout_seconds,
            )
            if response.status_code != HTTPStatus.OK:
                code = response.code or "unknown"
                return self._degraded(
                    candidates,
                    k,
                    f"api_error:{response.status_code}:{code}",
                )

            results = list(response.output.results or [])
            expected_count = min(k, len(candidates))
            if not results:
                return self._degraded(candidates, k, "empty_results")

            ranked: list[RetrievalItem] = []
            seen_indices: set[int] = set()
            for result in results:
                index = int(result.index)
                if index in seen_indices or not 0 <= index < len(candidates):
                    continue
                seen_indices.add(index)
                item = candidates[index]
                item.rerank_score = float(result.relevance_score)
                item.metadata["rerank_status"] = "dedicated"
                item.metadata["rerank_model"] = self._model
                ranked.append(item)

            if len(ranked) != expected_count:
                return self._degraded(candidates, k, "partial_results")

            ranked.sort(
                key=lambda item: item.rerank_score or 0.0,
                reverse=True,
            )
            logger.info(
                f"Dedicated rerank completed: model={self._model}, "
                f"input={len(candidates)}, output={len(ranked)}, "
                f"elapsed_ms={(time.time() - started) * 1000:.0f}"
            )
            return RerankResult(
                items=ranked[:k],
                status="applied",
                applied=True,
            )
        except asyncio.TimeoutError:
            return self._degraded(candidates, k, "timeout")
        except Exception as exc:
            return self._degraded(
                candidates,
                k,
                f"rerank_exception:{type(exc).__name__}",
            )

    def _degraded(
        self,
        candidates: list[RetrievalItem],
        k: int,
        reason: str,
    ) -> RerankResult:
        logger.warning(
            f"Dedicated rerank degraded to vector order: model={self._model}, "
            f"reason={reason}"
        )
        return self._vector_result(
            candidates,
            k,
            status="degraded",
            reason=reason,
            degraded=True,
        )

    @staticmethod
    def _vector_result(
        candidates: list[RetrievalItem],
        k: int,
        status: str,
        reason: str,
        degraded: bool = False,
    ) -> RerankResult:
        """Build a fallback that is never filtered by reranker thresholds."""
        for item in candidates:
            item.rerank_score = None
            item.metadata["rerank_status"] = "vector_fallback"
        items = sorted(
            candidates,
            key=lambda item: item.vector_score or 0.0,
            reverse=True,
        )[:k]
        return RerankResult(
            items=items,
            status=status,
            applied=False,
            degraded=degraded,
            reason=reason,
        )
