from __future__ import annotations

import asyncio
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from app.services.retrieval.context_builder import ContextBuilder
from app.services.retrieval.recall_service import RecallService
from app.services.retrieval.rerank_service import RerankService
from app.services.retrieval.retrieval_models import RerankResult, RetrievalItem
from app.services.retrieval.retrieval_service import RetrievalService, _dedup_candidates


def _items(count: int = 20) -> list[RetrievalItem]:
    return [
        RetrievalItem(
            chunk_id=f"doc_{index}:hash_{index}",
            source_id=f"doc_{index}",
            source=f"Author{index}_2020.pdf",
            chunk_index=0,
            content=f"unique evidence content {index}",
            vector_score=0.50 - index * 0.01,
            metadata={"content_hash": f"hash_{index}"},
        )
        for index in range(count)
    ]


def _rerank_response(
    scores: list[tuple[int, float]],
    *,
    status_code: int = HTTPStatus.OK,
    code: str | None = None,
):
    return SimpleNamespace(
        status_code=status_code,
        code=code,
        output=SimpleNamespace(
            results=[
                SimpleNamespace(index=index, relevance_score=score)
                for index, score in scores
            ]
        ),
    )


class _Recall:
    def __init__(self, candidates: list[RetrievalItem]) -> None:
        self.candidates = candidates
        self.candidate_count = None

    def recall(self, query: str, candidate_count: int):
        self.candidate_count = candidate_count
        return list(self.candidates)

    def expand_neighbors(self, selected, window: int, top_n: int):
        return selected


@pytest.mark.parametrize("failure", ["timeout", "exception", "http", "empty", "partial"])
def test_p04_rerank_failures_return_nonempty_vector_results(
    monkeypatch,
    failure: str,
) -> None:
    async def fake_call(**kwargs):
        if failure == "timeout":
            raise asyncio.TimeoutError()
        if failure == "exception":
            raise RuntimeError("rerank unavailable")
        if failure == "http":
            return _rerank_response(
                [],
                status_code=HTTPStatus.BAD_GATEWAY,
                code="BadGateway",
            )
        if failure == "empty":
            return _rerank_response([])
        return _rerank_response([(0, 0.9)])

    monkeypatch.setattr(
        "app.services.retrieval.rerank_service.AioTextReRank.call",
        fake_call,
    )
    rerank = RerankService()
    recall = _Recall(_items())
    service = RetrievalService(recall=recall, rerank=rerank)

    context, artifact = asyncio.run(service.retrieve("AVF evidence"))

    assert recall.candidate_count == 20
    assert artifact["rerank_status"] == "degraded"
    assert artifact["rerank_degraded"] is True
    assert artifact["threshold_applied"] is False
    assert artifact["selected_count"] == 5
    assert artifact["documents"]
    assert len(artifact["documents"]) == 5
    assert "[证据 1]" in context


def test_p04_wait_for_timeout_returns_top5_vector_results(monkeypatch) -> None:
    async def slow_call(**kwargs):
        await asyncio.sleep(0.05)
        return _rerank_response([(index, 1.0) for index in range(10)])

    monkeypatch.setattr(
        "app.services.retrieval.rerank_service.AioTextReRank.call",
        slow_call,
    )
    rerank = RerankService()
    rerank._timeout_seconds = 0.001
    service = RetrievalService(recall=_Recall(_items()), rerank=rerank)

    context, artifact = asyncio.run(service.retrieve("AVF evidence"))

    assert artifact["rerank_status"] == "degraded"
    assert artifact["rerank_reason"] == "timeout"
    assert artifact["threshold_applied"] is False
    assert len(artifact["documents"]) == 5
    assert "[证据 1]" in context


def test_rerank_processes_all_20_candidates_in_one_call_and_keeps_10(
    monkeypatch,
) -> None:
    calls = []

    async def valid_call(**kwargs):
        calls.append(kwargs)
        return _rerank_response(
            [(index, 1.0 - index * 0.01) for index in range(10)]
        )

    monkeypatch.setattr(
        "app.services.retrieval.rerank_service.AioTextReRank.call",
        valid_call,
    )
    candidates = _items()
    candidates[0].content = "x" * 1200 + "TAIL_MARKER"
    rerank = RerankService()

    result = asyncio.run(rerank.rerank("query", candidates, top_k=10))

    assert len(calls) == 1
    assert calls[0]["model"] == "qwen3-rerank"
    assert len(calls[0]["documents"]) == 20
    assert calls[0]["documents"][0].endswith("TAIL_MARKER")
    assert calls[0]["top_n"] == 10
    assert result.status == "applied"
    assert result.applied is True
    assert len(result.items) == 10


def test_all_llm_scores_below_threshold_use_top_n_fallback() -> None:
    class _LowScoreRerank:
        async def rerank(self, query, candidates, top_k):
            items = list(candidates[:top_k])
            for index, item in enumerate(items):
                item.rerank_score = 0.50 - index * 0.01
            return RerankResult(items=items, status="applied", applied=True)

    service = RetrievalService(
        recall=_Recall(_items()),
        rerank=_LowScoreRerank(),  # type: ignore[arg-type]
    )
    context, artifact = asyncio.run(service.retrieve("low score query"))

    assert artifact["threshold_applied"] is True
    assert artifact["threshold_fallback"] is True
    assert artifact["selected_count"] == 3
    assert "[证据 1]" in context


def test_all_search_modes_use_fixed_20_recall_and_10_rerank() -> None:
    service = RetrievalService(recall=_Recall([]))
    for mode in ("auto", "focused", "comparison", "broad", "unknown"):
        params = service._get_mode_params(mode)
        assert params["candidate_k"] == 20
        assert params["rerank_k"] == 10
        assert params["final_chunks"] == 5


def test_dedup_uses_complete_content_not_first_200_characters() -> None:
    common = "x" * 200
    first = RetrievalItem("a", "s1", "a.pdf", 0, common + "A", 0.4)
    second = RetrievalItem("b", "s2", "b.pdf", 0, common + "B", 0.5)
    duplicate = RetrievalItem("c", "s3", "c.pdf", 0, common + "A", 0.6)

    result = _dedup_candidates([first, second, duplicate])

    assert len(result) == 2
    assert duplicate in result
    assert second in result


def test_context_budget_includes_reference_section() -> None:
    builder = ContextBuilder()
    item = RetrievalItem(
        chunk_id="doc:hash",
        source_id="doc",
        source="Zhou_2023_long_source_name.pdf",
        chunk_index=0,
        content="evidence " * 200,
        rerank_score=0.9,
    )

    context, artifact = builder.build([item], max_chars=420)

    assert len(context) <= 420
    assert "参考文献" in context
    assert artifact.selected_count == 1
    assert artifact.documents[0].content.endswith("...")
    assert artifact.documents[0].content in context
    assert len(artifact.documents[0].content) < len(item.content)


def test_neighbors_are_queried_by_source_and_chunk_index() -> None:
    class _Store:
        def __init__(self) -> None:
            self.queries = []

        def get_document_rows(self, document_id: str):
            self.queries.append(document_id)
            return [
                {
                    "id": f"row-{index}",
                    "content": f"chunk {index}",
                    "metadata": {
                        "source_id": "doc",
                        "chunk_index": index,
                        "content_hash": f"hash{index}",
                        "chunk_id": f"doc:hash{index}",
                        "_file_name": "paper.pdf",
                    },
                }
                for index in (4, 5, 6)
            ]

    store = _Store()
    recall = RecallService(store_manager=store)
    selected = RetrievalItem(
        "doc:hash5", "doc", "paper.pdf", 5, "chunk 5", rerank_score=0.9
    )

    expanded = recall.expand_neighbors([selected], window=1, top_n=1)

    assert store.queries == ["doc"]
    assert {item.chunk_index for item in expanded} == {4, 5, 6}
    assert all(
        item.metadata.get("neighbor_of") == "doc:hash5"
        for item in expanded
        if item.chunk_index != 5
    )
