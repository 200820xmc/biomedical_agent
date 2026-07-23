from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.rag_agent_service import RagAgentService
from app.services.vector_store_manager import VectorStoreManager


class _FailingMilvusManager:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.close_calls = 0

    def connect(self):
        self.connect_calls += 1
        raise RuntimeError("milvus intentionally unavailable")

    def health_check(self) -> bool:
        return False

    def close(self) -> None:
        self.close_calls += 1


class _RecordingVectorStoreManager:
    def __init__(self) -> None:
        self.initialize_calls = 0
        self.shutdown_calls = 0

    def initialize(self):
        self.initialize_calls += 1
        raise AssertionError("连接失败后不应初始化VectorStore")

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeAgentService:
    def __init__(self) -> None:
        self.initialize_calls = 0
        self.cleanup_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


def test_service_constructors_do_not_initialize_external_clients() -> None:
    calls = {"embedding": 0, "vector_store": 0, "model": 0, "checkpointer": 0}

    def embedding_factory():
        calls["embedding"] += 1
        return object()

    def vector_store_factory(**kwargs):
        calls["vector_store"] += 1
        return object()

    def model_factory(**kwargs):
        calls["model"] += 1
        return object()

    def checkpointer_factory():
        calls["checkpointer"] += 1
        return object()

    _ = VectorStoreManager(
        milvus_client_manager=_FailingMilvusManager(),
        embedding_factory=embedding_factory,
        vector_store_factory=vector_store_factory,
    )
    _ = RagAgentService(
        model_factory=model_factory,
        tools=[],
        checkpointer_factory=checkpointer_factory,
    )

    assert calls == {"embedding": 0, "vector_store": 0, "model": 0, "checkpointer": 0}


def test_app_starts_degraded_and_chat_returns_503_when_milvus_is_down(
    monkeypatch,
) -> None:
    from app import main as main_module

    fake_milvus = _FailingMilvusManager()
    fake_vector_store = _RecordingVectorStoreManager()
    fake_agent = _FakeAgentService()
    monkeypatch.setattr(main_module, "milvus_manager", fake_milvus)
    monkeypatch.setattr(main_module, "vector_store_manager", fake_vector_store)
    monkeypatch.setattr(main_module, "rag_agent_service", fake_agent)

    with TestClient(main_module.create_app()) as client:
        health = client.get("/health")
        chat = client.post(
            "/api/chat",
            json={"Id": "test-session", "Question": "测试问题"},
        )

        assert health.status_code == 503
        health_data = health.json()["data"]
        assert health_data["startup_errors"]["knowledge_backend"]
        dependencies = health_data["external_dependencies"]
        assert dependencies["milvus"]["status"] == "disconnected"
        assert dependencies["dashscope"]["status"] in {
            "configured",
            "not_configured",
        }
        assert dependencies["xparse"]["status"] in {"available", "unavailable"}
        assert dependencies["mcp"] == {
            "status": "disabled",
            "required": False,
            "mode": "experimental_not_loaded",
        }
        assert "api_key" not in str(health.json()).lower()
        assert chat.status_code == 503
        assert "Milvus" in chat.json()["detail"]
        assert fake_milvus.connect_calls == 1
        assert fake_vector_store.initialize_calls == 0
        assert fake_agent.initialize_calls == 1

    assert fake_agent.cleanup_calls == 1
    assert fake_vector_store.shutdown_calls == 1
    assert fake_milvus.close_calls == 1
