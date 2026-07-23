from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.models.request import ChatRequest


class _FailingMilvusManager:
    def connect(self):
        raise RuntimeError("milvus unavailable for security test")

    def health_check(self) -> bool:
        return False

    def close(self) -> None:
        return None


class _VectorStoreManager:
    def initialize(self):
        raise AssertionError("Milvus失败后不应初始化VectorStore")

    def shutdown(self) -> None:
        return None


class _AgentService:
    def initialize(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None


def test_public_directory_index_route_is_removed() -> None:
    from app.main import create_app

    paths = {getattr(route, "path", "") for route in create_app().routes}
    assert "/api/index_directory" not in paths


def test_chat_request_enforces_length_and_safe_session_id() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(Id="session-ok", Question="x" * 1001)

    with pytest.raises(ValidationError):
        ChatRequest(Id="../../other-session", Question="正常问题")

    with pytest.raises(ValidationError):
        ChatRequest(Id="session-ok", Question="   ")


def test_security_headers_and_explicit_cors(monkeypatch) -> None:
    from app import main as main_module

    monkeypatch.setattr(main_module, "milvus_manager", _FailingMilvusManager())
    monkeypatch.setattr(main_module, "vector_store_manager", _VectorStoreManager())
    monkeypatch.setattr(main_module, "rag_agent_service", _AgentService())

    with TestClient(main_module.create_app()) as client:
        allowed = client.get("/", headers={"Origin": "http://localhost:9900"})
        blocked = client.get("/", headers={"Origin": "https://evil.example"})

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:9900"
    assert "access-control-allow-origin" not in blocked.headers
    assert "object-src 'none'" in allowed.headers["content-security-policy"]
    assert allowed.headers["x-content-type-options"] == "nosniff"
    assert allowed.headers["x-frame-options"] == "DENY"


def test_markdown_rendering_requires_dompurify() -> None:
    project_root = Path(__file__).resolve().parent.parent
    html = (project_root / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (project_root / "static" / "app.js").read_text(encoding="utf-8")

    assert "dompurify@3.2.6" in html
    assert "DOMPurify.sanitize" in javascript
    assert "typeof DOMPurify === 'undefined'" in javascript
    assert "this.apiBaseUrl = '/api'" in javascript


def test_upload_reads_only_limit_plus_one_byte(monkeypatch) -> None:
    from app.api import file as file_api

    class _Upload:
        filename = "oversized.txt"

        def __init__(self) -> None:
            self.requested_size = 0

        async def read(self, size: int) -> bytes:
            self.requested_size = size
            return b"x" * size

    upload = _Upload()
    monkeypatch.setattr(file_api, "MAX_FILE_SIZE", 8)

    with pytest.raises(HTTPException) as caught:
        asyncio.run(file_api.upload_file(upload))  # type: ignore[arg-type]

    assert caught.value.status_code == 413
    assert upload.requested_size == 9


def test_concurrency_limit_returns_429_when_queue_times_out(monkeypatch) -> None:
    from app import dependencies

    async def attempt() -> None:
        semaphore = asyncio.Semaphore(0)
        generator = dependencies._acquire_slot(semaphore, "测试")
        with pytest.raises(HTTPException) as caught:
            await anext(generator)
        assert caught.value.status_code == 429

    monkeypatch.setattr(dependencies.config, "request_queue_timeout_seconds", 0.01)
    asyncio.run(attempt())
