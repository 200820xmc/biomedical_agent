from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import LOGS_DIR, PROJECT_ROOT, STATIC_DIR, UPLOADS_DIR
from app.utils.logger import describe_text


class _FailingMilvus:
    def connect(self) -> None:
        raise RuntimeError("unavailable")

    def health_check(self) -> bool:
        return False

    def close(self) -> None:
        return None


class _VectorStore:
    def initialize(self) -> None:
        raise AssertionError("must not initialize after Milvus failure")

    def shutdown(self) -> None:
        return None


class _Agent:
    def initialize(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None


def test_project_paths_are_derived_from_project_root() -> None:
    assert PROJECT_ROOT == Path(__file__).resolve().parent.parent
    assert UPLOADS_DIR == PROJECT_ROOT / "uploads"
    assert LOGS_DIR == PROJECT_ROOT / "logs"
    assert STATIC_DIR == PROJECT_ROOT / "static"


def test_describe_text_never_contains_original_text() -> None:
    question = "这是一条不应出现在日志里的科研问题"
    description = describe_text(question, "question")

    assert question not in description
    assert f"question_len={len(question)}" in description
    assert "question_sha256=" in description
    assert len(description.rsplit("=", 1)[1]) == 16


def test_request_id_is_validated_and_returned(monkeypatch) -> None:
    from app import main as main_module

    monkeypatch.setattr(main_module, "milvus_manager", _FailingMilvus())
    monkeypatch.setattr(main_module, "vector_store_manager", _VectorStore())
    monkeypatch.setattr(main_module, "rag_agent_service", _Agent())

    with TestClient(main_module.create_app()) as client:
        supplied = client.get("/", headers={"X-Request-ID": "request_12345678"})
        generated = client.get("/", headers={"X-Request-ID": "bad"})

    assert supplied.headers["X-Request-ID"] == "request_12345678"
    assert len(generated.headers["X-Request-ID"]) == 32


def test_startup_files_do_not_hardcode_network_environment() -> None:
    run_server = (PROJECT_ROOT / "run_server.py").read_text(encoding="utf-8")
    windows_script = (PROJECT_ROOT / "start-windows.bat").read_text(
        encoding="utf-8"
    )

    for forbidden in ("HTTP_PROXY", "HTTPS_PROXY", "DASHSCOPE_API_BASE", "7890"):
        assert forbidden not in run_server
    assert "host=config.host" in run_server
    assert "port=config.port" in run_server
    assert "%~dp0" in windows_script
    assert ".venv\\Scripts\\python.exe" in windows_script
    assert "python run_server.py" not in windows_script


def test_mcp_is_not_imported_by_online_agent() -> None:
    agent_source = (PROJECT_ROOT / "app/services/rag_agent_service.py").read_text(
        encoding="utf-8"
    )
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "app.agent.mcp_client" not in agent_source
    assert (PROJECT_ROOT / "app/agent/mcp_client.py").exists()
    assert "experimental" in readme
    assert "不声称MCP已经落地" in readme


def test_research_question_is_not_interpolated_into_logs() -> None:
    files = [
        "app/api/chat.py",
        "app/services/rag_agent_service.py",
        "app/services/retrieval/retrieval_service.py",
        "app/services/vector_store_manager.py",
        "app/tools/knowledge_tool.py",
    ]
    combined = "\n".join(
        (PROJECT_ROOT / name).read_text(encoding="utf-8") for name in files
    )

    assert "query_preview" not in combined
    assert "{request.question}" not in combined
    assert "query='{query" not in combined
