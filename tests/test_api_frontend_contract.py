from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.chat import chat
from app.models.request import ChatRequest


class _FailingAgent:
    async def query(self, question: str, session_id: str):
        raise RuntimeError("model unavailable")


def test_quick_chat_failure_uses_http_500() -> None:
    request = ChatRequest(Id="session-1", Question="test")

    with pytest.raises(HTTPException) as caught:
        asyncio.run(chat(request, _FailingAgent(), None))  # type: ignore[arg-type]

    assert caught.value.status_code == 500
    assert caught.value.detail == "对话服务暂时不可用"


def test_frontend_supports_pdf_201_and_consistent_assistant_roles() -> None:
    root = Path(__file__).resolve().parent.parent
    html = (root / "static" / "index.html").read_text(encoding="utf-8")
    javascript = (root / "static" / "app.js").read_text(encoding="utf-8")

    assert 'accept=".txt,.md,.pdf"' in html
    assert "['.txt', '.md', '.pdf']" in javascript
    assert "data.code === 201" in javascript
    assert "等待解析入库" in javascript
    assert "msg.role === 'user' ? 'user' : 'assistant'" in javascript


def test_frontend_recognizes_tool_and_retrieval_sse_events() -> None:
    javascript = (
        Path(__file__).resolve().parent.parent / "static" / "app.js"
    ).read_text(encoding="utf-8")

    assert "sseMessage.type === 'tool_start'" in javascript
    assert "sseMessage.type === 'retrieval_complete'" in javascript


def test_frontend_persists_new_and_completed_chats_in_recent_history() -> None:
    javascript = (
        Path(__file__).resolve().parent.parent / "static" / "app.js"
    ).read_text(encoding="utf-8")

    assert "persistCurrentChatHistory()" in javascript
    assert javascript.count("this.persistCurrentChatHistory();") >= 2
    assert "localStorage.setItem('chatHistories'" in javascript
