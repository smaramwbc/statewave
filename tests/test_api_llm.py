"""Tests for the internal /v1/llm/complete endpoint.

The endpoint is a thin pass-through over `services.llm.acomplete`. We mock
that function to keep these tests provider-free and fast.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from server.app import create_app
from server.core.config import settings
from server.services.llm import LLMProviderError, LLMTimeoutError


async def _client_with(monkeypatch=None) -> AsyncClient:
    app = create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_complete_chat_happy_path(monkeypatch):
    monkeypatch.setattr(settings, "litellm_model", "gpt-4o-mini")
    with patch(
        "server.api.llm.acomplete",
        new_callable=AsyncMock,
        return_value="hello from llm",
    ) as mocked:
        async with await _client_with() as c:
            r = await c.post(
                "/v1/llm/complete",
                json={
                    "messages": [
                        {"role": "system", "content": "you are helpful"},
                        {"role": "user", "content": "hi"},
                    ],
                    "max_tokens": 50,
                    "temperature": 0.5,
                },
            )
    assert r.status_code == 200
    assert r.json() == {"reply": "hello from llm"}
    # acomplete called with (messages_list, max_tokens=..., temperature=...) — no
    # model kwarg, since callers must not pick the model.
    args, kwargs = mocked.call_args
    assert args[0] == [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    assert kwargs == {"max_tokens": 50, "temperature": 0.5}


@pytest.mark.asyncio
async def test_complete_chat_caller_cannot_pick_model(monkeypatch):
    """The schema does not accept a `model` field — caller-supplied values
    are silently dropped (Pydantic's default for unknown fields)."""
    monkeypatch.setattr(settings, "litellm_model", "gpt-4o-mini")
    with patch(
        "server.api.llm.acomplete", new_callable=AsyncMock, return_value="ok"
    ) as mocked:
        async with await _client_with() as c:
            r = await c.post(
                "/v1/llm/complete",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "evil/cheap-model",
                },
            )
    assert r.status_code == 200
    _, kwargs = mocked.call_args
    assert "model" not in kwargs


@pytest.mark.asyncio
async def test_complete_chat_returns_503_when_no_model(monkeypatch):
    monkeypatch.setattr(settings, "litellm_model", "")
    async with await _client_with() as c:
        r = await c.post(
            "/v1/llm/complete",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 503
    body = r.json()["error"]
    assert body["code"] == "llm_not_configured"


@pytest.mark.asyncio
async def test_complete_chat_provider_error_returns_502(monkeypatch):
    monkeypatch.setattr(settings, "litellm_model", "gpt-4o-mini")
    with patch(
        "server.api.llm.acomplete",
        new_callable=AsyncMock,
        side_effect=LLMProviderError("rate limit hit at https://internal-config"),
    ):
        async with await _client_with() as c:
            r = await c.post(
                "/v1/llm/complete",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
    assert r.status_code == 502
    body = r.json()["error"]
    assert body["code"] == "upstream_llm_error"
    # Don't echo upstream details (URLs, internal model identifiers, etc.)
    assert "internal-config" not in r.text


@pytest.mark.asyncio
async def test_complete_chat_timeout_returns_504(monkeypatch):
    monkeypatch.setattr(settings, "litellm_model", "gpt-4o-mini")
    with patch(
        "server.api.llm.acomplete",
        new_callable=AsyncMock,
        side_effect=LLMTimeoutError("timed out after 60s"),
    ):
        async with await _client_with() as c:
            r = await c.post(
                "/v1/llm/complete",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "upstream_llm_timeout"


@pytest.mark.asyncio
async def test_complete_chat_validation_rejects_empty_messages():
    async with await _client_with() as c:
        r = await c.post("/v1/llm/complete", json={"messages": []})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_complete_chat_validation_rejects_bad_role():
    async with await _client_with() as c:
        r = await c.post(
            "/v1/llm/complete",
            json={"messages": [{"role": "robot", "content": "hi"}]},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_complete_chat_validation_caps_message_count():
    async with await _client_with() as c:
        r = await c.post(
            "/v1/llm/complete",
            json={
                "messages": [
                    {"role": "user", "content": str(i)} for i in range(100)
                ]
            },
        )
    assert r.status_code == 422
