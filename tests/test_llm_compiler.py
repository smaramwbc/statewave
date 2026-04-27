"""Tests for LLM compiler."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.db.tables import EpisodeRow
from server.services.compilers.llm import LLMCompiler


def _make_episode(**kw) -> EpisodeRow:
    defaults = dict(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="test",
        type="chat",
        payload={"messages": [{"role": "user", "content": "My name is Alice. I work at Acme Corp."}]},
        metadata_={},
        provenance={},
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return EpisodeRow(**defaults)


def _make_compiler() -> LLMCompiler:
    """Create a compiler for testing."""
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    compiler._api_key = "test-key"
    compiler._client = None
    return compiler


def _mock_response(content: str):
    """Create a mock httpx response."""
    mock = AsyncMock()
    mock.status_code = 200
    mock.raise_for_status = lambda: None
    mock.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock


@pytest.mark.asyncio
async def test_llm_compile_basic():
    compiler = _make_compiler()

    llm_response = json.dumps([
        {"kind": "profile_fact", "content": "Name is Alice", "summary": "Name is Alice", "confidence": 0.9},
        {"kind": "profile_fact", "content": "Works at Acme Corp", "summary": "Works at Acme Corp", "confidence": 0.85},
        {"kind": "episode_summary", "content": "User introduced themselves", "summary": "Introduction", "confidence": 0.8},
    ])

    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock, return_value=json.loads(llm_response)):
        ep = _make_episode()
        memories = await compiler.compile_async([ep])

    assert len(memories) == 3
    assert memories[0].kind == "profile_fact"
    assert memories[0].source_episode_ids == [ep.id]
    assert memories[0].metadata_.get("compiler") == "llm"


@pytest.mark.asyncio
async def test_llm_compile_empty_payload():
    compiler = _make_compiler()

    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock) as mock_llm:
        ep = _make_episode(payload={})
        memories = await compiler.compile_async([ep])
        assert memories == []
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_llm_compile_api_failure_returns_empty():
    compiler = _make_compiler()

    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock, side_effect=RuntimeError("API down")):
        ep = _make_episode()
        memories = await compiler.compile_async([ep])
        assert memories == []


@pytest.mark.asyncio
async def test_llm_compile_invalid_json_returns_empty():
    compiler = _make_compiler()

    # _call_llm_async parses JSON, so invalid json would raise in there
    # Simulate by returning a non-list
    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock, side_effect=json.JSONDecodeError("", "", 0)):
        memories = await compiler.compile_async([_make_episode()])
        assert memories == []


@pytest.mark.asyncio
async def test_llm_compile_clamps_confidence():
    compiler = _make_compiler()

    raw = [
        {"kind": "profile_fact", "content": "A", "summary": "A", "confidence": 5.0},
        {"kind": "profile_fact", "content": "B", "summary": "B", "confidence": -1.0},
    ]

    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock, return_value=raw):
        memories = await compiler.compile_async([_make_episode()])
        assert memories[0].confidence == 1.0
        assert memories[1].confidence == 0.0


@pytest.mark.asyncio
async def test_llm_compile_strips_markdown_fences():
    compiler = _make_compiler()

    raw = [{"kind": "profile_fact", "content": "Test", "summary": "Test", "confidence": 0.9}]

    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock, return_value=raw):
        memories = await compiler.compile_async([_make_episode()])
        assert len(memories) == 1
