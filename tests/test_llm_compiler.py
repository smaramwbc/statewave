"""Tests for LLM compiler."""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_llm_compile_basic():
    compiler = _make_compiler()

    llm_response = [
        {"kind": "profile_fact", "content": "Name is Alice", "summary": "Name is Alice", "confidence": 0.9, "episode_index": 0},
        {"kind": "profile_fact", "content": "Works at Acme Corp", "summary": "Works at Acme Corp", "confidence": 0.85, "episode_index": 0},
        {"kind": "episode_summary", "content": "User introduced themselves", "summary": "Introduction", "confidence": 0.8, "episode_index": 0},
    ]

    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock, return_value=llm_response):
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
async def test_llm_compile_clamps_confidence():
    compiler = _make_compiler()

    raw = [
        {"kind": "profile_fact", "content": "A", "summary": "A", "confidence": 5.0, "episode_index": 0},
        {"kind": "profile_fact", "content": "B", "summary": "B", "confidence": -1.0, "episode_index": 0},
    ]

    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock, return_value=raw):
        memories = await compiler.compile_async([_make_episode()])
        assert memories[0].confidence == 1.0
        assert memories[1].confidence == 0.0


@pytest.mark.asyncio
async def test_llm_compile_batches_multiple_episodes():
    """Multiple small episodes should be batched into a single LLM call."""
    compiler = _make_compiler()

    raw = [
        {"kind": "profile_fact", "content": "Fact from ep0", "summary": "Fact", "confidence": 0.9, "episode_index": 0},
        {"kind": "profile_fact", "content": "Fact from ep1", "summary": "Fact", "confidence": 0.9, "episode_index": 1},
    ]

    ep0 = _make_episode(payload={"messages": [{"role": "user", "content": "Hello from episode 0"}]})
    ep1 = _make_episode(payload={"messages": [{"role": "user", "content": "Hello from episode 1"}]})

    with patch.object(compiler, '_call_llm_async', new_callable=AsyncMock, return_value=raw) as mock_llm:
        memories = await compiler.compile_async([ep0, ep1])
        # Should be 1 call (both episodes fit in one batch)
        assert mock_llm.call_count == 1
        assert len(memories) == 2
        assert memories[0].source_episode_ids == [ep0.id]
        assert memories[1].source_episode_ids == [ep1.id]
