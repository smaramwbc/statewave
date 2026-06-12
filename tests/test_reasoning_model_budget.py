"""Reasoning models (o-series, gpt-5.x, ...) spend part of the token budget on
hidden reasoning tokens before emitting output, so a tight `max_tokens` cap
truncates their JSON (→ compile 502) and a 1-token health ping can never finish.

These tests pin the two fixes:
  1. the compiler floors its per-batch budget at half the (configurable) ceiling,
     so even a single-episode batch leaves reasoning headroom;
  2. aping treats a max-tokens / output-limit rejection as reachable — that round
     trip still proved auth + connectivity.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from server.core.config import settings
from server.services import llm as llm_adapter
from server.services.compilers.llm import LLMCompiler


@pytest.mark.asyncio
async def test_single_episode_batch_gets_reasoning_headroom(monkeypatch):
    """A single-episode batch must request at least half the ceiling, not the old
    tight 1500 — otherwise a reasoning model truncates its output."""
    monkeypatch.setattr(settings, "litellm_compile_max_tokens", 16000)
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-5.4-mini"

    captured: dict = {}

    async def fake_acomplete_json(messages, *, model, temperature, max_tokens):
        captured["max_tokens"] = max_tokens
        return {"memories": []}

    with patch.object(llm_adapter, "acomplete_json", new=AsyncMock(side_effect=fake_acomplete_json)):
        await compiler._call_llm_async("some episode text", episode_count=1)

    assert captured["max_tokens"] == 8000  # half of 16000, not 1500


@pytest.mark.asyncio
async def test_budget_scales_with_batch_but_stays_under_ceiling(monkeypatch):
    monkeypatch.setattr(settings, "litellm_compile_max_tokens", 16000)
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    captured: dict = {}

    async def fake_acomplete_json(messages, *, model, temperature, max_tokens):
        captured["max_tokens"] = max_tokens
        return {"memories": []}

    with patch.object(llm_adapter, "acomplete_json", new=AsyncMock(side_effect=fake_acomplete_json)):
        await compiler._call_llm_async("text", episode_count=20)  # 20 * 1500 = 30000

    assert captured["max_tokens"] == 16000  # capped at the ceiling, never above the model's output limit


@pytest.mark.asyncio
async def test_ceiling_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "litellm_compile_max_tokens", 4000)  # e.g. a legacy small-output model
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4"
    captured: dict = {}

    async def fake_acomplete_json(messages, *, model, temperature, max_tokens):
        captured["max_tokens"] = max_tokens
        return {"memories": []}

    with patch.object(llm_adapter, "acomplete_json", new=AsyncMock(side_effect=fake_acomplete_json)):
        await compiler._call_llm_async("text", episode_count=1)

    assert captured["max_tokens"] == 2000  # half of the lowered ceiling


@pytest.mark.asyncio
async def test_aping_treats_output_limit_as_reachable():
    """A reasoning model that exhausts the 1-token ping budget on reasoning is
    still REACHABLE — auth + connectivity were proven. aping must not report it
    as a failure."""
    err = llm_adapter.LLMProviderError(
        "OpenAIException - Could not finish the message because max_tokens or model "
        "output limit was reached."
    )
    with patch.object(llm_adapter, "acomplete", new=AsyncMock(side_effect=err)):
        assert await llm_adapter.aping() is True


@pytest.mark.asyncio
async def test_aping_still_raises_real_failures():
    """A genuine auth/connectivity failure must still surface."""
    err = llm_adapter.LLMProviderError("AuthenticationError - invalid api key")
    with patch.object(llm_adapter, "acomplete", new=AsyncMock(side_effect=err)):
        with pytest.raises(llm_adapter.LLMProviderError):
            await llm_adapter.aping()
