"""Unit tests for ``acomplete_with_usage`` + ``_extract_usage``.

These power the token stats that ``/v1/llm/complete`` returns so first-party
consoles (the admin "Chat with Memory") can show usage without holding their
own provider key. The provider call is mocked so the tests stay fast and
credential-free; we only assert how usage is extracted and shaped.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.services import llm as llm_mod


def _fake_completion(content: str, usage):
    """Build a LiteLLM-shaped chat-completion response object."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.anyio
async def test_returns_reply_and_usage(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "litellm_model", "gpt-4o-mini")
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15)
    fake_litellm = MagicMock()
    fake_litellm.acompletion = AsyncMock(return_value=_fake_completion("hi there", usage))

    with patch.object(llm_mod, "_ensure_litellm", return_value=fake_litellm):
        reply, got = await llm_mod.acomplete_with_usage(
            [{"role": "user", "content": "hi"}], max_tokens=50
        )

    assert reply == "hi there"
    assert got == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}


@pytest.mark.anyio
async def test_usage_none_when_provider_omits_it(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "litellm_model", "ollama/llama3")
    fake_litellm = MagicMock()
    fake_litellm.acompletion = AsyncMock(return_value=_fake_completion("local", None))

    with patch.object(llm_mod, "_ensure_litellm", return_value=fake_litellm):
        reply, got = await llm_mod.acomplete_with_usage([{"role": "user", "content": "hi"}])

    assert reply == "local"
    assert got is None


@pytest.mark.anyio
async def test_derives_total_when_provider_omits_it(monkeypatch):
    """Some providers report prompt/completion but not total — derive it."""
    monkeypatch.setattr(llm_mod.settings, "litellm_model", "gpt-4o-mini")
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=None)
    fake_litellm = MagicMock()
    fake_litellm.acompletion = AsyncMock(return_value=_fake_completion("x", usage))

    with patch.object(llm_mod, "_ensure_litellm", return_value=fake_litellm):
        _, got = await llm_mod.acomplete_with_usage([{"role": "user", "content": "hi"}])

    assert got == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


@pytest.mark.anyio
async def test_all_zero_usage_collapses_to_none(monkeypatch):
    """An all-zero usage block is indistinguishable from "not reported" — we
    surface None so the UI shows "n/a" rather than a misleading 0."""
    monkeypatch.setattr(llm_mod.settings, "litellm_model", "gpt-4o-mini")
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    fake_litellm = MagicMock()
    fake_litellm.acompletion = AsyncMock(return_value=_fake_completion("x", usage))

    with patch.object(llm_mod, "_ensure_litellm", return_value=fake_litellm):
        _, got = await llm_mod.acomplete_with_usage([{"role": "user", "content": "hi"}])

    assert got is None
