"""Regression test for issue #115.

The LLM compiler must give the model the episode's real reference date so
relative phrases ("today", "this morning") resolve to the actual date
instead of a hallucinated one (the LoCoMo sample's "25 May 2023").
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.db.tables import EpisodeRow
from server.services.compilers.llm import _SYSTEM_PROMPT, LLMCompiler


# ---------------------------------------------------------------------------
# #115 — episode reference date reaches the LLM prompt
# ---------------------------------------------------------------------------


def _make_compiler() -> LLMCompiler:
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    return compiler


def _episode(**kw) -> EpisodeRow:
    defaults = dict(
        id=uuid.uuid4(),
        subject_id="alice_bob_conversation",
        source="test",
        type="chat",
        payload={"messages": [{"role": "user", "content": "Bob checked the weather this morning."}]},
        metadata_={},
        provenance={},
        created_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    defaults.update(kw)
    return EpisodeRow(**defaults)


@pytest.mark.asyncio
async def test_115_prompt_carries_episode_reference_date():
    """The episode's resolved timestamp must be in the text sent to the LLM."""
    compiler = _make_compiler()
    ep = _episode()

    captured = AsyncMock(return_value=[])
    with patch.object(compiler, "_call_llm_async", captured):
        await compiler.compile_async([ep])

    assert captured.await_count == 1
    prompt_text = captured.await_args.args[0]
    # 2026-05-16 is a Saturday — header is "recorded YYYY-MM-DD (Weekday)".
    assert "Episode 0 | recorded 2026-05-16 (Saturday)" in prompt_text
    # The bare, dateless header must no longer be emitted.
    assert "--- Episode 0 ---" not in prompt_text


@pytest.mark.asyncio
async def test_115_message_timestamp_overrides_episode_date():
    """payload.messages[0].timestamp wins over created_at (episode_valid_from)."""
    compiler = _make_compiler()
    ep = _episode(
        payload={
            "messages": [
                {"role": "user", "content": "Hi", "timestamp": "2024-03-02T09:00:00+00:00"}
            ]
        },
    )
    captured = AsyncMock(return_value=[])
    with patch.object(compiler, "_call_llm_async", captured):
        await compiler.compile_async([ep])

    assert "recorded 2024-03-02" in captured.await_args.args[0]


def test_115_system_prompt_forbids_inventing_dates():
    assert "NEVER invent, guess, or default a date" in _SYSTEM_PROMPT
    assert "recorded YYYY-MM-DD (Weekday)" in _SYSTEM_PROMPT
    # The old unconditional "omit the date" escape hatch is gone — it now
    # only applies when there is genuinely no reference date at all.
    assert "If no timestamp is available and no absolute date" not in _SYSTEM_PROMPT
