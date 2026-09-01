"""A truncated LLM response must not cost the whole subject (issue #375).

`acomplete_json` raises `LLMResponseError` when the provider's body will not
parse. That is a different class of failure from a transport/auth error: the
model answered, the answer is just unusable — typically because this particular
combination of episodes provoked an over-long response. Splitting the batch
shrinks each response, so one awkward episode costs itself rather than every
episode batched with it.

Before this, `_call_llm_async` flattened every provider error to `RuntimeError`,
so `_process_batch` could not tell the two apart and aborted the entire compile
— leaving the subject with zero memories and no way to recover, since a retry
re-ran the same batch.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.db.tables import EpisodeRow
from server.services.compilers.errors import CompilationError
from server.services.compilers.llm import LLMCompiler
from server.services.llm import LLMResponseError

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 5, 4, tzinfo=timezone.utc)


def _ep(text: str) -> EpisodeRow:
    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="test",
        type="conversation",
        payload={"text": text},
        metadata_={},
        provenance={},
        created_at=_NOW,
        occurred_at=_NOW,
    )


def _memory(content: str) -> dict:
    return {"kind": "profile_fact", "content": content, "summary": content}


async def test_irreducible_episode_raises_rather_than_being_swallowed(caplog):
    """An episode whose output will not parse even alone still raises — issue
    #201's guarantee that a failure never silently consumes episodes."""
    poison = _ep("poison")

    async def fake_call(text: str, count: int):
        raise LLMResponseError("LLM returned invalid JSON (3693 chars): {…")

    compiler = LLMCompiler()
    with patch.object(compiler, "_call_llm_async", new=AsyncMock(side_effect=fake_call)):
        with pytest.raises(LLMResponseError):
            await compiler._extract_with_split_retry([(poison, "poison")], "poison")


async def test_process_batch_names_the_offending_episode(caplog):
    """The whole point of #375: the operator learns WHICH episode is poison
    instead of an opaque \"provider unavailable or misconfigured\"."""
    poison = _ep("poison")

    async def fake_call(text: str, count: int):
        raise LLMResponseError("LLM returned invalid JSON (3693 chars): {…")

    compiler = LLMCompiler()
    sem = asyncio.Semaphore(1)
    with patch.object(compiler, "_call_llm_async", new=AsyncMock(side_effect=fake_call)):
        with pytest.raises(CompilationError) as excinfo:
            await compiler._process_batch([(poison, "poison")], sem)

    assert str(poison.id) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, LLMResponseError)


async def test_split_recovers_when_the_smaller_halves_parse():
    """The realistic case: the combination was too long, the halves are fine."""
    a, b = _ep("alpha"), _ep("beta")
    batch = [(a, "alpha"), (b, "beta")]
    calls: list[int] = []

    async def fake_call(text: str, count: int):
        calls.append(count)
        if count > 1:
            raise LLMResponseError("LLM returned invalid JSON (3693 chars): {…")
        return [_memory(text)]

    compiler = LLMCompiler()
    with patch.object(compiler, "_call_llm_async", new=AsyncMock(side_effect=fake_call)):
        out = await compiler._extract_with_split_retry(batch, "alpha\n\nbeta")

    assert calls == [2, 1, 1]  # tried together, then each alone
    # Both halves produced a memory: the poison combination cost nothing.
    assert len(out) == 2
    contents = " ".join(m["content"] for m in out)
    assert "alpha" in contents and "beta" in contents


async def test_transport_failures_are_not_split_retried():
    """A round-trip that could not RUN must abort immediately (issue #201) —
    splitting would just multiply a provider outage into N failing calls."""
    a, b = _ep("alpha"), _ep("beta")
    calls: list[int] = []

    async def fake_call(text: str, count: int):
        calls.append(count)
        raise RuntimeError("LLM completion timed out after 60.0s")

    compiler = LLMCompiler()
    with patch.object(compiler, "_call_llm_async", new=AsyncMock(side_effect=fake_call)):
        with pytest.raises(RuntimeError):
            await compiler._extract_with_split_retry([(a, "alpha"), (b, "beta")], "alpha\n\nbeta")

    assert calls == [2]  # no split, no retry


async def test_call_llm_async_preserves_the_unparseable_type():
    """The root cause of #375: `_call_llm_async` used to flatten EVERY provider
    error to a bare RuntimeError, so `_process_batch` structurally could not
    tell "answered with unusable output" from "could not reach the provider" —
    and treated both as fatal. The type must survive to reach the split path."""
    compiler = LLMCompiler()

    with patch(
        "server.services.llm.acomplete_json",
        new=AsyncMock(side_effect=LLMResponseError("LLM returned invalid JSON (3693 chars): {…")),
    ):
        with pytest.raises(LLMResponseError):
            await compiler._call_llm_async("some text", 1)


async def test_call_llm_async_still_flattens_transport_errors():
    """Transport/auth failures keep their existing surface — only the
    unparseable-output case is singled out."""
    from server.services.llm import StatewaveLLMError

    compiler = LLMCompiler()
    with patch(
        "server.services.llm.acomplete_json",
        new=AsyncMock(side_effect=StatewaveLLMError("LLM completion timed out after 60.0s")),
    ):
        with pytest.raises(RuntimeError) as excinfo:
            await compiler._call_llm_async("some text", 1)
    assert not isinstance(excinfo.value, LLMResponseError)
