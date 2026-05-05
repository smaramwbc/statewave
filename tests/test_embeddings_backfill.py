"""Tests for `server.services.embeddings.backfill`.

The helper is shared by two write paths: the LLM compile endpoint
(`POST /v1/memories/compile`) and the bulk-import path
(`_ingest_records_async`). Both must produce embedded memories with the
same guarantees, so the contract here is:

  * empty inputs → no-op (no provider call, no DB session)
  * mismatched id/text lengths → logged + no-op (defensive)
  * provider returns None (`STATEWAVE_EMBEDDING_PROVIDER=none`) → no-op
  * provider raises → swallow + log warning, never propagate
  * happy path → one provider call, one UPDATE per memory

These tests stub the provider + DB session factory so they run without
Postgres. The integration test that exercises the real ingest path
through `_ingest_records_async` lives separately.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from server.services.embeddings import backfill as backfill_mod


@pytest.mark.anyio
async def test_empty_inputs_are_a_noop():
    """No provider call when there's nothing to embed."""
    with patch.object(backfill_mod, "get_provider") as gp:
        await backfill_mod.generate_embeddings_background([], [])
        gp.assert_not_called()


@pytest.mark.anyio
async def test_mismatched_input_lengths_short_circuits():
    """Defensive guard: callers that pass mismatched ids/texts get a
    no-op + warning rather than a partial UPDATE that would silently
    desync the DB."""
    with patch.object(backfill_mod, "get_provider") as gp:
        await backfill_mod.generate_embeddings_background(
            [uuid4(), uuid4()], ["only one text"]
        )
        gp.assert_not_called()


@pytest.mark.anyio
async def test_disabled_provider_short_circuits():
    """When `STATEWAVE_EMBEDDING_PROVIDER=none`, `get_provider()` returns
    None and we must not touch the DB."""
    with (
        patch.object(backfill_mod, "get_provider", return_value=None),
        patch.object(backfill_mod, "get_session_factory") as sf,
    ):
        await backfill_mod.generate_embeddings_background(
            [uuid4()], ["something"]
        )
        sf.assert_not_called()


@pytest.mark.anyio
async def test_happy_path_writes_one_embedding_per_memory():
    """Single provider call, one UPDATE per memory, one commit."""
    ids = [uuid4(), uuid4(), uuid4()]
    texts = ["a", "b", "c"]
    fake_vectors = [[0.1] * 4, [0.2] * 4, [0.3] * 4]

    fake_provider = MagicMock()
    fake_provider.embed_texts = AsyncMock(return_value=fake_vectors)

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()
    fake_session.commit = AsyncMock()

    fake_session_ctx = MagicMock()
    fake_session_ctx.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session_ctx.__aexit__ = AsyncMock(return_value=None)
    fake_factory = MagicMock(return_value=fake_session_ctx)

    with (
        patch.object(backfill_mod, "get_provider", return_value=fake_provider),
        patch.object(backfill_mod, "get_session_factory", return_value=fake_factory),
    ):
        await backfill_mod.generate_embeddings_background(ids, texts)

    fake_provider.embed_texts.assert_awaited_once_with(texts)
    assert fake_session.execute.await_count == len(ids)
    fake_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_provider_failure_is_swallowed():
    """A flaky provider must NOT propagate — embeddings are an
    optimisation, retrieval falls back to non-vector ranking when
    they're missing. The exception is logged and the request that
    spawned this background task is unaffected."""
    fake_provider = MagicMock()
    fake_provider.embed_texts = AsyncMock(side_effect=RuntimeError("boom"))

    with patch.object(backfill_mod, "get_provider", return_value=fake_provider):
        # Must not raise.
        await backfill_mod.generate_embeddings_background(
            [uuid4()], ["something"]
        )


@pytest.mark.anyio
async def test_schedule_returns_none_when_nothing_to_do():
    """The convenience scheduler skips `asyncio.create_task` entirely on
    empty inputs — keeps the no-op path zero-cost."""
    task = backfill_mod.schedule_embedding_backfill([], [])
    assert task is None


@pytest.mark.anyio
async def test_schedule_returns_running_task():
    """Happy path: returns the scheduled task so callers can await it
    in tests (production callers fire-and-forget)."""
    fake_provider = MagicMock()
    fake_provider.embed_texts = AsyncMock(return_value=[[0.0] * 4])

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()
    fake_session.commit = AsyncMock()
    fake_session_ctx = MagicMock()
    fake_session_ctx.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session_ctx.__aexit__ = AsyncMock(return_value=None)
    fake_factory = MagicMock(return_value=fake_session_ctx)

    with (
        patch.object(backfill_mod, "get_provider", return_value=fake_provider),
        patch.object(backfill_mod, "get_session_factory", return_value=fake_factory),
    ):
        task = backfill_mod.schedule_embedding_backfill([uuid4()], ["x"])
        assert task is not None
        await task
        fake_provider.embed_texts.assert_awaited_once()
