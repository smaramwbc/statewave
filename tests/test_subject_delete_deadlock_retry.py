"""Subject purge retries once on a Postgres deadlock (main-CI flake 09-02).

The purge's multi-row DELETEs can cross lock order with another
multi-row writer on the same subject's rows (a still-draining compile
batch, an embedding backfill). Postgres aborts exactly one side with
SQLSTATE 40P01; before this the purge surfaced that as a raw 500 —
observed as `DELETE FROM memories` deadlocking in
tests/integration/test_semantic.py on main run 33638444055. A single
rollback-and-redo converges: the competing transaction has either
finished or loses the rematch.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import DBAPIError

from server.api import subjects as api_subjects

pytestmark = pytest.mark.asyncio


def _deadlock_error() -> DBAPIError:
    orig = Exception("deadlock detected\nDETAIL: Process 1 waits for ShareLock...")
    return DBAPIError("DELETE FROM memories ...", params=None, orig=orig)


def _other_db_error() -> DBAPIError:
    return DBAPIError("DELETE ...", params=None, orig=Exception("connection reset"))


def _wire(monkeypatch, delete_memories):
    async def two(_s, _subj, *, tenant_id):
        return 2

    async def none(_s, _subj, *, tenant_id):
        return 0

    monkeypatch.setattr(api_subjects.repo, "delete_episodes_by_subject", two)
    monkeypatch.setattr(api_subjects.repo, "delete_memories_by_subject", delete_memories)
    monkeypatch.setattr(api_subjects.repo, "delete_resolutions_by_subject", none)
    monkeypatch.setattr(api_subjects.repo, "delete_health_cache_by_subject", none)
    monkeypatch.setattr(api_subjects.repo, "delete_entities_by_subject", none)
    monkeypatch.setattr(api_subjects.webhooks, "fire", AsyncMock())


def _session():
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


async def test_purge_retries_once_on_deadlock_and_succeeds(monkeypatch):
    calls = {"n": 0}

    async def flaky_delete(_s, _subj, *, tenant_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _deadlock_error()
        return 3

    _wire(monkeypatch, flaky_delete)
    session = _session()

    resp = await api_subjects.delete_subject("subj", session=session, tenant_id=None)

    assert resp.memories_deleted == 3
    assert calls["n"] == 2
    session.rollback.assert_awaited_once()
    session.commit.assert_awaited_once()


async def test_purge_gives_up_after_second_deadlock(monkeypatch):
    async def always_deadlocks(_s, _subj, *, tenant_id):
        raise _deadlock_error()

    _wire(monkeypatch, always_deadlocks)
    session = _session()

    with pytest.raises(DBAPIError):
        await api_subjects.delete_subject("subj", session=session, tenant_id=None)
    session.rollback.assert_awaited_once()  # only between attempts, not after the raise


async def test_purge_does_not_retry_non_deadlock_errors(monkeypatch):
    calls = {"n": 0}

    async def breaks(_s, _subj, *, tenant_id):
        calls["n"] += 1
        raise _other_db_error()

    _wire(monkeypatch, breaks)
    session = _session()

    with pytest.raises(DBAPIError):
        await api_subjects.delete_subject("subj", session=session, tenant_id=None)
    assert calls["n"] == 1, "a non-deadlock DB error must surface immediately"
    session.rollback.assert_not_awaited()
