"""Async compile-start must attach to a live job, not race it (docs-refresh
failure class, 4th occurrence 2026-09-01).

An async compile job is an in-process asyncio task backed by a durable row.
Two failure shapes at compile-start on a busy subject:

* The previous job is ALIVE but slow — a second submission would start a
  concurrent drain over the same uncompiled-episode snapshot and
  double-compile them. Compile-start must return the live job's id instead.
* The previous job is ORPHANED — its process restarted (rolling deploy), the
  task died, the row says `running` forever. Compile-start must supersede it
  and start a fresh job that resumes the remaining episodes.

Liveness is the `heartbeat_at` column, bumped by the LLM compiler as each
internal batch completes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import server.services.compile_jobs_durable as durable
from server.services.compile_jobs_durable import ACTIVE_STALE_SECONDS, find_active_job

pytestmark = pytest.mark.asyncio

_NOW = datetime.now(timezone.utc)


def _row(status="running", heartbeat_age_s=10.0, job_id="j1"):
    hb = _NOW - timedelta(seconds=heartbeat_age_s) if heartbeat_age_s is not None else None
    return SimpleNamespace(
        id=job_id,
        subject_id="subj",
        tenant_id=None,
        status=status,
        memories_created=0,
        error=None,
        created_at=_NOW - timedelta(seconds=3600),
        started_at=None,
        completed_at=None,
        heartbeat_at=hb,
    )


def _session_returning(rows):
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    session.execute = AsyncMock(return_value=result)
    return session


def _patched_factory(session):
    return patch.object(durable, "get_session_factory", return_value=lambda: session)


# ---------------------------------------------------------------------------
# find_active_job
# ---------------------------------------------------------------------------


async def test_fresh_heartbeat_job_is_returned_as_live():
    row = _row(heartbeat_age_s=30.0)
    session = _session_returning([row])
    with _patched_factory(session):
        job = await find_active_job("subj")
    assert job is not None and job.id == "j1"
    assert row.status == "running", "a live job must not be touched"


async def test_stale_heartbeat_job_is_superseded():
    row = _row(heartbeat_age_s=ACTIVE_STALE_SECONDS + 60)
    session = _session_returning([row])
    with _patched_factory(session):
        job = await find_active_job("subj")
    assert job is None, "a stale job is dead — caller should submit a new one"
    assert row.status == "failed"
    assert "orphaned" in row.error
    assert row.completed_at is not None
    session.commit.assert_awaited()


async def test_no_active_rows_returns_none():
    session = _session_returning([])
    with _patched_factory(session):
        assert await find_active_job("subj") is None


async def test_pending_row_liveness_falls_back_to_created_at():
    """A job submitted moments ago has no heartbeat yet — it must still
    count as live or every attach would race the brand-new job."""
    row = _row(status="pending", heartbeat_age_s=None)
    row.created_at = _NOW - timedelta(seconds=5)
    session = _session_returning([row])
    with _patched_factory(session):
        job = await find_active_job("subj")
    assert job is not None and job.id == "j1"


async def test_second_fresh_row_is_left_alone():
    """Two fresh in-flight jobs are a pre-attach race artifact; neither task
    can be cancelled from here, so neither row may be marked failed."""
    newer, older = _row(job_id="new", heartbeat_age_s=10), _row(job_id="old", heartbeat_age_s=20)
    session = _session_returning([newer, older])
    with _patched_factory(session):
        job = await find_active_job("subj")
    assert job is not None and job.id == "new"
    assert older.status == "running"


# ---------------------------------------------------------------------------
# compile_memories: async branch attaches instead of racing
# ---------------------------------------------------------------------------


def _api():
    from server.api import memories as api_memories

    return api_memories


async def test_compile_start_attaches_to_live_job(monkeypatch):
    api_memories = _api()
    from server.schemas.requests import CompileMemoriesRequest
    from server.services.compile_jobs import CompileJob, JobStatus

    live = CompileJob(id="live-1", subject_id="subj", status=JobStatus.running)
    find = AsyncMock(return_value=live)
    submit = AsyncMock(side_effect=AssertionError("must not submit a second job"))
    monkeypatch.setattr(api_memories.compile_jobs, "find_active_job_durable", find)
    monkeypatch.setattr(api_memories.compile_jobs, "submit_job_durable", submit)

    body = CompileMemoriesRequest(**{"subject_id": "subj", "async": True})
    resp = await api_memories.compile_memories(body, session=MagicMock(), tenant_id=None)

    assert resp.status_code == 202
    import json

    payload = json.loads(resp.body)
    assert payload["job_id"] == "live-1"
    assert payload["attached"] is True
    find.assert_awaited_once_with("subj", tenant_id=None)


async def test_compile_start_submits_when_no_live_job(monkeypatch):
    api_memories = _api()
    from server.schemas.requests import CompileMemoriesRequest
    from server.services.compile_jobs import CompileJob

    submitted = CompileJob(id="new-1", subject_id="subj")
    monkeypatch.setattr(
        api_memories.compile_jobs, "find_active_job_durable", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        api_memories.compile_jobs, "submit_job_durable", AsyncMock(return_value=submitted)
    )

    async def fake_run_compile(*_a, **_kw):
        return None

    monkeypatch.setattr(api_memories, "_run_compile", fake_run_compile)

    body = CompileMemoriesRequest(**{"subject_id": "subj", "async": True})
    resp = await api_memories.compile_memories(body, session=MagicMock(), tenant_id=None)
    # Let the create_task'd noop settle before the loop closes.
    await asyncio.sleep(0)

    assert resp.status_code == 202
    import json

    payload = json.loads(resp.body)
    assert payload["job_id"] == "new-1"
    assert "attached" not in payload


# ---------------------------------------------------------------------------
# heartbeat threading: _run_compile → _compile_one_batch → compiler
# ---------------------------------------------------------------------------


async def test_run_compile_threads_a_working_heartbeat_cb(monkeypatch):
    api_memories = _api()

    seen_cbs = []

    async def fake_batch(_session, _subject_id, _tenant_id, _batch_size, progress_cb=None):
        seen_cbs.append(progress_cb)
        return ([], 0, 0)

    hb = AsyncMock()
    monkeypatch.setattr(api_memories, "_compile_one_batch", fake_batch)
    monkeypatch.setattr(api_memories.compile_jobs, "heartbeat_durable", hb)
    monkeypatch.setattr(api_memories.compile_jobs, "mark_running_durable", AsyncMock())
    monkeypatch.setattr(api_memories.compile_jobs, "update_progress_durable", AsyncMock())
    monkeypatch.setattr(api_memories.compile_jobs, "mark_completed_durable", AsyncMock())

    import server.db.engine as engine_module

    class _Ctx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *a):
            return None

    monkeypatch.setattr(engine_module, "get_session_factory", lambda: lambda: _Ctx())

    await api_memories._run_compile("subj", job_id="job-1", tenant_id=None)

    assert len(seen_cbs) == 1 and seen_cbs[0] is not None
    await seen_cbs[0]()
    hb.assert_awaited_once_with("job-1")


async def test_heartbeat_cb_failure_does_not_raise(monkeypatch):
    """A heartbeat write failure only degrades liveness detection — it must
    never kill a compile that is otherwise succeeding."""
    api_memories = _api()
    monkeypatch.setattr(
        api_memories.compile_jobs,
        "heartbeat_durable",
        AsyncMock(side_effect=RuntimeError("db hiccup")),
    )
    cb = api_memories._heartbeat_cb("job-1")
    await cb()  # must not raise


async def test_heartbeat_cb_is_none_without_job():
    api_memories = _api()
    assert api_memories._heartbeat_cb(None) is None


# ---------------------------------------------------------------------------
# LLM compiler: one ping per completed internal batch
# ---------------------------------------------------------------------------


async def test_llm_compiler_pings_progress_per_batch():
    import uuid

    from server.db.tables import EpisodeRow
    from server.services.compilers.llm import LLMCompiler

    def _ep(text):
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

    compiler = LLMCompiler()
    process = AsyncMock(return_value=[])
    cb = AsyncMock()
    # Two ~4000-char episodes exceed the 6000-char batch budget → 2 batches.
    episodes = [_ep("x" * 4000), _ep("y" * 4000)]
    with patch.object(compiler, "_process_batch", new=process):
        await compiler.compile_async(episodes, progress_cb=cb)

    assert process.await_count >= 1
    assert cb.await_count == process.await_count, "one liveness ping per completed batch"


async def test_llm_compiler_progress_cb_defaults_to_none():
    """Callers that don't care about liveness (sync path, tests) pass
    nothing and nothing extra runs."""
    import uuid

    from server.db.tables import EpisodeRow
    from server.services.compilers.llm import LLMCompiler

    ep = EpisodeRow(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="test",
        type="conversation",
        payload={"text": "hello"},
        metadata_={},
        provenance={},
        created_at=_NOW,
        occurred_at=_NOW,
    )
    compiler = LLMCompiler()
    with patch.object(compiler, "_process_batch", new=AsyncMock(return_value=[])):
        result = await compiler.compile_async([ep])
    assert result == []
