"""Regression tests for issue #134.

Before the fix, `POST /v1/memories/compile` silently processed at most
500 uncompiled episodes per call (the `list_uncompiled_episodes`
default), and the response carried no completion signal. Operators with
large episode backlogs saw the admin UI's `Memories` count plateau at
small round numbers (commonly ~500 or ~1000) with no error.

The fix:
  * `CompileMemoriesResponse` carries `has_more` and `remaining_episodes`.
  * Sync mode stays bounded per call (latency budget) but signals the
    backlog so the caller can loop.
  * Async mode (`async: true`) drains the subject by looping internally,
    capped by `settings.compile_max_iterations`.

These unit-level tests verify the schema and the drain loop without a
real database — the integration counterpart in
`tests/integration/test_issue_134_drain.py` covers the live flow.
"""

from __future__ import annotations

import pytest

from server.schemas.responses import CompileMemoriesResponse


def test_134_response_carries_drain_signal_with_safe_defaults():
    """Existing call sites that omit the new fields still construct
    a valid response — the defaults state "no more work pending"."""
    r = CompileMemoriesResponse(subject_id="s", memories_created=0, memories=[])
    assert r.has_more is False
    assert r.remaining_episodes == 0


def test_134_response_accepts_has_more_and_remaining():
    r = CompileMemoriesResponse(
        subject_id="s",
        memories_created=3,
        memories=[],
        has_more=True,
        remaining_episodes=42,
    )
    assert r.has_more is True
    assert r.remaining_episodes == 42


@pytest.mark.anyio
async def test_134_async_drain_loops_until_remaining_is_zero(monkeypatch):
    """`_run_compile` keeps calling `_compile_one_batch` until the batch
    reports `remaining == 0`. Each batch's `memories_created` accumulates
    into the final response and into the durable job progress."""
    from server.api import memories as api_memories

    # Three batches: 4 created + 2 remaining, 4 created + 1 remaining,
    # 1 created + 0 remaining → final total 9, has_more False.
    scripted = [
        ([], 4, 2),
        ([], 4, 1),
        ([], 1, 0),
    ]

    call_log = []

    async def fake_batch(_session, _subject_id, _tenant_id, _batch_size):
        call_log.append("batch")
        return scripted.pop(0)

    # `_run_compile` opens its own session via `get_session_factory()()`.
    # We don't care what session arrives at `_compile_one_batch` — only
    # that the loop drives it to drain.
    async def fake_factory_call():
        class _Ctx:
            async def __aenter__(self_):
                return object()  # opaque session sentinel

            async def __aexit__(self_, *a):
                return None

        return _Ctx()

    progress_updates = []

    async def fake_update_progress(job_id, total):
        progress_updates.append(total)

    async def fake_mark_running(_job_id):
        pass

    async def fake_mark_completed(_job_id, _total, _mems):
        pass

    monkeypatch.setattr(api_memories, "_compile_one_batch", fake_batch)
    monkeypatch.setattr(
        api_memories.compile_jobs, "update_progress_durable", fake_update_progress
    )
    monkeypatch.setattr(
        api_memories.compile_jobs, "mark_running_durable", fake_mark_running
    )
    monkeypatch.setattr(
        api_memories.compile_jobs, "mark_completed_durable", fake_mark_completed
    )

    # Patch `get_session_factory` so `_run_compile`'s `async with ... as session`
    # works without a DB.
    import server.db.engine as engine_module

    class _FakeFactory:
        def __call__(self_inner):
            class _Ctx:
                async def __aenter__(self_):
                    return object()

                async def __aexit__(self_, *a):
                    return None

            return _Ctx()

    monkeypatch.setattr(engine_module, "get_session_factory", lambda: _FakeFactory())

    result = await api_memories._run_compile("subj-x", job_id="job-1", tenant_id=None)

    assert len(call_log) == 3, "loop should run exactly one batch per scripted entry"
    assert result.memories_created == 9
    assert result.has_more is False
    assert result.remaining_episodes == 0
    # Progress published on first iteration even with zero creates, then on
    # every subsequent batch that created memories.
    assert progress_updates == [4, 8, 9]


@pytest.mark.anyio
async def test_134_async_drain_respects_iteration_cap(monkeypatch):
    """If the compiler keeps reporting work remaining (e.g. fails to
    advance `last_compiled_at`), `_run_compile` must stop at
    `settings.compile_max_iterations` instead of looping forever."""
    from server.api import memories as api_memories
    from server.core.config import settings

    async def never_drains(_session, _subject_id, _tenant_id, _batch_size):
        return [], 1, 999  # always 1 created, always 999 remaining

    async def noop(*_a, **_k):
        return None

    monkeypatch.setattr(api_memories, "_compile_one_batch", never_drains)
    monkeypatch.setattr(api_memories.compile_jobs, "update_progress_durable", noop)
    monkeypatch.setattr(api_memories.compile_jobs, "mark_running_durable", noop)
    monkeypatch.setattr(api_memories.compile_jobs, "mark_completed_durable", noop)

    import server.db.engine as engine_module

    class _FakeFactory:
        def __call__(self_inner):
            class _Ctx:
                async def __aenter__(self_):
                    return object()

                async def __aexit__(self_, *a):
                    return None

            return _Ctx()

    monkeypatch.setattr(engine_module, "get_session_factory", lambda: _FakeFactory())
    monkeypatch.setattr(settings, "compile_max_iterations", 5)

    result = await api_memories._run_compile("subj-stuck", job_id=None, tenant_id=None)

    # 5 iterations × 1 created per iteration → 5 total. has_more=True
    # because the loop bailed out without draining.
    assert result.memories_created == 5
    assert result.has_more is True
    assert result.remaining_episodes == 999
