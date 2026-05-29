"""Regression tests for issue #201.

A compile batch that fails for a configuration/provider reason (e.g. the LLM
compiler with no reachable key) used to be indistinguishable from a batch
that legitimately extracted nothing: `LLMCompiler._process_batch` swallowed
the error into an empty list, and `_compile_one_batch` marked the episodes
compiled regardless. The episodes were then permanently consumed — a later,
correctly configured recompile could not recover them.

The fix:
  * Provider/config failures raise `CompilationError` instead of returning [].
  * `_compile_one_batch` returns before `mark_episodes_compiled`/`commit` when
    that error propagates, so the episodes stay uncompiled and recoverable.
  * The `/v1/memories/compile` sync route surfaces a 502 (not a misleading
    `memories_created: 0`); async mode marks the durable job failed.
  * A legitimate empty extraction still marks the episodes compiled.

These are unit-level tests that exercise the control flow without a real
database, mirroring tests/test_issue_134_compile_drain.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.services.compilers.errors import CompilationError


class _FakeEpisode:
    def __init__(self) -> None:
        self.id = uuid.uuid4()
        self.subject_id = "user-1"
        self.payload = {"text": "My name is Alice and I work at Globex."}
        self.created_at = datetime(2026, 5, 29, tzinfo=timezone.utc)


class _FakeSession:
    """Minimal async-session stand-in — records commits, no DB."""

    def __init__(self) -> None:
        self.added: list = []
        self.commits = 0

    def add(self, row) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, row) -> None:  # pragma: no cover - no rows in these tests
        pass


# ---------------------------------------------------------------------------
# Compiler layer: a provider failure raises rather than returning [].
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_failure_raises_compilation_error():
    """The canonical no-key case: the provider round-trip fails, so
    `compile_async` raises `CompilationError` (not an empty list)."""
    from server.services.compilers.llm import LLMCompiler
    from server.services import llm as llm_adapter

    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"

    ep = _FakeEpisode()

    with patch.object(
        compiler,
        "_call_llm_async",
        new_callable=AsyncMock,
        side_effect=llm_adapter.LLMProviderError("401 missing key"),
    ):
        with pytest.raises(CompilationError, match="missing key"):
            await compiler.compile_async([ep])


# ---------------------------------------------------------------------------
# _compile_one_batch: failed batch must NOT mark episodes compiled.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_batch_does_not_consume_episodes(monkeypatch):
    """When the compiler raises, `mark_episodes_compiled` must never run —
    the episodes stay uncompiled and recoverable (issue #201)."""
    from server.api import memories as api_memories

    episodes = [_FakeEpisode()]
    marked: list = []

    async def fake_list(_session, _subject_id, *, tenant_id, limit):
        return episodes

    async def fake_mark(_session, ids):
        marked.append(ids)

    class _RaisingCompiler:
        async def compile_async(self, _eps):
            raise CompilationError("LLM compilation failed: no reachable key")

    monkeypatch.setattr(api_memories.repo, "list_uncompiled_episodes", fake_list)
    monkeypatch.setattr(api_memories.repo, "mark_episodes_compiled", fake_mark)
    monkeypatch.setattr(api_memories, "get_compiler", lambda: _RaisingCompiler())

    session = _FakeSession()
    with pytest.raises(CompilationError):
        await api_memories._compile_one_batch(session, "user-1", None, 10)

    assert marked == [], "failed batch must not mark episodes compiled"
    assert session.commits == 0, "failed batch must not commit"


@pytest.mark.asyncio
async def test_legitimate_empty_extraction_still_marks_compiled(monkeypatch):
    """A *successful* run that extracts nothing is the opposite case — the
    episodes ARE marked compiled so we don't reprocess them forever."""
    from server.api import memories as api_memories

    episodes = [_FakeEpisode()]
    marked: list = []

    async def fake_list(_session, _subject_id, *, tenant_id, limit):
        return episodes

    async def fake_mark(_session, ids):
        marked.append(ids)

    async def fake_count(_session, _subject_id, *, tenant_id):
        return 0

    async def fake_resolve(_session, _subject_id):
        return []

    async def fake_fire(*_a, **_k):
        return None

    class _EmptyCompiler:
        async def compile_async(self, _eps):
            return []  # legitimate "extracted nothing"

    monkeypatch.setattr(api_memories.repo, "list_uncompiled_episodes", fake_list)
    monkeypatch.setattr(api_memories.repo, "mark_episodes_compiled", fake_mark)
    monkeypatch.setattr(api_memories.repo, "count_uncompiled_episodes", fake_count)
    monkeypatch.setattr(api_memories, "get_compiler", lambda: _EmptyCompiler())
    monkeypatch.setattr(api_memories, "resolve_conflicts", fake_resolve)
    monkeypatch.setattr(api_memories, "schedule_embedding_backfill", lambda *_a, **_k: None)
    monkeypatch.setattr(api_memories.webhooks, "fire", fake_fire)

    session = _FakeSession()
    responses, created, remaining = await api_memories._compile_one_batch(
        session, "user-1", None, 10
    )

    assert created == 0
    assert responses == []
    assert marked == [[episodes[0].id]], "empty-but-successful run marks episodes compiled"
    assert session.commits == 1


# ---------------------------------------------------------------------------
# Route layer: sync 502, async job failed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_route_returns_502_on_compilation_error(monkeypatch):
    """Sync `/v1/memories/compile` surfaces a 502 with a clear error envelope
    rather than a misleading `memories_created: 0`."""
    import json

    from server.api import memories as api_memories
    from server.schemas.requests import CompileMemoriesRequest

    async def raising_batch(*_a, **_k):
        raise CompilationError("LLM compiler has no reachable key")

    monkeypatch.setattr(api_memories, "_compile_one_batch", raising_batch)

    body = CompileMemoriesRequest(subject_id="user-1")
    resp = await api_memories.compile_memories(body, session=_FakeSession(), tenant_id=None)

    assert resp.status_code == 502
    payload = json.loads(bytes(resp.body))
    assert payload["error"]["code"] == "compilation_failed"
    assert "no reachable key" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_async_job_marked_failed_on_compilation_error(monkeypatch):
    """Async drain marks the durable job failed (with the reason) when a
    batch raises `CompilationError`, instead of completing as zero-memory."""
    from server.api import memories as api_memories

    failed: list = []

    async def raising_batch(*_a, **_k):
        raise CompilationError("LLM compiler has no reachable key")

    async def fake_mark_running(_job_id):
        pass

    async def fake_mark_failed(job_id, error):
        failed.append((job_id, error))

    monkeypatch.setattr(api_memories, "_compile_one_batch", raising_batch)
    monkeypatch.setattr(api_memories.compile_jobs, "mark_running_durable", fake_mark_running)
    monkeypatch.setattr(api_memories.compile_jobs, "mark_failed_durable", fake_mark_failed)

    import server.db.engine as engine_module

    class _FakeFactory:
        def __call__(self_inner):
            class _Ctx:
                async def __aenter__(self_):
                    return _FakeSession()

                async def __aexit__(self_, *a):
                    return None

            return _Ctx()

    monkeypatch.setattr(engine_module, "get_session_factory", lambda: _FakeFactory())

    with pytest.raises(CompilationError):
        await api_memories._run_compile("user-1", job_id="job-1", tenant_id=None)

    assert len(failed) == 1
    assert failed[0][0] == "job-1"
    assert "no reachable key" in failed[0][1]
