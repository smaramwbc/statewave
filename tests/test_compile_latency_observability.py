"""Latency observability + timeout hygiene for the compile path (issue #380).

A provider latency regression turned a 6-minute pack compile into 25+
minutes and took a day to diagnose because nothing recorded where the
time went. These tests pin the three guards added in response:

* every provider call gets a PER-ATTEMPT litellm timeout so
  `num_retries` actually fits inside the outer `wait_for` window
  (before this, attempt 1 could consume the whole budget and retries
  were dead config);
* a call slower than `litellm_slow_call_seconds` logs `llm_slow_call` —
  the direct per-call evidence for a wall-time spike;
* reconcile chunks (strictly sequential, fail-open) use the tighter
  `reconcile_chunk_timeout_seconds` and count swallowed failures into
  `reconcile_done`;
* `_compile_one_batch` emits one `compile_phase_timings` line per batch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.core.config import settings

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeResp:
    def __init__(self, content="ok"):
        self.choices = [_FakeChoice(content)]


async def test_litellm_gets_a_per_attempt_timeout():
    """The kwargs handed to litellm must carry timeout = outer/(retries+1),
    so its internal retries fit inside the outer wait_for window."""
    from server.services import llm as llm_adapter

    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeResp()

    with patch.object(llm_adapter, "_ensure_litellm") as ensure:
        ensure.return_value = type("L", (), {"acompletion": staticmethod(fake_acompletion)})()
        await llm_adapter.acomplete([{"role": "user", "content": "hi"}])

    expected = settings.litellm_timeout_seconds / (settings.litellm_max_retries + 1)
    assert captured["timeout"] == pytest.approx(expected)
    assert captured["num_retries"] == settings.litellm_max_retries


async def test_slow_call_logs_warning(caplog):
    from server.services import llm as llm_adapter

    async def fake_acompletion(**kwargs):
        return _FakeResp()

    with patch.object(llm_adapter, "_ensure_litellm") as ensure, patch.object(
        settings, "litellm_slow_call_seconds", 0.0
    ):
        ensure.return_value = type("L", (), {"acompletion": staticmethod(fake_acompletion)})()
        with caplog.at_level("WARNING"):
            await llm_adapter.acomplete([{"role": "user", "content": "hi"}])

    assert any("llm_slow_call" in r.message for r in caplog.records)


async def test_fast_call_logs_no_slow_warning(caplog):
    from server.services import llm as llm_adapter

    async def fake_acompletion(**kwargs):
        return _FakeResp()

    with patch.object(llm_adapter, "_ensure_litellm") as ensure:
        ensure.return_value = type("L", (), {"acompletion": staticmethod(fake_acompletion)})()
        with caplog.at_level("WARNING"):
            await llm_adapter.acomplete([{"role": "user", "content": "hi"}])

    assert not any("llm_slow_call" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Reconcile: tighter chunk timeout + swallowed-failure counter
# ---------------------------------------------------------------------------


def _mem(content: str):
    from server.db.tables import MemoryRow

    return MemoryRow(
        id=uuid.uuid4(),
        subject_id="subj",
        kind="profile_fact",
        content=content,
        summary=content,
        source_episode_ids=[],
        created_at=_NOW,
        valid_from=_NOW,
        status="active",
        metadata_={},
    )


async def _run_reconcile(acomplete_json_mock, n_candidates=2):
    from server.services import reconcile as rec

    candidates = [_mem(f"fact {i}") for i in range(n_candidates)]
    session = AsyncMock()

    async def fake_existing(_session, _subject_id, *_a, **_kw):
        return [_mem("existing fact")]

    with patch.object(rec.repo, "list_active_memories_by_subject", new=fake_existing), patch.object(
        rec.llm_adapter, "acomplete_json", new=acomplete_json_mock
    ):
        return await rec.reconcile_compile_batch(session, "subj", candidates, tenant_id=None)


async def test_reconcile_passes_tight_chunk_timeout():
    captured = {}

    async def fake_json(messages, **kwargs):
        captured.update(kwargs)
        return {"decisions": []}

    await _run_reconcile(fake_json)
    assert captured["timeout"] == settings.reconcile_chunk_timeout_seconds
    assert settings.reconcile_chunk_timeout_seconds < settings.litellm_timeout_seconds


async def test_reconcile_counts_swallowed_failures(caplog):
    async def fake_json(messages, **kwargs):
        raise RuntimeError("provider fell over")

    with caplog.at_level("INFO"):
        kept, superseded = await _run_reconcile(fake_json, n_candidates=3)

    assert len(kept) == 3, "fail-open must keep the chunk wholesale"
    import json

    done = [r for r in caplog.records if "reconcile_done" in r.message]
    assert done
    payload = json.loads(done[0].message)
    assert payload["llm_failures"] == 1


# ---------------------------------------------------------------------------
# compile_phase_timings: one structured line per batch
# ---------------------------------------------------------------------------


async def test_compile_batch_emits_phase_timings(monkeypatch, caplog):
    from server.api import memories as api_memories
    from tests.test_compile_error_handling import _FakeEpisode, _FakeSession

    episodes = [_FakeEpisode()]

    async def fake_list(_session, _subject_id, *, tenant_id, limit):
        return episodes

    async def fake_mark(_session, ids):
        return None

    async def fake_count(_session, _subject_id, *, tenant_id):
        return 0

    async def fake_resolve(_session, _subject_id, *, tenant_id):
        return []

    async def fake_fire(*_a, **_kw):
        return None

    class _EmptyCompiler:
        async def compile_async(self, _eps, **_kw):
            return []

    monkeypatch.setattr(api_memories.repo, "list_uncompiled_episodes", fake_list)
    monkeypatch.setattr(api_memories.repo, "mark_episodes_compiled", fake_mark)
    monkeypatch.setattr(api_memories.repo, "count_uncompiled_episodes", fake_count)
    monkeypatch.setattr(api_memories, "resolve_conflicts", fake_resolve)
    monkeypatch.setattr(api_memories.webhooks, "fire", fake_fire)
    monkeypatch.setattr(api_memories, "get_compiler", lambda: _EmptyCompiler())

    with caplog.at_level("INFO"):
        await api_memories._compile_one_batch(_FakeSession(), "subj", None, 10)

    import json

    lines = [r for r in caplog.records if "compile_phase_timings" in r.message]
    assert len(lines) == 1
    payload = json.loads(lines[0].message)
    for key in (
        "extraction_seconds",
        "dedup_seconds",
        "reconcile_seconds",
        "conflicts_seconds",
        "commit_seconds",
        "entities_seconds",
        "total_seconds",
    ):
        assert key in payload, f"missing {key}"
    assert payload["episodes"] == 1
