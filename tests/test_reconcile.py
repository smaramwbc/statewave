"""Tests for the context-aware compile reconcile (Phase 4 + 5b).

Covers the decision-application logic (ADD / DUPLICATE / UPDATE / DELETE,
existing-target vs candidate-target) and the fail-open guarantees. The LLM
call and the existing-memory read are mocked, so these are pure unit tests of
the apply logic — no DB, no provider.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.db.tables import MemoryRow
from server.services import reconcile


def _mem(content: str, *, days_ago: int = 0, mid: uuid.UUID | None = None) -> MemoryRow:
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return MemoryRow(
        id=mid or uuid.uuid4(),
        subject_id="user-1",
        kind="profile_fact",
        content=content,
        summary=content[:200],
        confidence=0.8,
        valid_from=when,
        source_episode_ids=[uuid.uuid4()],
        metadata_={},
        status="active",
    )


def _patch(existing, decisions=None, llm_exc=None):
    """Patch the repo read + LLM call. Returns the context-manager stack via a
    helper that yields the patched acomplete_json mock."""
    list_mock = AsyncMock(return_value=existing)
    if llm_exc is not None:
        llm_mock = AsyncMock(side_effect=llm_exc)
    else:
        llm_mock = AsyncMock(return_value={"decisions": decisions or []})
    return (
        patch.object(reconcile.repo, "list_active_memories_by_subject", list_mock),
        patch.object(reconcile.llm_adapter, "acomplete_json", llm_mock),
        llm_mock,
    )


@pytest.mark.asyncio
async def test_all_add_keeps_everything():
    session = MagicMock()
    c0 = _mem("Alice likes coffee", days_ago=2)
    c1 = _mem("Alice hikes on weekends", days_ago=1)
    p_list, p_llm, _ = _patch([], decisions=[
        {"i": 0, "action": "ADD", "target": None},
        {"i": 1, "action": "ADD", "target": None},
    ])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c0, c1]
        )
    assert {m.id for m in kept} == {c0.id, c1.id}
    assert superseded == set()


@pytest.mark.asyncio
async def test_duplicate_of_existing_is_dropped():
    session = MagicMock()
    e0 = _mem("Alice lives in Munich", days_ago=30)
    c0 = _mem("Alice lives in Munich", days_ago=1)
    p_list, p_llm, _ = _patch([e0], decisions=[
        {"i": 0, "action": "DUPLICATE", "target": "E0"},
    ])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c0]
        )
    assert kept == []          # the duplicate is dropped
    assert superseded == set() # existing memory untouched


@pytest.mark.asyncio
async def test_update_of_existing_supersedes_old_keeps_new():
    session = MagicMock()
    e0 = _mem("Alice lives in Munich", days_ago=30)
    c0 = _mem("Alice moved to Berlin in 2024", days_ago=1)
    p_list, p_llm, _ = _patch([e0], decisions=[
        {"i": 0, "action": "UPDATE", "target": "E0"},
    ])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c0]
        )
    assert [m.id for m in kept] == [c0.id]   # new value kept
    assert superseded == {e0.id}             # stale value superseded


@pytest.mark.asyncio
async def test_delete_of_existing_supersedes_and_keeps_statement():
    session = MagicMock()
    e0 = _mem("Project uses Stripe", days_ago=20)
    c0 = _mem("Project stopped using Stripe", days_ago=1)
    p_list, p_llm, _ = _patch([e0], decisions=[
        {"i": 0, "action": "DELETE", "target": "E0"},
    ])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c0]
        )
    assert [m.id for m in kept] == [c0.id]
    assert superseded == {e0.id}


@pytest.mark.asyncio
async def test_intra_candidate_update_drops_older_candidate():
    session = MagicMock()
    # Two candidates in the SAME batch (no existing memory). The older one is
    # superseded by the newer — this is the LoCoMo single-batch case.
    c_old = _mem("API averages 250ms", days_ago=10)
    c_new = _mem("API now averages 90ms", days_ago=1)
    p_list, p_llm, _ = _patch([], decisions=[
        {"i": 0, "action": "ADD", "target": None},        # c_old (chrono idx 0)
        {"i": 1, "action": "UPDATE", "target": "N0"},     # c_new updates c_old
    ])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c_new, c_old]  # pass out of order; reconcile sorts
        )
    # c_old (N0) dropped, c_new (N1) kept; no EXISTING memory touched.
    assert [m.id for m in kept] == [c_new.id]
    assert superseded == set()


@pytest.mark.asyncio
async def test_failopen_on_llm_error_keeps_all():
    session = MagicMock()
    c0 = _mem("fact one", days_ago=2)
    c1 = _mem("fact two", days_ago=1)
    p_list, p_llm, _ = _patch([_mem("existing")], llm_exc=RuntimeError("boom"))
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c0, c1]
        )
    assert {m.id for m in kept} == {c0.id, c1.id}
    assert superseded == set()


@pytest.mark.asyncio
async def test_all_duplicate_of_existing_drops_all_legitimately():
    session = MagicMock()
    c0 = _mem("a", days_ago=2)
    c1 = _mem("b", days_ago=1)
    # Every candidate is already represented by an EXISTING memory → dropping
    # the whole batch is correct (the information is already stored). The
    # total-loss guard does NOT fire because existing memory is non-empty.
    p_list, p_llm, _ = _patch([_mem("x")], decisions=[
        {"i": 0, "action": "DUPLICATE", "target": "E0"},
        {"i": 1, "action": "DUPLICATE", "target": "E0"},
    ])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c0, c1]
        )
    assert kept == []
    assert superseded == set()


@pytest.mark.asyncio
async def test_large_batch_is_chunked_not_skipped():
    session = MagicMock()
    cands = [_mem(f"fact {i}", days_ago=i) for i in range(200)]
    # All-ADD decisions for every chunk (the mock returns the same shape each
    # call; only indices 0..39 are read per chunk, so a generous list works).
    p_list, p_llm, llm_mock = _patch([], decisions=[
        {"i": j, "action": "ADD", "target": None} for j in range(40)
    ])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", cands
        )
    assert len(kept) == 200                     # all kept
    assert superseded == set()
    assert llm_mock.await_count == 5            # 200 / chunk_size(40) = 5 calls


@pytest.mark.asyncio
async def test_single_candidate_no_existing_short_circuits():
    session = MagicMock()
    c0 = _mem("only fact")
    p_list, p_llm, llm_mock = _patch([], decisions=[])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c0]
        )
    assert [m.id for m in kept] == [c0.id]
    assert superseded == set()
    llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_target_degrades_to_add():
    session = MagicMock()
    e0 = _mem("existing fact", days_ago=5)
    c0 = _mem("new fact", days_ago=1)
    # UPDATE with an out-of-range / junk target → keep candidate, retire nothing.
    p_list, p_llm, _ = _patch([e0], decisions=[
        {"i": 0, "action": "UPDATE", "target": "E9"},
    ])
    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c0]
        )
    assert [m.id for m in kept] == [c0.id]
    assert superseded == set()
