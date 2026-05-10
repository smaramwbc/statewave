"""Integration tests for memory TTL — DB-backed cleanup + retrieval-filter behaviour.

Unit-level coverage (config parsing, compute_valid_to, compiler stamping) lives
in `tests/test_memory_ttl.py`.

The load-bearing test in this file is `test_retrieval_excludes_expired_pre_cleanup`:
it asserts the invariant the v0.7 TTL design is built around — an expired memory
must not reach `/v1/context` even between the hourly cleanup runs. That's the
analog of the negative test issue #49 (state-assembly receipts) calls out:
"Stale fact selected for a current query (`valid_to` passed)".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient

from server.db.tables import MemoryRow
from server.services.memory_ttl import cleanup_expired_memories


# ---------------------------------------------------------------------------
# Helpers — insert a memory directly via the session factory so the test
# controls valid_to precisely (compiling through the API would stamp valid_to
# from settings.kind_ttl_days which we'd then have to monkey-patch on the
# live FastAPI process — clunkier than the direct insert).
# ---------------------------------------------------------------------------


async def _insert_memory(
    session_factory,
    *,
    subject_id: str,
    valid_to: datetime | None,
    status: str = "active",
    kind: str = "episode_summary",
    content: str = "TTL integration probe",
) -> uuid.UUID:
    memory_id = uuid.uuid4()
    async with session_factory() as session:
        session.add(
            MemoryRow(
                id=memory_id,
                subject_id=subject_id,
                kind=kind,
                content=content,
                summary=content[:200],
                confidence=0.9,
                valid_from=datetime.now(timezone.utc) - timedelta(days=1),
                valid_to=valid_to,
                source_episode_ids=[],
                metadata_={},
                status=status,
            )
        )
        await session.commit()
    return memory_id


# ---------------------------------------------------------------------------
# cleanup_expired_memories — backstop tombstoning behaviour
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cleanup_tombstones_expired_active_memory(session_factory, subject_id: str):
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    memory_id = await _insert_memory(session_factory, subject_id=subject_id, valid_to=past)

    async with session_factory() as session:
        count = await cleanup_expired_memories(session)
        await session.commit()
    assert count >= 1

    async with session_factory() as session:
        row = await session.get(MemoryRow, memory_id)
        assert row is not None
        assert row.status == "tombstoned"


@pytest.mark.anyio
async def test_cleanup_leaves_unexpired_memory_alone(session_factory, subject_id: str):
    future = datetime.now(timezone.utc) + timedelta(days=7)
    memory_id = await _insert_memory(session_factory, subject_id=subject_id, valid_to=future)

    async with session_factory() as session:
        await cleanup_expired_memories(session)
        await session.commit()

    async with session_factory() as session:
        row = await session.get(MemoryRow, memory_id)
        assert row is not None
        assert row.status == "active"


@pytest.mark.anyio
async def test_cleanup_leaves_null_valid_to_alone(session_factory, subject_id: str):
    memory_id = await _insert_memory(session_factory, subject_id=subject_id, valid_to=None)

    async with session_factory() as session:
        await cleanup_expired_memories(session)
        await session.commit()

    async with session_factory() as session:
        row = await session.get(MemoryRow, memory_id)
        assert row is not None
        assert row.status == "active"


@pytest.mark.anyio
async def test_cleanup_leaves_already_tombstoned_alone(session_factory, subject_id: str):
    # Idempotency: re-running cleanup should not re-touch already-tombstoned rows.
    past = datetime.now(timezone.utc) - timedelta(days=1)
    memory_id = await _insert_memory(
        session_factory, subject_id=subject_id, valid_to=past, status="tombstoned"
    )

    async with session_factory() as session:
        count = await cleanup_expired_memories(session)
        await session.commit()
    assert count == 0  # nothing to do — already tombstoned

    async with session_factory() as session:
        row = await session.get(MemoryRow, memory_id)
        assert row.status == "tombstoned"


@pytest.mark.anyio
async def test_cleanup_leaves_superseded_alone(session_factory, subject_id: str):
    # Superseded memories carry valid_to from the conflict-resolution path;
    # they must not be re-stamped as tombstoned (semantics differ — superseded
    # is "replaced by a newer fact", tombstoned is "expired by TTL").
    past = datetime.now(timezone.utc) - timedelta(days=1)
    memory_id = await _insert_memory(
        session_factory, subject_id=subject_id, valid_to=past, status="superseded"
    )

    async with session_factory() as session:
        await cleanup_expired_memories(session)
        await session.commit()

    async with session_factory() as session:
        row = await session.get(MemoryRow, memory_id)
        assert row.status == "superseded"


# ---------------------------------------------------------------------------
# Retrieval — the load-bearing #49 negative-test analog
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_retrieval_excludes_expired_pre_cleanup(
    client: AsyncClient, session_factory, subject_id: str
):
    """`/v1/context` must NOT surface a memory whose valid_to has passed,
    even when the cleanup loop has not yet tombstoned it (status is still
    'active'). This is the design-load-bearing fence — without it,
    expired memories would pollute retrieval for up to one hour every
    cycle, and #49 receipts would record stale facts as "selected"."""

    # Active memory, but expired one hour ago. Cleanup hasn't run yet.
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    expired_id = await _insert_memory(
        session_factory,
        subject_id=subject_id,
        valid_to=past,
        status="active",
        kind="episode_summary",
        content="EXPIRED-MEMORY-MARKER-DO-NOT-SURFACE",
    )

    # Fresh memory, also active, no TTL.
    fresh_id = await _insert_memory(
        session_factory,
        subject_id=subject_id,
        valid_to=None,
        status="active",
        kind="episode_summary",
        content="FRESH-MEMORY-MARKER-SURFACE-ME",
    )

    resp = await client.post(
        "/v1/context",
        json={"subject_id": subject_id, "task": "any task"},
    )
    assert resp.status_code == 200
    body = resp.json()

    surfaced = " ".join(
        m.get("content", "") for m in body.get("episodes", []) + body.get("facts", [])
    ) + body.get("assembled_context", "")

    assert "EXPIRED-MEMORY-MARKER-DO-NOT-SURFACE" not in surfaced, (
        "expired memory leaked into /v1/context — TTL retrieval filter is broken"
    )
    assert "FRESH-MEMORY-MARKER-SURFACE-ME" in surfaced, (
        "fresh memory was incorrectly excluded — TTL retrieval filter is over-aggressive"
    )

    # Expired row is still present in the DB (soft-tombstone semantics —
    # rows persist for issue #49 receipts to reference).
    async with session_factory() as session:
        expired_row = await session.get(MemoryRow, expired_id)
        fresh_row = await session.get(MemoryRow, fresh_id)
        assert expired_row is not None
        assert fresh_row is not None
