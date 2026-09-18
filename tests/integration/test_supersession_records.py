"""The admin relationship view answers from the recorded decision (#419).

Before this, `GET /admin/subjects/{s}/memories/{id}/related` derived
supersedes/superseded_by at read time from "same subject, same kind, nearest
created_at on the other side of the status line". With more than one candidate
that derivation names the wrong memory, confidently. These tests stage exactly
that shape — two active memories newer than a superseded one — so a derived
answer and a recorded one differ, and assert the endpoint serves the recorded
one and says which it served.

Backfill is impossible (the decisions were never stored), so a memory with no
record must still get the old derivation, labelled "inferred".
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from server.db.tables import EpisodeRow, MemoryRow, SupersessionRecordRow

_NOW = datetime.now(timezone.utc)


def _memory(subject_id: str, content: str, *, days_ago: int, status: str = "active") -> MemoryRow:
    when = _NOW - timedelta(days=days_ago)
    return MemoryRow(
        id=uuid.uuid4(),
        subject_id=subject_id,
        kind="profile_fact",
        content=content,
        summary=content[:200],
        confidence=0.9,
        created_at=when,
        valid_from=when,
        source_episode_ids=[uuid.uuid4()],
        metadata_={},
        status=status,
    )


async def _seed_chain(session_factory, subject_id: str, *, with_record: bool):
    """One superseded memory and TWO active memories newer than it.

    The derivation picks the OLDEST active one after the target (`middle`);
    the recorded decision names the newest (`winner`). Any answer therefore
    identifies which source produced it.
    """
    retired = _memory(subject_id, "Alice lives in Munich", days_ago=30, status="superseded")
    middle = _memory(subject_id, "Alice enjoys hiking", days_ago=20)
    winner = _memory(subject_id, "Alice moved to Berlin", days_ago=10)
    async with session_factory() as session:
        session.add_all([retired, middle, winner])
        if with_record:
            session.add(
                SupersessionRecordRow(
                    subject_id=subject_id,
                    superseded_memory_id=retired.id,
                    superseding_memory_id=winner.id,
                    rule="lexical",
                    score=0.75,
                    threshold=0.6,
                    compile_job_id="job-1",
                )
            )
        await session.commit()
    return retired, middle, winner


@pytest.mark.anyio
async def test_related_serves_the_recorded_successor_not_the_derived_one(
    client: AsyncClient, subject_id: str, session_factory
):
    retired, middle, winner = await _seed_chain(session_factory, subject_id, with_record=True)

    r = await client.get(f"/admin/subjects/{subject_id}/memories/{retired.id}/related")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["relationship_source"] == "recorded"
    assert body["superseding_memory"]["id"] == str(winner.id)
    assert body["superseding_memory"]["id"] != str(middle.id), (
        "the read-time derivation's answer — the endpoint must not fall back to it "
        "when the decision itself is on record"
    )
    # The decision's own terms travel with it.
    assert body["superseding_memory"]["rule"] == "lexical"
    assert body["superseding_memory"]["score"] == pytest.approx(0.75)
    assert body["superseding_memory"]["threshold"] == pytest.approx(0.6)


@pytest.mark.anyio
async def test_related_falls_back_to_the_derivation_and_says_so(
    client: AsyncClient, subject_id: str, session_factory
):
    """Memories superseded before #419 have no record and cannot get one, so
    the panel keeps working — labelled as the guess it is."""
    retired, middle, _winner = await _seed_chain(session_factory, subject_id, with_record=False)

    r = await client.get(f"/admin/subjects/{subject_id}/memories/{retired.id}/related")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["relationship_source"] == "inferred"
    assert body["superseding_memory"]["id"] == str(middle.id)
    assert body["superseding_memory"]["rule"] is None


@pytest.mark.anyio
async def test_active_memory_lists_the_memories_it_actually_retired(
    client: AsyncClient, subject_id: str, session_factory
):
    """The other side of the panel. The derivation would hand `winner` every
    superseded memory of its kind that predates it; the records name the one
    it retired."""
    retired, middle, winner = await _seed_chain(session_factory, subject_id, with_record=True)
    bystander = _memory(subject_id, "Alice used to drink tea", days_ago=25, status="superseded")
    async with session_factory() as session:
        session.add(bystander)
        await session.commit()

    r = await client.get(f"/admin/subjects/{subject_id}/memories/{winner.id}/related")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["relationship_source"] == "recorded"
    assert [m["id"] for m in body["superseded_memories"]] == [str(retired.id)]
    assert body["superseded_memories"][0]["rule"] == "lexical"
    # Same kind, superseded, older than `winner` — the derivation would have
    # included it.
    assert str(bystander.id) not in [m["id"] for m in body["superseded_memories"]]


@pytest.mark.anyio
async def test_a_record_whose_successor_was_never_persisted_stays_recorded(
    client: AsyncClient, subject_id: str, session_factory
):
    """Reconcile can retire a memory with a candidate it later drops, so the
    successor may not exist. That is a known answer, not a missing one: the
    endpoint must not fall back to guessing a different memory."""
    retired = _memory(subject_id, "Alice lives in Munich", days_ago=30, status="superseded")
    middle = _memory(subject_id, "Alice enjoys hiking", days_ago=20)
    async with session_factory() as session:
        session.add_all([retired, middle])
        session.add(
            SupersessionRecordRow(
                subject_id=subject_id,
                superseded_memory_id=retired.id,
                superseding_memory_id=None,
                rule="reconcile_update",
            )
        )
        await session.commit()

    r = await client.get(f"/admin/subjects/{subject_id}/memories/{retired.id}/related")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["relationship_source"] == "recorded"
    assert body["superseding_memory"] is None


@pytest.mark.anyio
async def test_delete_subject_reaps_supersession_records(
    client: AsyncClient, subject_id: str, session_factory
):
    """"Delete all subject data" has no FK cascade to ride on. Records hold the
    subject id and its compile-time decisions, so they go with it."""
    async with session_factory() as session:
        session.add(
            EpisodeRow(
                id=uuid.uuid4(),
                subject_id=subject_id,
                source="test",
                type="conversation",
                payload={"text": "hi"},
            )
        )
        session.add(
            SupersessionRecordRow(
                subject_id=subject_id,
                superseded_memory_id=uuid.uuid4(),
                superseding_memory_id=uuid.uuid4(),
                rule="lexical",
                score=0.9,
                threshold=0.6,
            )
        )
        await session.commit()

    r = await client.delete(f"/v1/subjects/{subject_id}")
    assert r.status_code == 200, r.text

    async with session_factory() as session:
        remaining = await session.scalar(
            select(func.count())
            .select_from(SupersessionRecordRow)
            .where(SupersessionRecordRow.subject_id == subject_id)
        )
    assert remaining == 0, "supersession records must be deleted with the subject"


@pytest.mark.anyio
async def test_delete_subject_reap_is_tenant_scoped(
    client: AsyncClient, session_factory
):
    """The reap is a multi-row DELETE on a subject id shared across tenants —
    unscoped, it would take another tenant's audit rows with it."""
    subject = f"sr-iso-{uuid.uuid4().hex[:12]}"
    async with session_factory() as session:
        for tenant in ("tenant-a", "tenant-b"):
            session.add(
                SupersessionRecordRow(
                    subject_id=subject,
                    tenant_id=tenant,
                    superseded_memory_id=uuid.uuid4(),
                    superseding_memory_id=uuid.uuid4(),
                    rule="lexical",
                )
            )
        await session.commit()

    r = await client.delete(f"/v1/subjects/{subject}", headers={"X-Tenant-ID": "tenant-b"})
    assert r.status_code == 200, r.text

    async with session_factory() as session:
        survivors = (
            await session.execute(
                select(SupersessionRecordRow.tenant_id).where(
                    SupersessionRecordRow.subject_id == subject
                )
            )
        ).scalars().all()
    assert survivors == ["tenant-a"]
