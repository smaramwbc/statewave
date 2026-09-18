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


# --- regressions found in review ------------------------------------------


async def test_a_long_tenant_claim_key_does_not_abort_the_compile(session_factory):
    """Tenant-registered claim keys carry no length bound.

    Before the column was Text, a key longer than 256 characters turned the
    record INSERT into a truncation error. Because the record rides the
    compile batch's transaction, that rolled the whole batch back: no
    memories written, episodes never marked compiled, and the same failure
    on every retry. The key only ever lived in memory before this table
    existed, so nothing upstream bounds it.
    """
    from server.services.supersession import SupersessionDecision, record_supersessions

    subject_id = f"longkey-{uuid.uuid4().hex[:8]}"
    long_key = "custom." + ("k" * 400)

    async with session_factory() as session:
        record_supersessions(
            session,
            subject_id,
            [
                SupersessionDecision(
                    superseded_memory_id=uuid.uuid4(),
                    superseding_memory_id=uuid.uuid4(),
                    rule="claim_contradiction",
                    claim_key=long_key,
                )
            ],
        )
        await session.commit()

    async with session_factory() as session:
        stored = (
            await session.execute(
                select(SupersessionRecordRow).where(
                    SupersessionRecordRow.subject_id == subject_id
                )
            )
        ).scalar_one()
        assert stored.claim_key == long_key, "the key must survive unmodified"


async def test_real_conflict_resolution_writes_a_readable_row(client, session_factory):
    """Producer to real table, end to end.

    Every other test here hand-inserts the row or drives the producers
    against a mocked session, so no test ever put a producer-generated value
    into a real column. That gap is exactly why a column-width mismatch
    survived the whole suite. This closes the class: run the real conflict
    resolver against Postgres and read back what it wrote.
    """
    from server.services.conflicts import resolve_conflicts

    subject_id = f"real-{uuid.uuid4().hex[:8]}"
    async with session_factory() as session:
        older = _memory(subject_id, "the user lives in Berlin and works remotely", days_ago=5)
        newer = _memory(subject_id, "the user lives in Berlin and works remotely now", days_ago=1)
        session.add_all([older, newer])
        await session.commit()
        older_id, newer_id = older.id, newer.id

    async with session_factory() as session:
        superseded = await resolve_conflicts(session, subject_id, tenant_id=None)
        await session.commit()

    assert superseded, "the fixture is meant to produce a supersession"

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(SupersessionRecordRow).where(
                    SupersessionRecordRow.subject_id == subject_id
                )
            )
        ).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.superseded_memory_id == older_id
        assert row.superseding_memory_id == newer_id
        assert row.rule == "lexical"
        assert row.score is not None and row.threshold is not None
        assert row.score >= row.threshold
        # The owner's rule: ids and scores, never the text they came from.
        assert older.content not in (row.claim_key or "")
        assert row.details == {}


async def test_a_twice_updated_fact_reports_both_the_direct_and_the_live_memory(
    client, session_factory
):
    """A fact updated twice retires the intermediate too.

    `superseding_memory` is the decision that was recorded, so it stays the
    DIRECT successor even once that successor is itself retired: changing it
    would make the endpoint disagree with the audit record. `current_memory`
    follows the chain to the live row, so a panel can say "replaced by B,
    currently C" without every consumer reimplementing the walk.
    """
    from server.services.supersession import SupersessionDecision, record_supersessions

    subject_id = f"chain-{uuid.uuid4().hex[:8]}"
    async with session_factory() as session:
        a = _memory(subject_id, "the user is in CET", days_ago=9, status="superseded")
        b = _memory(subject_id, "the user is in GMT", days_ago=5, status="superseded")
        c = _memory(subject_id, "the user is in JST", days_ago=1)
        session.add_all([a, b, c])
        await session.commit()
        a_id, b_id, c_id = a.id, b.id, c.id

        record_supersessions(
            session,
            subject_id,
            [
                SupersessionDecision(
                    superseded_memory_id=a_id, superseding_memory_id=b_id, rule="lexical"
                ),
                SupersessionDecision(
                    superseded_memory_id=b_id, superseding_memory_id=c_id, rule="lexical"
                ),
            ],
        )
        await session.commit()

    resp = await client.get(f"/admin/subjects/{subject_id}/memories/{a_id}/related")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["relationship_source"] == "recorded"
    # The recorded decision, unchanged: B is what actually replaced A.
    assert body["superseding_memory"]["id"] == str(b_id)
    assert body["superseding_memory"]["status"] == "superseded"
    # And the answer an operator is looking for.
    assert body["current_memory"]["id"] == str(c_id)
    assert body["current_memory"]["status"] == "active"


async def test_no_current_memory_when_the_direct_successor_is_already_live(
    client, session_factory
):
    """Nothing to follow, so the field stays empty rather than echoing."""
    from server.services.supersession import SupersessionDecision, record_supersessions

    subject_id = f"chain1-{uuid.uuid4().hex[:8]}"
    async with session_factory() as session:
        old = _memory(subject_id, "the user is in CET", days_ago=5, status="superseded")
        new = _memory(subject_id, "the user is in JST", days_ago=1)
        session.add_all([old, new])
        await session.commit()
        old_id, new_id = old.id, new.id
        record_supersessions(
            session,
            subject_id,
            [
                SupersessionDecision(
                    superseded_memory_id=old_id, superseding_memory_id=new_id, rule="lexical"
                )
            ],
        )
        await session.commit()

    body = (
        await client.get(f"/admin/subjects/{subject_id}/memories/{old_id}/related")
    ).json()
    assert body["superseding_memory"]["id"] == str(new_id)
    assert body["current_memory"] is None
