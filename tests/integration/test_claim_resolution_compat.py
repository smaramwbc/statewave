"""Real-Postgres compatibility matrix for hybrid claim-keyed resolution.

Exercises the resolver against a live DB (mark_memories_superseded actually
updates rows), plus the externally-visible surfaces that must keep working:
metadata round-trip through the API, export/import of claim metadata + status,
and the invariant that reads never mutate supersession state.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

import tests.integration.conftest as _conftest
from server.db import repositories as repo
from server.db.tables import EpisodeRow, MemoryRow
from server.services.backup import export_subject, import_subject
from server.services.conflicts import resolve_conflicts
from server.services.context import assemble_context
from server.db.engine import set_engine_for_testing

_NOW = dt.datetime.now(dt.timezone.utc)


def _claim(key, value, *, valid_from=None, valid_to=None):
    env = {"schema_version": 1, "key": key, "value": value}
    if valid_from is not None:
        env["valid_from"] = valid_from.isoformat()
    if valid_to is not None:
        env["valid_to"] = valid_to.isoformat()
    return {"claim": env}


def _mem(subject_id, content, *, metadata=None, age_days=0, status="active"):
    return MemoryRow(
        id=uuid.uuid4(),
        subject_id=subject_id,
        kind="profile_fact",
        content=content,
        summary=content[:200],
        confidence=0.9,
        valid_from=_NOW - dt.timedelta(days=age_days),
        created_at=_NOW - dt.timedelta(days=age_days),
        source_episode_ids=[uuid.uuid4()],
        metadata_=metadata or {},
        status=status,
    )


async def _statuses(session_factory, subject_id):
    async with session_factory() as s:
        rows = await repo.list_memories_by_subject(s, subject_id, limit=500)
    return {r.id: r.status for r in rows}


# --- resolver on a live DB ----------------------------------------------------


@pytest.mark.anyio
async def test_employer_contradiction_supersedes_on_real_db(session_factory, subject_id):
    older = _mem(subject_id, "spent years at Acme", metadata=_claim("employer", "Acme"), age_days=10)
    newer = _mem(subject_id, "new gig at Globex", metadata=_claim("employer", "Globex"), age_days=0)
    async with session_factory() as s:
        s.add_all([older, newer])
        await s.commit()

    async with session_factory() as s:
        result = await resolve_conflicts(s, subject_id)
        await s.commit()

    assert older.id in result
    statuses = await _statuses(session_factory, subject_id)
    assert statuses[older.id] == "superseded"
    assert statuses[newer.id] == "active"


@pytest.mark.anyio
async def test_legacy_only_db_unchanged(session_factory, subject_id):
    older = _mem(subject_id, "my name is Alice", age_days=10)
    newer = _mem(subject_id, "my name is Alice Chen", age_days=0)
    distinct = _mem(subject_id, "I work at Globex Corporation", age_days=5)
    async with session_factory() as s:
        s.add_all([older, newer, distinct])
        await s.commit()

    async with session_factory() as s:
        result = await resolve_conflicts(s, subject_id)
        await s.commit()

    statuses = await _statuses(session_factory, subject_id)
    assert statuses[older.id] == "superseded"  # legacy Jaccard
    assert statuses[newer.id] == "active"
    assert statuses[distinct.id] == "active"  # distinct fact coexists


@pytest.mark.anyio
async def test_historical_and_current_coexist_on_real_db(session_factory, subject_id):
    hist = _mem(subject_id, "worked at Acme",
                metadata=_claim("employer", "Acme", valid_from=dt.datetime(2015, 1, 1, tzinfo=dt.timezone.utc),
                                valid_to=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)), age_days=10)
    curr = _mem(subject_id, "now at Globex",
                metadata=_claim("employer", "Globex", valid_from=dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)), age_days=0)
    async with session_factory() as s:
        s.add_all([hist, curr])
        await s.commit()

    async with session_factory() as s:
        result = await resolve_conflicts(s, subject_id)
        await s.commit()

    assert result == []
    statuses = await _statuses(session_factory, subject_id)
    assert statuses[hist.id] == "active" and statuses[curr.id] == "active"


@pytest.mark.anyio
async def test_mixed_keyed_unkeyed_and_malformed_remain_stable(session_factory, subject_id):
    keyed = _mem(subject_id, "at Globex", metadata=_claim("employer", "Globex"), age_days=0)
    malformed = _mem(subject_id, "random note", metadata={"claim": {"oops": True}}, age_days=1)
    unknown = _mem(subject_id, "shoe note", metadata=_claim("shoe_size", "10"), age_days=2)
    unkeyed = _mem(subject_id, "totally distinct fact about cats", age_days=3)
    async with session_factory() as s:
        s.add_all([keyed, malformed, unknown, unkeyed])
        await s.commit()

    async with session_factory() as s:
        result = await resolve_conflicts(s, subject_id)
        await s.commit()

    # Nothing conflicts: distinct content, single keyed value, malformed/unknown
    # fall back to legacy and don't match anything.
    assert result == []
    statuses = await _statuses(session_factory, subject_id)
    assert all(v == "active" for v in statuses.values())


# --- reads never mutate -------------------------------------------------------


@pytest.mark.anyio
async def test_reads_do_not_mutate_supersession_state(session_factory, subject_id):
    active = _mem(subject_id, "Stripe pricing is 2.9% plus 30 cents", age_days=0)
    superseded = _mem(subject_id, "Stripe pricing is 3.5% plus 35 cents", age_days=5, status="superseded")
    async with session_factory() as s:
        s.add_all([active, superseded])
        await s.commit()

    async with session_factory() as s:
        await assemble_context(s, subject_id, task="Stripe pricing", max_tokens=2000)

    statuses = await _statuses(session_factory, subject_id)
    assert statuses[active.id] == "active"
    assert statuses[superseded.id] == "superseded"  # unchanged by a read


# --- export/import round-trip -------------------------------------------------


@pytest.mark.anyio
async def test_export_import_preserves_claim_metadata_and_status(session_factory, subject_id):
    claimed = _mem(subject_id, "at Globex", metadata=_claim("employer", "Globex"), age_days=0)
    gone = _mem(subject_id, "at Acme", metadata=_claim("employer", "Acme"), age_days=5, status="superseded")
    async with session_factory() as s:
        s.add_all([claimed, gone])
        await s.commit()

    prev = set_engine_for_testing(_conftest._engine, _conftest._session_factory)
    try:
        doc = await export_subject(subject_id)
        target = f"copy-{uuid.uuid4().hex[:8]}"
        await import_subject(doc, target_subject_id=target, preserve_ids=False)
    finally:
        set_engine_for_testing(*prev)

    imported = await _statuses(session_factory, target)
    # Both statuses survived the round-trip (superseded not dropped, not reset).
    assert sorted(imported.values()) == ["active", "superseded"]
    async with session_factory() as s:
        rows = await repo.list_memories_by_subject(s, target, limit=500)
    claims = [r.metadata_.get("claim") for r in rows if r.metadata_.get("claim")]
    assert len(claims) == 2
    assert {c["value"] for c in claims} == {"Globex", "Acme"}
    # Stored keys round-trip VERBATIM — the resolver canonicalizes the alias at
    # resolve time, it never rewrites persisted metadata.
    assert all(c["key"] == "employer" for c in claims)


# --- API opaque pass-through --------------------------------------------------


@pytest.mark.anyio
async def test_claim_metadata_roundtrips_through_api(client, subject_id):
    md = _claim("employer", "Globex")
    md["unrelated"] = {"nested": [1, 2, 3]}
    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "chat",
            "type": "message",
            "payload": {"text": "I just joined Globex"},
            "metadata": md,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["metadata"] == md  # echoed verbatim, claim sub-key intact

    tl = await client.get("/v1/timeline", params={"subject_id": subject_id})
    assert tl.status_code == 200
    assert tl.json()["episodes"][0]["metadata"] == md
