"""Integration: subject_health_cache is isolated per (tenant_id, subject_id).

Proves tenant A cannot read, overwrite, or delete tenant B's cached health for
the same subject_id — the cross-tenant leak fixed by migration 0024 (the cache
used to be keyed by subject_id alone).
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from server.db import repositories as repo
from server.db.tables import SubjectHealthCacheRow


def _subject() -> str:
    return f"health-iso-{uuid.uuid4().hex[:12]}"


async def test_tenants_keep_separate_rows_for_same_subject(session_factory):
    subject = _subject()
    async with session_factory() as s:
        await repo.upsert_health_cache(s, subject, "at_risk", 30, tenant_id="tenant-a")
        await repo.upsert_health_cache(s, subject, "healthy", 95, tenant_id="tenant-b")
        await s.commit()

    async with session_factory() as s:
        a = await repo.get_health_cache(s, subject, tenant_id="tenant-a")
        b = await repo.get_health_cache(s, subject, tenant_id="tenant-b")

    assert a is not None and a.last_state == "at_risk" and a.last_score == 30
    assert b is not None and b.last_state == "healthy" and b.last_score == 95
    assert a.id != b.id  # two distinct rows for the same subject_id


async def test_upsert_does_not_overwrite_other_tenant(session_factory):
    subject = _subject()
    async with session_factory() as s:
        await repo.upsert_health_cache(s, subject, "at_risk", 30, tenant_id="tenant-a")
        await s.commit()
    # Tenant B writes the same subject_id — must NOT touch tenant A's row.
    async with session_factory() as s:
        await repo.upsert_health_cache(s, subject, "healthy", 99, tenant_id="tenant-b")
        await s.commit()

    async with session_factory() as s:
        a = await repo.get_health_cache(s, subject, tenant_id="tenant-a")

    assert a is not None and a.last_state == "at_risk" and a.last_score == 30


async def test_delete_is_tenant_scoped(session_factory):
    subject = _subject()
    async with session_factory() as s:
        await repo.upsert_health_cache(s, subject, "at_risk", 30, tenant_id="tenant-a")
        await repo.upsert_health_cache(s, subject, "healthy", 95, tenant_id="tenant-b")
        await s.commit()

    async with session_factory() as s:
        await repo.delete_health_cache_by_subject(s, subject, tenant_id="tenant-a")
        await s.commit()

    async with session_factory() as s:
        gone = await repo.get_health_cache(s, subject, tenant_id="tenant-a")
        survived = await repo.get_health_cache(s, subject, tenant_id="tenant-b")

    assert gone is None
    assert survived is not None and survived.last_state == "healthy"


async def test_repeated_upsert_same_tenant_updates_in_place(session_factory):
    subject = _subject()
    async with session_factory() as s:
        await repo.upsert_health_cache(s, subject, "watch", 60, tenant_id="tenant-a")
        await s.commit()
    async with session_factory() as s:
        await repo.upsert_health_cache(s, subject, "at_risk", 20, tenant_id="tenant-a")
        await s.commit()

    async with session_factory() as s:
        rows = (
            await s.execute(
                select(SubjectHealthCacheRow).where(
                    SubjectHealthCacheRow.subject_id == subject
                )
            )
        ).scalars().all()

    assert len(rows) == 1  # update in place, not a second row
    assert rows[0].last_state == "at_risk" and rows[0].last_score == 20


async def test_single_tenant_mode_still_works(session_factory):
    subject = _subject()
    async with session_factory() as s:
        await repo.upsert_health_cache(s, subject, "healthy", 100)  # tenant_id=None
        await s.commit()

    async with session_factory() as s:
        row = await repo.get_health_cache(s, subject)  # tenant_id=None

    assert row is not None and row.last_state == "healthy" and row.last_score == 100
