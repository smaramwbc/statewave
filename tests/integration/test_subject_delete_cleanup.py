"""Regression: deleting a subject must also reap its resolutions and
health-cache row.

delete_subject only deleted episodes and memories; there is no FK cascade, and
the existing delete_resolutions_by_subject / delete_health_cache_by_subject
helpers were never wired in. The leftovers keep open/resolved-session logic
treating the deleted subject's sessions as live, and a stale subject_health_cache
row (PK = subject_id) can suppress or forge a health alert if the id is reused.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from server.db import repositories as repo
from server.db.tables import EpisodeRow, ResolutionRow, SubjectHealthCacheRow


@pytest.mark.anyio
async def test_delete_subject_reaps_resolutions_and_health_cache(
    client: AsyncClient, subject_id: str, session_factory
):
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
            ResolutionRow(
                id=uuid.uuid4(),
                subject_id=subject_id,
                session_id="s1",
                status="resolved",
                resolution_summary="done",
            )
        )
        session.add(
            SubjectHealthCacheRow(subject_id=subject_id, last_state="healthy", last_score=100)
        )
        await session.commit()

    r = await client.delete(f"/v1/subjects/{subject_id}")
    assert r.status_code == 200

    async with session_factory() as session:
        res_count = await session.scalar(
            select(func.count())
            .select_from(ResolutionRow)
            .where(ResolutionRow.subject_id == subject_id)
        )
        hc_count = await session.scalar(
            select(func.count())
            .select_from(SubjectHealthCacheRow)
            .where(SubjectHealthCacheRow.subject_id == subject_id)
        )
    assert res_count == 0, "resolutions must be deleted with the subject"
    assert hc_count == 0, "health-cache row must be deleted with the subject"


@pytest.mark.anyio
async def test_delete_subject_health_cache_reap_is_tenant_scoped(
    client: AsyncClient, session_factory
):
    """Reaping the health-cache on delete must stay within the caller's tenant.

    delete_subject wired in delete_health_cache_by_subject WITHOUT threading
    tenant_id. Post-migration-0024 that helper treats tenant_id=None as "match
    every tenant", so deleting a subject under tenant B also wiped tenant A's
    cached health for the same subject_id — re-opening the exact cross-tenant
    leak migration 0024 just closed. delete_resolutions_by_subject is the same
    shape and is already scoped correctly here.
    """
    subject = f"del-iso-{uuid.uuid4().hex[:12]}"
    async with session_factory() as session:
        await repo.upsert_health_cache(session, subject, "at_risk", 30, tenant_id="tenant-a")
        await repo.upsert_health_cache(session, subject, "healthy", 95, tenant_id="tenant-b")
        await session.commit()

    r = await client.delete(f"/v1/subjects/{subject}", headers={"X-Tenant-ID": "tenant-b"})
    assert r.status_code == 200

    async with session_factory() as session:
        gone = await repo.get_health_cache(session, subject, tenant_id="tenant-b")
        survived = await repo.get_health_cache(session, subject, tenant_id="tenant-a")

    assert gone is None, "tenant B's own health-cache row should be reaped"
    assert survived is not None and survived.last_state == "at_risk", (
        "tenant A's health-cache row must survive a tenant-B subject delete"
    )
