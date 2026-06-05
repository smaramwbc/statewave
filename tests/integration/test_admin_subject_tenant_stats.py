from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from server.db.tables import EpisodeRow, MemoryRow, ResolutionRow, SubjectHealthCacheRow

pytestmark = pytest.mark.anyio


async def test_admin_subject_stats_are_scoped_by_tenant(client: AsyncClient, session_factory):
    subject_id = f"shared-admin-{uuid.uuid4().hex[:8]}"

    async with session_factory() as session:
        session.add_all(
            [
                EpisodeRow(
                    subject_id=subject_id,
                    tenant_id="tenant-a",
                    session_id="a-session",
                    source="test",
                    type="message",
                    payload={"text": "tenant a"},
                    metadata_={},
                    provenance={},
                ),
                EpisodeRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    session_id="b-session-1",
                    source="test",
                    type="message",
                    payload={"text": "tenant b first"},
                    metadata_={},
                    provenance={},
                ),
                EpisodeRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    session_id="b-session-2",
                    source="test",
                    type="message",
                    payload={"text": "tenant b second"},
                    metadata_={},
                    provenance={},
                ),
                MemoryRow(
                    subject_id=subject_id,
                    tenant_id="tenant-a",
                    kind="fact",
                    content="tenant a memory",
                    summary="tenant a memory",
                    confidence=1.0,
                    source_episode_ids=[],
                    metadata_={},
                    status="active",
                    sensitivity_labels=[],
                    suggested_labels=[],
                ),
                MemoryRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    kind="fact",
                    content="tenant b memory one",
                    summary="tenant b memory one",
                    confidence=1.0,
                    source_episode_ids=[],
                    metadata_={},
                    status="active",
                    sensitivity_labels=[],
                    suggested_labels=[],
                ),
                MemoryRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    kind="fact",
                    content="tenant b memory two",
                    summary="tenant b memory two",
                    confidence=1.0,
                    source_episode_ids=[],
                    metadata_={},
                    status="active",
                    sensitivity_labels=[],
                    suggested_labels=[],
                ),
                ResolutionRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    session_id="b-session-1",
                    status="open",
                    metadata_={},
                ),
                SubjectHealthCacheRow(
                    subject_id=subject_id,
                    tenant_id="tenant-a",
                    last_state="healthy",
                    last_score=92,
                ),
                SubjectHealthCacheRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    last_state="critical",
                    last_score=12,
                ),
            ]
        )
        await session.commit()

    tenant_a = await client.get(
        "/admin/subjects", params={"tenant_id": "tenant-a", "search": subject_id}
    )
    assert tenant_a.status_code == 200
    tenant_a_body = tenant_a.json()
    assert tenant_a_body["total"] == 1
    tenant_a_subject = tenant_a_body["subjects"][0]
    assert tenant_a_subject["episode_count"] == 1
    assert tenant_a_subject["memory_count"] == 1
    assert tenant_a_subject["session_count"] == 1
    assert tenant_a_subject["open_sessions"] == 0
    assert tenant_a_subject["health_state"] == "healthy"
    assert tenant_a_subject["health_score"] == 92

    tenant_b = await client.get(
        "/admin/subjects", params={"tenant_id": "tenant-b", "search": subject_id}
    )
    assert tenant_b.status_code == 200
    tenant_b_body = tenant_b.json()
    assert tenant_b_body["total"] == 1
    tenant_b_subject = tenant_b_body["subjects"][0]
    assert tenant_b_subject["episode_count"] == 2
    assert tenant_b_subject["memory_count"] == 2
    assert tenant_b_subject["session_count"] == 2
    assert tenant_b_subject["open_sessions"] == 1
    assert tenant_b_subject["health_state"] == "critical"
    assert tenant_b_subject["health_score"] == 12
