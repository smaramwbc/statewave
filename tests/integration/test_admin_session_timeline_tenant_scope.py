from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from server.db.tables import EpisodeRow, MemoryRow

pytestmark = pytest.mark.anyio


async def test_admin_session_timeline_metrics_are_scoped_by_tenant(
    client: AsyncClient, session_factory
):
    subject_id = f"shared-timeline-{uuid.uuid4().hex[:8]}"
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    async with session_factory() as session:
        tenant_a_episode = EpisodeRow(
            subject_id=subject_id,
            tenant_id="tenant-a",
            session_id=session_id,
            source="test",
            type="message",
            payload={"text": "tenant a"},
            metadata_={},
            provenance={},
        )
        session.add(tenant_a_episode)
        session.add_all(
            [
                EpisodeRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    session_id=session_id,
                    source="test",
                    type="message",
                    payload={"text": "tenant b first"},
                    metadata_={},
                    provenance={},
                ),
                EpisodeRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    session_id=session_id,
                    source="test",
                    type="message",
                    payload={"text": "tenant b second"},
                    metadata_={},
                    provenance={},
                ),
            ]
        )
        await session.flush()

        session.add_all(
            [
                MemoryRow(
                    subject_id=subject_id,
                    tenant_id="tenant-a",
                    kind="fact",
                    content="tenant a memory",
                    summary="tenant a memory",
                    confidence=1.0,
                    source_episode_ids=[tenant_a_episode.id],
                    metadata_={},
                    status="active",
                    sensitivity_labels=[],
                    suggested_labels=[],
                ),
                MemoryRow(
                    subject_id=subject_id,
                    tenant_id="tenant-b",
                    kind="fact",
                    content="tenant b imported memory",
                    summary="tenant b imported memory",
                    confidence=1.0,
                    source_episode_ids=[tenant_a_episode.id],
                    metadata_={},
                    status="active",
                    sensitivity_labels=[],
                    suggested_labels=[],
                ),
            ]
        )
        await session.commit()

    response = await client.get(
        f"/admin/subjects/{subject_id}/sessions/{session_id}/timeline",
        params={"tenant_id": "tenant-a"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["episode_count"] == 1
    episode_events = [event for event in body["events"] if event["event_type"] == "episode"]
    assert len(episode_events) == 1
    assert episode_events[0]["citing_memory_count"] == 1

    tenant_a_episode_id = str(tenant_a_episode.id)

    # Operator-global view (no tenant filter): the metrics must AGGREGATE across
    # tenants — all three episodes, and the tenant-a episode cited by BOTH
    # tenants' memories. This guards against a future regression where the
    # tenant filter leaks into the operator path and over-narrows the global
    # view (the behavior the fix is careful to preserve).
    global_response = await client.get(
        f"/admin/subjects/{subject_id}/sessions/{session_id}/timeline",
    )
    assert global_response.status_code == 200
    global_body = global_response.json()
    assert global_body["episode_count"] == 3
    global_episode_events = [e for e in global_body["events"] if e["event_type"] == "episode"]
    tenant_a_event = next(e for e in global_episode_events if e["id"] == tenant_a_episode_id)
    assert tenant_a_event["citing_memory_count"] == 2

    # Tenant-b filtered view: only tenant-b's two episodes, and neither is cited
    # by a tenant-b memory (both memories cite the tenant-a episode id), so the
    # count must be 0 — not the cross-tenant citation.
    tenant_b_response = await client.get(
        f"/admin/subjects/{subject_id}/sessions/{session_id}/timeline",
        params={"tenant_id": "tenant-b"},
    )
    assert tenant_b_response.status_code == 200
    tenant_b_body = tenant_b_response.json()
    assert tenant_b_body["episode_count"] == 2
    tenant_b_episode_events = [
        e for e in tenant_b_body["events"] if e["event_type"] == "episode"
    ]
    assert len(tenant_b_episode_events) == 2
    assert all(e["citing_memory_count"] == 0 for e in tenant_b_episode_events)
