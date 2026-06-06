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
