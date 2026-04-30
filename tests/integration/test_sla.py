"""Integration test: SLA tracking endpoint against real DB."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_sla_endpoint_returns_metrics(client: AsyncClient, subject_id: str):
    """Full lifecycle: ingest episodes + resolution → GET /sla returns computed metrics."""

    # Ingest user message
    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "session_id": "sess-sla-1",
            "source": "user",
            "type": "message",
            "payload": {"messages": [{"role": "user", "content": "I need help with billing"}]},
        },
    )
    assert resp.status_code == 201

    # Ingest agent response (simulates 2-min delay via ordering)
    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "session_id": "sess-sla-1",
            "source": "assistant",
            "type": "message",
            "payload": {
                "messages": [{"role": "assistant", "content": "Let me check your account"}]
            },
        },
    )
    assert resp.status_code == 201

    # Create resolution
    resp = await client.post(
        "/v1/resolutions",
        json={
            "subject_id": subject_id,
            "session_id": "sess-sla-1",
            "status": "resolved",
            "resolution_summary": "Billing issue clarified",
        },
    )
    assert resp.status_code == 200

    # Get SLA
    resp = await client.get(f"/v1/subjects/{subject_id}/sla")
    assert resp.status_code == 200
    data = resp.json()

    assert data["subject_id"] == subject_id
    assert data["total_sessions"] == 1
    assert data["resolved_sessions"] == 1
    assert data["open_sessions"] == 0
    assert len(data["sessions"]) == 1

    session_sla = data["sessions"][0]
    assert session_sla["session_id"] == "sess-sla-1"
    assert session_sla["status"] == "resolved"
    assert session_sla["first_message_at"] is not None
    assert session_sla["first_response_at"] is not None
    assert session_sla["first_response_seconds"] is not None
    assert session_sla["first_response_seconds"] >= 0
    assert session_sla["resolution_seconds"] is not None


@pytest.mark.anyio
async def test_sla_custom_threshold(client: AsyncClient, subject_id: str):
    """Custom threshold via query param triggers breach on tight deadline."""

    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "session_id": "sess-sla-thresh",
            "source": "user",
            "type": "message",
            "payload": {"messages": [{"role": "user", "content": "Help please"}]},
        },
    )
    assert resp.status_code == 201

    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "session_id": "sess-sla-thresh",
            "source": "assistant",
            "type": "message",
            "payload": {"messages": [{"role": "assistant", "content": "On it"}]},
        },
    )
    assert resp.status_code == 201

    # Resolution threshold of 0 hours — any resolution time is a breach
    resp = await client.get(
        f"/v1/subjects/{subject_id}/sla",
        params={"resolution_threshold_hours": 0.0001},
    )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_sla_empty_subject(client: AsyncClient, subject_id: str):
    """SLA for non-existent subject returns empty metrics."""
    resp = await client.get(f"/v1/subjects/{subject_id}/sla")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_sessions"] == 0
    assert data["sessions"] == []
