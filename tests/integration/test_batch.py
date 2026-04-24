"""Integration tests for POST /v1/episodes/batch.

Requires a running Postgres with `statewave_test` database.
Uses the shared integration conftest fixtures (client, subject_id).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def _make_episodes(subject_id: str, count: int = 5) -> list[dict]:
    return [
        {
            "subject_id": subject_id,
            "source": "batch-test",
            "type": "conversation",
            "payload": {
                "messages": [
                    {"role": "user", "content": f"Batch message {i}"},
                    {"role": "assistant", "content": f"Batch response {i}"},
                ]
            },
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_batch_ingest_happy_path(client: AsyncClient, subject_id: str):
    """Batch-ingest multiple episodes and verify all are created."""
    episodes = _make_episodes(subject_id, count=5)
    resp = await client.post("/v1/episodes/batch", json={"episodes": episodes})
    assert resp.status_code == 201, resp.text

    data = resp.json()
    assert data["episodes_created"] == 5
    assert len(data["episodes"]) == 5

    # All episodes belong to the right subject
    for ep in data["episodes"]:
        assert ep["subject_id"] == subject_id
        assert ep["source"] == "batch-test"
        assert ep["id"]  # UUID assigned
        assert ep["created_at"]

    # IDs are unique
    ids = [ep["id"] for ep in data["episodes"]]
    assert len(set(ids)) == 5


@pytest.mark.anyio
async def test_batch_episodes_appear_on_timeline(client: AsyncClient, subject_id: str):
    """Batch-ingested episodes should appear on the subject timeline."""
    episodes = _make_episodes(subject_id, count=3)
    resp = await client.post("/v1/episodes/batch", json={"episodes": episodes})
    assert resp.status_code == 201

    # Check timeline
    resp = await client.get("/v1/timeline", params={"subject_id": subject_id})
    assert resp.status_code == 200
    timeline = resp.json()
    assert len(timeline["episodes"]) == 3


@pytest.mark.anyio
async def test_batch_then_compile(client: AsyncClient, subject_id: str):
    """Batch-ingest episodes, compile, and verify memories are produced."""
    episodes = [
        {
            "subject_id": subject_id,
            "source": "batch-test",
            "type": "conversation",
            "payload": {
                "messages": [
                    {"role": "user", "content": "My name is Bob and I work at Acme Inc."},
                    {"role": "assistant", "content": "Nice to meet you, Bob!"},
                ]
            },
        },
        {
            "subject_id": subject_id,
            "source": "batch-test",
            "type": "conversation",
            "payload": {
                "messages": [
                    {"role": "user", "content": "I prefer dark mode in all applications."},
                    {"role": "assistant", "content": "Noted, dark mode preference saved."},
                ]
            },
        },
    ]
    resp = await client.post("/v1/episodes/batch", json={"episodes": episodes})
    assert resp.status_code == 201

    # Compile
    resp = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["memories_created"] > 0

    # Memories should include extracted facts
    contents = [m["content"].lower() for m in data["memories"]]
    assert any("bob" in c for c in contents), f"Expected 'bob' in {contents}"


@pytest.mark.anyio
async def test_batch_mixed_subjects(client: AsyncClient):
    """Batch can contain episodes for different subjects."""
    episodes = [
        {
            "subject_id": "batch-subj-a",
            "source": "test",
            "type": "msg",
            "payload": {"text": "hello from A"},
        },
        {
            "subject_id": "batch-subj-b",
            "source": "test",
            "type": "msg",
            "payload": {"text": "hello from B"},
        },
    ]
    resp = await client.post("/v1/episodes/batch", json={"episodes": episodes})
    assert resp.status_code == 201

    data = resp.json()
    subjects = {ep["subject_id"] for ep in data["episodes"]}
    assert subjects == {"batch-subj-a", "batch-subj-b"}

    # Each subject's timeline should have exactly 1 episode
    for sid in ["batch-subj-a", "batch-subj-b"]:
        resp = await client.get("/v1/timeline", params={"subject_id": sid})
        assert resp.status_code == 200
        assert len(resp.json()["episodes"]) >= 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_batch_empty_list_rejected(client: AsyncClient):
    """Empty episodes list should return 422."""
    resp = await client.post("/v1/episodes/batch", json={"episodes": []})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_batch_over_100_rejected(client: AsyncClient):
    """More than 100 episodes should return 422."""
    episodes = [
        {"subject_id": "flood", "source": "t", "type": "t", "payload": {"x": 1}}
        for _ in range(101)
    ]
    resp = await client.post("/v1/episodes/batch", json={"episodes": episodes})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_batch_missing_required_field(client: AsyncClient):
    """Missing required fields in an episode should return 422."""
    resp = await client.post("/v1/episodes/batch", json={
        "episodes": [{"subject_id": "x"}]
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_batch_invalid_subject_id(client: AsyncClient):
    """Empty subject_id should return 422."""
    resp = await client.post("/v1/episodes/batch", json={
        "episodes": [{"subject_id": "", "source": "s", "type": "t", "payload": {"k": "v"}}]
    })
    assert resp.status_code == 422
