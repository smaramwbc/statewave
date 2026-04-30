"""Tests for admin subject explorer endpoints."""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.anyio


async def test_list_subjects_empty(client: AsyncClient):
    """List subjects returns empty when no data."""
    resp = await client.get("/admin/subjects")
    assert resp.status_code == 200
    data = resp.json()
    assert data["subjects"] == []
    assert data["total"] == 0


async def test_list_subjects_with_data(client: AsyncClient):
    """List subjects returns data after episode ingestion."""
    # Create an episode first
    ep_resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": "test_user_1",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )
    assert ep_resp.status_code == 201

    # List subjects
    resp = await client.get("/admin/subjects")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    subjects = data["subjects"]
    assert any(s["subject_id"] == "test_user_1" for s in subjects)


async def test_list_subjects_search(client: AsyncClient):
    """Search filters subjects by ID."""
    # Create test subjects
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "searchable_user",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "other_user",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )

    # Search for specific subject
    resp = await client.get("/admin/subjects", params={"search": "searchable"})
    assert resp.status_code == 200
    data = resp.json()
    subjects = data["subjects"]
    assert all("searchable" in s["subject_id"] for s in subjects)


async def test_list_subjects_pagination(client: AsyncClient):
    """Pagination works correctly."""
    # Create multiple subjects
    for i in range(5):
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": f"paginate_user_{i}",
                "source": "test",
                "type": "message",
                "payload": {"text": f"msg {i}"},
            },
        )

    # Get first page with limit 2
    resp = await client.get("/admin/subjects", params={"limit": 2, "offset": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["subjects"]) <= 2
    assert data["limit"] == 2
    assert data["offset"] == 0


async def test_get_subject_detail(client: AsyncClient):
    """Get detail for a specific subject."""
    # Create subject with episodes
    for i in range(3):
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": "detail_user",
                "source": "test",
                "type": "message",
                "payload": {"text": f"message {i}"},
            },
        )

    # Get detail
    resp = await client.get("/admin/subjects/detail_user")
    assert resp.status_code == 200
    data = resp.json()
    assert data["subject_id"] == "detail_user"
    assert data["summary"]["episode_count"] >= 3
    assert data["summary"]["memory_count"] >= 0


async def test_get_subject_detail_not_found(client: AsyncClient):
    """Return 404 for nonexistent subject."""
    resp = await client.get("/admin/subjects/nonexistent_subject_xyz")
    assert resp.status_code == 404


async def test_list_subject_memories(client: AsyncClient):
    """List memories for a subject."""
    # Create episode and compile
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "memory_user",
            "source": "test",
            "type": "message",
            "payload": {"text": "I prefer dark mode"},
        },
    )
    await client.post("/v1/memories/compile", json={"subject_id": "memory_user"})

    # List memories
    resp = await client.get("/admin/subjects/memory_user/memories")
    assert resp.status_code == 200
    data = resp.json()
    assert "memories" in data
    assert "total" in data


async def test_list_subject_episodes(client: AsyncClient):
    """List episodes for a subject."""
    # Create episodes
    for i in range(3):
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": "episode_list_user",
                "source": "test",
                "type": "message",
                "payload": {"text": f"message {i}"},
            },
        )

    # List episodes
    resp = await client.get("/admin/subjects/episode_list_user/episodes")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["episodes"]) >= 3
    assert data["total"] >= 3
