"""Integration tests for async compile flow (durable jobs).

Tests the full async compilation lifecycle against a real database:
- submit async compile → 202 with job_id
- poll status → eventually reaches terminal state
- successful job creates memories
- job state survives in-memory cache clear (Postgres durability)
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_async_compile_returns_202(client: AsyncClient, subject_id: str):
    """POST /v1/memories/compile with async=true returns 202 + job_id."""
    # Ingest an episode first
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "support-chat",
            "payload": {
                "messages": [
                    {"role": "user", "content": "My name is TestUser"},
                    {"role": "assistant", "content": "Hi TestUser!"},
                ]
            },
        },
    )

    r = await client.post(
        "/v1/memories/compile",
        json={
            "subject_id": subject_id,
            "async": True,
        },
    )
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "pending"
    assert body["subject_id"] == subject_id


@pytest.mark.anyio
async def test_async_compile_reaches_completed(client: AsyncClient, subject_id: str):
    """Async compile job eventually reaches 'completed' status."""
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "support-chat",
            "payload": {
                "messages": [
                    {"role": "user", "content": "I work at Acme Corp"},
                    {"role": "assistant", "content": "Noted!"},
                ]
            },
        },
    )

    r = await client.post(
        "/v1/memories/compile",
        json={
            "subject_id": subject_id,
            "async": True,
        },
    )
    job_id = r.json()["job_id"]

    # Poll until terminal (max 5s)
    for _ in range(50):
        await asyncio.sleep(0.1)
        r = await client.get(f"/v1/memories/compile/{job_id}")
        assert r.status_code == 200
        if r.json()["status"] in ("completed", "failed"):
            break

    status = r.json()
    assert status["status"] == "completed"
    assert status["memories_created"] >= 1


@pytest.mark.anyio
async def test_async_compile_creates_memories(client: AsyncClient, subject_id: str):
    """After async compile completes, memories are searchable."""
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "support-chat",
            "payload": {
                "messages": [
                    {"role": "user", "content": "My email is test@example.com"},
                    {"role": "assistant", "content": "Got it!"},
                ]
            },
        },
    )

    r = await client.post(
        "/v1/memories/compile",
        json={
            "subject_id": subject_id,
            "async": True,
        },
    )
    job_id = r.json()["job_id"]

    # Wait for completion
    for _ in range(50):
        await asyncio.sleep(0.1)
        r = await client.get(f"/v1/memories/compile/{job_id}")
        if r.json()["status"] == "completed":
            break

    # Verify memories exist
    r = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "limit": 50,
        },
    )
    assert r.status_code == 200
    assert len(r.json()["memories"]) >= 1


@pytest.mark.anyio
async def test_async_compile_no_episodes_completes_immediately(
    client: AsyncClient, subject_id: str
):
    """Async compile with no uncompiled episodes still completes (0 memories)."""
    r = await client.post(
        "/v1/memories/compile",
        json={
            "subject_id": subject_id,
            "async": True,
        },
    )
    job_id = r.json()["job_id"]

    for _ in range(30):
        await asyncio.sleep(0.1)
        r = await client.get(f"/v1/memories/compile/{job_id}")
        if r.json()["status"] == "completed":
            break

    assert r.json()["status"] == "completed"
    assert r.json()["memories_created"] == 0


@pytest.mark.anyio
async def test_job_status_survives_memory_clear(client: AsyncClient, subject_id: str):
    """Job status is readable from Postgres even after in-memory cache is cleared."""
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "note",
            "payload": {"text": "test data"},
        },
    )

    r = await client.post(
        "/v1/memories/compile",
        json={
            "subject_id": subject_id,
            "async": True,
        },
    )
    job_id = r.json()["job_id"]

    # Wait for completion
    for _ in range(50):
        await asyncio.sleep(0.1)
        r = await client.get(f"/v1/memories/compile/{job_id}")
        if r.json()["status"] in ("completed", "failed"):
            break

    # Clear the in-memory store
    from server.services.compile_jobs import _jobs

    _jobs.clear()

    # Should still be able to read from Postgres
    r = await client.get(f"/v1/memories/compile/{job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


@pytest.mark.anyio
async def test_nonexistent_job_returns_404(client: AsyncClient):
    """Polling a nonexistent job_id returns 404."""
    r = await client.get("/v1/memories/compile/nonexist")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_sync_compile_still_works(client: AsyncClient, subject_id: str):
    """Sync compile (no async flag) still returns 200 with results directly."""
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "support-chat",
            "payload": {
                "messages": [
                    {"role": "user", "content": "I prefer Python"},
                    {"role": "assistant", "content": "Python it is!"},
                ]
            },
        },
    )

    r = await client.post(
        "/v1/memories/compile",
        json={
            "subject_id": subject_id,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "memories_created" in body
    assert body["memories_created"] >= 1
    assert "job_id" not in body
