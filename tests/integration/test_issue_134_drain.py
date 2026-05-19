"""Integration tests for issue #134 — compile drains the subject.

Exercises the real DB-backed flow with a small `compile_batch_size` so
the test ingests just enough episodes to force multiple batches without
slowing the suite.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient


async def _ingest(client: AsyncClient, subject_id: str, n: int) -> None:
    for i in range(n):
        r = await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "chat.note",
                "payload": {"text": f"Episode {i}: customer prefers dark mode"},
            },
        )
        assert r.status_code in (200, 201), r.text


@pytest.mark.anyio
async def test_134_sync_compile_reports_has_more_when_batch_size_exceeded(
    client: AsyncClient, subject_id: str, monkeypatch
):
    """With 5 uncompiled episodes and `compile_batch_size=2`, the first
    sync call must report `has_more=True` and `remaining_episodes>0`."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "compile_batch_size", 2)
    await _ingest(client, subject_id, 5)

    r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["has_more"] is True
    assert body["remaining_episodes"] >= 3


@pytest.mark.anyio
async def test_134_sync_compile_loop_drains_subject(
    client: AsyncClient, subject_id: str, monkeypatch
):
    """A client that loops on `has_more` ends at `has_more=False` and
    has compiled the whole subject."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "compile_batch_size", 2)
    await _ingest(client, subject_id, 5)

    total_created = 0
    for _ in range(20):  # generous loop bound — 5 episodes × 2/batch = ≤3 calls
        r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
        assert r.status_code == 200, r.text
        body = r.json()
        total_created += body["memories_created"]
        if not body["has_more"]:
            assert body["remaining_episodes"] == 0
            break
    else:
        pytest.fail("sync compile loop did not drain in 20 iterations")

    # Heuristic compiler emits at least one memory per episode for this
    # ingest shape (a "preference" sentence). We only assert "non-zero"
    # because exact count depends on the heuristic — the contract under
    # test is the drain signal, not the compiler output count.
    assert total_created > 0


@pytest.mark.anyio
async def test_134_async_compile_drains_subject_in_one_job(
    client: AsyncClient, subject_id: str, monkeypatch
):
    """`async: true` must process the entire backlog inside one job —
    no caller-side loop required."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "compile_batch_size", 2)
    await _ingest(client, subject_id, 5)

    r = await client.post(
        "/v1/memories/compile", json={"subject_id": subject_id, "async": True}
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    # Poll for completion. Each batch is fast; 10s is generous.
    for _ in range(100):
        await asyncio.sleep(0.1)
        status_r = await client.get(f"/v1/memories/compile/{job_id}")
        assert status_r.status_code == 200, status_r.text
        body = status_r.json()
        if body["status"] in ("completed", "failed"):
            break
    else:
        pytest.fail(f"async compile job {job_id} did not finish within 10s")

    assert body["status"] == "completed", body
    assert body["memories_created"] > 0

    # The whole backlog should be drained — a follow-up sync compile
    # finds nothing to do.
    follow_up = await client.post(
        "/v1/memories/compile", json={"subject_id": subject_id}
    )
    assert follow_up.status_code == 200
    follow_body = follow_up.json()
    assert follow_body["memories_created"] == 0
    assert follow_body["has_more"] is False
    assert follow_body["remaining_episodes"] == 0
