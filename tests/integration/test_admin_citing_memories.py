"""Regression: the reverse-provenance lookup must not 500 on a hit.

GET /admin/subjects/{subject_id}/episodes/{episode_id}/citing-memories built
its ``MemoryListItem`` response without the required ``sensitivity_labels`` /
``suggested_labels`` fields, so it raised a response-model ValidationError
(HTTP 500) the moment any memory actually cited the episode — i.e. exactly the
case the endpoint exists to serve. It only "worked" when the result was empty.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_citing_memories_returns_200_when_memories_cite_episode(
    client: AsyncClient, subject_id: str
):
    # Ingest an episode and capture its id.
    r = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "conversation",
            "payload": {
                "messages": [
                    {"role": "user", "content": "My name is Alice and I work at Initech."}
                ]
            },
        },
    )
    assert r.status_code == 201
    episode_id = r.json()["id"]

    # Compile → the derived memories carry this episode in source_episode_ids.
    r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert r.status_code == 200
    assert r.json()["memories_created"] > 0

    # Reverse-provenance lookup must succeed (previously 500).
    r = await client.get(
        f"/admin/subjects/{subject_id}/episodes/{episode_id}/citing-memories"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] >= 1
    assert len(body["memories"]) >= 1
    first = body["memories"][0]
    # The governance fields that were previously omitted must be present.
    assert "sensitivity_labels" in first
    assert "suggested_labels" in first
