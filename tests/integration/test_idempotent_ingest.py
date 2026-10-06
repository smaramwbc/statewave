"""Episode ingest is idempotent: re-ingesting the same idempotency_key is a
no-op, not a duplicate. This is what makes re-running a connector seed (or a
retried webhook) safe — without it, a repo seeded N times held N× the episodes.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import AsyncClient


def _episode(subject_id: str, key: str, text: str = "hello") -> dict:
    return {
        "subject_id": subject_id,
        "source": "git",
        "type": "git.commit",
        "payload": {"text": text},
        "idempotency_key": key,
    }


@pytest.mark.anyio
async def test_same_key_does_not_duplicate(client: AsyncClient, subject_id: str):
    first = await client.post("/v1/episodes", json=_episode(subject_id, "git:commit:abc"))
    assert first.status_code == 201
    second = await client.post("/v1/episodes", json=_episode(subject_id, "git:commit:abc", text="changed"))
    assert second.status_code in (200, 201)
    # Same key → same row returned, and the timeline holds exactly one episode.
    assert second.json()["id"] == first.json()["id"]

    timeline = await client.get(f"/v1/timeline?subject_id={subject_id}")
    episodes = timeline.json().get("episodes") or timeline.json().get("items") or []
    assert len(episodes) == 1


@pytest.mark.anyio
async def test_key_in_metadata_is_honored_for_legacy_clients(client: AsyncClient, subject_id: str):
    # Older connectors stash the key in metadata rather than the top-level field.
    ep = {
        "subject_id": subject_id,
        "source": "git",
        "type": "git.commit",
        "payload": {"text": "x"},
        "metadata": {"idempotency_key": "git:commit:legacy"},
    }
    await client.post("/v1/episodes", json=ep)
    await client.post("/v1/episodes", json=ep)
    timeline = await client.get(f"/v1/timeline?subject_id={subject_id}")
    episodes = timeline.json().get("episodes") or timeline.json().get("items") or []
    assert len(episodes) == 1


@pytest.mark.anyio
async def test_distinct_keys_and_keyless_all_insert(client: AsyncClient, subject_id: str):
    await client.post("/v1/episodes", json=_episode(subject_id, "git:commit:a"))
    await client.post("/v1/episodes", json=_episode(subject_id, "git:commit:b"))
    # No key → never de-duped (live-chat ingest), even when identical.
    keyless = {"subject_id": subject_id, "source": "chat", "type": "chat.msg", "payload": {"text": "hi"}}
    await client.post("/v1/episodes", json=keyless)
    await client.post("/v1/episodes", json=keyless)
    timeline = await client.get(f"/v1/timeline?subject_id={subject_id}")
    episodes = timeline.json().get("episodes") or timeline.json().get("items") or []
    assert len(episodes) == 4  # 2 distinct keys + 2 keyless


@pytest.mark.anyio
@patch("server.api.episodes.webhooks.fire")
async def test_idempotent_replay_returns_200_and_does_not_fire_webhook(mock_fire, client: AsyncClient, subject_id: str):
    # First insert -> 201 Created and fires webhook
    first = await client.post("/v1/episodes", json=_episode(subject_id, "git:commit:123"))
    assert first.status_code == 201
    assert mock_fire.call_count == 1
    
    # Second insert with same key -> 200 OK and does NOT fire webhook again
    second = await client.post("/v1/episodes", json=_episode(subject_id, "git:commit:123", text="changed"))
    assert second.status_code == 200
    assert mock_fire.call_count == 1

@pytest.mark.anyio
@patch("server.api.episodes.webhooks.fire")
async def test_batch_idempotent_replay_counts_and_webhooks(mock_fire, client: AsyncClient, subject_id: str):
    batch = {
        "episodes": [
            _episode(subject_id, "batch:1"),
            _episode(subject_id, "batch:2"),
        ]
    }
    
    first = await client.post("/v1/episodes/batch", json=batch)
    assert first.status_code == 201
    assert first.json()["episodes_created"] == 2
    assert mock_fire.call_count == 1
    
    second = await client.post("/v1/episodes/batch", json=batch)
    assert second.status_code == 201
    assert second.json()["episodes_created"] == 0
    assert mock_fire.call_count == 1
