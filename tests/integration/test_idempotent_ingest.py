"""Episode ingest is idempotent: re-ingesting the same idempotency_key is a
no-op, not a duplicate. This is what makes re-running a connector seed (or a
retried webhook) safe — without it, a repo seeded N times held N× the episodes.
"""

from __future__ import annotations

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
