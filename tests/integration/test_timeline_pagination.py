"""GET /v1/timeline previously called the repo layer with no limit/offset
exposed, silently capping at 100 episodes and 100 memories with no way
to page further and no signal that the response was truncated (#331).
This covers: bounded limit/offset params, has_more flags per collection,
and that paging with offset returns every row exactly once.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from server.db.tables import EpisodeRow, MemoryRow

pytestmark = pytest.mark.anyio


async def _seed(session_factory, subject_id: str, n_episodes: int, n_memories: int):
    async with session_factory() as session:
        episode_ids = []
        for i in range(n_episodes):
            row = EpisodeRow(
                subject_id=subject_id,
                source="test",
                type="message",
                payload={"i": i},
                metadata_={},
                provenance={},
            )
            session.add(row)
            episode_ids.append(row)
        await session.flush()

        for i in range(n_memories):
            session.add(
                MemoryRow(
                    subject_id=subject_id,
                    kind="profile_fact",
                    content=f"memory {i}",
                    summary=f"memory {i}",
                    confidence=1.0,
                    source_episode_ids=[episode_ids[0].id] if episode_ids else [],
                    metadata_={},
                    status="active",
                    sensitivity_labels=[],
                    suggested_labels=[],
                )
            )
        await session.commit()


async def test_has_more_is_true_when_a_collection_exceeds_the_page(
    client: AsyncClient, session_factory
):
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=5, n_memories=2)

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "limit": 3, "offset": 0}
    )
    assert response.status_code == 200
    body = response.json()

    assert len(body["episodes"]) == 3
    assert body["episodes_has_more"] is True
    assert len(body["memories"]) == 2
    assert body["memories_has_more"] is False


async def test_has_more_is_false_on_the_last_page(client: AsyncClient, session_factory):
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=5, n_memories=0)

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "limit": 3, "offset": 3}
    )
    assert response.status_code == 200
    body = response.json()

    assert len(body["episodes"]) == 2
    assert body["episodes_has_more"] is False


async def test_paging_with_offset_covers_every_episode_exactly_once(
    client: AsyncClient, session_factory
):
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=7, n_memories=0)

    seen: list[str] = []
    limit = 3
    offset = 0
    for _ in range(5):  # safety bound
        response = await client.get(
            "/v1/timeline", params={"subject_id": subject_id, "limit": limit, "offset": offset}
        )
        assert response.status_code == 200
        body = response.json()
        page = body["episodes"]
        if not page:
            break
        seen.extend(e["id"] for e in page)
        if not body["episodes_has_more"]:
            break
        offset += limit

    assert len(seen) == 7
    assert len(set(seen)) == 7


async def test_limit_out_of_bounds_is_422(client: AsyncClient, session_factory):
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "limit": 0}
    )
    assert response.status_code == 422

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "limit": 500}
    )
    assert response.status_code == 422

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "offset": -1}
    )
    assert response.status_code == 422