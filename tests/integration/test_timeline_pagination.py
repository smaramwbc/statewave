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
from sqlalchemy import delete

from server.db.tables import EpisodeRow, MemoryRow

pytestmark = pytest.mark.anyio


_SEEDED: list[str] = []


@pytest.fixture(autouse=True)
async def _cleanup_seeded(session_factory):
    """Delete the rows each test seeded.

    The suite shares one database and a subject is derived from the episode and
    memory rows that mention it, so a test that seeds and leaves permanently
    enlarges `GET /v1/subjects` and eventually fails a test in another file.
    The cost stays in the file that creates it.
    """
    yield
    subject_ids, _SEEDED[:] = list(_SEEDED), []
    if not subject_ids:
        return
    async with session_factory() as session:
        await session.execute(delete(MemoryRow).where(MemoryRow.subject_id.in_(subject_ids)))
        await session.execute(delete(EpisodeRow).where(EpisodeRow.subject_id.in_(subject_ids)))
        await session.commit()


async def _seed(session_factory, subject_id: str, n_episodes: int, n_memories: int):
    _SEEDED.append(subject_id)
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

async def test_exactly_limit_rows_is_not_has_more(client: AsyncClient, session_factory):
    """The limit+1 sentinel exists for this boundary: a page that is exactly full
    must still report has_more=False when nothing lies beyond it."""
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=3, n_memories=3)

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "limit": 3, "offset": 0}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["episodes"]) == 3 and body["episodes_has_more"] is False
    assert len(body["memories"]) == 3 and body["memories_has_more"] is False


async def test_offset_at_total_returns_an_empty_last_page(client: AsyncClient, session_factory):
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=4, n_memories=4)

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "limit": 4, "offset": 4}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["episodes"] == [] and body["episodes_has_more"] is False
    assert body["memories"] == [] and body["memories_has_more"] is False


async def test_memories_has_more_is_true_when_memories_exceed_the_page(
    client: AsyncClient, session_factory
):
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=1, n_memories=5)

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "limit": 2, "offset": 0}
    )
    body = response.json()
    assert len(body["memories"]) == 2 and body["memories_has_more"] is True
    assert len(body["episodes"]) == 1 and body["episodes_has_more"] is False


async def test_rows_sharing_a_timestamp_page_without_gaps_or_repeats(
    client: AsyncClient, session_factory
):
    """Rows written in one transaction share a single now() for created_at and
    occurred_at, so ORDER BY on the timestamps alone is not a total order and
    OFFSET pages could skip or repeat rows. The id tiebreak makes paging exact:
    every seeded row must appear exactly once across the pages."""
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=23, n_memories=17)

    seen_episodes: list[str] = []
    seen_memories: list[str] = []
    offset = 0
    while True:
        response = await client.get(
            "/v1/timeline", params={"subject_id": subject_id, "limit": 5, "offset": offset}
        )
        assert response.status_code == 200
        body = response.json()
        seen_episodes += [e["id"] for e in body["episodes"]]
        seen_memories += [m["id"] for m in body["memories"]]
        if not body["episodes_has_more"] and not body["memories_has_more"]:
            break
        offset += 5

    assert len(seen_episodes) == 23 and len(set(seen_episodes)) == 23
    assert len(seen_memories) == 17 and len(set(seen_memories)) == 17
