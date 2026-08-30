"""`GET /v1/timeline` returned a subject's OLDEST records and nothing else.

The repository caps each collection at 100 rows in ascending order, and the
route exposed no way to change that, so once a subject passed 100 episodes its
recent history became unreachable through this endpoint — the wrong half to
lose for a consumer asking what has happened lately. `newest_first=true` takes
the window from the other end. `/v1/context` and `/v1/handoff` already pass the
same repository flag; this covers the route that could not.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import delete

from server.db.tables import EpisodeRow, MemoryRow

pytestmark = pytest.mark.anyio

# Subjects seeded by the tests below, drained after each one. See `_cleanup_seeded`.
_SEEDED: list[str] = []


@pytest.fixture(autouse=True)
async def _cleanup_seeded(session_factory):
    """Delete the rows each test seeded.

    The suite shares one database and a subject is derived from the episode and
    memory rows that mention it, so a test that seeds and leaves permanently
    enlarges `GET /v1/subjects`. That listing pages at 50 by default and
    `test_edge_cases.py::test_list_subjects` asserts its own subject is on the
    first page — leaked subjects eventually fail a test in another file, which
    is a miserable thing to debug. The cost stays in the file that creates it.
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
    """Seed `n_episodes` + `n_memories` rows, each one minute after the last.

    The timestamps are explicit because both columns default to `now()`, which
    Postgres holds constant for a transaction: rows written in one commit would
    share a timestamp, their relative order would be unspecified, and every
    ordering assertion below would pass for the wrong reason.
    """
    _SEEDED.append(subject_id)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    async with session_factory() as session:
        first_episode = None
        for i in range(n_episodes):
            row = EpisodeRow(
                subject_id=subject_id,
                source="test",
                type="message",
                payload={"i": i},
                occurred_at=base + timedelta(minutes=i),
                metadata_={},
                provenance={},
            )
            session.add(row)
            if first_episode is None:
                first_episode = row
        await session.flush()

        for i in range(n_memories):
            session.add(
                MemoryRow(
                    subject_id=subject_id,
                    kind="profile_fact",
                    content=f"memory {i}",
                    summary=f"memory {i}",
                    confidence=1.0,
                    created_at=base + timedelta(minutes=i),
                    source_episode_ids=[first_episode.id] if first_episode is not None else [],
                    metadata_={},
                    status="active",
                    sensitivity_labels=[],
                    suggested_labels=[],
                )
            )
        await session.commit()


async def test_newest_first_returns_the_most_recent_window(client: AsyncClient, session_factory):
    """Past the 100-row cap, the recent end is only reachable with the flag."""
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=105, n_memories=0)

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "newest_first": "true"}
    )

    assert response.status_code == 200
    ordinals = [e["payload"]["i"] for e in response.json()["episodes"]]
    assert len(ordinals) == 100
    assert ordinals == list(range(5, 105))


async def test_default_still_returns_the_oldest_window(client: AsyncClient, session_factory):
    """Omitting the parameter must behave exactly as before."""
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=105, n_memories=0)

    response = await client.get("/v1/timeline", params={"subject_id": subject_id})

    ordinals = [e["payload"]["i"] for e in response.json()["episodes"]]
    assert ordinals == list(range(0, 100))


async def test_newest_first_page_is_still_ascending(client: AsyncClient, session_factory):
    """Which end the window is taken from is separate from the order within it."""
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=6, n_memories=0)

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "newest_first": "true"}
    )

    occurred = [e["occurred_at"] for e in response.json()["episodes"]]
    assert occurred == sorted(occurred)


async def test_newest_first_applies_to_memories_too(client: AsyncClient, session_factory):
    """The parameter is about the timeline, so it must not silently mean
    "episodes only" — a caller asking for recent history and getting the oldest
    memories back has no way to tell."""
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=0, n_memories=105)

    response = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "newest_first": "true"}
    )

    contents = [m["content"] for m in response.json()["memories"]]
    assert len(contents) == 100
    assert contents[0] == "memory 5"
    assert contents[-1] == "memory 104"


async def test_newest_first_is_unaffected_below_the_cap(client: AsyncClient, session_factory):
    """Under the cap both directions see the whole history, in the same order."""
    subject_id = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, subject_id, n_episodes=7, n_memories=0)

    default = await client.get("/v1/timeline", params={"subject_id": subject_id})
    newest = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "newest_first": "true"}
    )

    assert [e["id"] for e in default.json()["episodes"]] == [
        e["id"] for e in newest.json()["episodes"]
    ]


async def test_newest_first_respects_the_subject_boundary(client: AsyncClient, session_factory):
    """A recent-window query must not widen to another subject."""
    mine = f"test-{uuid.uuid4().hex[:12]}"
    theirs = f"test-{uuid.uuid4().hex[:12]}"
    await _seed(session_factory, mine, n_episodes=3, n_memories=0)
    await _seed(session_factory, theirs, n_episodes=3, n_memories=0)

    response = await client.get("/v1/timeline", params={"subject_id": mine, "newest_first": "true"})

    body = response.json()
    assert body["subject_id"] == mine
    assert len(body["episodes"]) == 3
