"""GET /v1/timeline returned superseded and expired memories inside the
caller's `limit` budget, with no consumer-reachable way to ask for only the
surviving state (#370). `status=active` restricts the memories collection to
rows that are currently authoritative — status `active` and not past their
`valid_to` — while the default `all` keeps the previous behaviour unchanged.
Episodes are raw history and are never filtered.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete

from server.db.tables import EpisodeRow, MemoryRow

pytestmark = pytest.mark.anyio


_SEEDED: list[str] = []


@pytest.fixture(autouse=True)
async def _cleanup_seeded(session_factory):
    yield
    subject_ids, _SEEDED[:] = list(_SEEDED), []
    if not subject_ids:
        return
    async with session_factory() as session:
        await session.execute(delete(MemoryRow).where(MemoryRow.subject_id.in_(subject_ids)))
        await session.execute(delete(EpisodeRow).where(EpisodeRow.subject_id.in_(subject_ids)))
        await session.commit()


async def _seed(session_factory, subject_id: str, rows: list[dict]):
    """Seed memories described by dicts of (status, expired) flags, in order."""
    _SEEDED.append(subject_id)
    async with session_factory() as session:
        episode = EpisodeRow(
            subject_id=subject_id,
            source="test",
            type="message",
            payload={},
            metadata_={},
            provenance={},
        )
        session.add(episode)
        await session.flush()
        base = datetime.now(timezone.utc) - timedelta(minutes=len(rows))
        for i, spec in enumerate(rows):
            valid_to = None
            if spec.get("expired"):
                valid_to = datetime.now(timezone.utc) - timedelta(hours=1)
            session.add(
                MemoryRow(
                    created_at=base + timedelta(seconds=i),
                    subject_id=subject_id,
                    kind="profile_fact",
                    content=f"fact {i} ({spec['status']})",
                    summary=f"fact {i}",
                    confidence=1.0,
                    source_episode_ids=[episode.id],
                    metadata_={},
                    status=spec["status"],
                    valid_to=valid_to,
                    sensitivity_labels=[],
                    suggested_labels=[],
                )
            )
        await session.commit()


def _subject() -> str:
    return f"tl-active-{uuid.uuid4().hex[:12]}"


async def test_default_still_returns_every_row(client, session_factory):
    subject_id = _subject()
    await _seed(
        session_factory,
        subject_id,
        [{"status": "superseded"}, {"status": "active"}, {"status": "active", "expired": True}],
    )
    resp = await client.get("/v1/timeline", params={"subject_id": subject_id})
    assert resp.status_code == 200
    assert len(resp.json()["memories"]) == 3


async def test_status_active_returns_only_authoritative_rows(client, session_factory):
    subject_id = _subject()
    await _seed(
        session_factory,
        subject_id,
        [
            {"status": "superseded"},
            {"status": "active"},
            {"status": "active", "expired": True},
            {"status": "expired"},
            {"status": "active"},
        ],
    )
    resp = await client.get(
        "/v1/timeline", params={"subject_id": subject_id, "status": "active"}
    )
    assert resp.status_code == 200
    body = resp.json()
    texts = [m["content"] for m in body["memories"]]
    assert texts == ["fact 1 (active)", "fact 4 (active)"]
    assert body["memories_has_more"] is False
    # episodes are raw history — never filtered by memory status
    assert len(body["episodes"]) == 1


async def test_pagination_counts_only_included_rows(client, session_factory):
    """`limit`/`has_more` must apply to the filtered set: 99 superseded rows
    ahead of the surviving fact cannot push it out of reach (#370's motivating
    failure)."""
    subject_id = _subject()
    await _seed(
        session_factory,
        subject_id,
        [{"status": "superseded"} for _ in range(99)] + [{"status": "active"}],
    )
    resp = await client.get(
        "/v1/timeline",
        params={"subject_id": subject_id, "status": "active", "limit": 1},
    )
    body = resp.json()
    assert [m["content"] for m in body["memories"]] == ["fact 99 (active)"]
    assert body["memories_has_more"] is False


async def test_active_filter_composes_with_newest_first(client, session_factory):
    subject_id = _subject()
    await _seed(
        session_factory,
        subject_id,
        [
            {"status": "active"},
            {"status": "superseded"},
            {"status": "active"},
            {"status": "superseded"},
            {"status": "active"},
        ],
    )
    resp = await client.get(
        "/v1/timeline",
        params={
            "subject_id": subject_id,
            "status": "active",
            "newest_first": "true",
            "limit": 2,
        },
    )
    body = resp.json()
    # the two most recent ACTIVE rows, ascending, with more remaining
    assert [m["content"] for m in body["memories"]] == ["fact 2 (active)", "fact 4 (active)"]
    assert body["memories_has_more"] is True


async def test_unknown_status_is_rejected(client):
    resp = await client.get(
        "/v1/timeline", params={"subject_id": "whatever", "status": "superseded"}
    )
    assert resp.status_code == 422
