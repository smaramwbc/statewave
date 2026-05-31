"""Regression: the bounded episode fetch used by /v1/context and handoff must
return the subject's MOST-RECENT `limit` episodes, not its oldest.

`list_episodes_by_subject` orders by occurred_at ascending, so a `limit`
smaller than the subject's lifetime episode count returned the OLDEST `limit`
rows. /v1/context (candidate pool, "Recent interactions" rendering, recency
boost, repeat-issue detection) and the handoff brief's cross-session "recent
context" both call it with limit=30 and assume recent episodes — so for any
subject with >30 episodes they silently operated on the oldest 30.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from server.db import repositories as repo
from server.db.tables import EpisodeRow


@pytest.mark.anyio
async def test_newest_first_returns_recent_window_ascending(session_factory, subject_id):
    base = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    async with session_factory() as session:
        for i in range(40):
            session.add(
                EpisodeRow(
                    id=uuid.uuid4(),
                    subject_id=subject_id,
                    source="test",
                    type="conversation",
                    payload={"text": f"ep {i}"},
                    occurred_at=base + dt.timedelta(days=i),
                )
            )
        await session.commit()

    async with session_factory() as session:
        rows = await repo.list_episodes_by_subject(
            session, subject_id, limit=30, newest_first=True
        )

    texts = [r.payload["text"] for r in rows]
    assert len(rows) == 30
    # The most-recent 30 (ep 10..39), returned in ascending chronological order.
    assert texts[0] == "ep 10"
    assert texts[-1] == "ep 39"
    assert [r.occurred_at for r in rows] == sorted(r.occurred_at for r in rows)
    # The genuinely recent episode is present; the oldest is excluded.
    assert "ep 39" in texts
    assert "ep 0" not in texts


@pytest.mark.anyio
async def test_default_still_returns_oldest_window(session_factory, subject_id):
    """Backward-compat: without newest_first, the historical ascending/oldest
    behaviour is preserved for any other caller."""
    base = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    async with session_factory() as session:
        for i in range(40):
            session.add(
                EpisodeRow(
                    id=uuid.uuid4(),
                    subject_id=subject_id,
                    source="test",
                    type="conversation",
                    payload={"text": f"ep {i}"},
                    occurred_at=base + dt.timedelta(days=i),
                )
            )
        await session.commit()

    async with session_factory() as session:
        rows = await repo.list_episodes_by_subject(session, subject_id, limit=30)

    texts = [r.payload["text"] for r in rows]
    assert texts[0] == "ep 0"
    assert texts[-1] == "ep 29"
