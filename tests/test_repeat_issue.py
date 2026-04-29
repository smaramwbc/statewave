"""Tests for repeat-issue detection in context assembly."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.context import (
    _extract_issue_keywords,
    _session_keyword_overlap,
    assemble_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)


def _make_episode_row(
    *,
    session_id: str | None = None,
    text: str = "hello",
    minutes_ago: int = 0,
):
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="chat",
        type="message",
        payload={"messages": [{"role": "user", "content": text}]},
        metadata_={},
        provenance={},
        session_id=session_id,
        created_at=_BASE - timedelta(minutes=minutes_ago),
    )


def _make_resolution(session_id: str, resolution_summary: str):
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        session_id=session_id,
        status="resolved",
        summary=resolution_summary,
        resolution_summary=resolution_summary,
        resolved_by="agent",
        metadata_={},
        created_at=_BASE - timedelta(hours=1),
    )


@contextmanager
def _mock_repos(episodes, *, resolved_sessions=None, resolutions=None):
    with (
        patch(
            "server.services.context.repo.search_memories",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "server.services.context.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes,
        ),
        patch(
            "server.services.context.repo.search_memories_by_embedding",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("server.services.context.get_embedding_provider", return_value=None),
        patch(
            "server.services.context.repo.get_resolved_session_ids",
            new_callable=AsyncMock,
            return_value=resolved_sessions or set(),
        ),
        patch(
            "server.services.context.repo.get_open_session_ids",
            new_callable=AsyncMock,
            return_value=set(),
        ),
        patch(
            "server.services.context.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=resolutions or [],
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestExtractIssueKeywords:
    def test_removes_stopwords(self):
        kw = _extract_issue_keywords("the user is having a problem")
        assert "the" not in kw
        assert "is" not in kw
        assert "problem" in kw

    def test_lowercases(self):
        kw = _extract_issue_keywords("Payment FAILED error")
        assert "payment" in kw
        assert "failed" in kw

    def test_filters_short_words(self):
        kw = _extract_issue_keywords("I am ok no")
        # Words <= 2 chars should be excluded
        assert "am" not in kw
        assert "ok" not in kw
        assert "no" not in kw


class TestSessionKeywordOverlap:
    def test_identical_sets(self):
        s = {"payment", "failed", "error"}
        assert _session_keyword_overlap(s, s) == 1.0

    def test_no_overlap(self):
        assert _session_keyword_overlap({"login", "error"}, {"billing", "invoice"}) == 0.0

    def test_partial_overlap(self):
        overlap = _session_keyword_overlap(
            {"payment", "failed", "checkout"},
            {"payment", "failed", "refund"},
        )
        # 2 common / 3 in smaller set = 0.666...
        assert overlap > 0.5

    def test_empty_sets(self):
        assert _session_keyword_overlap(set(), {"a", "b"}) == 0.0
        assert _session_keyword_overlap({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# Integration tests for repeat-issue boost in context assembly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeat_issue_boosts_resolved_session():
    """Resolved session with keyword overlap should be boosted above normal episodes."""
    # Prior resolved session about "payment failed"
    old_ep = _make_episode_row(
        session_id="sess-old", text="my payment failed at checkout", minutes_ago=120
    )
    # Current session about "payment failed" again
    current_ep = _make_episode_row(
        session_id="sess-current", text="payment failed again on checkout", minutes_ago=2
    )
    # Unrelated old episode
    unrelated_ep = _make_episode_row(
        session_id="sess-other", text="how do I change my username", minutes_ago=60
    )

    resolutions = [_make_resolution("sess-old", "Reset payment gateway token")]

    with _mock_repos(
        [old_ep, current_ep, unrelated_ep],
        resolved_sessions={"sess-old"},
        resolutions=resolutions,
    ):
        result = await assemble_context(
            AsyncMock(), "user-1", "payment failed", max_tokens=4000, session_id="sess-current"
        )

    # Current session ep should be first (session boost), old payment ep should be
    # boosted above unrelated ep due to repeat-issue detection
    session_ids = [e.session_id for e in result.episodes]
    assert session_ids.index("sess-old") < session_ids.index("sess-other")


@pytest.mark.asyncio
async def test_no_false_boost_without_overlap():
    """Resolved session without keyword overlap should NOT get repeat boost."""
    old_ep = _make_episode_row(
        session_id="sess-old", text="billing invoice question", minutes_ago=120
    )
    current_ep = _make_episode_row(
        session_id="sess-current", text="login authentication error", minutes_ago=2
    )

    with _mock_repos(
        [old_ep, current_ep],
        resolved_sessions={"sess-old"},
        resolutions=[_make_resolution("sess-old", "Sent updated invoice")],
    ):
        result = await assemble_context(
            AsyncMock(), "user-1", "login error", max_tokens=4000, session_id="sess-current"
        )

    # Current session ep should rank first; old ep has resolved penalty, no repeat boost
    assert result.episodes[0].session_id == "sess-current"


@pytest.mark.asyncio
async def test_repeat_resolved_boost_higher_than_plain_repeat():
    """Resolution with summary should get higher boost than without."""
    # Two old sessions about same topic
    old_with_resolution = _make_episode_row(
        session_id="sess-resolved", text="payment gateway timeout error", minutes_ago=200
    )
    old_without_resolution = _make_episode_row(
        session_id="sess-no-res", text="payment gateway timeout happening", minutes_ago=180
    )
    current_ep = _make_episode_row(
        session_id="sess-current", text="payment gateway timeout again", minutes_ago=1
    )

    resolutions = [_make_resolution("sess-resolved", "Restarted gateway service")]

    with _mock_repos(
        [old_with_resolution, old_without_resolution, current_ep],
        resolved_sessions={"sess-resolved", "sess-no-res"},
        resolutions=resolutions,
    ):
        result = await assemble_context(
            AsyncMock(), "user-1", "payment timeout", max_tokens=4000, session_id="sess-current"
        )

    # Session with resolution summary should rank above session without
    session_ids = [e.session_id for e in result.episodes]
    assert session_ids.index("sess-resolved") < session_ids.index("sess-no-res")
