"""Tests for session-aware context assembly."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.context import (
    _RECENCY_MAX,
    _SESSION_BOOST,
    assemble_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode_row(
    *,
    session_id: str | None = None,
    source: str = "chat",
    type_: str = "message",
    text: str = "hello",
    minutes_ago: int = 0,
):
    """Create a fake episode row matching the ORM shape."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        source=source,
        type=type_,
        payload={"messages": [{"role": "user", "content": text}]},
        metadata_={},
        provenance={},
        session_id=session_id,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


@contextmanager
def _mock_repos(episodes, *, resolved_sessions=None):
    """Patch all repo calls used by assemble_context."""
    with (
        patch(
            "server.services.context.repo.search_memories", new_callable=AsyncMock, return_value=[]
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
            return_value=[],
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Session boost scoring
# ---------------------------------------------------------------------------


def test_session_boost_constant_is_significant():
    """Session boost should be large enough to override recency for same-bucket episodes."""
    # A non-session episode with max recency: _EPISODE_PRIORITY + _RECENCY_MAX = 8.0
    # A session episode with zero recency: _EPISODE_PRIORITY + 0 + _SESSION_BOOST = 9.0
    assert _SESSION_BOOST > _RECENCY_MAX, "Session boost must exceed max recency to matter"


def test_session_boost_value():
    assert _SESSION_BOOST == 6.0


# ---------------------------------------------------------------------------
# Full assembly — session-aware behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_episodes_boosted_in_context():
    """Episodes with matching session_id should appear in context even if older."""
    current_session = "sess-abc"
    old_session_ep = _make_episode_row(
        session_id=current_session, text="earlier in session", minutes_ago=60
    )
    recent_other_ep = _make_episode_row(session_id="sess-other", text="recent other", minutes_ago=1)

    with _mock_repos([old_session_ep, recent_other_ep]):
        result = await assemble_context(
            AsyncMock(), "user-1", "help with billing", max_tokens=4000, session_id=current_session
        )

    ep_ids = {str(e.id) for e in result.episodes}
    assert str(old_session_ep.id) in ep_ids
    assert str(recent_other_ep.id) in ep_ids


@pytest.mark.asyncio
async def test_session_grouping_in_assembled_text():
    """When session_id is provided, assembled text should have session headers."""
    current_session = "sess-123"
    ep_in_session = _make_episode_row(session_id=current_session, text="session msg", minutes_ago=5)
    ep_other = _make_episode_row(session_id="sess-old", text="old msg", minutes_ago=30)

    with _mock_repos([ep_in_session, ep_other]):
        result = await assemble_context(
            AsyncMock(), "user-1", "help user", max_tokens=4000, session_id=current_session
        )

    assert f"### Current session ({current_session})" in result.assembled_context
    assert "### Previous interactions" in result.assembled_context


@pytest.mark.asyncio
async def test_no_session_id_preserves_flat_rendering():
    """Without session_id, episodes render flat (no session headers)."""
    ep1 = _make_episode_row(session_id="sess-a", text="msg1", minutes_ago=5)
    ep2 = _make_episode_row(session_id="sess-b", text="msg2", minutes_ago=10)

    with _mock_repos([ep1, ep2]):
        result = await assemble_context(AsyncMock(), "user-1", "help user", max_tokens=4000)

    assert "### Current session" not in result.assembled_context
    assert "### Previous interactions" not in result.assembled_context
    assert "## Recent interactions" in result.assembled_context


@pytest.mark.asyncio
async def test_session_metadata_in_response():
    """Response should include session info for sessions represented."""
    ep1 = _make_episode_row(session_id="sess-a", text="msg1", minutes_ago=5)
    ep2 = _make_episode_row(session_id="sess-a", text="msg2", minutes_ago=3)
    ep3 = _make_episode_row(session_id="sess-b", text="msg3", minutes_ago=30)

    with _mock_repos([ep1, ep2, ep3]):
        result = await assemble_context(
            AsyncMock(), "user-1", "help user", max_tokens=4000, session_id="sess-a"
        )

    assert len(result.sessions) == 2
    sess_a = next(s for s in result.sessions if s.session_id == "sess-a")
    assert sess_a.episode_count == 2
    sess_b = next(s for s in result.sessions if s.session_id == "sess-b")
    assert sess_b.episode_count == 1


@pytest.mark.asyncio
async def test_token_budget_respected_with_session_boost():
    """Even with session boost, token budget must not be exceeded."""
    episodes = [
        _make_episode_row(session_id="sess-x", text="x" * 200, minutes_ago=i) for i in range(20)
    ]

    with _mock_repos(episodes):
        result = await assemble_context(
            AsyncMock(), "user-1", "help", max_tokens=200, session_id="sess-x"
        )

    assert result.token_estimate <= 200
    assert len(result.episodes) < 20


@pytest.mark.asyncio
async def test_no_session_episodes_still_works():
    """If no episodes have session_id set, everything still works."""
    ep1 = _make_episode_row(session_id=None, text="no session", minutes_ago=5)

    with _mock_repos([ep1]):
        result = await assemble_context(
            AsyncMock(), "user-1", "help", max_tokens=4000, session_id="sess-current"
        )

    assert len(result.episodes) == 1
    assert len(result.sessions) == 0


@pytest.mark.asyncio
async def test_resolved_session_episodes_penalized():
    """Episodes from resolved sessions should be deprioritized."""
    resolved_ep = _make_episode_row(
        session_id="sess-resolved", text="old billing problem fixed", minutes_ago=5
    )
    active_ep = _make_episode_row(
        session_id="sess-active", text="active login error", minutes_ago=10
    )

    with _mock_repos([resolved_ep, active_ep], resolved_sessions={"sess-resolved"}):
        result = await assemble_context(
            AsyncMock(), "user-1", "help", max_tokens=4000, session_id="sess-active"
        )

    # Both included (budget allows), but active session ep should be first
    assert len(result.episodes) == 2
    assert result.episodes[0].session_id == "sess-active"
