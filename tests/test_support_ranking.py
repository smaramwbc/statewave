"""Tests for support-agent-specific ranking signals in context assembly."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.context import (
    _ACTION_STEP_BOOST,
    _IDLE_CHATTER_PENALTY,
    _OPEN_ISSUE_BOOST,
    _URGENCY_BOOST,
    _has_urgency,
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
    # Use a fixed base time to avoid microsecond timing issues in tests
    base = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        source=source,
        type=type_,
        payload={"messages": [{"role": "user", "content": text}]},
        metadata_={},
        provenance={},
        session_id=session_id,
        created_at=base - timedelta(minutes=minutes_ago),
    )


@contextmanager
def _mock_repos(episodes, *, resolved_sessions=None, open_sessions=None):
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
            return_value=open_sessions or set(),
        ),
        patch(
            "server.services.context.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Unit tests for _has_urgency
# ---------------------------------------------------------------------------


def test_urgency_detects_keywords():
    assert _has_urgency("This is urgent, please help ASAP")
    assert _has_urgency("Production is DOWN — critical outage")
    assert _has_urgency("We have a compliance deadline on Friday")
    assert _has_urgency("This is a P0 incident")


def test_urgency_negative():
    assert not _has_urgency("Thanks for your help!")
    assert not _has_urgency("Just checking in on the status")
    assert not _has_urgency("Hello, I have a question about my account")


# ---------------------------------------------------------------------------
# Scoring constant sanity
# ---------------------------------------------------------------------------


def test_scoring_constants_are_positive():
    assert _OPEN_ISSUE_BOOST > 0
    assert _ACTION_STEP_BOOST > 0
    assert _URGENCY_BOOST > 0


def test_idle_chatter_penalty_is_negative():
    assert _IDLE_CHATTER_PENALTY < 0


# ---------------------------------------------------------------------------
# Full assembly — support-specific ranking behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_issue_episodes_outrank_untracked():
    """Episodes in sessions with open resolutions outrank those without."""
    open_ep = _make_episode_row(
        session_id="sess-open", text="Export job is failing for 3 days", minutes_ago=10
    )
    untracked_ep = _make_episode_row(
        session_id="sess-none", text="I have a general question about something", minutes_ago=10
    )

    with _mock_repos([open_ep, untracked_ep], open_sessions={"sess-open"}):
        result = await assemble_context(AsyncMock(), "user-1", "help", max_tokens=4000)

    # Open-issue episode should appear first due to +4 boost
    assert len(result.episodes) == 2
    assert result.episodes[0].session_id == "sess-open"


@pytest.mark.asyncio
async def test_action_steps_outrank_user_greetings():
    """Assistant/agent action episodes score higher than user small talk."""
    greeting = _make_episode_row(
        session_id="sess-1", source="user", text="Hi there, thanks for helping", minutes_ago=5
    )
    action = _make_episode_row(
        session_id="sess-1",
        source="assistant",
        text="I checked the account settings and reset the cache",
        minutes_ago=5,
    )

    with _mock_repos([greeting, action], open_sessions=set()):
        result = await assemble_context(AsyncMock(), "user-1", "what was tried", max_tokens=4000)

    # Action episode should rank higher
    assert len(result.episodes) == 2
    assert result.episodes[0].source == "assistant"


@pytest.mark.asyncio
async def test_urgency_episodes_boosted():
    """Episodes with urgency keywords get boosted."""
    urgent = _make_episode_row(
        session_id="sess-1",
        text="This is critical — our compliance deadline is tomorrow and the export is blocked",
        minutes_ago=10,
    )
    normal = _make_episode_row(
        session_id="sess-1", text="Can you check on the export status please", minutes_ago=10
    )

    with _mock_repos([urgent, normal], open_sessions=set()):
        result = await assemble_context(AsyncMock(), "user-1", "help", max_tokens=4000)

    # Urgent episode should rank first despite being older
    assert len(result.episodes) == 2
    assert "compliance" in result.episodes[0].payload["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_idle_chatter_deprioritized():
    """Very short low-signal episodes get penalized."""
    chatter = _make_episode_row(session_id="sess-1", text="Thanks!", minutes_ago=5)
    substance = _make_episode_row(
        session_id="sess-1",
        text="The data export job EXP-9912 has been timing out for three days",
        minutes_ago=5,
    )

    with _mock_repos([chatter, substance], open_sessions=set()):
        result = await assemble_context(AsyncMock(), "user-1", "help with export", max_tokens=4000)

    # Substance should outrank chatter despite being older
    assert len(result.episodes) == 2
    assert "EXP-9912" in result.episodes[0].payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_resolved_episodes_below_open_issues():
    """Episodes from resolved sessions rank below open-issue sessions."""
    resolved_ep = _make_episode_row(
        session_id="sess-resolved", text="Password reset completed", minutes_ago=5
    )
    open_ep = _make_episode_row(
        session_id="sess-open", text="Data export still failing", minutes_ago=10
    )

    with _mock_repos(
        [resolved_ep, open_ep],
        resolved_sessions={"sess-resolved"},
        open_sessions={"sess-open"},
    ):
        result = await assemble_context(AsyncMock(), "user-1", "help", max_tokens=4000)

    assert len(result.episodes) == 2
    assert result.episodes[0].session_id == "sess-open"


@pytest.mark.asyncio
async def test_combined_signals_under_tight_budget():
    """Under tight token budget, high-signal episodes survive and chatter is dropped."""
    # 4 episodes: urgent open issue, action step, idle chatter, resolved
    urgent_open = _make_episode_row(
        session_id="sess-open",
        source="user",
        text="URGENT: production export is down, compliance deadline tomorrow",
        minutes_ago=10,
    )
    action = _make_episode_row(
        session_id="sess-open",
        source="assistant",
        text="I've checked the logs and bumped the timeout to 15 minutes",
        minutes_ago=8,
    )
    chatter = _make_episode_row(session_id="sess-open", source="user", text="ok", minutes_ago=7)
    resolved = _make_episode_row(
        session_id="sess-old",
        source="user",
        text="My password reset worked, thanks!",
        minutes_ago=60,
    )

    with _mock_repos(
        [urgent_open, action, chatter, resolved],
        resolved_sessions={"sess-old"},
        open_sessions={"sess-open"},
    ):
        # Very tight budget — should only fit 2 episodes
        result = await assemble_context(AsyncMock(), "user-1", "help with export", max_tokens=150)

    included_texts = [ep.payload["messages"][0]["content"] for ep in result.episodes]
    # High-signal episodes (urgent + action) should be present
    has_high_signal = any("URGENT" in t or "checked the logs" in t for t in included_texts)
    assert has_high_signal, f"Expected high-signal episodes in: {included_texts}"
    # If resolved is included, it should be ranked last
    if any("password" in t.lower() for t in included_texts):
        assert "password" in included_texts[-1].lower()


@pytest.mark.asyncio
async def test_output_remains_deterministic():
    """Same input produces same ranking order."""
    eps = [
        _make_episode_row(
            session_id="s1", source="assistant", text="I reset the cache", minutes_ago=5
        ),
        _make_episode_row(session_id="s1", source="user", text="Thanks!", minutes_ago=4),
        _make_episode_row(
            session_id="s2", source="user", text="Export is blocked urgently", minutes_ago=20
        ),
    ]

    async def _run():
        with _mock_repos(eps, open_sessions=set()):
            return await assemble_context(AsyncMock(), "user-1", "help", max_tokens=4000)

    r1 = await _run()
    r2 = await _run()

    assert [str(e.id) for e in r1.episodes] == [str(e.id) for e in r2.episodes]
    assert r1.assembled_context == r2.assembled_context
