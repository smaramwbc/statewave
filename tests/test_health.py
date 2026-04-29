"""Tests for customer health scoring."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.health import compute_health


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc)


def _make_resolution(
    session_id: str,
    status: str = "open",
    *,
    resolved_at: datetime | None = None,
):
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        session_id=session_id,
        status=status,
        resolution_summary="Fixed it" if status == "resolved" else None,
        resolved_at=resolved_at,
        metadata_={},
        created_at=_NOW - timedelta(days=5),
        updated_at=_NOW - timedelta(days=1),
    )


def _make_episode(
    session_id: str,
    text: str = "hello",
    *,
    days_ago: int = 0,
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
        created_at=_NOW - timedelta(days=days_ago),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_history_is_healthy():
    """Subject with no episodes or resolutions should be healthy (score 100)."""
    with (
        patch(
            "server.services.health.repo.list_resolutions", new_callable=AsyncMock, return_value=[]
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    assert result.score == 100
    assert result.state == "healthy"
    assert result.factors == []


@pytest.mark.asyncio
async def test_unresolved_issues_worsen_health():
    """Open/unresolved sessions should reduce score."""
    resolutions = [
        _make_resolution("sess-1", "open"),
        _make_resolution("sess-2", "open"),
    ]
    episodes = [
        _make_episode("sess-1", "need help"),
        _make_episode("sess-2", "still broken"),
    ]

    with (
        patch(
            "server.services.health.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=resolutions,
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes,
        ),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    assert result.score < 100
    assert result.state in ("watch", "at_risk")
    signals = [f.signal for f in result.factors]
    assert "unresolved_issues" in signals


@pytest.mark.asyncio
async def test_repeated_issues_worsen_health():
    """Open issue matching prior resolved issue keywords should trigger repeated_issues."""
    resolutions = [
        _make_resolution("sess-old", "resolved", resolved_at=_NOW - timedelta(days=30)),
        _make_resolution("sess-new", "open"),
    ]
    episodes = [
        _make_episode("sess-old", "payment gateway timeout error", days_ago=30),
        _make_episode("sess-new", "payment gateway timeout happening again", days_ago=0),
    ]

    with (
        patch(
            "server.services.health.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=resolutions,
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes,
        ),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    signals = [f.signal for f in result.factors]
    assert "repeated_issues" in signals
    assert result.score < 80  # Significant penalty


@pytest.mark.asyncio
async def test_recently_resolved_accounts_score_better():
    """Account with recent resolution should score better than one with unresolved recurring issues."""
    # Good account: all resolved recently
    good_resolutions = [
        _make_resolution("sess-1", "resolved", resolved_at=_NOW - timedelta(days=2)),
        _make_resolution("sess-2", "resolved", resolved_at=_NOW - timedelta(days=5)),
    ]
    good_episodes = [
        _make_episode("sess-1", "login issue", days_ago=3),
        _make_episode("sess-2", "billing question", days_ago=6),
    ]

    # Bad account: unresolved recurring
    bad_resolutions = [
        _make_resolution("sess-a", "resolved", resolved_at=_NOW - timedelta(days=30)),
        _make_resolution("sess-b", "open"),
    ]
    bad_episodes = [
        _make_episode("sess-a", "payment failed at checkout", days_ago=30),
        _make_episode("sess-b", "payment failed at checkout again", days_ago=0),
    ]

    with (
        patch(
            "server.services.health.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=good_resolutions,
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=good_episodes,
        ),
    ):
        good_result = await compute_health(AsyncMock(), "user-good")

    with (
        patch(
            "server.services.health.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=bad_resolutions,
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=bad_episodes,
        ),
    ):
        bad_result = await compute_health(AsyncMock(), "user-bad")

    assert good_result.score > bad_result.score
    assert good_result.state == "healthy"
    assert bad_result.state in ("watch", "at_risk")


@pytest.mark.asyncio
async def test_deterministic_scoring():
    """Same inputs should always produce same output."""
    resolutions = [
        _make_resolution("sess-1", "open"),
        _make_resolution("sess-2", "resolved", resolved_at=_NOW - timedelta(days=3)),
    ]
    episodes = [
        _make_episode("sess-1", "urgent issue blocker", days_ago=1),
        _make_episode("sess-2", "billing question", days_ago=5),
    ]

    results = []
    for _ in range(3):
        with (
            patch(
                "server.services.health.repo.list_resolutions",
                new_callable=AsyncMock,
                return_value=resolutions,
            ),
            patch(
                "server.services.health.repo.list_episodes_by_subject",
                new_callable=AsyncMock,
                return_value=episodes,
            ),
        ):
            results.append(await compute_health(AsyncMock(), "user-1"))

    assert all(r.score == results[0].score for r in results)
    assert all(r.state == results[0].state for r in results)
    assert all(len(r.factors) == len(results[0].factors) for r in results)


@pytest.mark.asyncio
async def test_explainable_output():
    """Factors should have signal, impact, and detail."""
    resolutions = [_make_resolution("sess-1", "open")]
    episodes = [_make_episode("sess-1", "urgent help needed", days_ago=0)]

    with (
        patch(
            "server.services.health.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=resolutions,
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes,
        ),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    assert len(result.factors) > 0
    for factor in result.factors:
        assert factor.signal
        assert factor.impact != 0
        assert factor.detail


@pytest.mark.asyncio
async def test_escalation_penalty():
    """Episodes with urgency keywords should trigger escalation penalty."""
    resolutions = []
    episodes = [
        _make_episode("sess-1", "this is urgent and critical", days_ago=0),
        _make_episode("sess-1", "we have an outage blocker", days_ago=0),
    ]

    with (
        patch(
            "server.services.health.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=resolutions,
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes,
        ),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    signals = [f.signal for f in result.factors]
    assert "escalations" in signals
    assert result.score < 100


@pytest.mark.asyncio
async def test_idle_open_issue_penalty():
    """Open issue with no activity for 7+ days should be penalized."""
    resolutions = [_make_resolution("sess-1", "open")]
    episodes = [_make_episode("sess-1", "need help", days_ago=10)]

    with (
        patch(
            "server.services.health.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=resolutions,
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes,
        ),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    signals = [f.signal for f in result.factors]
    assert "idle_open_issue" in signals


@pytest.mark.asyncio
async def test_high_resolution_rate_bonus():
    """Account with >80% resolution rate should get bonus."""
    resolutions = [
        _make_resolution("sess-1", "resolved", resolved_at=_NOW - timedelta(days=20)),
        _make_resolution("sess-2", "resolved", resolved_at=_NOW - timedelta(days=15)),
        _make_resolution("sess-3", "resolved", resolved_at=_NOW - timedelta(days=10)),
    ]
    episodes = [
        _make_episode("sess-1", "q1", days_ago=20),
        _make_episode("sess-2", "q2", days_ago=15),
        _make_episode("sess-3", "q3", days_ago=10),
    ]

    with (
        patch(
            "server.services.health.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=resolutions,
        ),
        patch(
            "server.services.health.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes,
        ),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    signals = [f.signal for f in result.factors]
    assert "high_resolution_rate" in signals
    assert result.state == "healthy"


# ---------------------------------------------------------------------------
# SLA-based health signals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sla_resolution_breach_penalizes_health():
    """A session resolved after 30h should trigger sla_resolution_breaches penalty."""
    t0 = _NOW - timedelta(hours=30)
    resolution = _make_resolution("s1", "resolved", resolved_at=_NOW)
    # Episode from user 30h before resolution
    ep = SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="user",
        type="message",
        payload={"messages": [{"role": "user", "content": "help"}]},
        metadata_={},
        provenance={},
        session_id="s1",
        created_at=t0,
    )

    with (
        patch("server.services.health.repo.list_resolutions", new_callable=AsyncMock, return_value=[resolution]),
        patch("server.services.health.repo.list_episodes_by_subject", new_callable=AsyncMock, return_value=[ep]),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    signals = [f.signal for f in result.factors]
    assert "sla_resolution_breaches" in signals
    breach_factor = next(f for f in result.factors if f.signal == "sla_resolution_breaches")
    assert breach_factor.impact == -10


@pytest.mark.asyncio
async def test_slow_first_response_penalizes_health():
    """Avg first response > 10 min should trigger slow_first_response penalty."""
    t0 = _NOW - timedelta(hours=1)
    resolution = _make_resolution("s1", "resolved", resolved_at=_NOW)
    ep_user = SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="user",
        type="message",
        payload={"messages": [{"role": "user", "content": "help"}]},
        metadata_={},
        provenance={},
        session_id="s1",
        created_at=t0,
    )
    ep_agent = SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="assistant",
        type="message",
        payload={"messages": [{"role": "assistant", "content": "hi"}]},
        metadata_={},
        provenance={},
        session_id="s1",
        created_at=t0 + timedelta(minutes=15),  # 15 min response
    )

    with (
        patch("server.services.health.repo.list_resolutions", new_callable=AsyncMock, return_value=[resolution]),
        patch("server.services.health.repo.list_episodes_by_subject", new_callable=AsyncMock, return_value=[ep_user, ep_agent]),
    ):
        result = await compute_health(AsyncMock(), "user-1")

    signals = [f.signal for f in result.factors]
    assert "slow_first_response" in signals
