"""Tests for SLA tracking service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.sla import compute_sla


def _ep(source: str, session_id: str, created_at: datetime):
    return SimpleNamespace(
        source=source, session_id=session_id, created_at=created_at
    )


def _resolution(session_id: str, status: str, resolved_at: datetime | None = None):
    return SimpleNamespace(session_id=session_id, status=status, resolved_at=resolved_at)


NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_first_response_time_computed():
    t0 = NOW
    t1 = NOW + timedelta(minutes=2)
    episodes = [_ep("user", "s1", t0), _ep("assistant", "s1", t1)]
    resolutions = [_resolution("s1", "resolved", t1)]

    with patch("server.services.sla.repo") as mock_repo:
        mock_repo.list_episodes_by_subject = AsyncMock(return_value=episodes)
        mock_repo.list_resolutions = AsyncMock(return_value=resolutions)
        result = await compute_sla(AsyncMock(), "sub1")

    assert result.sessions[0].first_response_seconds == 120.0
    assert result.sessions[0].first_response_breached is False


@pytest.mark.asyncio
async def test_first_response_breach():
    t0 = NOW
    t1 = NOW + timedelta(minutes=10)
    episodes = [_ep("user", "s1", t0), _ep("agent", "s1", t1)]
    resolutions = []

    with patch("server.services.sla.repo") as mock_repo:
        mock_repo.list_episodes_by_subject = AsyncMock(return_value=episodes)
        mock_repo.list_resolutions = AsyncMock(return_value=resolutions)
        result = await compute_sla(AsyncMock(), "sub1")

    assert result.sessions[0].first_response_breached is True
    assert result.first_response_breach_count == 1


@pytest.mark.asyncio
async def test_resolution_time_computed():
    t0 = NOW
    t1 = NOW + timedelta(minutes=1)
    t2 = NOW + timedelta(hours=2)
    episodes = [_ep("user", "s1", t0), _ep("assistant", "s1", t1)]
    resolutions = [_resolution("s1", "resolved", t2)]

    with patch("server.services.sla.repo") as mock_repo:
        mock_repo.list_episodes_by_subject = AsyncMock(return_value=episodes)
        mock_repo.list_resolutions = AsyncMock(return_value=resolutions)
        result = await compute_sla(AsyncMock(), "sub1")

    assert result.sessions[0].resolution_seconds == 7200.0
    assert result.sessions[0].resolution_breached is False
    assert result.resolved_sessions == 1


@pytest.mark.asyncio
async def test_resolution_breach():
    t0 = NOW
    t1 = NOW + timedelta(minutes=1)
    t2 = NOW + timedelta(hours=30)
    episodes = [_ep("user", "s1", t0), _ep("assistant", "s1", t1)]
    resolutions = [_resolution("s1", "resolved", t2)]

    with patch("server.services.sla.repo") as mock_repo:
        mock_repo.list_episodes_by_subject = AsyncMock(return_value=episodes)
        mock_repo.list_resolutions = AsyncMock(return_value=resolutions)
        result = await compute_sla(AsyncMock(), "sub1")

    assert result.sessions[0].resolution_breached is True
    assert result.resolution_breach_count == 1


@pytest.mark.asyncio
async def test_open_session_no_resolution_seconds():
    t0 = NOW
    t1 = NOW + timedelta(minutes=1)
    episodes = [_ep("user", "s1", t0), _ep("assistant", "s1", t1)]
    resolutions = [_resolution("s1", "open", None)]

    with patch("server.services.sla.repo") as mock_repo:
        mock_repo.list_episodes_by_subject = AsyncMock(return_value=episodes)
        mock_repo.list_resolutions = AsyncMock(return_value=resolutions)
        result = await compute_sla(AsyncMock(), "sub1")

    assert result.sessions[0].resolution_seconds is None
    assert result.sessions[0].open_duration_seconds is not None
    assert result.open_sessions == 1


@pytest.mark.asyncio
async def test_multiple_sessions_averaged():
    t0 = NOW
    eps = [
        _ep("user", "s1", t0),
        _ep("assistant", "s1", t0 + timedelta(minutes=2)),
        _ep("user", "s2", t0),
        _ep("assistant", "s2", t0 + timedelta(minutes=4)),
    ]
    resolutions = [
        _resolution("s1", "resolved", t0 + timedelta(hours=1)),
        _resolution("s2", "resolved", t0 + timedelta(hours=3)),
    ]

    with patch("server.services.sla.repo") as mock_repo:
        mock_repo.list_episodes_by_subject = AsyncMock(return_value=eps)
        mock_repo.list_resolutions = AsyncMock(return_value=resolutions)
        result = await compute_sla(AsyncMock(), "sub1")

    assert result.total_sessions == 2
    assert result.avg_first_response_seconds == 180.0  # (120 + 240) / 2
    assert result.avg_resolution_seconds == 7200.0  # (3600 + 10800) / 2


@pytest.mark.asyncio
async def test_no_episodes_returns_empty():
    with patch("server.services.sla.repo") as mock_repo:
        mock_repo.list_episodes_by_subject = AsyncMock(return_value=[])
        mock_repo.list_resolutions = AsyncMock(return_value=[])
        result = await compute_sla(AsyncMock(), "sub1")

    assert result.total_sessions == 0
    assert result.sessions == []


@pytest.mark.asyncio
async def test_custom_thresholds():
    t0 = NOW
    t1 = NOW + timedelta(minutes=2)
    episodes = [_ep("user", "s1", t0), _ep("assistant", "s1", t1)]
    resolutions = []

    with patch("server.services.sla.repo") as mock_repo:
        mock_repo.list_episodes_by_subject = AsyncMock(return_value=episodes)
        mock_repo.list_resolutions = AsyncMock(return_value=resolutions)
        # 1 minute threshold — should breach
        result = await compute_sla(
            AsyncMock(), "sub1", first_response_threshold=timedelta(minutes=1)
        )

    assert result.sessions[0].first_response_breached is True
