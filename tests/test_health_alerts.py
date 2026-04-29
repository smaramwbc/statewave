"""Tests for proactive health alerts via webhooks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.health import HealthFactor, HealthResult
from server.services.health_alerts import check_and_alert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _health(state: str, score: int, factors=None) -> HealthResult:
    return HealthResult(
        subject_id="user-1",
        score=score,
        state=state,
        factors=factors or [],
    )


def _cached(state: str, score: int):
    return SimpleNamespace(subject_id="user-1", last_state=state, last_score=score)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degraded_healthy_to_watch_fires_webhook():
    """healthy → watch should emit subject.health_degraded."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("healthy", 100),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("watch", 55))

    assert result == "subject.health_degraded"
    mock_fire.assert_called_once()
    call_args = mock_fire.call_args
    assert call_args[0][0] == "subject.health_degraded"
    payload = call_args[0][1]
    assert payload["previous_state"] == "healthy"
    assert payload["current_state"] == "watch"
    assert payload["score"] == 55


@pytest.mark.asyncio
async def test_degraded_watch_to_at_risk_fires_webhook():
    """watch → at_risk should emit."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("watch", 55),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("at_risk", 20))

    assert result == "subject.health_degraded"
    mock_fire.assert_called_once()
    payload = mock_fire.call_args[0][1]
    assert payload["previous_state"] == "watch"
    assert payload["current_state"] == "at_risk"


@pytest.mark.asyncio
async def test_degraded_healthy_to_at_risk_fires_webhook():
    """healthy → at_risk (direct jump) should emit."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("healthy", 100),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("at_risk", 15))

    assert result == "subject.health_degraded"
    payload = mock_fire.call_args[0][1]
    assert payload["previous_state"] == "healthy"
    assert payload["current_state"] == "at_risk"
    assert payload["score"] == 15


@pytest.mark.asyncio
async def test_unchanged_state_does_not_fire():
    """Same state should NOT emit a webhook."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("watch", 55),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("watch", 50))

    assert result is None
    mock_fire.assert_not_called()


@pytest.mark.asyncio
async def test_improved_at_risk_to_watch_fires():
    """at_risk → watch should emit subject.health_improved."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("at_risk", 20),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("watch", 55))

    assert result == "subject.health_improved"
    mock_fire.assert_called_once()
    payload = mock_fire.call_args[0][1]
    assert payload["previous_state"] == "at_risk"
    assert payload["current_state"] == "watch"
    assert payload["score"] == 55


@pytest.mark.asyncio
async def test_improved_watch_to_healthy_fires():
    """watch → healthy should emit subject.health_improved."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("watch", 55),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("healthy", 95))

    assert result == "subject.health_improved"
    payload = mock_fire.call_args[0][1]
    assert payload["previous_state"] == "watch"
    assert payload["current_state"] == "healthy"


@pytest.mark.asyncio
async def test_improved_at_risk_to_healthy_fires():
    """at_risk → healthy (direct jump) should emit subject.health_improved."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("at_risk", 15),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("healthy", 100))

    assert result == "subject.health_improved"
    payload = mock_fire.call_args[0][1]
    assert payload["previous_state"] == "at_risk"
    assert payload["current_state"] == "healthy"
    assert payload["score"] == 100


@pytest.mark.asyncio
async def test_first_time_unhealthy_fires():
    """No prior cache (new subject) going to watch should fire."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=None,  # No cached state — defaults to "healthy"
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("at_risk", 25))

    assert result == "subject.health_degraded"
    payload = mock_fire.call_args[0][1]
    assert payload["previous_state"] == "healthy"
    assert payload["current_state"] == "at_risk"


@pytest.mark.asyncio
async def test_first_time_healthy_does_not_fire():
    """New subject that's healthy should NOT fire."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("healthy", 100))

    assert result is None
    mock_fire.assert_not_called()


@pytest.mark.asyncio
async def test_payload_contains_factors():
    """Webhook payload should include top factors."""
    factors = [
        HealthFactor(signal="unresolved_issues", impact=-30, detail="2 open sessions"),
        HealthFactor(signal="repeated_issues", impact=-20, detail="Recurring pattern"),
        HealthFactor(signal="escalations", impact=-10, detail="1 urgency episode"),
        HealthFactor(signal="extra", impact=-5, detail="Should be excluded (>3)"),
    ]

    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("healthy", 100),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ),
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        await check_and_alert(AsyncMock(), _health("at_risk", 20, factors))

    payload = mock_fire.call_args[0][1]
    assert len(payload["factors"]) == 3  # Capped at 3
    assert payload["factors"][0]["signal"] == "unresolved_issues"
    assert payload["factors"][0]["impact"] == -30
    assert payload["factors"][0]["detail"] == "2 open sessions"
    assert "generated_at" in payload


@pytest.mark.asyncio
async def test_cache_updated_regardless_of_alert():
    """Cache should be updated even when no alert fires (unchanged state)."""
    with (
        patch(
            "server.services.health_alerts.repo.get_health_cache",
            new_callable=AsyncMock,
            return_value=_cached("watch", 55),
        ),
        patch(
            "server.services.health_alerts.repo.upsert_health_cache",
            new_callable=AsyncMock,
        ) as mock_upsert,
        patch(
            "server.services.health_alerts.webhooks.fire",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_fire,
    ):
        result = await check_and_alert(AsyncMock(), _health("watch", 50))

    assert result is None
    mock_fire.assert_not_called()
    mock_upsert.assert_called_once_with(
        mock_upsert.call_args[0][0],
        "user-1",
        "watch",
        50,
        tenant_id=None,
    )
