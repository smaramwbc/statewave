"""Tests for reliable webhook delivery service."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.services import webhooks


@pytest.fixture(autouse=True)
def _reset_webhook():
    """Reset webhook config between tests."""
    webhooks.configure(None)
    yield
    webhooks.configure(None)


async def test_fire_noop_when_no_url():
    """No event persisted when webhook URL is not configured."""
    webhooks.configure(None)
    result = await webhooks.fire("episode.created", {"id": "123"})
    assert result is None


async def test_fire_persists_event_when_url_set():
    """Event is persisted to DB when URL is configured."""
    webhooks.configure("http://example.com/hook")

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("server.services.webhooks.async_session_factory", return_value=mock_session):
        event_id = await webhooks.fire("episode.created", {"id": "123"})

    assert event_id is not None
    assert isinstance(event_id, uuid.UUID)
    mock_session.add.assert_called_once()
    row = mock_session.add.call_args[0][0]
    assert row.event == "episode.created"
    assert row.status == "pending"
    assert row.payload["data"]["id"] == "123"
    mock_session.commit.assert_called_once()


async def test_fire_uses_provided_session():
    """When a session is passed, event is added without commit (caller controls tx)."""
    webhooks.configure("http://example.com/hook")

    mock_session = MagicMock()
    event_id = await webhooks.fire("episode.created", {"id": "123"}, db=mock_session)

    assert event_id is not None
    mock_session.add.assert_called_once()
    # Should NOT commit — caller's responsibility
    mock_session.commit.assert_not_called()


def test_backoff_increases_exponentially():
    """Backoff schedule grows with attempt number."""
    b1 = webhooks._backoff_seconds(1)
    b2 = webhooks._backoff_seconds(2)
    b3 = webhooks._backoff_seconds(3)
    # With jitter (0.5-1.5), base is 30, 120, 480
    assert 15 <= b1 <= 45
    assert 60 <= b2 <= 180
    assert 240 <= b3 <= 720

