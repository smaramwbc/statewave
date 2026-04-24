"""Tests for webhooks service."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from server.services import webhooks


@pytest.fixture(autouse=True)
def _reset_webhook():
    """Reset webhook config between tests."""
    webhooks.configure(None)
    yield
    webhooks.configure(None)


async def test_fire_noop_when_no_url():
    """No error when webhook URL is not configured."""
    webhooks.configure(None)
    await webhooks.fire("episode.created", {"id": "123"})
    # Should silently do nothing


async def test_fire_creates_task_when_url_set():
    webhooks.configure("http://example.com/hook")
    with patch("server.services.webhooks._deliver", new_callable=AsyncMock) as mock_deliver:
        await webhooks.fire("episode.created", {"id": "123"})
        # Give the task a chance to run
        await asyncio.sleep(0.05)
        mock_deliver.assert_called_once()
        body = mock_deliver.call_args[0][0]
        assert body["event"] == "episode.created"
        assert body["data"]["id"] == "123"
        assert "timestamp" in body
