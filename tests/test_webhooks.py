"""Tests for reliable webhook delivery service."""

from __future__ import annotations

import uuid
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

    def mock_factory():
        return mock_session

    with patch("server.services.webhooks.get_session_factory", return_value=mock_factory):
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


# ── Event-type filter (STATEWAVE_WEBHOOK_EVENTS) ───────────────────────────

# Every event type the server emits — kept in lockstep with
# server.core.webhook_events.KNOWN_WEBHOOK_EVENTS.
ALL_EVENT_TYPES = [
    "episode.created",
    "episodes.batch_created",
    "memories.compiled",
    "subject.deleted",
    "subject.health_degraded",
    "subject.health_improved",
]


async def test_fire_no_filter_delivers_all_event_types():
    """With no filter configured, every event type is enqueued."""
    webhooks.configure("http://example.com/hook")  # events defaults to None
    assert webhooks.event_filter() == frozenset()

    for event in ALL_EVENT_TYPES:
        session = MagicMock()
        result = await webhooks.fire(event, {"x": 1}, db=session)
        assert result is not None, f"{event} should be enqueued"
        session.add.assert_called_once()


async def test_fire_single_event_filter():
    """A one-event filter enqueues that type and drops every other."""
    webhooks.configure("http://example.com/hook", events=["memories.compiled"])

    allowed = MagicMock()
    assert await webhooks.fire("memories.compiled", {}, db=allowed) is not None
    allowed.add.assert_called_once()

    for blocked in ("episode.created", "subject.deleted", "subject.health_degraded"):
        session = MagicMock()
        assert await webhooks.fire(blocked, {}, db=session) is None
        session.add.assert_not_called()


async def test_fire_multiple_event_filter():
    """A multi-event filter enqueues exactly the listed types."""
    allowed_events = ["memories.compiled", "subject.deleted"]
    webhooks.configure("http://example.com/hook", events=allowed_events)

    for event in allowed_events:
        session = MagicMock()
        assert await webhooks.fire(event, {}, db=session) is not None
        session.add.assert_called_once()

    for blocked in ("episode.created", "episodes.batch_created", "subject.health_improved"):
        session = MagicMock()
        assert await webhooks.fire(blocked, {}, db=session) is None
        session.add.assert_not_called()


async def test_fire_filtered_event_is_not_persisted():
    """A filtered-out event never touches the delivery queue."""
    webhooks.configure("http://example.com/hook", events=["memories.compiled"])
    session = MagicMock()
    result = await webhooks.fire("subject.deleted", {"subject_id": "u1"}, db=session)
    assert result is None
    session.add.assert_not_called()
    session.commit.assert_not_called()


async def test_filter_decision_is_event_type_only():
    """Tenant isolation: the filter keys on event type alone, never on
    payload contents. Payload data — including any tenant/subject id —
    can neither smuggle a blocked event through nor block an allowed one,
    so the filter introduces no cross-tenant coupling."""
    webhooks.configure("http://example.com/hook", events=["memories.compiled"])

    # An allowed event is enqueued whichever tenant/subject the payload names.
    for tenant in ("acme", "globex", None):
        session = MagicMock()
        payload = {"subject_id": "s1", "tenant_id": tenant}
        assert await webhooks.fire("memories.compiled", payload, db=session) is not None
        session.add.assert_called_once()

    # A blocked event stays blocked whatever the payload contains.
    for tenant in ("acme", "globex", None):
        session = MagicMock()
        payload = {"subject_id": "s1", "tenant_id": tenant}
        assert await webhooks.fire("subject.deleted", payload, db=session) is None
        session.add.assert_not_called()


async def test_fire_noop_without_url_even_with_filter():
    """No URL short-circuits before the filter — nothing is enqueued."""
    webhooks.configure(None, events=["memories.compiled"])
    assert await webhooks.fire("memories.compiled", {}, db=MagicMock()) is None


def test_event_filter_accessor_reflects_configuration():
    webhooks.configure("http://example.com/hook", events=["subject.deleted"])
    assert webhooks.event_filter() == frozenset({"subject.deleted"})
    # Reconfiguring without events clears the filter.
    webhooks.configure("http://example.com/hook")
    assert webhooks.event_filter() == frozenset()
