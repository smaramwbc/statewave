"""Tests for reliable webhook delivery service."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
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


async def test_fire_persists_tenant_id_without_mutating_payload():
    """Tenant-aware callers should get filterable queue rows without changing payload."""
    webhooks.configure("http://example.com/hook")

    payload = {"id": "123", "subject_id": "s1"}
    mock_session = MagicMock()

    event_id = await webhooks.fire(
        "episode.created", payload, db=mock_session, tenant_id="tenant-a"
    )

    assert event_id is not None
    mock_session.add.assert_called_once()
    row = mock_session.add.call_args[0][0]
    assert row.tenant_id == "tenant-a"
    assert row.payload["data"] == payload
    assert "tenant_id" not in row.payload["data"]


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


# ── Delivery failure classification ────────────────────────────────────────


class _FakeAsyncClient:
    def __init__(self, status_code: int):
        self._status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        return SimpleNamespace(status_code=self._status_code, text="response body")


def _webhook_row() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        event="episode.created",
        payload={"event": "episode.created", "data": {"id": "123"}},
        status="pending",
        attempts=0,
        max_attempts=5,
        next_attempt_at=None,
        last_attempt_at=None,
        last_error=None,
        http_status=None,
        delivered_at=None,
    )


async def test_permanent_client_error_dead_letters_without_retry(monkeypatch):
    """Permanent 4xx responses should not consume all retry attempts."""
    webhooks.configure("http://example.com/hook")
    row = _webhook_row()

    monkeypatch.setattr("httpx.AsyncClient", lambda timeout: _FakeAsyncClient(400))

    await webhooks._attempt_delivery(row, MagicMock())

    assert row.attempts == 1
    assert row.http_status == 400
    assert row.status == "dead_letter"
    assert row.next_attempt_at is None
    assert "HTTP 400" in row.last_error


async def test_retryable_client_error_still_schedules_retry(monkeypatch):
    """Rate-limit style 4xx responses should keep the existing retry path."""
    webhooks.configure("http://example.com/hook")
    row = _webhook_row()

    monkeypatch.setattr("httpx.AsyncClient", lambda timeout: _FakeAsyncClient(429))
    monkeypatch.setattr(webhooks, "_backoff_seconds", lambda attempt: 30)

    await webhooks._attempt_delivery(row, MagicMock())

    assert row.attempts == 1
    assert row.http_status == 429
    assert row.status == "pending"
    assert row.next_attempt_at is not None
    assert "HTTP 429" in row.last_error


# ── get_event_status tenant filter ─────────────────────────────────────────


class _FakeGetSession:
    """Async-context session whose .get() returns a fixed row."""

    def __init__(self, row):
        self._row = row

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, _model, _key):
        return self._row


def _status_row(tenant_id: str | None):
    import datetime as _dt

    return SimpleNamespace(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        event="episode.created",
        status="pending",
        attempts=0,
        max_attempts=5,
        last_attempt_at=None,
        next_attempt_at=None,
        last_error=None,
        http_status=None,
        created_at=_dt.datetime(2026, 6, 1, tzinfo=_dt.timezone.utc),
        delivered_at=None,
    )


async def test_get_event_status_returns_event_for_matching_tenant():
    row = _status_row("tenant-a")
    with patch(
        "server.services.webhooks.get_session_factory", return_value=lambda: _FakeGetSession(row)
    ):
        result = await webhooks.get_event_status(row.id, tenant_id="tenant-a")
    assert result is not None and result["id"] == str(row.id)


async def test_get_event_status_hides_event_from_other_tenant():
    # A tenant filter that doesn't match the row's tenant must read as not-found,
    # mirroring the list/purge filter semantics.
    row = _status_row("tenant-a")
    with patch(
        "server.services.webhooks.get_session_factory", return_value=lambda: _FakeGetSession(row)
    ):
        result = await webhooks.get_event_status(row.id, tenant_id="tenant-b")
    assert result is None


async def test_get_event_status_without_tenant_filter_returns_event():
    # No filter = operator global view (unchanged default behavior).
    row = _status_row("tenant-a")
    with patch(
        "server.services.webhooks.get_session_factory", return_value=lambda: _FakeGetSession(row)
    ):
        result = await webhooks.get_event_status(row.id)
    assert result is not None and result["id"] == str(row.id)


# ── purge_events (issue #279) ────────────────────────────────────────────────
#
# purge_events() is an operator-facing DESTRUCTIVE delete. These tests lock in
# its two safety guards (no-filter, terminal-status) and its filter scoping so a
# future refactor that drops a guard turns CI red.


class _CaptureSession:
    """Async-context session that records the statement passed to execute() and
    reports a fixed rowcount, so tests can assert what the DELETE targets
    without a live database."""

    def __init__(self, rowcount: int = 0):
        self.executed = None
        self.committed = False
        self._rowcount = rowcount

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, statement):
        self.executed = statement
        return SimpleNamespace(rowcount=self._rowcount)

    async def commit(self):
        self.committed = True


def _purge_sql(session) -> str:
    """The captured DELETE rendered with literal values inlined, for asserting
    on the WHERE clause."""
    from sqlalchemy.dialects import postgresql

    return str(
        session.executed.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


async def test_purge_events_requires_at_least_one_filter():
    """No-filter guard: purge_events() with no filters raises ValueError instead
    of wiping every delivered + dead-letter event in one call."""
    with pytest.raises(ValueError, match="at least one filter"):
        await webhooks.purge_events()


async def test_purge_events_rejects_non_terminal_status():
    """Terminal-status guard: 'pending' is rejected — those events may still be
    in flight in the delivery loop."""
    with pytest.raises(ValueError, match="must be one of"):
        await webhooks.purge_events(status="pending")


async def test_purge_events_status_filter_targets_only_that_status():
    """purge_events(status='delivered') deletes only delivered rows (dead_letter
    is left untouched) and returns the deleted row count."""
    session = _CaptureSession(rowcount=3)
    with patch(
        "server.services.webhooks.get_session_factory", return_value=lambda: session
    ):
        count = await webhooks.purge_events(status="delivered")

    assert count == 3
    assert session.committed
    sql = _purge_sql(session)
    assert "'delivered'" in sql
    assert "'dead_letter'" not in sql  # only the requested status is purged


async def test_purge_events_without_status_targets_all_terminal_only():
    """With only an event_type filter, both terminal statuses are in scope and
    'pending' never is."""
    session = _CaptureSession(rowcount=2)
    with patch(
        "server.services.webhooks.get_session_factory", return_value=lambda: session
    ):
        count = await webhooks.purge_events(event_type="episode.created")

    assert count == 2
    sql = _purge_sql(session)
    assert "'delivered'" in sql and "'dead_letter'" in sql
    assert "'pending'" not in sql
    assert "'episode.created'" in sql  # event_type filter applied


async def test_purge_events_tenant_filter_applied():
    """tenant_id filter scopes the delete to a single tenant."""
    session = _CaptureSession(rowcount=1)
    with patch(
        "server.services.webhooks.get_session_factory", return_value=lambda: session
    ):
        count = await webhooks.purge_events(tenant_id="tenant-a")

    assert count == 1
    assert "'tenant-a'" in _purge_sql(session)
