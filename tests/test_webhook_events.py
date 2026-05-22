"""Tests for the webhook event-type filter (STATEWAVE_WEBHOOK_EVENTS).

Covers the `parse_webhook_event_filter` parser/validator and its
integration into `Settings`. The delivery-path behaviour (which events
actually reach the queue) is in `test_webhooks.py`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.core.config import Settings
from server.core.webhook_events import (
    KNOWN_WEBHOOK_EVENTS,
    parse_webhook_event_filter,
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_known_events_non_empty():
    """The vocabulary must list the event types the server emits."""
    assert KNOWN_WEBHOOK_EVENTS
    assert "memories.compiled" in KNOWN_WEBHOOK_EVENTS
    assert "subject.deleted" in KNOWN_WEBHOOK_EVENTS
    assert "subject.health_degraded" in KNOWN_WEBHOOK_EVENTS


# ---------------------------------------------------------------------------
# parse_webhook_event_filter
# ---------------------------------------------------------------------------

def test_parse_none_and_empty_mean_no_filter():
    assert parse_webhook_event_filter(None) == []
    assert parse_webhook_event_filter("") == []


def test_parse_single_event_string():
    assert parse_webhook_event_filter("memories.compiled") == ["memories.compiled"]


def test_parse_comma_separated_string():
    assert parse_webhook_event_filter("memories.compiled,subject.deleted") == [
        "memories.compiled",
        "subject.deleted",
    ]


def test_parse_strips_whitespace_and_blank_entries():
    assert parse_webhook_event_filter(" memories.compiled ,, subject.deleted ") == [
        "memories.compiled",
        "subject.deleted",
    ]


def test_parse_accepts_list_input_and_sorts():
    assert parse_webhook_event_filter(["subject.deleted", "memories.compiled"]) == [
        "memories.compiled",
        "subject.deleted",
    ]


def test_parse_deduplicates():
    assert parse_webhook_event_filter("memories.compiled,memories.compiled") == [
        "memories.compiled",
    ]


def test_parse_rejects_unknown_event():
    with pytest.raises(ValueError, match="unknown event type"):
        parse_webhook_event_filter("bogus.event")


def test_parse_rejects_unknown_mixed_with_known():
    """One bad entry rejects the whole filter — there is no partial apply."""
    with pytest.raises(ValueError, match="bogus.event"):
        parse_webhook_event_filter("memories.compiled,bogus.event")


def test_parse_rejects_wrong_type():
    with pytest.raises(ValueError, match="comma-separated string or a"):
        parse_webhook_event_filter(123)


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------

def test_settings_default_is_no_filter():
    """No STATEWAVE_WEBHOOK_EVENTS set → empty filter → deliver everything."""
    assert Settings(webhook_events="").webhook_event_filter == []


def test_settings_parses_comma_separated():
    settings = Settings(webhook_events="memories.compiled,subject.deleted")
    assert settings.webhook_event_filter == ["memories.compiled", "subject.deleted"]


def test_settings_rejects_unknown_event_at_construction():
    """An unknown event type fails fast — the server refuses to start."""
    with pytest.raises(ValidationError, match="unknown event type"):
        Settings(webhook_events="not.a.real.event")
