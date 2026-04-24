"""Tests for memory conflict resolution."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.services.conflicts import _are_conflicting, resolve_conflicts
from server.db.tables import MemoryRow


def _make_memory(
    content: str = "test",
    kind: str = "profile_fact",
    status: str = "active",
    created_at: datetime | None = None,
) -> MemoryRow:
    return MemoryRow(
        id=uuid.uuid4(),
        subject_id="user-1",
        kind=kind,
        content=content,
        summary=content[:200],
        confidence=0.8,
        valid_from=created_at or datetime.now(timezone.utc),
        source_episode_ids=[uuid.uuid4()],
        metadata_={},
        status=status,
    )


def test_are_conflicting_high_overlap():
    a = _make_memory("my name is Alice Chen")
    b = _make_memory("my name is Alice Chen-Smith")
    # "my name is Alice" overlap is high
    assert _are_conflicting(a, b) is True


def test_are_conflicting_no_overlap():
    a = _make_memory("my name is Alice")
    b = _make_memory("I work at Globex Corporation")
    assert _are_conflicting(a, b) is False


def test_are_conflicting_empty_content():
    a = _make_memory("")
    b = _make_memory("something")
    assert _are_conflicting(a, b) is False


def test_are_conflicting_same_content():
    a = _make_memory("I prefer email notifications")
    b = _make_memory("I prefer email notifications")
    assert _are_conflicting(a, b) is True


def test_are_conflicting_stricter_for_non_facts():
    a = _make_memory("The user asked about billing", kind="episode_summary")
    b = _make_memory("The user asked about billing issues", kind="episode_summary")
    # episode_summary has stricter threshold (0.8)
    # These share most words but threshold is higher
    assert _are_conflicting(a, b) is True  # still high overlap


async def test_resolve_conflicts_marks_older_superseded():
    now = datetime.now(timezone.utc)
    older = _make_memory("my name is Alice", created_at=now - timedelta(days=5))
    newer = _make_memory("my name is Alice Chen", created_at=now)

    with patch("server.services.conflicts.repo") as mock_repo:
        mock_repo.list_active_memories_by_subject = AsyncMock(return_value=[older, newer])
        mock_repo.mark_memories_superseded = AsyncMock()

        session = AsyncMock()
        result = await resolve_conflicts(session, "user-1")

    assert older.id in result
    assert newer.id not in result
    mock_repo.mark_memories_superseded.assert_called_once()


async def test_resolve_conflicts_no_conflicts():
    a = _make_memory("my name is Alice")
    b = _make_memory("I work at Globex Corporation")

    with patch("server.services.conflicts.repo") as mock_repo:
        mock_repo.list_active_memories_by_subject = AsyncMock(return_value=[a, b])
        mock_repo.mark_memories_superseded = AsyncMock()

        session = AsyncMock()
        result = await resolve_conflicts(session, "user-1")

    assert result == []
    mock_repo.mark_memories_superseded.assert_not_called()
