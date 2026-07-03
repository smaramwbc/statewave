"""Unit tests for tenant scoping logic (no DB required).

Tests the _tenant_filter helper and verifies that repository function
signatures accept tenant_id correctly.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from server.db.repositories import _tenant_filter, get_episodes_by_ids
from server.db.tables import EpisodeRow, MemoryRow
from server.core.dependencies import get_tenant_id


class _ScalarRows:
    def all(self):
        return []


class _ExecuteResult:
    def scalars(self):
        return _ScalarRows()


class _CaptureSession:
    def __init__(self):
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _ExecuteResult()


class TestTenantFilter:
    """Tests for the _tenant_filter helper."""

    def test_none_tenant_returns_stmt_unchanged(self):
        """When tenant_id is None, statement is not modified."""
        stmt = MagicMock()
        result = _tenant_filter(stmt, EpisodeRow.tenant_id, None)
        assert result is stmt
        stmt.where.assert_not_called()

    def test_tenant_id_adds_where_clause(self):
        """When tenant_id is set, a where clause is added."""
        stmt = MagicMock()
        stmt.where.return_value = stmt
        _tenant_filter(stmt, EpisodeRow.tenant_id, "tenant-a")
        stmt.where.assert_called_once()


@pytest.mark.asyncio
async def test_get_episodes_by_ids_applies_tenant_filter():
    session = _CaptureSession()

    await get_episodes_by_ids(session, [uuid.uuid4()], tenant_id="tenant-a")

    compiled = session.statement.compile(dialect=postgresql.dialect())
    assert "episodes.tenant_id =" in str(compiled)
    assert "tenant-a" in compiled.params.values()


class TestGetTenantIdDependency:
    """Tests for the FastAPI tenant dependency."""

    def test_returns_none_when_no_tenant(self):
        """Without tenant in state, returns None."""
        request = MagicMock()
        request.state.tenant_id = None
        assert get_tenant_id(request) is None

    def test_returns_tenant_id_from_state(self):
        """Returns tenant_id from request state."""
        request = MagicMock()
        request.state.tenant_id = "org-123"
        assert get_tenant_id(request) == "org-123"

    def test_returns_none_when_attr_missing(self):
        """Handles missing attribute gracefully."""
        request = MagicMock(spec=[])
        del request.state
        # get_tenant_id uses getattr with default
        request.state = MagicMock(spec=[])
        assert get_tenant_id(request) is None


class TestEpisodeRowHasTenantId:
    """Verify ORM model has tenant_id field."""

    def test_episode_row_has_tenant_id(self):
        row = EpisodeRow(
            subject_id="test",
            tenant_id="t1",
            source="test",
            type="msg",
            payload={},
        )
        assert row.tenant_id == "t1"

    def test_memory_row_has_tenant_id(self):
        row = MemoryRow(
            subject_id="test",
            tenant_id="t1",
            kind="profile_fact",
            content="test",
        )
        assert row.tenant_id == "t1"

    def test_tenant_id_defaults_none(self):
        row = EpisodeRow(
            subject_id="test",
            source="test",
            type="msg",
            payload={},
        )
        assert row.tenant_id is None
