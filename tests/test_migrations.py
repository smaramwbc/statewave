"""Tests for migration safety utilities and endpoint."""

from __future__ import annotations


import pytest

from server.services.migrations import (
    EXPECTED_HEAD,
    MigrationStatus,
    _resolve_pending,
    check_migration_status,
    get_all_revisions,
)


class TestMigrationStatus:
    def test_compatible_summary(self):
        s = MigrationStatus(is_compatible=True, current_revision=EXPECTED_HEAD)
        assert s.summary == "Schema is up to date"
        assert not s.needs_migration

    def test_pending_summary(self):
        s = MigrationStatus(
            current_revision="0010",
            pending_count=2,
            pending_revisions=["0011", "0012_add_health_cache"],
        )
        assert "2 pending" in s.summary
        assert s.needs_migration

    def test_error_summary(self):
        s = MigrationStatus(error="connection refused")
        assert "ERROR" in s.summary


def test_get_all_revisions():
    """Verify we can introspect the migration chain."""
    revs = get_all_revisions()
    assert len(revs) == 14
    assert revs[0] == "0001"
    assert revs[-1] == EXPECTED_HEAD


@pytest.mark.asyncio
async def test_check_migration_status_no_url(monkeypatch):
    """Should return error when no DB URL is supplied via arg or env."""
    monkeypatch.delenv("STATEWAVE_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    status = await check_migration_status(database_url="")
    assert status.error is not None


def test_resolve_pending_at_head():
    """When current == head, is_compatible should be True."""
    status = MigrationStatus(current_revision=EXPECTED_HEAD)
    result = _resolve_pending(status)
    assert result.is_compatible
    assert result.pending_count == 0


def test_resolve_pending_behind():
    """When current is behind head, should list pending revisions."""
    status = MigrationStatus(current_revision="0010")
    result = _resolve_pending(status)
    assert not result.is_compatible
    assert result.pending_count == 4
    assert "0011" in result.pending_revisions
    assert EXPECTED_HEAD in result.pending_revisions


def test_resolve_pending_fresh_db():
    """When current is None, all revisions are pending."""
    status = MigrationStatus(current_revision=None)
    result = _resolve_pending(status)
    assert result.pending_count == 14


def test_resolve_pending_unknown_revision():
    """When current revision is unknown, should set error."""
    status = MigrationStatus(current_revision="unknown_rev_xyz")
    result = _resolve_pending(status)
    assert result.error is not None
    assert "not found" in result.error
