"""Tests for migration safety utilities and endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from server.services.migrations import (
    EXPECTED_HEAD,
    MigrationStatus,
    _resolve_pending,
    check_migration_status,
    get_all_revisions,
    get_script_directory,
)


@pytest.fixture(scope="module")
def embedding_model_migration():
    """The loaded 0032 revision module, via the same ScriptDirectory this
    file already uses to introspect revisions (never a raw import, the
    filename starts with a digit and is not a normal Python module path)."""
    return get_script_directory().get_revision("0032_memories_embedding_model").module


class TestConfiguredEmbeddingModel:
    """#421 backfill precedence: system_settings (global_db) before
    server.core.config.settings (env, already folding in .env). Rows are
    (key, value) tuples, matching what `dict(result.fetchall())` in the
    real function consumes."""

    def _bind(self, rows):
        bind = MagicMock()
        bind.execute.return_value.fetchall.return_value = rows
        return bind

    def test_no_overrides_falls_back_to_env_settings_stub(self, embedding_model_migration, monkeypatch):
        from server.core.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "stub")
        result = embedding_model_migration._configured_embedding_model(self._bind([]))
        assert result == "stub"

    def test_no_overrides_falls_back_to_env_settings_litellm(self, embedding_model_migration, monkeypatch):
        from server.core.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "litellm")
        monkeypatch.setattr(settings, "litellm_embedding_model", "text-embedding-3-small")
        result = embedding_model_migration._configured_embedding_model(self._bind([]))
        assert result == "text-embedding-3-small"

    def test_global_db_provider_override_wins_over_env(self, embedding_model_migration, monkeypatch):
        """Regression for the maintainer's review: an admin-UI-configured
        deployment (litellm set via system_settings, never via env/.env)
        must not backfill as "stub"."""
        from server.core.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "stub")
        monkeypatch.setattr(settings, "litellm_embedding_model", "text-embedding-3-small")
        rows = [
            ("embedding_provider", "litellm"),
            ("litellm_embedding_model", "admin-configured-model"),
        ]
        result = embedding_model_migration._configured_embedding_model(self._bind(rows))
        assert result == "admin-configured-model"

    def test_global_db_provider_override_with_env_model_fallback(
        self, embedding_model_migration, monkeypatch
    ):
        """Only the provider is overridden in system_settings; the model
        name still falls back to env/.env settings."""
        from server.core.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "stub")
        monkeypatch.setattr(settings, "litellm_embedding_model", "env-configured-model")
        rows = [("embedding_provider", "litellm")]
        result = embedding_model_migration._configured_embedding_model(self._bind(rows))
        assert result == "env-configured-model"

    def test_provider_none_returns_none(self, embedding_model_migration, monkeypatch):
        from server.core.config import settings

        monkeypatch.setattr(settings, "embedding_provider", "none")
        result = embedding_model_migration._configured_embedding_model(self._bind([]))
        assert result is None


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
    assert len(revs) == 32
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
    assert result.pending_count == 22
    assert "0011" in result.pending_revisions
    assert EXPECTED_HEAD in result.pending_revisions


def test_resolve_pending_fresh_db():
    """When current is None, all revisions are pending."""
    status = MigrationStatus(current_revision=None)
    result = _resolve_pending(status)
    assert result.pending_count == 32


def test_resolve_pending_unknown_revision():
    """When current revision is unknown, should set error."""
    status = MigrationStatus(current_revision="unknown_rev_xyz")
    result = _resolve_pending(status)
    assert result.error is not None
    assert "not found" in result.error
