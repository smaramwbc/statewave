"""Integration test: the #421 `embedding_model` backfill migration (0032)
against a real, freshly-migrated Postgres database.

`tests/test_migrations.py::TestConfiguredEmbeddingModel` exercises
`_configured_embedding_model` in isolation with a mocked `bind`, which pins
the precedence logic but never runs the migration itself through Alembic:
a mock can't catch a bug in how that logic is actually wired into
`alembic upgrade head` (wrong revision order, a column/table name typo, a
JSONB round trip through `system_settings` that behaves differently than
a hand-built mock). This is the maintainer's ask on PR #449: a test that
runs `alembic upgrade head` against a seeded table, covering the
config-precedence fix (review issue 1) end to end.

Requires a live Postgres (`STATEWAVE_DATABASE_URL` / `DATABASE_URL`), same
as the rest of this file's suite once CI's "Run migrations" step has run.
Each test creates and drops its own throwaway database on that same
server, so it never touches the schema the surrounding unit tests read.

`alembic/env.py` re-resolves `STATEWAVE_DATABASE_URL`/`DATABASE_URL` from
the process environment on every run (`resolve_database_url()`, called
with no argument) rather than trusting whatever URL is already set on the
`Config` object handed to `command.upgrade`, so pointing a Python-driven
migration at the throwaway database means overriding the env var itself
for the duration of the call, not just `Config.set_main_option`.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest
from alembic import command
from sqlalchemy.engine.url import make_url

from server.services.migrations import get_alembic_config, resolve_database_url

# The revision immediately before 0032: seed data at this schema, then let
# the test upgrade to head so 0032 is the only migration under test.
_PRE_BACKFILL_REVISION = "0031_supersession_records"


def _require_live_db() -> str:
    url = resolve_database_url()
    if not url or "asyncpg" not in url:
        pytest.skip("no live Postgres resolved via STATEWAVE_DATABASE_URL/DATABASE_URL")
    return url


def _asyncpg_dsn(url: str, db_name: str) -> str:
    """Plain `postgresql://` DSN asyncpg's own `connect()` accepts (it
    rejects the SQLAlchemy `+asyncpg` driver suffix)."""
    parsed = make_url(url)
    return f"postgresql://{parsed.username}:{parsed.password}@{parsed.host}:{parsed.port}/{db_name}"


def _sqlalchemy_asyncpg_url(url: str, db_name: str) -> str:
    """`postgresql+asyncpg://` URL for `STATEWAVE_DATABASE_URL`, the form
    `alembic/env.py`'s `async_engine_from_config` requires."""
    parsed = make_url(url)
    return (
        f"postgresql+asyncpg://{parsed.username}:{parsed.password}"
        f"@{parsed.host}:{parsed.port}/{db_name}"
    )


def _run_alembic_against(base_url: str, db_name: str, revision: str) -> None:
    """Runs `command.upgrade` with `STATEWAVE_DATABASE_URL` pointed at the
    throwaway database for the duration of the call, then restores
    whatever was there before (env.py reads the env var itself, ignoring
    the Config object's own `sqlalchemy.url`)."""
    prior = os.environ.get("STATEWAVE_DATABASE_URL")
    os.environ["STATEWAVE_DATABASE_URL"] = _sqlalchemy_asyncpg_url(base_url, db_name)
    try:
        cfg = get_alembic_config()
        # `alembic/env.py` runs `fileConfig(config.config_file_name)` when
        # this is set, which is meant for one-shot CLI invocations. Inside a
        # long-lived pytest process it calls `logging.config.fileConfig`
        # with its default `disable_existing_loggers=True`, which disables
        # every logger not named in alembic.ini's `[loggers]` section,
        # process-wide. That silently broke `caplog` for every
        # warning-log test collected after this one the first time this
        # file ran as part of the full suite (`test_embeddings.py::
        # test_stub_provider_logs_loud_warning_at_first_use` among them).
        # Clearing it here is a no-op for the migration itself, which needs
        # no CLI log formatting.
        cfg.config_file_name = None
        command.upgrade(cfg, revision)
    finally:
        if prior is None:
            os.environ.pop("STATEWAVE_DATABASE_URL", None)
        else:
            os.environ["STATEWAVE_DATABASE_URL"] = prior


@pytest.fixture
def migration_db_name():
    """Creates a throwaway database and migrates it to the revision just
    before 0032, then yields its name for the test body to seed and
    finish upgrading. `command.upgrade` is driven from plain sync code,
    never from inside an `async def` test: Alembic's own `env.py`
    drives its async engine with `asyncio.run`, which raises if a loop is
    already running in this thread."""
    base_url = _require_live_db()
    db_name = f"statewave_migtest_{uuid.uuid4().hex[:12]}"

    async def _create():
        conn = await asyncpg.connect(dsn=_asyncpg_dsn(base_url, "postgres"))
        try:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
        finally:
            await conn.close()

    asyncio.run(_create())

    _run_alembic_against(base_url, db_name, _PRE_BACKFILL_REVISION)

    yield db_name

    async def _drop():
        conn = await asyncpg.connect(dsn=_asyncpg_dsn(base_url, "postgres"))
        try:
            await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await conn.close()

    asyncio.run(_drop())


def _seed_and_upgrade(base_url: str, db_name: str, *, system_settings_rows, embedding):
    """Seeds `system_settings` overrides plus one `memories` row at the
    pre-0032 schema, runs the real migration to head, and returns the
    resulting `embedding_model` value for that row."""
    memory_id = uuid.uuid4()

    async def _seed():
        conn = await asyncpg.connect(dsn=_asyncpg_dsn(base_url, db_name))
        try:
            for key, value, category in system_settings_rows:
                await conn.execute(
                    "INSERT INTO system_settings (key, value, category) VALUES ($1, $2::jsonb, $3)",
                    key,
                    value,
                    category,
                )
            await conn.execute(
                """
                INSERT INTO memories (
                    id, subject_id, kind, content, source_episode_ids, metadata, embedding
                ) VALUES ($1, $2, $3, $4, '{}', '{}'::jsonb, $5)
                """,
                memory_id,
                "test-subject",
                "profile_fact",
                "seeded row for the #421 migration test",
                str(embedding),
            )
        finally:
            await conn.close()

    asyncio.run(_seed())

    _run_alembic_against(base_url, db_name, "head")

    async def _read():
        conn = await asyncpg.connect(dsn=_asyncpg_dsn(base_url, db_name))
        try:
            return await conn.fetchval(
                "SELECT embedding_model FROM memories WHERE id = $1", memory_id
            )
        finally:
            await conn.close()

    return asyncio.run(_read())


def test_system_settings_override_wins_over_env_default(migration_db_name):
    """Review issue 1: an admin-UI-configured deployment (litellm set via
    `system_settings`, never via env/`.env`) must backfill the model it is
    actually running, not the pydantic `Settings` default ("stub"). This
    was the concrete Helm/`.env`-install failure mode the maintainer
    described: the migration used to read `os.environ` only and stamp
    every such row "stub", turning the whole corpus into a false positive
    mismatch."""
    base_url = resolve_database_url()
    result = _seed_and_upgrade(
        base_url,
        migration_db_name,
        system_settings_rows=[
            ("embedding_provider", '"litellm"', "embeddings"),
            ("litellm_embedding_model", '"admin-configured-model"', "embeddings"),
        ],
        embedding=[0.01] * 1536,
    )
    assert result == "admin-configured-model"


def test_unrecognized_provider_leaves_the_column_null(migration_db_name, monkeypatch):
    """Where the effective provider genuinely cannot be attributed to a
    live model ("none", or anything unrecognized), the migration must
    leave `embedding_model` NULL rather than guess. NULL is exempt from
    mismatch detection by design, and a guessed value is worse than no
    value, per the maintainer's review."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "embedding_provider", "none")

    base_url = resolve_database_url()
    result = _seed_and_upgrade(
        base_url,
        migration_db_name,
        system_settings_rows=[],
        embedding=[0.02] * 1536,
    )
    assert result is None
