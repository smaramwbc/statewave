"""Migration safety utilities.

Shared logic for preflight checks, startup guards, and admin endpoints.
Provides schema version introspection without requiring the full app to boot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

# Expected head revision — update this when adding new migrations
EXPECTED_HEAD = "0014_query_embedding_cache"

# Path to alembic.ini relative to the repo root
_ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"


@dataclass
class MigrationStatus:
    """Result of a migration state check."""

    current_revision: str | None = None
    expected_head: str = EXPECTED_HEAD
    pending_count: int = 0
    pending_revisions: list[str] = field(default_factory=list)
    is_compatible: bool = False
    error: str | None = None

    @property
    def needs_migration(self) -> bool:
        return self.pending_count > 0

    @property
    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        if self.is_compatible:
            return "Schema is up to date"
        return f"{self.pending_count} pending migration(s): {self.current_revision} → {self.expected_head}"


def get_alembic_config() -> Config:
    """Build Alembic config from alembic.ini."""
    import os

    cfg = Config(str(_ALEMBIC_INI))
    # Allow DATABASE_URL override
    if os.environ.get("DATABASE_URL"):
        cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    return cfg


def get_script_directory() -> ScriptDirectory:
    """Get the Alembic script directory for revision introspection."""
    return ScriptDirectory.from_config(get_alembic_config())


def get_all_revisions() -> list[str]:
    """Return ordered list of all revision IDs from base to head."""
    script = get_script_directory()
    revs = []
    for rev in script.walk_revisions():
        revs.append(rev.revision)
    revs.reverse()  # walk_revisions goes head→base
    return revs


async def check_migration_status(database_url: str | None = None) -> MigrationStatus:
    """Check current DB revision against expected head.

    This creates its own connection (no dependency on the app engine)
    so it can be used in preflight scripts and startup guards.
    """
    import os

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url:
        return MigrationStatus(error="DATABASE_URL not set")

    status = MigrationStatus()

    try:
        engine = create_async_engine(url, pool_pre_ping=True)
        async with engine.connect() as conn:
            # Check if alembic_version table exists
            result = await conn.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT FROM information_schema.tables "
                    "  WHERE table_name = 'alembic_version'"
                    ")"
                )
            )
            table_exists = result.scalar()

            if not table_exists:
                status.current_revision = None
                status.pending_count = len(get_all_revisions())
                status.pending_revisions = get_all_revisions()
                return status

            result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            row = result.first()
            status.current_revision = row[0] if row else None

        await engine.dispose()
    except Exception as exc:
        status.error = str(exc)[:300]
        return status

    # Calculate pending migrations
    return _resolve_pending(status)


def _resolve_pending(status: MigrationStatus) -> MigrationStatus:
    """Given a status with current_revision set, calculate pending info."""
    all_revs = get_all_revisions()
    if status.current_revision is None:
        status.pending_revisions = all_revs
        status.pending_count = len(all_revs)
    elif status.current_revision == EXPECTED_HEAD:
        status.is_compatible = True
        status.pending_count = 0
    else:
        try:
            idx = all_revs.index(status.current_revision)
            status.pending_revisions = all_revs[idx + 1 :]
            status.pending_count = len(status.pending_revisions)
        except ValueError:
            status.error = (
                f"Current revision '{status.current_revision}' not found in migration chain"
            )

    return status
