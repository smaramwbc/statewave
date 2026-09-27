"""memories.embedding_model: record which model produced each vector (#421).

Revision ID: 0032_memories_embedding_model
Revises: 0031_supersession_records
Create Date: 2026-09-18

The embedding model is a single global (or tenant-overridden) setting; nothing
recorded which model produced a given `memories.embedding` vector. A model
swap to a different dimension already fails loudly at the fixed `vector(N)`
column, but a same-dimension swap was accepted silently, and its vectors
compared against the previous model's in the same HNSW index with no signal
anything had changed.

This adds a nullable `embedding_model` column. NULL means "unknown
provenance" (a legacy row, or an import that did not carry the field) and is
never treated as a mismatch by the read path: a same-dimension swap is only
detectable going forward from whatever point provenance starts being
recorded, and treating every pre-existing legacy row as suspect would warn
forever about a mismatch that mostly is not one.

Existing rows that already carry a vector ARE backfilled here, from whatever
embedding provider/model this deployment is actually configured with at
migration time: `system_settings` (the admin-UI override layer, #26) first,
falling back to `server.core.config.settings` (env vars *and* `.env`),
the same effective-config resolution the rest of the app uses, minus the
per-tenant step (see `_configured_embedding_model` below for why). Earlier
this read `os.environ` directly on the stated assumption that "migrations
don't import the `server` package": false even at the time, since
`alembic/env.py` already imports `server.db.tables` and
`server.services.migrations`. A raw `os.environ` read never sees a
`.env`-only value or a DB-held admin override, so it silently backfilled
"stub" for every `.env` install and every admin-UI-configured deployment,
which is worse than leaving the column NULL: NULL is exempt from mismatch
detection by design, but a wrongly-stamped "stub" makes every real-model
row in the corpus read as a positive mismatch. Where the model still can't
be determined (an unrecognized/`none` provider), this leaves the column
NULL rather than guessing.

That said, this remains a deliberate best-effort assumption, not a fact
this migration can verify: it is correct for the overwhelming majority of
existing rows (they were produced by whatever the deployment has been
running), and it is exactly what stops every upgraded deployment from
warning about a mismatch on its entire pre-existing corpus the moment this
column exists. A deployment that changed models WITHOUT this column
tracking it has already silently mixed vectors before this migration ever
runs; this backfill does not and cannot detect that after the fact, it
only avoids inventing a false one.

subject_entities intentionally NOT touched here: its embedding writer
(`upsert_entity_with_link`) forks into three paths (exact-match, semantic
merge, insert-or-update) and doing the same provenance tracking justice
there is a separate change.

Reversible: downgrade drops the column.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0032_memories_embedding_model"
down_revision: Union[str, None] = "0031_supersession_records"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Rows are backfilled in bounded batches rather than one UPDATE over the
# whole table. `memories.embedding` sits behind an HNSW index (0013), so a
# single UPDATE across a large corpus does full index-maintenance work in
# one uninterruptible statement, and at scale that can comfortably clear the
# Helm migration Job's 600s deadline, and because it is still one statement
# in one migration transaction, a kill mid-run rolls back everything rather
# than leaving partial progress.
_BACKFILL_BATCH_SIZE = 5000


def _configured_embedding_model(bind) -> str | None:
    """Best-effort resolution of the model presently producing embeddings.

    Mirrors the precedence `server.core.dynamic_settings.get_setting()`
    documents (tenant_override -> global_db -> env -> hardcoded default),
    minus the tenant step: `server.services.embeddings.get_provider()`
    builds one process-wide singleton from global config only and never
    consults a tenant override, so a tenant-scoped `litellm_embedding_model`
    row isn't the model actually stamping fresh vectors either; mirroring
    tenant precedence here would claim a precision the running server
    doesn't have.

    Reads `system_settings` (the admin-UI override layer, #26) directly so
    an operator-configured deployment backfills the value actually set
    there, then falls back to `server.core.config.settings`, which loads
    `.env` itself, unlike a raw `os.environ` read, for the env/.env step.
    `alembic/env.py` already imports `server.db.tables` and
    `server.services.migrations`, so importing `server.core.config` here
    is the same thing this chain already does elsewhere.

    A deployment topology that hands this migration's process a narrower
    env than the application itself sees (e.g. a Helm migration Job given
    only `STATEWAVE_DATABASE_URL`) and has never written a `system_settings`
    override either is a chart-wiring gap outside what any in-process read
    can recover. The "none"/unrecognized branch below leaves the column
    NULL rather than guessing in that case.
    """
    from sqlalchemy import select

    from server.core.config import settings as env_settings
    from server.core.dynamic_settings import system_settings

    overrides: dict[str, object] = {}
    try:
        # Selecting through the real `system_settings` Table (rather than a
        # hand-typed `sa.text` SELECT) makes SQLAlchemy apply the JSONB
        # column's own result decoding, so this reads the same Python value
        # `apply_global_override` wrote, independent of what the driver
        # returns for a raw-text query.
        result = bind.execute(
            select(system_settings.c.key, system_settings.c.value).where(
                system_settings.c.key.in_(("embedding_provider", "litellm_embedding_model"))
            )
        )
        overrides = dict(result.fetchall())
    except Exception:
        # `system_settings` is created by migration 0026, strictly earlier
        # in this chain, so this is defensive (e.g. an offline `--sql`
        # render with no live connection) rather than an expected path.
        overrides = {}

    provider = overrides.get("embedding_provider", env_settings.embedding_provider)
    if provider == "litellm":
        return overrides.get("litellm_embedding_model", env_settings.litellm_embedding_model)
    if provider == "stub":
        return "stub"
    return None  # "none" (or unrecognized): no live provider to attribute rows to


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("embedding_model", sa.Text(), nullable=True),
    )
    bind = op.get_bind()
    current_model = _configured_embedding_model(bind)
    if current_model is not None:
        update_stmt = sa.text(
            "UPDATE memories SET embedding_model = :model "
            "WHERE id IN ("
            "  SELECT id FROM memories "
            "  WHERE embedding IS NOT NULL AND embedding_model IS NULL "
            "  LIMIT :batch_size"
            ")"
        ).bindparams(
            sa.bindparam("model", value=current_model, type_=sa.Text()),
            sa.bindparam("batch_size", value=_BACKFILL_BATCH_SIZE, type_=sa.Integer()),
        )
        while True:
            result = bind.execute(update_stmt)
            if (result.rowcount or 0) < _BACKFILL_BATCH_SIZE:
                break


def downgrade() -> None:
    op.drop_column("memories", "embedding_model")
