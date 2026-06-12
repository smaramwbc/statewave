"""episodes.idempotency_key — make ingest idempotent (de-dup re-runs)

Revision ID: 0025_episodes_idempotency_key
Revises: 0024_health_cache_tenant_iso
Create Date: 2026-06-12

The server ignored the ``idempotency_key`` connectors send, so re-running a seed
(or any repeated ingest) inserted DUPLICATE episodes — a repo seeded 3× held 3×
the episodes. This migration makes ingest idempotent:

  * Add a first-class ``idempotency_key`` column.
  * Backfill it from ``metadata->>'idempotency_key'``, where connectors
    historically stashed the key (so existing rows get de-duplicated too).
  * DELETE duplicates, keeping the earliest row per
    ``(tenant_id, subject_id, idempotency_key)``.
  * Add a partial unique index (``NULLS NOT DISTINCT``, PostgreSQL 15+/PG16) so
    future ingests with the same key collapse to one row. Episodes WITHOUT a key
    are unconstrained — the partial predicate excludes them.

NOTE: the de-dup DELETEs episode rows. Memories already compiled from those
duplicates are NOT touched; recompile affected subjects to collapse memory
duplication too. Reversible: downgrade drops the index + column but cannot
restore deleted duplicates (which were redundant by definition).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0025_episodes_idempotency_key"
down_revision: Union[str, None] = "0024_health_cache_tenant_iso"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("episodes", sa.Column("idempotency_key", sa.String(length=512), nullable=True))

    # Backfill from where connectors historically stored the key.
    op.execute(
        """
        UPDATE episodes
        SET idempotency_key = metadata->>'idempotency_key'
        WHERE idempotency_key IS NULL
          AND metadata ? 'idempotency_key'
          AND metadata->>'idempotency_key' IS NOT NULL
          AND metadata->>'idempotency_key' <> ''
        """
    )

    # De-duplicate: keep the earliest row per (tenant_id, subject_id, key).
    # PARTITION groups NULL tenant_id together (single-tenant mode), matching the
    # NULLS NOT DISTINCT semantics of the unique index created below.
    op.execute(
        """
        DELETE FROM episodes e
        USING (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY tenant_id, subject_id, idempotency_key
                       ORDER BY created_at, id
                   ) AS rn
            FROM episodes
            WHERE idempotency_key IS NOT NULL
        ) d
        WHERE e.id = d.id AND d.rn > 1
        """
    )

    # Partial unique index — enforce only when a key is present; NULLS NOT
    # DISTINCT treats a NULL tenant_id as a value so single-tenant rows still
    # de-dup. Requires PG15+ (statewave runs PG16: pgvector/pgvector:pg16).
    op.execute(
        """
        CREATE UNIQUE INDEX ix_episodes_idempotency
        ON episodes (tenant_id, subject_id, idempotency_key)
        NULLS NOT DISTINCT
        WHERE idempotency_key IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_episodes_idempotency")
    op.drop_column("episodes", "idempotency_key")
