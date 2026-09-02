"""subject_entities unique identity — dedup then enforce in the database.

Revision ID: 0030_subject_entities_unique
Revises: 0029_compile_jobs_heartbeat
Create Date: 2026-09-02

`upsert_entity_with_link` was select-then-insert over non-unique indexes,
so two CONCURRENT entity-population runs for the same subject could
silently insert duplicate rows for the same (subject, tenant,
normalized-text) identity (issue #383). The invariant now lives in the
database: existing duplicates are merged (union of linked_memory_ids;
the surviving row is the one with an embedding, else the oldest), then a
unique expression index is created. COALESCE folds NULL tenant_id into
one value — a plain multi-column unique index treats NULLs as distinct
and would not protect untenanted subjects, which are the common case.

The upsert itself switches to INSERT ... ON CONFLICT in the same change,
so concurrent writers converge instead of erroring.
"""

from __future__ import annotations

from alembic import op

revision = "0030_subject_entities_unique"
down_revision = "0029_compile_jobs_heartbeat"
branch_labels = None
depends_on = None

_RANKED = """
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY subject_id, COALESCE(tenant_id, ''), entity_normalized
               ORDER BY (embedding IS NULL), created_at, id
           ) AS rn,
           subject_id,
           COALESCE(tenant_id, '') AS tenant_key,
           entity_normalized
    FROM subject_entities
"""


def upgrade() -> None:
    # 1. Merge every duplicate group's linked_memory_ids into its keeper.
    op.execute(
        f"""
        WITH ranked AS ({_RANKED}),
        keepers AS (SELECT * FROM ranked WHERE rn = 1),
        merged AS (
            SELECT k.id AS keeper_id,
                   (
                       SELECT array_agg(DISTINCT mid)
                       FROM subject_entities se2
                       JOIN ranked r2 ON r2.id = se2.id,
                       LATERAL unnest(se2.linked_memory_ids) AS mid
                       WHERE r2.subject_id = k.subject_id
                         AND r2.tenant_key = k.tenant_key
                         AND r2.entity_normalized = k.entity_normalized
                   ) AS all_mids
            FROM keepers k
        )
        UPDATE subject_entities se
        SET linked_memory_ids = m.all_mids,
            updated_at = now()
        FROM merged m
        WHERE se.id = m.keeper_id
          AND se.linked_memory_ids IS DISTINCT FROM m.all_mids
        """
    )
    # 2. Drop the non-keepers.
    op.execute(
        f"""
        WITH ranked AS ({_RANKED})
        DELETE FROM subject_entities
        WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
        """
    )
    # 3. Enforce the identity from here on.
    op.execute(
        """
        CREATE UNIQUE INDEX uq_subject_entities_identity
        ON subject_entities (subject_id, (COALESCE(tenant_id, '')), entity_normalized)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_subject_entities_identity")
    # The dedup merge is not reversible.
