"""subject_health_cache: per-(tenant_id, subject_id) identity — tenant isolation

Revision ID: 0024_health_cache_tenant_iso
Revises: 0023_receipts_policy_snapshot
Create Date: 2026-06-01

`subject_health_cache` used `subject_id` as its sole primary key (migration
0012). In a multi-tenant deployment, two tenants caching health for the same
`subject_id` collided on one row: tenant B's upsert silently overwrote tenant
A's cached state, and a read returned whichever tenant wrote last. Health
state/score is tenant data, so this is a cross-tenant read/write leak — the
same isolation class fixed in #206 (conflicts) and #207 (compile jobs).

The fix mirrors the policy_bundles per-tenant fix (migration 0019):

  * Add a synthetic UUID `id` column as the new primary key — random per row,
    no semantic meaning; the row's *identity* becomes the UUID.

  * Drop the old single-column PK on `subject_id`.

  * Keep a plain index on `subject_id` for the single-tenant lookup path
    (where no tenant filter is applied).

  * Add a composite unique index on `(tenant_id, subject_id)` with
    `NULLS NOT DISTINCT` so `(NULL, 's')` can't be duplicated in single-tenant
    mode, while `(NULL, 's')` and `('acme', 's')` stay distinct (correct
    per-tenant scoping). Requires PostgreSQL 15+; statewave runs PG16 (see the
    CI workflow + the pgvector/pgvector:pg16 image).

Reversible: downgrade restores the single-column PK, but DOES NOT collapse
rows that share a `subject_id` across tenants — a downgrade applied to a
database with cross-tenant cache rows will fail on the PK re-add. The cache is
regenerated on the next health check, so operators rolling back should first
`DELETE FROM subject_health_cache WHERE tenant_id IS NOT NULL`.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision: str = "0024_health_cache_tenant_iso"
down_revision: Union[str, None] = "0023_receipts_policy_snapshot"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add the synthetic id column. server_default applies on INSERT so new
    #    rows get a UUID automatically; existing rows are backfilled below.
    op.add_column(
        "subject_health_cache",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
    )
    # 2. Backfill rows present before the ALTER (server_default doesn't apply
    #    to them).
    op.execute("UPDATE subject_health_cache SET id = gen_random_uuid() WHERE id IS NULL")

    # 3. Swap the PK from subject_id to the synthetic id.
    op.drop_constraint("subject_health_cache_pkey", "subject_health_cache", type_="primary")
    op.create_primary_key("subject_health_cache_pkey", "subject_health_cache", ["id"])

    # 4. Plain index on subject_id (no longer implied by a PK) for the
    #    single-tenant lookup path.
    op.create_index(
        "ix_subject_health_cache_subject_id", "subject_health_cache", ["subject_id"]
    )

    # 5. Composite unique index. NULLS NOT DISTINCT (PG15+) makes (NULL, 's')
    #    equal to (NULL, 's') so single-tenant rows can't be duplicated, while
    #    (NULL, 's') and ('acme', 's') stay distinct. Raw SQL because alembic
    #    1.13's op.create_index doesn't surface the keyword.
    op.execute(
        "CREATE UNIQUE INDEX ix_subject_health_cache_tenant_subject "
        "ON subject_health_cache (tenant_id, subject_id) NULLS NOT DISTINCT"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_subject_health_cache_tenant_subject")
    op.drop_index("ix_subject_health_cache_subject_id", table_name="subject_health_cache")
    op.drop_constraint("subject_health_cache_pkey", "subject_health_cache", type_="primary")
    # The single-column PK only succeeds if no cross-tenant duplicates exist —
    # see the module docstring.
    op.create_primary_key("subject_health_cache_pkey", "subject_health_cache", ["subject_id"])
    op.drop_column("subject_health_cache", "id")
