"""policy_bundles: composite (tenant_id, bundle_hash) uniqueness — closes #79

Revision ID: 0019_per_tenant_bundles
Revises: 0018_sensitivity_labels
Create Date: 2026-05-14

v1 of #50 used `bundle_hash` as the sole primary key on
`policy_bundles`. Because bundles are content-addressed, two tenants
uploading the same YAML produce the same hash → only one row could
exist for that hash, and the second tenant's upload silently
re-bound the existing row's `tenant_id` to whatever was last
uploaded. The active-bundle resolver then failed to find a bundle
for the tenant whose row got hijacked. Caught in prod smoke testing
of the enforce-mode endpoint (the smoke had to use tenant-unique
YAML as a workaround); filed as #79.

The fix lets multiple rows share a `bundle_hash` as long as their
`tenant_id` differs:

  * Add a synthetic UUID `id` column as the new primary key. Random
    per row, no semantic meaning — content addressing stays on
    `bundle_hash`, the row's *identity* is the UUID.

  * Drop the old single-column PK on `bundle_hash`.

  * Add a composite unique index on `(tenant_id, bundle_hash)` with
    `NULLS NOT DISTINCT` so two rows with `(NULL, 'X')` conflict
    (correct for globals), but `(NULL, 'X')` and `('acme', 'X')`
    are distinct (correct for tenant scoping). Requires PostgreSQL
    15+ for `NULLS NOT DISTINCT`; statewave runs on PG16 (see CI
    workflow + the pgvector/pgvector:pg16 image).

  * Drop the obsolete `ix_policy_bundles_tenant_active` index
    created by 0017 — it's superseded by the new composite unique
    index (which covers the tenant_id lead column for the existing
    active-lookup query pattern). Re-created on downgrade.

The migration is reversible: downgrade restores the single-column
PK, but DOES NOT collapse rows that share a hash across tenants — a
downgrade applied to a database with cross-tenant bundles will fail
on the PK re-add. Operators rolling back should first run
`DELETE FROM policy_bundles WHERE tenant_id IS NOT NULL` after
exporting any policies they care about, then apply the downgrade.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision: str = "0019_per_tenant_bundles"
down_revision: Union[str, None] = "0018_sensitivity_labels"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add the synthetic id column. server_default applies on INSERT
    #    so new rows get a UUID automatically; existing rows below get
    #    populated by the explicit UPDATE.
    op.add_column(
        "policy_bundles",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
    )
    # 2. Backfill existing rows. server_default doesn't apply to rows
    #    already present at ALTER TABLE time — the explicit UPDATE
    #    fills them in.
    op.execute("UPDATE policy_bundles SET id = gen_random_uuid() WHERE id IS NULL")

    # 3. Drop the old PK and the obsolete tenant_active index that
    #    was added by 0017 (now subsumed by the composite unique
    #    index we're about to add).
    op.drop_index("ix_policy_bundles_tenant_active", table_name="policy_bundles")
    op.drop_constraint("policy_bundles_pkey", "policy_bundles", type_="primary")

    # 4. Add the new PK on `id`.
    op.create_primary_key("policy_bundles_pkey", "policy_bundles", ["id"])

    # 5. Composite unique index. `NULLS NOT DISTINCT` (PG15+) makes
    #    (NULL, 'X') equal to (NULL, 'X') so globals can't be
    #    duplicated, while still allowing (NULL, 'X') alongside
    #    ('acme', 'X'). Raw SQL is used because alembic 1.13's
    #    `op.create_index` doesn't surface the keyword directly.
    op.execute(
        "CREATE UNIQUE INDEX ix_policy_bundles_tenant_hash "
        "ON policy_bundles (tenant_id, bundle_hash) NULLS NOT DISTINCT"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_policy_bundles_tenant_hash")
    op.drop_constraint("policy_bundles_pkey", "policy_bundles", type_="primary")
    # The single-column PK only succeeds if no cross-tenant duplicates
    # exist. Operators rolling back must first drop tenant-scoped
    # bundles — see the module docstring.
    op.create_primary_key("policy_bundles_pkey", "policy_bundles", ["bundle_hash"])
    op.create_index(
        "ix_policy_bundles_tenant_active",
        "policy_bundles",
        ["tenant_id", "active"],
    )
    op.drop_column("policy_bundles", "id")
