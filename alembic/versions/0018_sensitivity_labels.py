"""memories.sensitivity_labels — per-memory capability tags for the
policy layer (issue #50).

Revision ID: 0018_sensitivity_labels
Revises: 0017_receipts_and_policy
Create Date: 2026-05-12

Adds a first-class `sensitivity_labels TEXT[]` column to `memories`
plus a GIN index for the array-overlap (`&&`) queries the policy
evaluator runs on the hot path. Labels are tenant-supplied capability
tags (e.g. `pii`, `financial`, `secret`) consumed by the policy bundle
loaded from `policy_bundles` (table created in 0017 with the receipt
migration).

The column is NOT NULL with a `'{}'::text[]` server default so every
existing row stays valid (untagged) without a backfill. Untagged
memories pass through the policy evaluator with no rule match, which
collapses to default-allow — they keep behaving exactly as they did
in v0.7. This is the load-bearing property: tenants can roll into the
new policy layer without first having to tag everything.

A first-class column was chosen over a JSONB metadata field for two
reasons:
  1. GIN-indexed overlap queries (`sensitivity_labels && '{pii}'`) run
     in milliseconds. JSONB array membership runs the same query
     orders of magnitude slower at scale.
  2. The policy evaluator runs on every assembly call. Keeping the
     filter inputs typed at the DB layer means a malformed value
     (e.g. someone stuffing a string instead of a list into JSONB)
     can't silently bypass policy.

Reverse-compatible: downgrade drops the column and its index.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY


revision: str = "0018_sensitivity_labels"
down_revision: Union[str, None] = "0017_receipts_and_policy"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column(
            "sensitivity_labels",
            ARRAY(sa.String()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
    )
    # GIN index for array-overlap queries — `sensitivity_labels && '{pii}'`
    # is the per-memory predicate the policy evaluator runs at assembly
    # time, so it must be cheap.
    op.create_index(
        "ix_memories_sensitivity_labels",
        "memories",
        ["sensitivity_labels"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_memories_sensitivity_labels", table_name="memories")
    op.drop_column("memories", "sensitivity_labels")
