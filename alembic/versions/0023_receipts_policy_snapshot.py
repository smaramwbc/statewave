"""receipts.policy_snapshot — embedded policy YAML for as-of replay (v0.9 #159).

Revision ID: 0023_receipts_policy_snapshot
Revises: 0022_memories_suggested_labels
Create Date: 2026-05-25

v0.8 stored ``policy_bundle_hash`` on every receipt — enough to tell
"which bundle was active when this receipt was emitted" but NOT enough
to replay the decision later: the operator may have deleted or
overwritten that bundle row, and a hash without the rules attached is
opaque.

v0.9 (#159) adds ``policy_snapshot``, a self-contained JSONB envelope
that ships the bundle's YAML alongside its hash and the capture
timestamp:

    {
      "bundle_hash": "<sha256>" | null,
      "bundle_yaml": "<verbatim YAML>" | null,
      "captured_at": "<ISO-8601>"
    }

A null inner pair records "no policy bundle was active" — distinct
from the column being NULL (which means "pre-v0.9 receipt, never had
a snapshot"). The replay endpoint refuses to operate on the latter
(returns 422 ``unreplayable: missing_policy_snapshot``); the former
replays cleanly against the no-policy fallback.

The column is nullable so existing v0.8 receipts pass through the
migration untouched — they remain queryable / verifiable, only the
replay surface refuses them. Reversible: downgrade drops the column.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "0023_receipts_policy_snapshot"
down_revision: Union[str, None] = "0022_memories_suggested_labels"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "receipts",
        sa.Column(
            "policy_snapshot",
            JSONB,
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("receipts", "policy_snapshot")
