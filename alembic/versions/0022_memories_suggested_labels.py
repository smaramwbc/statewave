"""memories.suggested_labels — heuristic-derived label hints, separate
from authoritative sensitivity_labels (v0.9 #158).

Revision ID: 0022_memories_suggested_labels
Revises: 0021_receipts_signature_keyid
Create Date: 2026-05-25

v0.8 introduced ``memories.sensitivity_labels`` (0018) as the tenant-
authoritative input to the policy evaluator. v0.9 (#158) adds an
*orthogonal* second column, ``memories.suggested_labels``, populated
automatically by detector heuristics at ingest time.

The split is deliberate and load-bearing:

  * ``sensitivity_labels`` are the source of truth the policy bundle
    reads on the hot assembly path. Tenants own them; only humans (or
    explicit tenant SDK calls) write them.

  * ``suggested_labels`` are advisory. The compiler stamps them when
    a detector matches at ingest. The policy evaluator never reads
    them. Nothing destructive (retention, redaction, refusal) keys
    off them in v0.9. They exist so an operator UI can review and
    promote/dismiss them into ``sensitivity_labels`` later.

This separation means a noisy detector can never tighten policy on
real traffic — the worst it can do is surface a noisy suggestion in
the admin UI. Promotion is a deliberate, auditable action.

Both columns share the same shape (TEXT[] with `'{}'::text[]` default,
GIN index for overlap queries) so the eventual promotion path is a
trivial array assignment. The GIN index supports the admin endpoint
``GET /admin/memories/with-suggested-labels`` which surfaces every
memory carrying at least one suggestion, keyed off
``suggested_labels && '{<label>}'``.

Reverse-compatible: downgrade drops the column and its index.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY


revision: str = "0022_memories_suggested_labels"
down_revision: Union[str, None] = "0021_receipts_signature_keyid"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column(
            "suggested_labels",
            ARRAY(sa.String()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
    )
    op.create_index(
        "ix_memories_suggested_labels",
        "memories",
        ["suggested_labels"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_memories_suggested_labels", table_name="memories")
    op.drop_column("memories", "suggested_labels")
