"""receipts: status + tombstoned_at for the v0.9 retention worker

Revision ID: 0020_receipts_retention
Revises: 0019_per_tenant_bundles
Create Date: 2026-05-25

v0.8 shipped the receipt-retention surface (`tenant_configs.config ->>
'receipt_retention_days'`) but explicitly deferred the worker to v0.9.
This migration adds the row-level state the worker needs to
soft-tombstone receipts past their tenant's retention window:

  * ``receipts.status``     — string enum, default 'active'. Matches
    the memory `status` vocabulary (`active | superseded | tombstoned`
    — though receipts only ever move active → tombstoned today; the
    other values are reserved for parity with the memories surface).
  * ``receipts.tombstoned_at`` — nullable timestamp; stamped when the
    worker transitions a row to `tombstoned`. Memories don't store
    this separately, but receipts are themselves audit artifacts so
    preserving "when was this audit record retired?" is part of the
    audit trail.

A **partial index** on `(tenant_id, created_at) WHERE status='active'`
keeps the retention scan cheap as the table grows — the scan only
visits not-yet-tombstoned rows, which is the small set even at scale
(active rows are bounded by the retention window).

The migration is reversible. Downgrade preserves the data (no DELETE),
just drops the columns + index. A re-upgrade re-stamps every existing
row as 'active' which is the correct "not-yet-tombstoned" default.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0020_receipts_retention"
down_revision: Union[str, None] = "0019_per_tenant_bundles"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # status: default 'active' (server_default applies on INSERT for new
    # rows; the backfill below covers rows that pre-date this migration).
    op.add_column(
        "receipts",
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
    )
    # tombstoned_at: nullable timestamp. Populated by the retention
    # worker on the active -> tombstoned transition.
    op.add_column(
        "receipts",
        sa.Column(
            "tombstoned_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # Partial index — only active rows are scanned by the retention
    # worker; tombstoned rows are excluded. Keeps the index narrow
    # even on tables with millions of tombstoned receipts.
    op.execute(
        "CREATE INDEX ix_receipts_active_tenant_created "
        "ON receipts (tenant_id, created_at) WHERE status = 'active'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_receipts_active_tenant_created")
    op.drop_column("receipts", "tombstoned_at")
    op.drop_column("receipts", "status")
