"""state-assembly receipts + per-tenant config + policy bundle reservation

Revision ID: 0017_receipts_and_policy
Revises: 0016_memory_status_tombstoned
Create Date: 2026-05-12

Lands three tables for issue #49 (state-assembly receipts) and the
forward-looking surfaces issue #50 (sensitivity-label policy) needs to
slot into without a second migration:

  * `receipts` — one row per assembly call when emission fires; the
    immutable audit artifact. Indexed for the two real query patterns:
    by id (point lookup) and by (tenant_id, subject_id, created_at)
    for time-range listing.

  * `tenant_configs` — first per-tenant configuration table on the
    schema. JSONB `config` column so future per-tenant knobs (retention,
    rate-limit tiers, policy mode) don't each need their own migration.
    Carries `version: int` for optimistic concurrency once a write API
    lands and `updated_at` for audit.

  * `policy_bundles` — placeholder for the policy YAML store that #50
    will populate. v1 of receipts ships with `policy.policy_bundle_hash
    = NULL`; #50 wires the real value once policies exist. Table is
    created now so #50 doesn't have to migrate again.

The `receipts` table is **append-only by convention** at the service
layer. Operators wanting hard enforcement should additionally grant
the service role INSERT+SELECT only on this table — documented in
docs/state-assembly-receipts.md.

Reverse-compatible: downgrade drops all three tables. Receipts already
written are lost on downgrade — by design, since v1 deployments
running without the table can't represent them.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "0017_receipts_and_policy"
down_revision: Union[str, None] = "0016_memory_status_tombstoned"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -----------------------------------------------------------------
    # receipts — primary artifact for issue #49
    # -----------------------------------------------------------------
    #
    # `receipt_id` is a ULID stored as TEXT (not UUID) — ULIDs sort by
    # creation time, which gives us a free natural index on creation
    # order without a separate created_at lookup. The column carries
    # an explicit length cap (26) matching the canonical ULID format.
    #
    # `body` is the full strict-superset receipt JSON. We persist the
    # whole document rather than splitting it across many columns
    # because (a) the schema will grow as new modes ship (as_of_replay,
    # eval_run) and we don't want one migration per field, and (b)
    # consumers nearly always want the whole receipt at once — there's
    # no read pattern that selects a few fields. The columns we DO
    # break out are exactly the ones we filter or join on.
    op.create_table(
        "receipts",
        sa.Column("receipt_id", sa.String(26), primary_key=True),
        sa.Column("parent_receipt_id", sa.String(26), nullable=True),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("tenant_id", sa.String(256), nullable=True),
        sa.Column("subject_id", sa.String(256), nullable=False),
        sa.Column("query_id", sa.String(64), nullable=True),
        sa.Column("task_id", sa.String(64), nullable=True),
        sa.Column("context_hash", sa.String(64), nullable=False),
        sa.Column("context_size_bytes", sa.Integer, nullable=False),
        sa.Column("policy_bundle_hash", sa.String(64), nullable=True),
        sa.Column("region", sa.String(64), nullable=True),
        sa.Column("receipt_signature", sa.String(128), nullable=True),
        sa.Column("body", JSONB, nullable=False),
        sa.Column(
            "as_of",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Time-range listing by tenant+subject is the primary read pattern
    # for audit dashboards. Compose the index on (tenant_id,
    # subject_id, created_at DESC) so range scans walk the index in
    # newest-first order without a separate sort.
    op.create_index(
        "ix_receipts_tenant_subject_created",
        "receipts",
        ["tenant_id", "subject_id", "created_at"],
    )

    # -----------------------------------------------------------------
    # tenant_configs — per-tenant settings (receipts mode, retention,
    # policy mode in #50, future knobs)
    # -----------------------------------------------------------------
    #
    # `config` is a JSONB document, schema-validated at the Pydantic
    # layer. Examples:
    #   {
    #     "receipts": "on_request",         # always | on_request | never
    #     "receipt_retention_days": 0,      # 0 = forever
    #     "policy_mode": "log_only",        # log_only | enforce (set by #50)
    #     "require_caller_identity": false  # set by #50
    #   }
    #
    # `version` is incremented on every write — admin UI will use it
    # for optimistic-concurrency edits once a write endpoint lands.
    op.create_table(
        "tenant_configs",
        sa.Column("tenant_id", sa.String(256), primary_key=True),
        sa.Column("config", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # -----------------------------------------------------------------
    # policy_bundles — reserved for issue #50
    # -----------------------------------------------------------------
    #
    # Stores immutable policy YAML bundles addressed by content hash.
    # Receipts reference `policy_bundle_hash`, making "what did policy
    # version abc123 say on date Y?" answerable forever — the load-
    # bearing replay feature for compliance and research. Empty in v1;
    # #50 ships the writer + activation logic.
    op.create_table(
        "policy_bundles",
        sa.Column("bundle_hash", sa.String(64), primary_key=True),
        sa.Column("yaml_content", sa.Text, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("tenant_id", sa.String(256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_policy_bundles_tenant_active",
        "policy_bundles",
        ["tenant_id", "active"],
    )


def downgrade() -> None:
    op.drop_index("ix_policy_bundles_tenant_active", table_name="policy_bundles")
    op.drop_table("policy_bundles")
    op.drop_table("tenant_configs")
    op.drop_index("ix_receipts_tenant_subject_created", table_name="receipts")
    op.drop_table("receipts")
