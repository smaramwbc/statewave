"""system_settings — DB-backed override layer for env-driven configuration

Revision ID: 0026_system_settings
Revises: 0025_episodes_idempotency_key
Create Date: 2026-06-13

Adds three tables that let operators override env-driven settings at runtime
via the admin UI instead of redeploying for every config change:

  * ``system_settings`` — global key → JSON value overrides
  * ``system_settings_audit`` — append-only change log (who/when/before/after)
  * ``tenant_settings`` — per-tenant overrides for the small subset of
    settings where per-tenant behaviour is meaningful (LLM provider, webhook
    URL, rate limits)

Precedence resolved at read time: tenant_override → global_db → env → hardcoded
default. Env continues to be honoured for anything not overridden in the DB,
so existing deployments keep working without a single config change.

NOT in this migration:

  * Receipt signing keys stay env-only (deliberate — `core/config.py` has a
    comment forbidding DB persistence)
  * Database URL, host, port, region, strict_schema stay env-only
    (changing them at runtime would brick the running process)

Reversible: ``downgrade`` drops all three tables (data loss).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0026_system_settings"
down_revision: Union[str, None] = "0025_episodes_idempotency_key"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "system_settings",
        sa.Column("key", sa.String(length=128), primary_key=True),
        sa.Column("value", JSONB, nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column(
            "is_secret",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("set_by", sa.String(length=256), nullable=True),
        sa.Column(
            "set_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "system_settings_audit",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("old_value", JSONB, nullable=True),
        sa.Column("new_value", JSONB, nullable=True),
        # 'patch' | 'delete' (reset to env) | 'test' (probe, no persistence)
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("tenant_id", sa.String(length=256), nullable=True),
        sa.Column("changed_by", sa.String(length=256), nullable=True),
        sa.Column(
            "changed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("note", sa.Text(), nullable=True),
    )
    # Audit queries are "show me the history for setting X" — index keyed
    # on (key, changed_at DESC) makes that O(log n) instead of full scan.
    op.create_index(
        "ix_system_settings_audit_key_changed_at",
        "system_settings_audit",
        ["key", sa.text("changed_at DESC")],
    )

    op.create_table(
        "tenant_settings",
        sa.Column("tenant_id", sa.String(length=256), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", JSONB, nullable=False),
        sa.Column(
            "is_secret",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("set_by", sa.String(length=256), nullable=True),
        sa.Column(
            "set_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("tenant_id", "key"),
    )


def downgrade() -> None:
    op.drop_table("tenant_settings")
    op.drop_index(
        "ix_system_settings_audit_key_changed_at",
        table_name="system_settings_audit",
    )
    op.drop_table("system_settings_audit")
    op.drop_table("system_settings")
