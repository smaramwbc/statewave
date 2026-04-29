"""Add tenant_id to episodes, memories, webhook_events, subject_snapshots.

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add tenant_id to episodes
    op.add_column("episodes", sa.Column("tenant_id", sa.String(256), nullable=True))
    op.create_index("ix_episodes_tenant_subject", "episodes", ["tenant_id", "subject_id"])

    # Add tenant_id to memories
    op.add_column("memories", sa.Column("tenant_id", sa.String(256), nullable=True))
    op.create_index("ix_memories_tenant_subject", "memories", ["tenant_id", "subject_id"])

    # Add tenant_id to webhook_events
    op.add_column("webhook_events", sa.Column("tenant_id", sa.String(256), nullable=True))
    op.create_index("ix_webhook_events_tenant", "webhook_events", ["tenant_id"])

    # Add tenant_id to subject_snapshots
    op.add_column("subject_snapshots", sa.Column("tenant_id", sa.String(256), nullable=True))
    op.create_index("ix_snapshots_tenant", "subject_snapshots", ["tenant_id"])

    # Add tenant_id index to compile_jobs (column already exists)
    op.create_index("ix_compile_jobs_tenant", "compile_jobs", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_compile_jobs_tenant", table_name="compile_jobs")
    op.drop_index("ix_snapshots_tenant", table_name="subject_snapshots")
    op.drop_column("subject_snapshots", "tenant_id")
    op.drop_index("ix_webhook_events_tenant", table_name="webhook_events")
    op.drop_column("webhook_events", "tenant_id")
    op.drop_index("ix_memories_tenant_subject", table_name="memories")
    op.drop_column("memories", "tenant_id")
    op.drop_index("ix_episodes_tenant_subject", table_name="episodes")
    op.drop_column("episodes", "tenant_id")
