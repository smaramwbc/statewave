"""compile_jobs.heartbeat_at — liveness signal for async compile jobs.

Revision ID: 0029_compile_jobs_heartbeat
Revises: 0028_subject_entities
Create Date: 2026-09-01

An async compile job is an in-process asyncio task backed by a durable
row. When the process restarts mid-job (rolling deploy), the task dies
but the row stays `running` forever — indistinguishable, until now, from
a job that is merely slow. The compile worker bumps `heartbeat_at` as
each internal LLM batch completes; compile-start uses it to attach to a
live job on the same subject instead of racing it, and to supersede a
row whose heartbeat has gone stale.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0029_compile_jobs_heartbeat"
down_revision = "0028_subject_entities"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "compile_jobs",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("compile_jobs", "heartbeat_at")
