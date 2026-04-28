"""Add compile_jobs table for durable async compilation.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-28
"""

from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "compile_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("subject_id", sa.String, nullable=False, index=True),
        sa.Column("tenant_id", sa.String, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("memories_created", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_compile_jobs_status", "compile_jobs", ["status"])
    op.create_index("ix_compile_jobs_created_at", "compile_jobs", ["created_at"])


def downgrade() -> None:
    op.drop_table("compile_jobs")
