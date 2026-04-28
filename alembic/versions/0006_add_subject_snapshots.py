"""add subject_snapshots table

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-27
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "subject_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("source_subject_id", sa.String(256), nullable=False),
        sa.Column("episode_count", sa.Integer(), nullable=False),
        sa.Column("memory_count", sa.Integer(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_snapshots_name_version", "subject_snapshots", ["name", "version"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_snapshots_name_version", table_name="subject_snapshots")
    op.drop_table("subject_snapshots")
