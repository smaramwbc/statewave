"""add last_compiled_at to episodes

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("episodes", sa.Column("last_compiled_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_episodes_uncompiled", "episodes", ["subject_id"], postgresql_where=sa.text("last_compiled_at IS NULL"))


def downgrade() -> None:
    op.drop_index("ix_episodes_uncompiled")
    op.drop_column("episodes", "last_compiled_at")
