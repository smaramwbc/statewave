"""Add session_id to episodes for session-aware context assembly.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("episodes", sa.Column("session_id", sa.String(256), nullable=True))
    op.create_index("ix_episodes_session", "episodes", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_episodes_session", table_name="episodes")
    op.drop_column("episodes", "session_id")
