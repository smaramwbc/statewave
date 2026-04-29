"""add embedding column as text (no pgvector dependency)

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add embedding as TEXT if it doesn't already exist (covers cases where
    # pgvector migration 0001 added it as vector type, or it was skipped).
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'memories' AND column_name = 'embedding'"
    ))
    if result.fetchone() is None:
        op.add_column("memories", sa.Column("embedding", sa.Text(), nullable=True))
    else:
        # Column exists (likely as vector type from 0001) — convert to TEXT
        # Drop the ivfflat index first if it exists
        op.execute("DROP INDEX IF EXISTS ix_memories_embedding_cosine")
        op.execute("ALTER TABLE memories ALTER COLUMN embedding TYPE TEXT USING embedding::TEXT")


def downgrade() -> None:
    op.drop_column("memories", "embedding")
