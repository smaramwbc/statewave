"""add ivfflat index on memory embeddings

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Only create index if pgvector extension is installed
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")
    )
    if result.scalar() == 0:
        return

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_memories_embedding_cosine
        ON memories
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memories_embedding_cosine")
