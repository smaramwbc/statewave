"""add ivfflat index on memory embeddings

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-24
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # IVFFlat index for cosine distance on memory embeddings.
    # lists=100 is suitable for up to ~100k memories; increase for larger datasets.
    # Only indexes rows where embedding IS NOT NULL, so existing data is fine.
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
