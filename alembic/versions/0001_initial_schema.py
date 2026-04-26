"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if pgvector is available before trying to create it
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT count(*) FROM pg_available_extensions WHERE name = 'vector'"
    ))
    has_vector = result.scalar() > 0

    if has_vector:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "episodes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("subject_id", sa.String(256), nullable=False),
        sa.Column("source", sa.String(256), nullable=False),
        sa.Column("type", sa.String(128), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("provenance", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_episodes_subject_id", "episodes", ["subject_id"])
    op.create_index("ix_episodes_subject_created", "episodes", ["subject_id", "created_at"])

    op.create_table(
        "memories",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("subject_id", sa.String(256), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_episode_ids", postgresql.ARRAY(sa.UUID()), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_subject_id", "memories", ["subject_id"])
    op.create_index("ix_memories_subject_kind", "memories", ["subject_id", "kind"])

    # Add the vector column separately if pgvector is available
    if has_vector:
        op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS embedding")
        op.execute("ALTER TABLE memories ADD COLUMN embedding vector(1536)")


def downgrade() -> None:
    op.drop_table("memories")
    op.drop_table("episodes")
    op.execute("DROP EXTENSION IF EXISTS vector")
