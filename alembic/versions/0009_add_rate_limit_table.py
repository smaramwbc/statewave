"""Add rate_limit_hits table for distributed rate limiting.

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rate_limit_hits",
        sa.Column("key", sa.String(256), nullable=False),
        sa.Column("window_start", sa.BigInteger, nullable=False),
        sa.Column("hit_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("key", "window_start"),
    )
    op.create_index("ix_rate_limit_hits_window", "rate_limit_hits", ["window_start"])


def downgrade() -> None:
    op.drop_index("ix_rate_limit_hits_window", table_name="rate_limit_hits")
    op.drop_table("rate_limit_hits")
