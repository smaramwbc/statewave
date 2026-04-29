"""add subject_health_cache table

Revision ID: 0012
Revises: 0011_add_resolutions_table
Create Date: 2026-04-29
"""

from alembic import op
import sqlalchemy as sa

revision = "0012_add_health_cache"
down_revision = "0011_add_resolutions_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subject_health_cache",
        sa.Column("subject_id", sa.String(256), primary_key=True),
        sa.Column("tenant_id", sa.String(256), nullable=True),
        sa.Column("last_state", sa.String(32), nullable=False),
        sa.Column("last_score", sa.Integer, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("subject_health_cache")
