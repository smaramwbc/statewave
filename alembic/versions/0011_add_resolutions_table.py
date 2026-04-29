"""Add resolutions table.

Revision ID: 0011
Revises: 0010
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "resolutions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("subject_id", sa.String(256), nullable=False),
        sa.Column("session_id", sa.String(256), nullable=False),
        sa.Column("tenant_id", sa.String(256), nullable=True),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default="open",
        ),
        sa.Column("resolution_summary", sa.Text, nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_", JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_resolutions_subject_id", "resolutions", ["subject_id"])
    op.create_index("ix_resolutions_session_id", "resolutions", ["session_id"])
    op.create_index("ix_resolutions_tenant_id", "resolutions", ["tenant_id"])
    op.create_index(
        "ix_resolutions_subject_status",
        "resolutions",
        ["subject_id", "status"],
    )


def downgrade() -> None:
    op.drop_table("resolutions")
