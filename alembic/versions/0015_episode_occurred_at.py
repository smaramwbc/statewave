"""episodes.occurred_at — first-class source-event timestamp

Revision ID: 0015
Revises: 0014_query_embedding_cache
Create Date: 2026-05-09

Until this migration the only timestamp on an episode was `created_at`,
which the database fills in at insertion time via `server_default=now()`.
That conflated two semantically-distinct concepts:

  * **created_at** — when the row landed in our database
  * **occurred_at** — when the source event actually happened

For live-chat ingest these are the same thing. For everything else they're
not: a backfilled GitHub issue from 2024, a Slack message replayed from a
historical export, a Zendesk ticket imported from another system. In each
case the connector knows the real source-event time, and timeline-style
queries ("what did the customer say in March", "show me activity around
the outage") are wrong if we sort by `created_at`.

This migration:
  * Adds `occurred_at timestamptz` to `episodes`, NOT NULL with a
    server-default of `now()` so existing connector code that doesn't
    supply the field keeps working unchanged.
  * Backfills existing rows so `occurred_at = created_at` — a sensible
    default since for those rows we have no better information.
  * Adds a composite (subject_id, occurred_at) index for the timeline
    query path; we keep the existing (subject_id, created_at) index
    because batch insert flows still order by created_at and search
    flows still want stable insertion order as a tiebreak.

Reverse-compatible: clients that don't pass `occurred_at` are unaffected
(the column defaults to now() server-side, identical to created_at). The
downgrade drops the column and the new index.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0015_episode_occurred_at"
down_revision: Union[str, None] = "0014_query_embedding_cache"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add the column with a server-default so the table stays writeable
    #    while we backfill. The DEFAULT applies to any rows inserted between
    #    the ADD COLUMN and the backfill UPDATE on busy systems.
    op.add_column(
        "episodes",
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # 2. Backfill existing rows so each episode's occurred_at matches its
    #    created_at. For docs/demo packs imported before this migration,
    #    there's no better signal; the alternative would be guessing.
    op.execute("UPDATE episodes SET occurred_at = created_at")

    # 3. Composite index for timeline-style queries — they filter by subject
    #    and order by occurred_at. Keep the existing (subject_id, created_at)
    #    index because the search path still uses created_at as a tiebreak.
    op.create_index(
        "ix_episodes_subject_occurred",
        "episodes",
        ["subject_id", "occurred_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_episodes_subject_occurred", table_name="episodes")
    op.drop_column("episodes", "occurred_at")
