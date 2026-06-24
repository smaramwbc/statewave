"""memories.content_tsvector — generated tsvector + GIN index for BM25-style retrieval.

Revision ID: 0027_memories_content_tsvector
Revises: 0026_system_settings
Create Date: 2026-06-18

Adds a Postgres-generated `tsvector` column derived from `content`, plus a
GIN index on it. This is the foundation for hybrid retrieval (semantic +
BM25) — phase 1 of improving cross-session retrieval on LongMemEval /
BEAM, where the strongest published numbers come from systems that blend
pgvector cosine similarity with BM25 keyword rank and entity-boost.

When retrieval is scored on a mix of semantic similarity, BM25, and
entity boost, our pure-pgvector path under-finds memories whose surface
form matches the question wording verbatim — LongMemEval
single_session_user / multi_session questions hinge on exact terms from
the user's utterance (names, numbers, dates).

Implementation:
  * `content_tsvector tsvector GENERATED ALWAYS AS (to_tsvector('english',
    content)) STORED` — Postgres maintains it automatically on every
    insert / update. Zero application-side bookkeeping; zero risk of
    drift.
  * GIN index for fast `@@` queries via `plainto_tsquery('english', ...)`
    in the new hybrid repo function (next commit).
  * STORED (not VIRTUAL) so the GIN index works — index-only-on-stored
    is required for the @@ planner path.

Reverse-compatible: downgrade drops the column and the index. The
existing pure-semantic path keeps working untouched in either
direction.
"""

from typing import Sequence, Union

from alembic import op


revision: str = "0027_memories_content_tsvector"
down_revision: Union[str, None] = "0026_system_settings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE memories
        ADD COLUMN content_tsvector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )
    op.execute(
        """
        CREATE INDEX ix_memories_content_tsvector
        ON memories USING gin (content_tsvector)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_memories_content_tsvector")
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS content_tsvector")
