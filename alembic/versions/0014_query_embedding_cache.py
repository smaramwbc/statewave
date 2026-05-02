"""query_embedding_cache table — cross-machine deduplication of embed_query calls

Revision ID: 0014
Revises: 0013_pgvector_native
Create Date: 2026-05-02

Adds a tiny Postgres-backed table that lets multiple Fly machines share the
result of a successful `provider.embed_query(text)` call. Without this,
each machine has its own in-process LRU cache and the LB can scatter the
same demo query across machines, repeatedly paying the full OpenAI
embedding round-trip (observed at 0.5–30s, occasional 30s spikes that
exceeded the dev-proxy timeout).

Schema notes:
  * (text_key, model) composite PK — same text + different embedding model
    is a different cache entry. Lets model rotations not return stale data.
  * embedding vector(1536) — reuses the pgvector type already required for
    memory storage (migration 0013). No new infra dependency.
  * expires_at indexed so the opportunistic cleanup at write time can do
    a fast range delete.
  * No subject_id / tenant_id — query embeddings are universal (same text
    → same OpenAI vector regardless of who's asking) and the cache key is
    only the text + model, so there's no data-leakage risk and no need
    for tenant scoping.

Hard-requires pgvector — same as 0013. The migration assumes the
extension is already enabled (it is, by 0013).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0014_query_embedding_cache"
down_revision: Union[str, None] = "0013_pgvector_native"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector must already be installed (0013 enforces it). The CREATE
    # below uses the vector(1536) type via raw SQL — Alembic's
    # autogenerate doesn't know pgvector types, so we use op.execute.
    op.execute(
        """
        CREATE TABLE query_embedding_cache (
            text_key text NOT NULL,
            model text NOT NULL,
            embedding vector(1536) NOT NULL,
            expires_at timestamptz NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (text_key, model)
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_query_embedding_cache_expires_at "
        "ON query_embedding_cache (expires_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_query_embedding_cache_expires_at")
    op.execute("DROP TABLE IF EXISTS query_embedding_cache")
