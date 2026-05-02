"""pgvector-native semantic retrieval — embedding column as vector(1536) + HNSW index

Revision ID: 0013
Revises: 0012
Create Date: 2026-05-02

Migrates `memories.embedding` from TEXT (with Python-side cosine compute) to
`vector(1536)` (native pgvector cosine via the `<=>` operator) and adds an
HNSW index for fast approximate nearest-neighbor search.

This reverses migration 0004's pragmatic "no pgvector dependency" stance.
ADR-001 says pgvector IS the data store; running on stock Postgres without
the extension is no longer supported. Deployments must install the
`postgresql-NN-pgvector` package and ensure the extension is available
before applying this migration. The migration fails loudly if it isn't.

The cast `embedding::vector(1536)` works because pgvector accepts JSON-array
TEXT (`[0.1, 0.2, ...]`) as a valid input format — which is exactly what the
existing background embedding task wrote. No data conversion script needed.

Index choice: HNSW over IVFFlat for this corpus size. HNSW gives sub-ms
nearest-neighbor lookups without an offline training step, at the cost of
slower writes — acceptable here because compile is async and embeddings
are written from a background task.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0013_pgvector_native"
down_revision: Union[str, None] = "0012_add_health_cache"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Hard-require pgvector. If the extension isn't available the migration
    # raises before any column changes, leaving the schema untouched.
    available = conn.execute(sa.text(
        "SELECT count(*) FROM pg_available_extensions WHERE name = 'vector'"
    )).scalar()
    if not available:
        raise RuntimeError(
            "pgvector extension is not available in this Postgres install. "
            "Install the `postgresql-N-pgvector` package on the database "
            "container (or use a Postgres image that bundles pgvector) "
            "before applying migration 0013. See deployment/pgvector-setup.md."
        )

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Convert any null-but-string-empty edge cases to NULL so the cast
    # doesn't blow up on garbage data. Belt-and-suspenders — the writers
    # never insert empty strings, but old test/dev data might.
    op.execute(
        "UPDATE memories SET embedding = NULL "
        "WHERE embedding IS NOT NULL AND embedding = ''"
    )

    # ALTER ... USING embedding::vector(1536). pgvector parses the
    # existing TEXT format ('[0.1, 0.2, ...]') natively — no migration
    # script needed. Memories with malformed embeddings will fail the
    # cast; in practice they don't exist (the only writer is the
    # background task which always writes well-formed lists).
    op.execute(
        "ALTER TABLE memories "
        "ALTER COLUMN embedding TYPE vector(1536) "
        "USING embedding::vector(1536)"
    )

    # HNSW index for cosine similarity. m=16, ef_construction=64 are
    # pgvector defaults that work well for ~1k–1M vectors at our
    # dimensions. Statement timeout extended for the build — HNSW
    # construction is the slow part of this migration on large tables.
    op.execute("SET LOCAL statement_timeout = '20min'")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_memories_embedding_hnsw "
        "ON memories USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    # Reverse path: drop index, cast vector → text. The text format
    # pgvector emits is '[0.1,0.2,...]' which round-trips with the
    # background task's writer (it stored the same format).
    op.execute("DROP INDEX IF EXISTS ix_memories_embedding_hnsw")
    op.execute(
        "ALTER TABLE memories "
        "ALTER COLUMN embedding TYPE TEXT "
        "USING embedding::text"
    )
    # Don't DROP EXTENSION vector — other tables/use-cases might use it,
    # and dropping it on rollback is a sharper edge than necessary.
