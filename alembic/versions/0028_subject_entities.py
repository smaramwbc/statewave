"""subject_entities — per-subject entity store with linked_memory_ids for cross-session lookup (Phase 2 of the cross-session retrieval improvements).

Revision ID: 0028_subject_entities
Revises: 0027_memories_content_tsvector
Create Date: 2026-06-18

Adds a separate entity collection keyed on (subject_id, entity_normalized)
with per-entity embeddings and an array of memory ids the entity appears
in. The entity-boost retrieval path relies on this: it looks up
query-matching entities, then boosts the score of every memory in any
matched entity's linked_memory_ids. This is the architectural mechanism
that turns isolated embedding vectors into a CROSS-SESSION join graph —
without paying the full cost of a Neo4j + LLM-built knowledge graph.

This migration adds ONLY the storage. The Phase 2 compile-time hook
(populating the table) lands in the next commit; the Phase 3
retrieval-time entity-boost lookup (using the table) lands in the commit
after that. Reverse-compatible end-to-end: until the population hook
runs, the table is empty + the new retrieval path is a no-op; pure
semantic + BM25 (Phase 1) still works.

Why a separate table (vs. a JSONB column on memories): entities live in
their own collection (`{collection}_entities`) for three reasons —
(1) one entity links to MANY memories naturally as
an N-to-M relation; (2) entity-level cosine dedup needs an entity
embedding column; (3) HNSW index on entity embeddings lets the
retrieval lane query entities first then map back to memories, instead
of scanning all memories.

Schema choices:
  * `entity_text` — canonical form (preserves casing for display)
  * `entity_normalized` — lowercased + whitespace-collapsed (the dedup key)
  * `entity_kind` — spaCy NER label (PERSON / ORG / GPE / DATE / etc) when
    available, NULL otherwise. Optional; some retrievers care, most don't.
  * `embedding` — pgvector, dim from settings.EMBEDDING_DIMENSIONS, HNSW-indexed
  * `linked_memory_ids` — UUID[]. Server-default to '{}' so we never
    have to special-case nullability.

Indexes:
  * `ix_subject_entities_subject_normalized` — composite for the "do I
    already have entity X for subject Y" exact-match lookup
  * `ix_subject_entities_embedding` — HNSW for semantic dedup at write
    AND query-time entity matching at read
  * `ix_subject_entities_linked_memory_ids` — GIN, so "find every entity
    pointing at memory M" is fast for memory-side cleanup paths

Reverse-compatible: downgrade drops the table + every index. No churn on
existing tables.
"""

from typing import Sequence, Union
import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, UUID

# Same constant the rest of the schema uses (server/db/tables.py). Fixed at
# 1536 (text-embedding-3-small), matching the vector(1536) columns from 0013.
EMBEDDING_DIMENSIONS = 1536


revision: str = "0028_subject_entities"
down_revision: Union[str, None] = "0027_memories_content_tsvector"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector + pgcrypto (for gen_random_uuid) are already available
    # from 0013/earlier migrations. No extension work needed here.
    op.create_table(
        "subject_entities",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("subject_id", sa.String(256), nullable=False),
        sa.Column("tenant_id", sa.String(256), nullable=True),
        sa.Column("entity_text", sa.Text, nullable=False),
        sa.Column("entity_normalized", sa.Text, nullable=False),
        sa.Column("entity_kind", sa.String(32), nullable=True),
        sa.Column(
            "linked_memory_ids",
            ARRAY(UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    # Add the pgvector column via raw SQL (same pattern as
    # 0013_pgvector_native.py — keeps alembic decoupled from the
    # pgvector SQLAlchemy adapter at migration time).
    op.execute(
        f"ALTER TABLE subject_entities ADD COLUMN embedding vector({EMBEDDING_DIMENSIONS})"
    )

    # Subject scoping — the hottest filter on every read.
    op.create_index(
        "ix_subject_entities_subject_id", "subject_entities", ["subject_id"]
    )
    # Composite for "exact-match dedup": before computing an embedding for
    # an extracted entity, we first check `(subject_id, entity_normalized)`
    # — a string-equality dedup is cheaper than the cosine probe and
    # eliminates ~80% of trivial duplicates ("John", "JOHN", "john ").
    op.create_index(
        "ix_subject_entities_subject_normalized",
        "subject_entities",
        ["subject_id", "entity_normalized"],
    )
    # HNSW for semantic dedup at write (cosine ≥ 0.95 → merge) and for
    # entity-boost lookup at read (find entities matching query text).
    op.execute(
        """
        CREATE INDEX ix_subject_entities_embedding
        ON subject_entities USING hnsw (embedding vector_cosine_ops)
        """
    )
    # GIN on linked_memory_ids so memory-side cascades (e.g. when a
    # memory is deleted, find its entity rows to scrub the linkage)
    # don't have to seq-scan the whole table.
    op.create_index(
        "ix_subject_entities_linked_memory_ids",
        "subject_entities",
        ["linked_memory_ids"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_subject_entities_linked_memory_ids", table_name="subject_entities"
    )
    op.execute("DROP INDEX IF EXISTS ix_subject_entities_embedding")
    op.drop_index(
        "ix_subject_entities_subject_normalized", table_name="subject_entities"
    )
    op.drop_index("ix_subject_entities_subject_id", table_name="subject_entities")
    op.drop_table("subject_entities")
