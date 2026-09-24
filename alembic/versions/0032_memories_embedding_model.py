"""memories.embedding_model: record which model produced each vector (#421).

Revision ID: 0032_memories_embedding_model
Revises: 0031_supersession_records
Create Date: 2026-09-18

The embedding model is a single global (or tenant-overridden) setting; nothing
recorded which model produced a given `memories.embedding` vector. A model
swap to a different dimension already fails loudly at the fixed `vector(N)`
column, but a same-dimension swap was accepted silently, and its vectors
compared against the previous model's in the same HNSW index with no signal
anything had changed.

This adds a nullable `embedding_model` column. NULL means "unknown
provenance" (a legacy row, or an import that did not carry the field) and is
never treated as a mismatch by the read path: a same-dimension swap is only
detectable going forward from whatever point provenance starts being
recorded, and treating every pre-existing legacy row as suspect would warn
forever about a mismatch that mostly is not one.

Existing rows that already carry a vector ARE backfilled here, from whatever
embedding provider/model this deployment is configured with at migration
time (`STATEWAVE_EMBEDDING_PROVIDER` / `STATEWAVE_LITELLM_EMBEDDING_MODEL`,
read directly from the environment, since migrations do not import the `server`
package). That is a deliberate assumption, not a fact this migration can
verify: it is correct for the overwhelming majority of existing rows (they
were produced by whatever the deployment has been running), and it is
exactly what stops every upgraded deployment from warning about a mismatch
on its entire pre-existing corpus the moment this column exists. A
deployment that changed models WITHOUT this column tracking it has already
silently mixed vectors before this migration ever runs; this backfill does
not and cannot detect that after the fact, it only avoids inventing a false
one.

subject_entities intentionally NOT touched here: its embedding writer
(`upsert_entity_with_link`) forks into three paths (exact-match, semantic
merge, insert-or-update) and doing the same provenance tracking justice
there is a separate change.

Reversible: downgrade drops the column.
"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0032_memories_embedding_model"
down_revision: Union[str, None] = "0031_supersession_records"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _configured_embedding_model() -> str | None:
    """Best-effort mirror of `server.core.config.Settings`' embedding
    defaults, read straight from the environment. Migrations deliberately
    don't import the `server` package (no other revision does), so this
    duplicates the two defaults rather than instantiating `Settings`."""
    provider = os.environ.get("STATEWAVE_EMBEDDING_PROVIDER", "stub")
    if provider == "litellm":
        return os.environ.get("STATEWAVE_LITELLM_EMBEDDING_MODEL", "text-embedding-3-small")
    if provider == "stub":
        return "stub"
    return None  # "none" (or unrecognized): no live provider to attribute rows to


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("embedding_model", sa.Text(), nullable=True),
    )
    current_model = _configured_embedding_model()
    if current_model is not None:
        op.execute(
            sa.text(
                "UPDATE memories SET embedding_model = :model "
                "WHERE embedding IS NOT NULL AND embedding_model IS NULL"
            ).bindparams(sa.bindparam("model", value=current_model, type_=sa.Text()))
        )


def downgrade() -> None:
    op.drop_column("memories", "embedding_model")
