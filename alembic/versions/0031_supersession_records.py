"""supersession_records — keep the decision that retired a memory, not just its outcome.

Revision ID: 0031_supersession_records
Revises: 0030_subject_entities_unique
Create Date: 2026-09-18

Superseding a memory wrote only the outcome: `status = 'superseded'` and
`valid_to`. The rule that fired, the score it fired at and the link to the
memory that replaced it were computed and thrown away, so the admin
relationship view had to GUESS a successor from same-subject/same-kind/
created_at ordering — and with several superseded memories of one kind it
guessed wrong (issue #419). This table holds the decision itself, written by
every producer at the moment it decides.

WHAT IS NOT IN HERE — and must not be added later
-------------------------------------------------
No memory content, no summary, no claim VALUES, no LLM rationale text. A row
carries ids, a rule name, a claim KEY (a registry key such as
"user.timezone"), numbers and a job id. Anything a reader wants to render is
joined from `memories`, which is also what subject deletion reaps. Subject
text outliving subject deletion is an open defect (#423); a second table
holding the same text would widen it.

THESE ARE NOT RECEIPTS
----------------------
A receipt attests to bytes delivered to a caller and is chained, hashed and
signed accordingly (docs/state-assembly-receipts.md). A supersession record
is a compile-time decision about stored state that no caller was handed.
Records never enter the receipt chain, are never signed, and nothing here is
hashed into a receipt body. Verification of receipts is untouched by this
migration.

Schema choices
--------------
  * A separate table, not columns on `memories`: the claim path supersedes
    ACROSS kind, one winner can retire several losers, and #414 will want to
    record a decision about a pair of memories where neither row is the
    natural owner of the record.
  * NO foreign keys, deliberately. Reconcile can accept a candidate that
    supersedes a committed memory and then drop that same candidate in a
    later chunk, so the successor may be a memory that never reaches the
    table — an FK would abort the compile. An FK cascade would also erase
    these rows when a memory is deleted, which is the opposite of the point.
    `superseding_memory_id` is therefore nullable: NULL means the decision
    named a successor that was never persisted.
  * `details` JSONB (default '{}') is the room #414 needs to record a
    non-supersession decision without a second migration. Nothing writes it
    today.

Reverse-compatible: the table is write-only for producers and read-only for
the admin view, which falls back to its existing derivation when a memory has
no record (backfill is impossible — the decisions were never stored).
Downgrade drops the table and its indexes; no existing table is touched.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0031_supersession_records"
down_revision: Union[str, None] = "0030_subject_entities_unique"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "supersession_records",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("subject_id", sa.String(256), nullable=False),
        sa.Column("tenant_id", sa.String(256), nullable=True),
        # The memory that lost. Always set: a record exists because this row
        # was marked superseded.
        sa.Column("superseded_memory_id", UUID(as_uuid=True), nullable=False),
        # The memory that replaced it, or NULL when the successor was never
        # persisted (see the FK note above).
        sa.Column("superseding_memory_id", UUID(as_uuid=True), nullable=True),
        # The producer's own strategy string, not a re-coined vocabulary:
        # "claim_contradiction" / "claim_duplicate" / "lexical" are the
        # strings conflicts.py already logs; reconcile's LLM actions are
        # recorded as "reconcile_update" / "reconcile_delete".
        sa.Column("rule", sa.String(64), nullable=False),
        # Claim path only: the canonical registry KEY. Never the value, and
        # never the v2 contradiction bucket, whose entity component can carry
        # personal data.
        sa.Column("claim_key", sa.String(256), nullable=True),
        # Scored rules only (lexical overlap): the similarity and the
        # threshold it was compared against, both from the one computation
        # that made the call.
        sa.Column("score", sa.Float, nullable=True),
        sa.Column("threshold", sa.Float, nullable=True),
        # compile_jobs.id when the compile ran as a tracked job; NULL for a
        # synchronous compile, which has no job row.
        sa.Column("compile_job_id", sa.String(36), nullable=True),
        sa.Column(
            "details",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    # "What replaced this memory" — newest first, so the lookup is a single
    # index read even for a memory superseded more than once across recompiles.
    op.create_index(
        "ix_supersession_records_superseded",
        "supersession_records",
        ["superseded_memory_id", "created_at"],
    )
    # "What did this memory replace" — the active-memory side of the panel.
    op.create_index(
        "ix_supersession_records_superseding",
        "supersession_records",
        ["superseding_memory_id", "created_at"],
    )
    # Subject deletion reaps these rows explicitly (there is no FK cascade to
    # ride on), so the cascade must not seq-scan the table.
    op.create_index(
        "ix_supersession_records_subject_id",
        "supersession_records",
        ["subject_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_supersession_records_subject_id", table_name="supersession_records")
    op.drop_index("ix_supersession_records_superseding", table_name="supersession_records")
    op.drop_index("ix_supersession_records_superseded", table_name="supersession_records")
    op.drop_table("supersession_records")
