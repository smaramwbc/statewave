"""memories.status — rename `deleted` to `tombstoned` (defensive)

Revision ID: 0016_memory_status_tombstoned
Revises: 0015_episode_occurred_at
Create Date: 2026-05-10

The Python `MemoryStatus` enum had three values until v0.7:
`active | superseded | deleted`. The `deleted` value was aspirational —
no code path ever wrote it (grep across the v0.6.0 codebase confirmed
this) — so this migration is **almost** a no-op for any real deployment.

It is shipped anyway because:

  1. **Defence against external writers.** The status column is a plain
     `String(32)` (no DB-level enum constraint), so anything writing to
     `memories.status` outside our service code — a manual SQL fix-up,
     a third-party importer, an old branch — could have stamped
     `'deleted'` in the wild. This migration normalises any such row
     to the new `'tombstoned'` value the v0.7 TTL cleanup uses.
  2. **Vocabulary alignment with issue #49.** State-assembly receipts
     specify `supersession_status: active | superseded | tombstoned`.
     Renaming now means we never have to migrate a value later when
     receipts ship.
  3. **Visible breadcrumb.** Future readers tracing why the enum value
     changed land on this migration's commit + docstring instead of
     having to chase a rename across the codebase.

Reverse-compatible: the downgrade flips `'tombstoned'` rows back to
`'deleted'`. v0.7 introduces `tombstoned` for TTL-expired memories so
running the downgrade on a TTL-active deployment will lose the
distinction between "never expired" and "actively tombstoned by TTL"
— operators rolling back should disable TTL first
(`STATEWAVE_KIND_TTL_DAYS=''`) to keep the inverse meaningful.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0016_memory_status_tombstoned"
down_revision: Union[str, None] = "0015_episode_occurred_at"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE memories SET status = 'tombstoned' WHERE status = 'deleted'")


def downgrade() -> None:
    op.execute("UPDATE memories SET status = 'deleted' WHERE status = 'tombstoned'")
