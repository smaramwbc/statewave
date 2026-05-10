"""Memory TTL / expiry — per-kind global expiry windows.

Two responsibilities:

  1. **Stamp** `valid_to` on freshly-compiled memories whose kind has an
     operator-configured TTL. Called from each compiler (heuristic + LLM)
     at the single point where `MemoryRow` is constructed.

  2. **Tombstone** active memories whose `valid_to` has passed. Called
     from the hourly `_cleanup_loop` in `server.app`. Soft-delete only:
     rows are kept so the source data is still resolvable for audit and
     for the receipts surface (issue #49). A future hard-purge knob
     (post-v0.7) can reclaim storage on tombstoned rows older than a
     grace window.

The retrieval path in `server.db.repositories` independently filters
`(valid_to IS NULL OR valid_to > now())` so an expired-but-not-yet-
cleaned-up memory still cannot surface in `/v1/context` between cleanup
runs. The cleanup loop is a backstop, not the load-bearing fence.

Per-subject / per-tenant TTL policies are intentionally out of scope
here — that's the policy layer in issue #50. v0.7 ships per-kind
globals only so the simple primitive lands first; the policy layer
will eventually subsume this with a richer expression and treat this
module's per-kind windows as the default-fallback rule.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.tables import MemoryRow

logger = structlog.stdlib.get_logger()


def compute_valid_to(
    kind: str,
    valid_from: datetime,
    kind_ttl_days: Mapping[str, int],
) -> datetime | None:
    """Return the `valid_to` to stamp on a fresh memory of `kind`.

    Returns `None` when the kind has no TTL configured (forever-valid),
    matching the existing semantics of `MemoryRow.valid_to`. Returns
    `valid_from + ttl_days` otherwise.

    The function is pure / side-effect-free and does not consult global
    settings — callers pass the dict explicitly so tests can construct
    isolated scenarios without monkey-patching the settings singleton.
    """
    days = kind_ttl_days.get(kind)
    if not days:
        return None
    if valid_from.tzinfo is None:
        valid_from = valid_from.replace(tzinfo=timezone.utc)
    return valid_from + timedelta(days=days)


async def cleanup_expired_memories(session: AsyncSession) -> int:
    """Tombstone active memories whose `valid_to` has passed.

    Single statement — atomic, replica-safe (multiple cleanup loops
    running against the same DB just write the same idempotent UPDATE).
    Returns the number of rows transitioned. Soft-tombstone only:
    `status = 'tombstoned'`, no `DELETE`, the row stays.

    Caller is responsible for committing the session.
    """
    stmt = (
        update(MemoryRow)
        .where(MemoryRow.status == "active")
        .where(MemoryRow.valid_to.isnot(None))
        .where(MemoryRow.valid_to < datetime.now(timezone.utc))
        .values(status="tombstoned")
    )
    result = await session.execute(stmt)
    count = result.rowcount or 0
    if count:
        logger.info("memory_ttl_tombstoned", count=count)
    return count
