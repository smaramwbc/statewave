"""Memory conflict resolution service.

Detects conflicting memories for the same subject and kind, superseding
older ones when a newer memory covers the same topic. Uses content similarity
(word overlap or embedding distance) to detect conflicts.

Strategy:
- Group active memories by (subject_id, kind).
- Within each group, compare each pair for similarity.
- When two memories conflict, the older one is marked as "superseded"
  with valid_to set to the newer memory's valid_from.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.tables import MemoryRow

logger = structlog.stdlib.get_logger()

# Similarity threshold for word-overlap conflict detection (0–1)
_WORD_OVERLAP_THRESHOLD = 0.6


async def resolve_conflicts(
    session: AsyncSession,
    subject_id: str,
) -> list[uuid.UUID]:
    """Detect and resolve conflicting memories. Returns IDs of superseded memories."""
    memories = await repo.list_active_memories_by_subject(session, subject_id)
    if len(memories) < 2:
        return []

    # Group by kind — conflicts only make sense within the same kind
    by_kind: dict[str, list[MemoryRow]] = {}
    for m in memories:
        by_kind.setdefault(m.kind, []).append(m)

    superseded_ids: list[uuid.UUID] = []

    for kind, group in by_kind.items():
        if len(group) < 2:
            continue
        # Sort by created_at ascending so we supersede older ones
        group.sort(key=lambda m: m.created_at or datetime.min.replace(tzinfo=timezone.utc))

        for i in range(len(group)):
            if group[i].status != "active":
                continue
            for j in range(i + 1, len(group)):
                if group[j].status != "active":
                    continue
                if _are_conflicting(group[i], group[j]):
                    # Supersede the older memory
                    superseded_ids.append(group[i].id)
                    group[i].status = "superseded"
                    group[i].valid_to = group[j].valid_from or datetime.now(timezone.utc)
                    logger.info(
                        "memory_superseded",
                        old_id=str(group[i].id),
                        new_id=str(group[j].id),
                        kind=kind,
                    )
                    break  # This memory is already superseded, move on

    if superseded_ids:
        await repo.mark_memories_superseded(session, superseded_ids)

    return superseded_ids


def _are_conflicting(older: MemoryRow, newer: MemoryRow) -> bool:
    """Determine if two memories of the same kind conflict.

    For profile_facts: high word overlap means the newer one replaces the older.
    For other kinds: require very high overlap.
    """
    threshold = _WORD_OVERLAP_THRESHOLD
    if older.kind != "profile_fact":
        threshold = 0.8  # stricter for non-facts

    older_tokens = set(older.content.lower().split())
    newer_tokens = set(newer.content.lower().split())

    if not older_tokens or not newer_tokens:
        return False

    # Jaccard similarity
    intersection = len(older_tokens & newer_tokens)
    union = len(older_tokens | newer_tokens)
    similarity = intersection / union if union else 0.0

    return similarity >= threshold
