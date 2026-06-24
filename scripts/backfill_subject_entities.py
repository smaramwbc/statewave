"""Backfill subject_entities for existing compiled memories.

Phase 2 of the cross-session retrieval improvements added the `subject_entities`
table + a compile-time hook that populates it for newly-compiled
memories. This script is the one-off path for memories that pre-date
the column — Phase 3 retrieval (entity boost) has nothing to query
until subjects have entries in the entity store, and re-compiling
from scratch would be wasteful when the memories are already there.

Usage:
    docker compose exec api python scripts/backfill_subject_entities.py \
        --bench locomo

    docker compose exec api python scripts/backfill_subject_entities.py \
        --subject bench:locomo:conv-26

Idempotent — `upsert_entity_with_link` dedups on (subject_id,
entity_normalized) exact match and on cosine ≥ 0.95 semantic match,
and skips already-linked memory_ids.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog
from sqlalchemy import select

from server.db.engine import get_session_factory
from server.db.tables import MemoryRow
from server.services.entities import backfill_entities_for_subject

logger = structlog.stdlib.get_logger()


async def _subjects_for_bench(bench: str) -> list[str]:
    """All subject_ids matching `bench:<bench>:%` pattern."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(MemoryRow.subject_id)
            .where(MemoryRow.subject_id.like(f"bench:{bench}:%"))
            .where(MemoryRow.status == "active")
            .distinct()
        )
        result = await session.execute(stmt)
        return sorted({row[0] for row in result.all()})


async def _backfill_one(subject_id: str, batch_size: int = 50) -> int:
    """Backfill one subject. Returns count of entity upserts."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        return await backfill_entities_for_subject(
            session, subject_id, batch_size=batch_size
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bench", choices=["locomo", "longmemeval", "beam"])
    group.add_argument("--subject", help="Exact subject_id")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = await _subjects_for_bench(args.bench)
        if args.max_subjects:
            subjects = subjects[: args.max_subjects]

    print(f"Backfilling entities for {len(subjects)} subject(s)…", flush=True)
    total_entities = 0
    for i, sid in enumerate(subjects, 1):
        try:
            n = await _backfill_one(sid, batch_size=args.batch_size)
            total_entities += n
            print(f"  [{i}/{len(subjects)}] {sid}: {n} entity upserts", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(subjects)}] {sid}: FAILED — {e}", flush=True)

    print(f"\nDone. {total_entities} total entity upserts across {len(subjects)} subjects.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
