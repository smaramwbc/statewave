"""Compile-time + backfill entity population for the subject_entities store.

Phase 2 of the cross-session retrieval improvements. This module glues three pieces together
that were built separately:

  - `entity_extraction.extract_entities` — LLM-based NER per memory
  - `embeddings.get_provider().embed_texts` — batched entity embeddings
  - `repositories.upsert_entity_with_link` — dedup + linkage at write

Public surface:
  - `populate_entities_for_memories(...)` — call this from inside the
    compile transaction (after memory commit) for newly-compiled rows.
    Idempotent on memory_id linkage (re-running won't duplicate the
    edge). Fan-out keeps total wall-time bounded by the slowest LLM
    extraction call, not the sum.

  - `backfill_entities_for_subject(...)` — one-off path for existing
    pre-Phase-2 compiled memories. Reads memories from the DB,
    extracts entities, populates the store. Safe to re-run.

Both paths gracefully degrade — if the LLM provider is unreachable,
extraction returns an empty list, the function logs at WARNING, and
the caller proceeds (memories were committed; the entity store stays
empty for those memories; Phase 3 retrieval falls back to pure
hybrid (semantic + BM25 from Phase 1)).

Why this split (population vs storage): the entity-boost step in the
retrieval path expects the entity store to be populated and silently
no-ops on empty; population guarantees that contract. The Phase 3
retrieval path doesn't care HOW the store got populated — at-compile-time
vs backfill is operational.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.services.embeddings import get_provider as get_embedding_provider
from server.services.entity_extraction import ExtractedEntity, extract_entities

logger = structlog.stdlib.get_logger()


@dataclass(frozen=True)
class MemoryForEntities:
    """Minimal shape the entity-population path needs — id + the text we
    extract entities from. Keeps the caller decoupled from the full
    MemoryRow object (useful for the backfill loop that processes memories
    in pages without keeping the whole row in memory)."""

    id: uuid.UUID
    content: str


# Per-call concurrency for the LLM extraction fan-out. Bounded so a
# pathologically large compile batch doesn't blow up the operator's
# LLM provider rate limit. 8 is conservative — typical compile batches
# are 20-50 memories; doubling concurrency gives no wall-time benefit
# when the bottleneck is provider round-trip latency.
_EXTRACT_CONCURRENCY = 8


async def _extract_one(
    memory: MemoryForEntities, semaphore: asyncio.Semaphore
) -> tuple[uuid.UUID, list[ExtractedEntity]]:
    async with semaphore:
        entities = await extract_entities(memory.content)
    return memory.id, entities


async def populate_entities_for_memories(
    session: AsyncSession,
    memories: list[MemoryForEntities],
    *,
    subject_id: str,
    tenant_id: str | None,
) -> int:
    """Extract + dedup + persist entities for a batch of just-compiled
    memories. Returns the count of distinct entity rows touched (newly
    inserted OR an existing row had memory_id appended to its
    linked_memory_ids).

    Caller MUST have already committed the memory rows in the same
    session. Entity inserts use the same session and are committed by
    the caller — keeping it in one transaction means a rollback in the
    compile path also rolls back the entity writes, no partial state.

    Order of operations (the Phase 7 entity-linking sequence):
      1. Parallel LLM extraction (fan-out, _EXTRACT_CONCURRENCY workers)
      2. Collect all distinct (text, normalized, kind) entities
      3. Batch-embed in ONE provider call (memory bandwidth, not latency)
      4. Per-(memory_id, entity) call upsert_entity_with_link, which
         does exact-match then cosine-similarity dedup against existing
         entities for the subject

    If the embedding provider is None (operator has embedding_provider=
    "none"), entities still get extracted + stored, but the dedup is
    string-only (no semantic merge) and Phase 3 retrieval can't use
    them — log at INFO so the operator notices.
    """
    if not memories:
        return 0

    # ── Step 1: parallel LLM extraction ─────────────────────────────────
    semaphore = asyncio.Semaphore(_EXTRACT_CONCURRENCY)
    tasks = [_extract_one(m, semaphore) for m in memories]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # ── Step 2: flatten + dedup entity text within this batch ────────────
    # (Same surface form across two memories gets ONE embedding call,
    # then both memories link to the same entity row at upsert time.)
    entities_by_memory: dict[uuid.UUID, list[ExtractedEntity]] = {}
    distinct_texts: dict[str, ExtractedEntity] = {}
    for memory_id, entities in results:
        if not entities:
            continue
        entities_by_memory[memory_id] = entities
        for e in entities:
            if e.normalized not in distinct_texts:
                distinct_texts[e.normalized] = e

    if not distinct_texts:
        return 0

    # ── Step 3: batch embed ─────────────────────────────────────────────
    provider = get_embedding_provider()
    embeddings_by_normalized: dict[str, list[float] | None] = {
        normalized: None for normalized in distinct_texts
    }
    if provider is not None:
        try:
            # Embed the CANONICAL (cased) form — same way the existing
            # memory embeddings are produced from memory.content. This
            # makes the entity↔memory similarity space consistent.
            normalized_keys = list(distinct_texts.keys())
            texts_to_embed = [distinct_texts[k].text for k in normalized_keys]
            embeddings = await provider.embed_texts(texts_to_embed)
            for normalized, vector in zip(normalized_keys, embeddings):
                embeddings_by_normalized[normalized] = vector
        except Exception:
            logger.warning(
                "entity_embedding_failed",
                subject_id=subject_id,
                count=len(distinct_texts),
                exc_info=True,
            )
            # Fall through — we still write the entities, just without
            # vectors. Dedup degenerates to exact-string-match (no
            # semantic merge). Phase 3 retrieval skips them.
    else:
        logger.info(
            "entity_embedding_skipped_no_provider",
            subject_id=subject_id,
            count=len(distinct_texts),
        )

    # ── Step 4: per-memory upsert ───────────────────────────────────────
    touched = 0
    for memory_id, entities in entities_by_memory.items():
        for e in entities:
            embedding = embeddings_by_normalized.get(e.normalized)
            try:
                await repo.upsert_entity_with_link(
                    session,
                    subject_id=subject_id,
                    tenant_id=tenant_id,
                    entity_text=e.text,
                    entity_normalized=e.normalized,
                    entity_kind=e.kind,
                    embedding=embedding,
                    memory_id=memory_id,
                )
                touched += 1
            except Exception:
                logger.warning(
                    "entity_upsert_failed",
                    subject_id=subject_id,
                    entity=e.normalized,
                    memory_id=str(memory_id),
                    exc_info=True,
                )
                # Continue with the next entity — one bad row doesn't
                # break the compile.
    return touched


async def backfill_entities_for_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    batch_size: int = 50,
) -> int:
    """One-off: extract + persist entities for every existing active
    memory of the subject. Returns total entity upsert count.

    Used to populate `subject_entities` for memories that pre-date this
    column — bench runs against a server that compiled subjects before
    Phase 2 landed need this before Phase 3 retrieval can do anything.

    Pages through memories `batch_size` at a time, commits per page,
    re-entrant on failure (the dedup means re-running upserts the same
    edges idempotently). For bench scales (~200 memories per subject),
    this completes in seconds; for production scales it's the operator's
    call when to run it.
    """
    total_touched = 0
    offset_id: uuid.UUID | None = None
    while True:
        # Page by created_at (stable order) with a tail keyset; the
        # repo helper that's been here forever uses asc ordering, so
        # we can just `list_memories_by_subject` and slice.
        page = await repo.list_memories_by_subject(
            session, subject_id, tenant_id=tenant_id, limit=batch_size * 100
        )
        # Slice manually so we control offset_id paging without a new
        # repo function. (For backfill the full list is fine to load.)
        if offset_id is not None:
            page = [m for m in page if m.created_at > _row_created_at(page, offset_id)]
        if not page:
            break
        page = list(page)[:batch_size]
        memories = [MemoryForEntities(id=m.id, content=m.content) for m in page]
        touched = await populate_entities_for_memories(
            session, memories, subject_id=subject_id, tenant_id=tenant_id
        )
        await session.commit()
        total_touched += touched
        if len(page) < batch_size:
            break
        offset_id = page[-1].id
    return total_touched


def _row_created_at(rows, target_id):
    for r in rows:
        if r.id == target_id:
            return r.created_at
    # Shouldn't happen; defensive — treat as "very old" so we keep the rest.
    return None
