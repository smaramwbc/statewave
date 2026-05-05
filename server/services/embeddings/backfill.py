"""Background embedding generation for newly-inserted memories.

Two write paths in Statewave create memories:
  * `POST /v1/memories/compile` — LLM/heuristic extraction from episodes.
  * `_ingest_records_async` (memory_packs) — bulk import of bundled or
    operator-supplied starter packs / .swmem archives.

Both paths used to embed inconsistently — the compile path scheduled a
fire-and-forget embedding task, the import path silently wrote
`embedding=NULL` for every row (the bundled JSONL ships without
embeddings, and there was no post-insert embed step). That asymmetry
broke vector retrieval on every clean import. This module is the shared
implementation both paths now use, so the rule is uniform: every memory
written by the server gets an embedding scheduled, and any row that
arrived with one already pre-computed is left alone.

Design:
  * Fire-and-forget via `asyncio.create_task` — embedding latency must
    never be in the user's request path. Failures are logged at WARNING,
    not propagated; a missing embedding degrades retrieval but does not
    break the request.
  * Row-level granularity — pass only memories that need embedding.
    Callers filter `embedding is None` themselves, so this function
    doesn't have to round-trip the DB to discover which rows are stale.
  * Provider-aware no-op — when the configured provider is `none`, this
    function returns immediately without touching the DB.
"""

from __future__ import annotations

import asyncio
from typing import Sequence
from uuid import UUID

import structlog
from sqlalchemy import update

from server.db.engine import get_session_factory
from server.db.tables import MemoryRow
from server.services.embeddings import get_provider

logger = structlog.stdlib.get_logger()


async def generate_embeddings_background(
    memory_ids: Sequence[UUID | str],
    texts: Sequence[str],
) -> None:
    """Compute embeddings for the given memories and persist them.

    Designed to be scheduled with `asyncio.create_task(...)` from a
    request handler — never `await`ed in the request path. Both inputs
    must be the same length and ordered consistently.

    No-op when:
      * `memory_ids` / `texts` is empty,
      * the embedding provider is disabled (`STATEWAVE_EMBEDDING_PROVIDER=none`),
      * the provider call fails — failure is logged and swallowed; the
        memories stay with `embedding IS NULL` and retrieval falls back
        to its non-vector path. Re-running compile or a future write
        will retry.
    """
    if not memory_ids or not texts:
        return
    if len(memory_ids) != len(texts):
        logger.warning(
            "background_embedding_mismatched_inputs",
            ids=len(memory_ids),
            texts=len(texts),
        )
        return

    provider = get_provider()
    if provider is None:
        return

    try:
        embeddings = await provider.embed_texts(list(texts))
        async with get_session_factory()() as session:
            for mid, emb in zip(memory_ids, embeddings):
                await session.execute(
                    update(MemoryRow).where(MemoryRow.id == mid).values(embedding=emb)
                )
            await session.commit()
        logger.info("embeddings_generated_background", count=len(embeddings))
    except Exception:
        logger.warning("background_embedding_failed", exc_info=True)


def schedule_embedding_backfill(
    memory_ids: Sequence[UUID | str],
    texts: Sequence[str],
) -> asyncio.Task | None:
    """Convenience: schedule `generate_embeddings_background` on the
    running event loop and return the task handle (or None when there's
    nothing to do).

    Callers in request handlers should use this rather than importing
    `asyncio.create_task` themselves — keeps the fire-and-forget pattern
    in one place and makes it greppable.
    """
    if not memory_ids or not texts:
        return None
    return asyncio.create_task(generate_embeddings_background(memory_ids, texts))
