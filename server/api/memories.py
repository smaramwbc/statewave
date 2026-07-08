"""Memory routes — compile and search."""

from __future__ import annotations

import asyncio
import functools
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.db.tables import MemoryRow
from server.schemas.requests import CompileMemoriesRequest, SetMemoryLabelsRequest
from server.schemas.responses import CompileMemoriesResponse, MemoryResponse, SearchMemoriesResponse
from server.services.compilers import CompilationError, get_compiler
from server.services.embeddings import get_provider as get_embedding_provider
from server.services.embeddings.backfill import schedule_embedding_backfill
from server.services.conflicts import resolve_conflicts
from server.services import webhooks
from server.services import compile_jobs
from server.core.tracing import span
from server.core.dependencies import get_tenant_id

logger = structlog.stdlib.get_logger()

router = APIRouter(prefix="/v1/memories", tags=["memories"])

# Client-facing error text for compile failures. Deliberately static and
# free of any exception detail: the underlying exception can carry provider
# internals (URLs, config, partial keys, stack traces) that must not reach an
# external caller (CWE-209). The full `str(exc)` is logged server-side at every
# raise site; the client only needs the actionable next step. `CompilationError`
# is an expected, recoverable misconfiguration, so its message points the caller
# at the fix; the catch-all path stays vague because the cause is unknown.
_COMPILE_FAILED_MESSAGE = (
    "Memory compilation could not run because the configured compiler or "
    "provider is unavailable or misconfigured. The affected episodes were left "
    "uncompiled — fix the provider configuration and retry. See server logs for "
    "the underlying error."
)
_COMPILE_INTERNAL_ERROR_MESSAGE = "An internal error occurred during compilation."


async def _compile_one_batch(
    session: AsyncSession, subject_id: str, tenant_id: str | None, batch_size: int
) -> tuple[list[MemoryResponse], int, int]:
    """Compile ONE batch of uncompiled episodes for `subject_id`.

    Returns `(memory_responses, memories_created, remaining_episodes)`.
    `remaining_episodes` is the count of still-uncompiled episodes AFTER
    this batch was marked compiled — feeds `has_more` in the response and
    the drain loop in `_run_compile`.

    Does its own commit so each batch is durable independently — if the
    process dies mid-drain, no episode is lost or double-counted.

    Raises `CompilationError` (propagated from the compiler) when extraction
    could not run — e.g. the LLM compiler has no reachable provider key. In
    that case the episodes are deliberately left UNCOMPILED: we return before
    `mark_episodes_compiled`/`commit`, so a later, correctly configured
    recompile reprocesses them instead of the failure silently consuming them
    (issue #201). An empty `new_rows` from a *successful* run is the opposite
    case — a legitimate "extracted nothing" — and does mark the episodes
    compiled.
    """
    from server.core.config import settings

    episodes = await repo.list_uncompiled_episodes(
        session, subject_id, tenant_id=tenant_id, limit=batch_size
    )
    if not episodes:
        return [], 0, 0

    compiler = get_compiler()
    # A CompilationError here propagates intentionally: nothing below this
    # point runs, so the episodes are never marked compiled or committed.
    if hasattr(compiler, "compile_async"):
        new_rows = await compiler.compile_async(list(episodes))
    else:
        loop = asyncio.get_running_loop()
        new_rows = await loop.run_in_executor(
            None, functools.partial(compiler.compile, list(episodes))
        )

    # Near-duplicate dedup (runs BEFORE reconcile). Full-conversation windowed
    # compile produces many restated/overlapping facts; this cheap, deterministic
    # pass collapses them so (a) retrieval isn't diluted by near-duplicates and
    # (b) reconcile + entity population run on a far smaller set. Non-destructive
    # (provenance unioned) and fail-open — returns the candidates unchanged on any
    # error. Stapled embeddings on survivors are reused for the memory.embedding
    # column below. See server/services/dedup.py.
    if settings.dedup_compile_enabled and new_rows:
        try:
            from server.services.dedup import dedup_candidates

            new_rows = await dedup_candidates(new_rows)
        except Exception:
            logger.warning("dedup_failed", subject_id=subject_id, exc_info=True)

    # Phase 4 + 5b — context-aware reconcile. One LLM call
    # decides, per freshly-extracted candidate, whether it is new, a duplicate
    # (dropped), a newer value of an existing memory, or a contradiction — and
    # supersedes the stale/contradicted existing memories. Runs BEFORE the
    # candidates are added to the session, so the "existing" view it reads is
    # exactly the committed memory (candidates can't pollute it). Fail-open:
    # `reconcile_compile_batch` returns the candidates unchanged on any error,
    # so a reconcile failure can never lose a memory. Gated for ablation/bench.
    reconcile_superseded_ids: set = set()
    if settings.reconcile_compile_enabled and new_rows:
        try:
            from server.services.reconcile import reconcile_compile_batch

            new_rows, reconcile_superseded_ids = await reconcile_compile_batch(
                session, subject_id, new_rows, tenant_id=tenant_id
            )
        except Exception:
            logger.warning("reconcile_failed", subject_id=subject_id, exc_info=True)
            reconcile_superseded_ids = set()

    for row in new_rows:
        row.tenant_id = tenant_id
        session.add(row)
    if reconcile_superseded_ids:
        await repo.mark_memories_superseded(session, list(reconcile_superseded_ids))
        logger.info(
            "reconcile_superseded",
            subject_id=subject_id,
            superseded=len(reconcile_superseded_ids),
        )
    await repo.mark_episodes_compiled(session, [ep.id for ep in episodes])

    superseded_ids = await resolve_conflicts(session, subject_id, tenant_id=tenant_id)
    if superseded_ids:
        logger.info("conflicts_resolved", superseded=len(superseded_ids))

    await session.commit()
    for row in new_rows:
        await session.refresh(row)

    # Only backfill rows that don't already carry an embedding. Dedup's semantic
    # pass computes and staples embeddings onto its surviving canonicals, so those
    # are persisted with the row above and need no second embedding round-trip.
    rows_needing_embedding = [r for r in new_rows if getattr(r, "embedding", None) is None]
    schedule_embedding_backfill(
        [row.id for row in rows_needing_embedding],
        [row.content for row in rows_needing_embedding],
    )

    # Phase 2 of the cross-session retrieval improvements: populate the per-subject
    # entity store from this batch's newly-compiled memories so Phase 3
    # retrieval (entity boost) has something to query. Best-effort — a
    # failure here must not roll back the compile commit above. The function
    # fans out LLM extraction calls in parallel + does ONE batched embedding
    # round-trip. It is one LLM call per memory, the dominant cost on large
    # full-conversation compiles, so it is skipped entirely when entity-boost
    # retrieval is not in use (settings.entity_population_enabled).
    if settings.entity_population_enabled:
        try:
            from server.services.entities import (
                MemoryForEntities,
                populate_entities_for_memories,
            )

            active_new_rows = [r for r in new_rows if r.id not in superseded_ids]
            touched = await populate_entities_for_memories(
                session,
                [MemoryForEntities(id=r.id, content=r.content) for r in active_new_rows],
                subject_id=subject_id,
                tenant_id=tenant_id,
            )
            if touched:
                await session.commit()
                logger.info(
                    "entities_populated",
                    subject_id=subject_id,
                    touched=touched,
                    memories=len(active_new_rows),
                )
        except Exception:
            # Phase 3 retrieval handles an empty entity store as a no-op,
            # so the compile result for the caller is identical with or
            # without this step. Logged at WARNING for operator visibility.
            logger.warning("entity_population_failed", subject_id=subject_id, exc_info=True)

    await webhooks.fire(
        "memories.compiled",
        {
            "subject_id": subject_id,
            "memories_created": len(new_rows),
        },
        tenant_id=tenant_id,
    )

    remaining = await repo.count_uncompiled_episodes(
        session, subject_id, tenant_id=tenant_id
    )
    return [_to_response(r) for r in new_rows], len(new_rows), remaining


async def _run_compile(
    subject_id: str, job_id: str | None = None, tenant_id: str | None = None
) -> CompileMemoriesResponse:
    """Async compile path — drains the subject batch by batch (issue #134).

    The async caller asked us not to block them. In return we promise the
    job actually finishes the work: we loop `_compile_one_batch` until
    `remaining_episodes == 0`, accumulate `memories_created`, and update
    the durable job row each iteration so polling clients see progress.
    Bounded by `settings.compile_max_iterations` so a misbehaving compiler
    can't burn forever.
    """
    from server.core.config import settings
    from server.db.engine import get_session_factory

    if job_id:
        await compile_jobs.mark_running_durable(job_id)

    total_created = 0
    last_batch_responses: list[MemoryResponse] = []
    last_remaining = 0
    try:
        async with get_session_factory()() as session:
            for iteration in range(settings.compile_max_iterations):
                batch_responses, created, remaining = await _compile_one_batch(
                    session, subject_id, tenant_id, settings.compile_batch_size
                )
                total_created += created
                last_batch_responses = batch_responses
                last_remaining = remaining

                if job_id and (created or iteration == 0):
                    await compile_jobs.update_progress_durable(job_id, total_created)

                # Drain on `remaining` alone — an empty `batch_responses`
                # with `remaining > 0` means the compiler produced no rows
                # this batch (rare but possible: all episodes filtered by
                # the compiler), and we should keep going. The iteration
                # cap below guards against a compiler that never advances.
                if remaining == 0:
                    break
            else:
                # Loop exhausted without draining — surface, don't silently
                # hide it. The job completes (we did compile a lot) but the
                # log entry tells the operator they hit the iteration cap.
                logger.warning(
                    "compile_drain_iteration_cap_hit",
                    subject_id=subject_id,
                    iterations=settings.compile_max_iterations,
                    total_created=total_created,
                    remaining=last_remaining,
                )

        result = CompileMemoriesResponse(
            subject_id=subject_id,
            memories_created=total_created,
            memories=last_batch_responses,
            has_more=last_remaining > 0,
            remaining_episodes=last_remaining,
        )

        if job_id:
            await compile_jobs.mark_completed_durable(
                job_id,
                total_created,
                [m.model_dump(mode="json") for m in last_batch_responses],
            )

        return result

    except CompilationError as exc:
        # Extraction could not run (e.g. provider misconfig / no key). This is
        # an expected, recoverable failure mode, not a code bug — log it
        # without a stacktrace. Episodes from the failed batch were left
        # uncompiled (issue #201); any batches that already succeeded in this
        # drain are committed. Mark the job failed so the polling client sees
        # the reason instead of a clean zero-memory completion.
        logger.warning("compile_unavailable", subject_id=subject_id, error=str(exc))
        if job_id:
            # `job.error` is returned to the polling client by
            # `get_compile_status`, so store the static message — never the raw
            # exception (CWE-209). The detail is in the log line above.
            await compile_jobs.mark_failed_durable(job_id, _COMPILE_FAILED_MESSAGE)
        raise
    except Exception:
        logger.error("compile_failed", subject_id=subject_id, exc_info=True)
        if job_id:
            await compile_jobs.mark_failed_durable(job_id, _COMPILE_INTERNAL_ERROR_MESSAGE)
        raise


@router.post("/compile", summary="Compile memories from episodes")
async def compile_memories(
    body: CompileMemoriesRequest,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Compile new memories from unprocessed episodes.

    Sync mode processes at most `STATEWAVE_COMPILE_BATCH_SIZE` (default
    500) uncompiled episodes per call and returns `has_more=True` plus
    `remaining_episodes` whenever the backlog isn't drained. Clients can
    loop until `has_more` is False, or pass `"async": true` and let the
    server drain the whole subject in a durable background job.
    """
    from server.core.config import settings

    with span("compile_memories", {"subject_id": body.subject_id, "async": body.async_mode}):
        if body.async_mode:
            # Async mode — return job_id immediately, compile in background (durable).
            # The background task drains the subject; the client polls
            # `/v1/memories/compile/{job_id}` for completion.
            job = await compile_jobs.submit_job_durable(body.subject_id, tenant_id=tenant_id)
            asyncio.create_task(_run_compile(body.subject_id, job.id, tenant_id=tenant_id))
            return JSONResponse(
                status_code=202,
                content={
                    "job_id": job.id,
                    "status": "pending",
                    "subject_id": body.subject_id,
                },
            )

        # Sync mode — bounded per-call: process at most one batch, then
        # report `has_more` so the caller knows whether to loop. Bounded
        # latency is the trade-off for not surprising long-standing
        # sync clients with multi-minute compile calls.
        try:
            memory_responses, created, remaining = await _compile_one_batch(
                session, body.subject_id, tenant_id, settings.compile_batch_size
            )
        except CompilationError as exc:
            # Extraction could not run (e.g. LLM compiler with no reachable
            # key). The episodes were left uncompiled (issue #201); surface
            # the failure as a 5xx instead of a misleading
            # `memories_created: 0` so the caller knows to fix the provider
            # config and retry rather than assuming there was nothing to
            # compile.
            logger.warning("compile_unavailable", subject_id=body.subject_id, error=str(exc))
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "code": "compilation_failed",
                        "message": _COMPILE_FAILED_MESSAGE,
                    }
                },
            )
        return CompileMemoriesResponse(
            subject_id=body.subject_id,
            memories_created=created,
            memories=memory_responses,
            has_more=remaining > 0,
            remaining_episodes=remaining,
        )


@router.get("/compile/{job_id}", summary="Check compile job status")
async def get_compile_status(
    job_id: str,
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Poll for the status of an async compile job (durable — survives restarts)."""
    job = await compile_jobs.get_job_durable(job_id, tenant_id=tenant_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found or expired"})

    response: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status.value,
        "subject_id": job.subject_id,
    }
    if job.status == compile_jobs.JobStatus.completed:
        response["memories_created"] = job.memories_created
        response["memories"] = job.memories
    elif job.status == compile_jobs.JobStatus.failed:
        response["error"] = job.error

    return JSONResponse(content=response)


@router.get("/search", response_model=SearchMemoriesResponse, summary="Search memories")
async def search_memories(
    subject_id: str = Query(...),
    kind: str | None = Query(None),
    query: str | None = Query(None, alias="q"),
    semantic: bool = Query(False, description="Use semantic similarity search when available"),
    hybrid: bool = Query(
        True,
        description=(
            "Blend semantic cosine with Postgres BM25 (ts_rank_cd) and entity "
            "boost for hybrid retrieval. Requires semantic=true and a non-empty "
            "query. Default flipped to True on 2026-06-19 — v10 bench validated "
            "this as a strict improvement across LoCoMo (+2.1), LongMemEval "
            "(+16.0 vs Phase-1 hybrid), and BEAM (+1.8). Pass hybrid=False to "
            "force the pre-2026-06-19 pure-pgvector path."
        ),
    ),
    entity_weight: float = Query(
        0.0,
        ge=0.0,
        le=10.0,
        description=(
            "Weight of the entity-boost lane in the hybrid blend. Default 0.0 — "
            "the entity lane is OFF by default. The published Statewave "
            "benchmark results (LoCoMo 0.905, LongMemEval 0.967) ran with "
            "entity_weight=0, and the entity lane showed no generalizable gain "
            "(it can regress temporal_reasoning on generic-entity questions). "
            "Pass a positive value (e.g. 0.3-1.0) to weight entity matches more "
            "heavily for entity-centric workloads. Only consulted when "
            "hybrid=true."
        ),
    ),
    entity_max_distance: float = Query(
        0.3,
        ge=0.0,
        le=2.0,
        description=(
            "Maximum cosine distance for an entity to count as 'matching' the "
            "query. Default 0.3 as of 2026-06-19 — corresponds to cosine "
            "similarity ≥ 0.7, tight enough to reject weak matches that "
            "pollute the boost lane on summarization / temporal questions. "
            "Pass 0.5 for the pre-2026-06-19 looser threshold. Only consulted "
            "when hybrid=true."
        ),
    ),
    rerank: bool = Query(
        False,
        description=(
            "LLM-rerank a hybrid candidate pool to surface the precise answer "
            "fact (single-fact precision). Retrieves `rerank_pool` candidates, "
            "an LLM scores relevance to the query, returns the best `limit`. "
            "Requires hybrid=true + a query. Fail-open to hybrid order."
        ),
    ),
    rerank_pool: int = Query(
        60, ge=1, le=1000,
        description="Candidate pool size fed to the reranker when rerank=true.",
    ),
    limit: int = Query(20, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Search a subject's memories by text or semantic similarity.

    The `search_mode` field in the response reports which path actually ran,
    so callers can tell whether semantic search really executed:
    - `semantic`: embedding (or hybrid) search ran.
    - `text`: plain text search — either `semantic` was not requested, or it
      was requested without a `q` query (in which case it is ignored).
    - `text_fallback`: `semantic` was requested with a `q`, but could not run
      (no embedding provider configured, or the provider errored), so text
      search ran instead. Check the server logs for the underlying cause.
    """
    with span(
        "search_memories",
        {
            "subject_id": subject_id,
            "semantic": semantic,
            "hybrid": hybrid,
            "entity_weight": entity_weight,
            "entity_max_distance": entity_max_distance,
            "rerank": rerank,
        },
    ):
        # `search_mode` records which path actually ran (issue #281): text is
        # the default; the semantic branch upgrades it to "semantic" on success
        # or "text_fallback" if semantic was requested but could not run.
        search_mode = "text"
        # Try semantic / hybrid search if requested and query text is provided
        if semantic and query:
            provider = get_embedding_provider()
            if provider:
                try:
                    # Cross-machine query embedding cache — same path as
                    # /v1/context. Repeated /v1/memories/search?semantic=
                    # calls cluster-wide pay the provider round-trip once.
                    from server.db.engine import get_session_factory
                    from server.services.embeddings.query_cache import cached_embed_query
                    query_embedding = await cached_embed_query(
                        get_session_factory(), provider, query
                    )
                    if hybrid:
                        # Hybrid retrieval: semantic + BM25 +
                        # entity-boost. See repositories.search_memories_hybrid
                        # for the blend formula. `entity_weight=0` disables
                        # the entity lane, falling back to Phase-1
                        # (semantic+BM25) — useful for ablations.
                        # When reranking, pull a larger candidate pool by hybrid
                        # similarity, then let the LLM reranker pick the best
                        # `limit` (fixes single-fact precision — the answer fact
                        # ranking mid-pack). Otherwise retrieve `limit` directly.
                        pool = max(limit, rerank_pool) if rerank else limit
                        hybrid_results = await repo.search_memories_hybrid(
                            session,
                            subject_id,
                            query,
                            query_embedding,
                            tenant_id=tenant_id,
                            kind=kind,
                            limit=pool,
                            entity_weight=entity_weight,
                            use_entity_boost=entity_weight > 0.0,
                            entity_max_distance=entity_max_distance,
                        )
                        rows = [row for row, _score, _bd in hybrid_results]
                        if rerank and len(rows) > limit:
                            from server.services.reranker import rerank_memories

                            rows = await rerank_memories(query, rows, limit)
                        else:
                            rows = rows[:limit]
                        return SearchMemoriesResponse(
                            memories=[_to_response(row) for row in rows],
                            search_mode="semantic",
                        )
                    results = await repo.search_memories_by_embedding(
                        session,
                        subject_id,
                        query_embedding,
                        tenant_id=tenant_id,
                        kind=kind,
                        limit=limit,
                    )
                    return SearchMemoriesResponse(
                        memories=[_to_response(row) for row, _dist in results],
                        search_mode="semantic",
                    )
                except Exception:
                    logger.warning("semantic_search_failed_falling_back", exc_info=True)
                    # Semantic was requested but errored — fall through to text
                    # search and tell the caller it was a fallback.
                    search_mode = "text_fallback"
            else:
                # Semantic requested with a query, but no embedding provider is
                # configured — text search runs instead of semantic.
                search_mode = "text_fallback"

        # Default: exact/text search
        rows = await repo.search_memories(
            session, subject_id, tenant_id=tenant_id, kind=kind, query=query, limit=limit
        )
        return SearchMemoriesResponse(
            memories=[_to_response(r) for r in rows], search_mode=search_mode
        )


def _to_response(row) -> MemoryResponse:
    return MemoryResponse(
        id=row.id,
        subject_id=row.subject_id,
        kind=row.kind,
        content=row.content,
        summary=row.summary,
        confidence=row.confidence,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        source_episode_ids=row.source_episode_ids or [],
        metadata=row.metadata_,
        status=row.status,
        sensitivity_labels=list(getattr(row, "sensitivity_labels", None) or []),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.patch("/{memory_id}/labels", response_model=MemoryResponse)
async def set_memory_labels(
    memory_id: uuid.UUID,
    body: SetMemoryLabelsRequest,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Replace a memory's `sensitivity_labels` with the supplied list.

    Tenant-scoped: a memory belonging to another tenant returns 404,
    not 403, so a tenant cannot probe another tenant's id space by
    PATCHing an id and looking at the error code.

    Labels are deduplicated, lowercased, and stripped of surrounding
    whitespace before write — operator-supplied strings are
    notoriously inconsistent, and the policy evaluator does exact
    match, so normalizing at the write boundary is the only place to
    do it safely. An empty list clears all labels (the memory becomes
    untagged → policy default-allow).
    """
    # Canonicalize labels so policy evaluation is stable regardless of
    # how the operator typed them. Cap at 32 entries (Pydantic
    # validation already enforces this; defensive recheck here).
    normalized = sorted({lbl.strip().lower() for lbl in body.sensitivity_labels if lbl.strip()})
    if len(normalized) > 32:
        raise HTTPException(status_code=400, detail="too many labels (max 32)")

    stmt = select(MemoryRow).where(MemoryRow.id == memory_id)
    if tenant_id is not None:
        stmt = stmt.where(MemoryRow.tenant_id == tenant_id)
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="memory not found")

    update_stmt = (
        update(MemoryRow)
        .where(MemoryRow.id == memory_id)
        .values(sensitivity_labels=normalized)
    )
    if tenant_id is not None:
        update_stmt = update_stmt.where(MemoryRow.tenant_id == tenant_id)
    await session.execute(update_stmt)
    await session.commit()

    # Re-fetch so the response carries the post-write timestamp.
    result = await session.execute(stmt)
    row = result.scalar_one()
    logger.info(
        "memory_labels_set",
        memory_id=str(memory_id),
        tenant_id=tenant_id,
        labels=normalized,
    )
    return _to_response(row)
