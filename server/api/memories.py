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
from server.services.compilers import get_compiler
from server.services.embeddings import get_provider as get_embedding_provider
from server.services.embeddings.backfill import schedule_embedding_backfill
from server.services.conflicts import resolve_conflicts
from server.services import webhooks
from server.services import compile_jobs
from server.core.tracing import span
from server.core.dependencies import get_tenant_id

logger = structlog.stdlib.get_logger()

router = APIRouter(prefix="/v1/memories", tags=["memories"])


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
    """
    episodes = await repo.list_uncompiled_episodes(
        session, subject_id, tenant_id=tenant_id, limit=batch_size
    )
    if not episodes:
        return [], 0, 0

    compiler = get_compiler()
    if hasattr(compiler, "compile_async"):
        new_rows = await compiler.compile_async(list(episodes))
    else:
        loop = asyncio.get_running_loop()
        new_rows = await loop.run_in_executor(
            None, functools.partial(compiler.compile, list(episodes))
        )

    for row in new_rows:
        row.tenant_id = tenant_id
        session.add(row)
    await repo.mark_episodes_compiled(session, [ep.id for ep in episodes])

    superseded_ids = await resolve_conflicts(session, subject_id)
    if superseded_ids:
        logger.info("conflicts_resolved", superseded=len(superseded_ids))

    await session.commit()
    for row in new_rows:
        await session.refresh(row)

    schedule_embedding_backfill(
        [row.id for row in new_rows],
        [row.content for row in new_rows],
    )

    await webhooks.fire(
        "memories.compiled",
        {
            "subject_id": subject_id,
            "memories_created": len(new_rows),
        },
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

    except Exception as exc:
        logger.error("compile_failed", subject_id=subject_id, exc_info=True)
        if job_id:
            await compile_jobs.mark_failed_durable(job_id, str(exc))
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
        memory_responses, created, remaining = await _compile_one_batch(
            session, body.subject_id, tenant_id, settings.compile_batch_size
        )
        return CompileMemoriesResponse(
            subject_id=body.subject_id,
            memories_created=created,
            memories=memory_responses,
            has_more=remaining > 0,
            remaining_episodes=remaining,
        )


@router.get("/compile/{job_id}", summary="Check compile job status")
async def get_compile_status(job_id: str):
    """Poll for the status of an async compile job (durable — survives restarts)."""
    job = await compile_jobs.get_job_durable(job_id)
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
    limit: int = Query(20, le=100),
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    with span("search_memories", {"subject_id": subject_id, "semantic": semantic}):
        # Try semantic search if requested and query text is provided
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
                    results = await repo.search_memories_by_embedding(
                        session,
                        subject_id,
                        query_embedding,
                        tenant_id=tenant_id,
                        kind=kind,
                        limit=limit,
                    )
                    return SearchMemoriesResponse(
                        memories=[_to_response(row) for row, _dist in results]
                    )
                except Exception:
                    logger.warning("semantic_search_failed_falling_back", exc_info=True)
                    # Fall through to text search

        # Default: exact/text search
        rows = await repo.search_memories(
            session, subject_id, tenant_id=tenant_id, kind=kind, query=query, limit=limit
        )
        return SearchMemoriesResponse(memories=[_to_response(r) for r in rows])


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
