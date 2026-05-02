"""Memory routes — compile and search."""

from __future__ import annotations

import asyncio
import functools
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.schemas.requests import CompileMemoriesRequest
from server.schemas.responses import CompileMemoriesResponse, MemoryResponse, SearchMemoriesResponse
from server.services.compilers import get_compiler
from server.services.embeddings import get_provider as get_embedding_provider
from server.services.conflicts import resolve_conflicts
from server.services import webhooks
from server.services import compile_jobs
from server.core.tracing import span
from server.core.dependencies import get_tenant_id

logger = structlog.stdlib.get_logger()

router = APIRouter(prefix="/v1/memories", tags=["memories"])


async def _run_compile(
    subject_id: str, job_id: str | None = None, tenant_id: str | None = None
) -> CompileMemoriesResponse:
    """Core compilation logic — used by both sync and async paths."""
    from server.db.engine import get_session_factory

    if job_id:
        await compile_jobs.mark_running_durable(job_id)

    try:
        async with get_session_factory()() as session:
            episodes = await repo.list_uncompiled_episodes(session, subject_id, tenant_id=tenant_id)
            if not episodes:
                result = CompileMemoriesResponse(
                    subject_id=subject_id, memories_created=0, memories=[]
                )
                if job_id:
                    await compile_jobs.mark_completed_durable(job_id, 0, [])
                return result

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

            await webhooks.fire(
                "memories.compiled",
                {
                    "subject_id": subject_id,
                    "memories_created": len(new_rows),
                },
            )

            memory_responses = [_to_response(r) for r in new_rows]
            result = CompileMemoriesResponse(
                subject_id=subject_id,
                memories_created=len(new_rows),
                memories=memory_responses,
            )

            if job_id:
                await compile_jobs.mark_completed_durable(
                    job_id, len(new_rows), [m.model_dump(mode="json") for m in memory_responses]
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

    Pass `"async": true` to return immediately with a job_id for polling.
    """
    with span("compile_memories", {"subject_id": body.subject_id, "async": body.async_mode}):
        if body.async_mode:
            # Async mode — return job_id immediately, compile in background (durable)
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

        # Sync mode — block until compilation is done (backward compatible)
        episodes = await repo.list_uncompiled_episodes(
            session, body.subject_id, tenant_id=tenant_id
        )
        if not episodes:
            return CompileMemoriesResponse(
                subject_id=body.subject_id, memories_created=0, memories=[]
            )

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

        superseded_ids = await resolve_conflicts(session, body.subject_id)
        if superseded_ids:
            logger.info("conflicts_resolved", superseded=len(superseded_ids))

        await session.commit()
        for row in new_rows:
            await session.refresh(row)

        # Generate embeddings in background (don't block response)
        memory_ids = [row.id for row in new_rows]
        memory_texts = [row.content for row in new_rows]
        asyncio.create_task(_generate_embeddings_background(memory_ids, memory_texts))

        await webhooks.fire(
            "memories.compiled",
            {
                "subject_id": body.subject_id,
                "memories_created": len(new_rows),
            },
        )

        return CompileMemoriesResponse(
            subject_id=body.subject_id,
            memories_created=len(new_rows),
            memories=[_to_response(r) for r in new_rows],
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


async def _generate_embeddings_background(memory_ids: list, texts: list[str]) -> None:
    """Generate embeddings for memories in the background (non-blocking).

    Writes go through the ORM so the pgvector SQLAlchemy adapter handles
    list[float] → vector(1536) serialization. Previously this used raw SQL
    with str(emb), which worked because pgvector parses JSON-array TEXT —
    but the ORM path is type-safe and survives future column-type changes.
    """
    from sqlalchemy import update

    from server.db.engine import get_session_factory
    from server.db.tables import MemoryRow

    provider = get_embedding_provider()
    if not provider or not texts:
        return

    try:
        embeddings = await provider.embed_texts(texts)
        async with get_session_factory()() as session:
            for mid, emb in zip(memory_ids, embeddings):
                await session.execute(
                    update(MemoryRow)
                    .where(MemoryRow.id == mid)
                    .values(embedding=emb)
                )
            await session.commit()
        logger.info("embeddings_generated_background", count=len(embeddings))
    except Exception:
        logger.warning("background_embedding_failed", exc_info=True)


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
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
