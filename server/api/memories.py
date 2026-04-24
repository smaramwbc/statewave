"""Memory routes — compile and search."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.schemas.requests import CompileMemoriesRequest
from server.schemas.responses import CompileMemoriesResponse, MemoryResponse, SearchMemoriesResponse
from server.services.compilers import get_compiler
from server.services.embeddings import get_provider as get_embedding_provider
from server.services.conflicts import resolve_conflicts
from server.services import webhooks
from server.core.tracing import span

logger = structlog.stdlib.get_logger()

router = APIRouter(prefix="/v1/memories", tags=["memories"])


@router.post("/compile", response_model=CompileMemoriesResponse, summary="Compile memories from episodes")
async def compile_memories(
    body: CompileMemoriesRequest,
    session: AsyncSession = Depends(get_session),
):
    """Compile new memories from unprocessed episodes. Idempotent — recompiling the same subject produces no duplicates."""
    with span("compile_memories", {"subject_id": body.subject_id}):
        # Only compile episodes that haven't been compiled yet (idempotent)
        episodes = await repo.list_uncompiled_episodes(session, body.subject_id)
        if not episodes:
            return CompileMemoriesResponse(
                subject_id=body.subject_id,
                memories_created=0,
                memories=[],
            )
        new_rows = get_compiler().compile(list(episodes))

        # Generate embeddings if provider is available
        provider = get_embedding_provider()
        if provider and new_rows:
            texts = [row.content for row in new_rows]
            try:
                embeddings = await provider.embed_texts(texts)
                for row, emb in zip(new_rows, embeddings):
                    row.embedding = emb
                logger.info("embeddings_generated", count=len(embeddings), provider=type(provider).__name__)
            except Exception:
                logger.warning("embedding_generation_failed", exc_info=True)
                # Continue without embeddings — graceful degradation

        for row in new_rows:
            session.add(row)
        # Mark episodes as compiled so they won't be reprocessed
        await repo.mark_episodes_compiled(
            session, [ep.id for ep in episodes]
        )

        # Auto-resolve memory conflicts before committing (single transaction)
        superseded_ids = await resolve_conflicts(session, body.subject_id)
        if superseded_ids:
            logger.info("conflicts_resolved", superseded=len(superseded_ids))

        await session.commit()
        for row in new_rows:
            await session.refresh(row)

        await webhooks.fire("memories.compiled", {
            "subject_id": body.subject_id,
            "memories_created": len(new_rows),
        })

        return CompileMemoriesResponse(
            subject_id=body.subject_id,
            memories_created=len(new_rows),
            memories=[_to_response(r) for r in new_rows],
        )


@router.get("/search", response_model=SearchMemoriesResponse, summary="Search memories")
async def search_memories(
    subject_id: str = Query(...),
    kind: str | None = Query(None),
    query: str | None = Query(None, alias="q"),
    semantic: bool = Query(False, description="Use semantic similarity search when available"),
    limit: int = Query(20, le=100),
    session: AsyncSession = Depends(get_session),
):
    with span("search_memories", {"subject_id": subject_id, "semantic": semantic}):
        # Try semantic search if requested and query text is provided
        if semantic and query:
            provider = get_embedding_provider()
            if provider:
                try:
                    query_embedding = await provider.embed_query(query)
                    results = await repo.search_memories_by_embedding(
                        session, subject_id, query_embedding, kind=kind, limit=limit,
                    )
                    return SearchMemoriesResponse(
                        memories=[_to_response(row) for row, _dist in results]
                    )
                except Exception:
                    logger.warning("semantic_search_failed_falling_back", exc_info=True)
                    # Fall through to text search

        # Default: exact/text search
        rows = await repo.search_memories(session, subject_id, kind=kind, query=query, limit=limit)
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
