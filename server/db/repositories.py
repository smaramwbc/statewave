"""Data-access layer. All SQL lives here.

Tenant scoping: when tenant_id is provided, all queries are filtered to
that tenant. When tenant_id is None (single-tenant mode), no filter is
applied — preserving backward compatibility for local/single-tenant use.
"""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.tables import EpisodeRow, MemoryRow, ResolutionRow, SubjectHealthCacheRow


def _tenant_filter(stmt, column, tenant_id: str | None):
    """Apply tenant filter to a query when tenant_id is set."""
    if tenant_id is not None:
        return stmt.where(column == tenant_id)
    return stmt


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


async def insert_episode(session: AsyncSession, row: EpisodeRow) -> EpisodeRow:
    session.add(row)
    await session.flush()
    return row


async def list_episodes_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 100,
) -> Sequence[EpisodeRow]:
    stmt = (
        select(EpisodeRow)
        .where(EpisodeRow.subject_id == subject_id)
        .order_by(EpisodeRow.created_at.asc())
        .limit(limit)
    )
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_uncompiled_episodes(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 500,
) -> Sequence[EpisodeRow]:
    """Fetch episodes that have never been compiled."""
    stmt = (
        select(EpisodeRow)
        .where(EpisodeRow.subject_id == subject_id)
        .where(EpisodeRow.last_compiled_at.is_(None))
        .order_by(EpisodeRow.created_at.asc())
        .limit(limit)
    )
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def mark_episodes_compiled(
    session: AsyncSession,
    episode_ids: list[uuid.UUID],
) -> None:
    """Mark episodes as compiled so they won't be reprocessed."""
    if not episode_ids:
        return
    stmt = (
        update(EpisodeRow)
        .where(EpisodeRow.id.in_(episode_ids))
        .values(last_compiled_at=text("now()"))
    )
    await session.execute(stmt)


async def get_episodes_by_ids(
    session: AsyncSession,
    ids: list[uuid.UUID],
) -> Sequence[EpisodeRow]:
    if not ids:
        return []
    stmt = select(EpisodeRow).where(EpisodeRow.id.in_(ids))
    result = await session.execute(stmt)
    return result.scalars().all()


async def delete_episodes_by_subject(
    session: AsyncSession, subject_id: str, *, tenant_id: str | None = None
) -> int:
    stmt = delete(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.rowcount  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


async def insert_memory(session: AsyncSession, row: MemoryRow) -> MemoryRow:
    session.add(row)
    await session.flush()
    return row


async def search_memories(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    kind: str | None = None,
    query: str | None = None,
    limit: int = 20,
) -> Sequence[MemoryRow]:
    stmt = (
        select(MemoryRow)
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
    )
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    if kind:
        stmt = stmt.where(MemoryRow.kind == kind)
    if query:
        stmt = stmt.where(MemoryRow.content.ilike(f"%{query}%"))
    stmt = stmt.order_by(MemoryRow.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_memories_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 100,
) -> Sequence[MemoryRow]:
    stmt = (
        select(MemoryRow)
        .where(MemoryRow.subject_id == subject_id)
        .order_by(MemoryRow.created_at.asc())
        .limit(limit)
    )
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def delete_memories_by_subject(
    session: AsyncSession, subject_id: str, *, tenant_id: str | None = None
) -> int:
    stmt = delete(MemoryRow).where(MemoryRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.rowcount  # type: ignore[return-value]


async def list_active_memories_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 500,
) -> Sequence[MemoryRow]:
    """Fetch active memories for a subject (for conflict resolution)."""
    stmt = (
        select(MemoryRow)
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
        .order_by(MemoryRow.created_at.asc())
        .limit(limit)
    )
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def mark_memories_superseded(
    session: AsyncSession,
    memory_ids: list[uuid.UUID],
) -> None:
    """Mark memories as superseded (conflict resolution)."""
    if not memory_ids:
        return
    stmt = (
        update(MemoryRow)
        .where(MemoryRow.id.in_(memory_ids))
        .values(status="superseded", updated_at=text("now()"))
    )
    await session.execute(stmt)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


async def search_memories_by_embedding(
    session: AsyncSession,
    subject_id: str,
    query_embedding: list[float],
    *,
    tenant_id: str | None = None,
    kind: str | None = None,
    limit: int = 20,
) -> list[tuple[MemoryRow, float]]:
    """Find memories by cosine distance. Returns (row, distance) tuples.

    Uses pgvector's native `<=>` cosine-distance operator. The HNSW index
    on `memories.embedding` (created in migration 0013) makes this an
    indexed nearest-neighbor lookup — sub-millisecond at our corpus
    sizes — instead of the previous fetch-all-and-cosine-in-Python path.

    Distance is in [0, 2] where 0 is identical and 2 is opposite (cosine
    distance, not similarity); callers convert to [0, 1] similarity if
    they need it. Lower is better; the query orders ascending and limits.
    """
    distance_expr = MemoryRow.embedding.cosine_distance(query_embedding)
    stmt = (
        select(MemoryRow, distance_expr.label("distance"))
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
        .where(MemoryRow.embedding.isnot(None))
    )
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    if kind:
        stmt = stmt.where(MemoryRow.kind == kind)
    stmt = stmt.order_by(distance_expr).limit(limit)
    result = await session.execute(stmt)
    return [(row, float(distance)) for row, distance in result.all()]


# ---------------------------------------------------------------------------
# Subject listing
# ---------------------------------------------------------------------------


async def list_subjects(
    session: AsyncSession,
    *,
    tenant_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Return distinct subject IDs with episode and memory counts."""
    # Base episode query, tenant-scoped
    ep_base = select(EpisodeRow.subject_id, func.count().label("episode_count"))
    if tenant_id is not None:
        ep_base = ep_base.where(EpisodeRow.tenant_id == tenant_id)
    ep_count = ep_base.group_by(EpisodeRow.subject_id).subquery()

    # Base memory query, tenant-scoped
    mem_base = select(MemoryRow.subject_id, func.count().label("memory_count"))
    if tenant_id is not None:
        mem_base = mem_base.where(MemoryRow.tenant_id == tenant_id)
    mem_count = mem_base.group_by(MemoryRow.subject_id).subquery()

    # UNION of subject_ids from both tables (tenant-scoped)
    ep_subjects = select(EpisodeRow.subject_id)
    mem_subjects = select(MemoryRow.subject_id)
    if tenant_id is not None:
        ep_subjects = ep_subjects.where(EpisodeRow.tenant_id == tenant_id)
        mem_subjects = mem_subjects.where(MemoryRow.tenant_id == tenant_id)
    all_subjects = ep_subjects.union(mem_subjects).subquery()

    stmt = (
        select(
            all_subjects.c.subject_id,
            func.coalesce(ep_count.c.episode_count, 0).label("episode_count"),
            func.coalesce(mem_count.c.memory_count, 0).label("memory_count"),
        )
        .outerjoin(ep_count, all_subjects.c.subject_id == ep_count.c.subject_id)
        .outerjoin(mem_count, all_subjects.c.subject_id == mem_count.c.subject_id)
        .where(all_subjects.c.subject_id.not_like("_snapshot/%"))
        .where(all_subjects.c.subject_id.not_like("_bootstrap_tmp/%"))
        .order_by(all_subjects.c.subject_id)
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return [
        {
            "subject_id": row.subject_id,
            "episode_count": row.episode_count,
            "memory_count": row.memory_count,
        }
        for row in result.all()
    ]


# ---------------------------------------------------------------------------
# Resolutions
# ---------------------------------------------------------------------------


async def upsert_resolution(session: AsyncSession, row: ResolutionRow) -> ResolutionRow:
    """Insert or update a resolution (keyed by subject_id + session_id + tenant_id)."""
    # Check for existing resolution on same subject+session
    stmt = select(ResolutionRow).where(
        ResolutionRow.subject_id == row.subject_id,
        ResolutionRow.session_id == row.session_id,
    )
    if row.tenant_id is not None:
        stmt = stmt.where(ResolutionRow.tenant_id == row.tenant_id)
    else:
        stmt = stmt.where(ResolutionRow.tenant_id.is_(None))

    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.status = row.status
        existing.resolution_summary = row.resolution_summary
        existing.resolved_at = row.resolved_at
        existing.metadata_ = row.metadata_
        await session.flush()
        return existing

    session.add(row)
    await session.flush()
    return row


async def list_resolutions(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> Sequence[ResolutionRow]:
    """List resolutions for a subject, optionally filtered by status."""
    stmt = select(ResolutionRow).where(ResolutionRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, ResolutionRow.tenant_id, tenant_id)
    if status:
        stmt = stmt.where(ResolutionRow.status == status)
    stmt = stmt.order_by(ResolutionRow.updated_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_resolved_session_ids(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> set[str]:
    """Return session_ids that are marked as resolved for a subject."""
    stmt = (
        select(ResolutionRow.session_id)
        .where(ResolutionRow.subject_id == subject_id)
        .where(ResolutionRow.status == "resolved")
    )
    stmt = _tenant_filter(stmt, ResolutionRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return {row[0] for row in result.all()}


async def get_open_session_ids(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> set[str]:
    """Return session_ids that have an open (unresolved) resolution."""
    stmt = (
        select(ResolutionRow.session_id)
        .where(ResolutionRow.subject_id == subject_id)
        .where(ResolutionRow.status == "open")
    )
    stmt = _tenant_filter(stmt, ResolutionRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return {row[0] for row in result.all()}


async def delete_resolutions_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> int:
    """Delete all resolutions for a subject (used in subject deletion)."""
    stmt = delete(ResolutionRow).where(ResolutionRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, ResolutionRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.rowcount


# ---------------------------------------------------------------------------
# Health cache
# ---------------------------------------------------------------------------


async def get_health_cache(
    session: AsyncSession,
    subject_id: str,
) -> SubjectHealthCacheRow | None:
    """Get cached health state for a subject."""
    stmt = select(SubjectHealthCacheRow).where(SubjectHealthCacheRow.subject_id == subject_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def upsert_health_cache(
    session: AsyncSession,
    subject_id: str,
    state: str,
    score: int,
    *,
    tenant_id: str | None = None,
) -> None:
    """Update or insert cached health state."""
    from datetime import datetime, timezone

    existing = await get_health_cache(session, subject_id)
    if existing:
        existing.last_state = state
        existing.last_score = score
        existing.updated_at = datetime.now(timezone.utc)
    else:
        row = SubjectHealthCacheRow(
            subject_id=subject_id,
            tenant_id=tenant_id,
            last_state=state,
            last_score=score,
            updated_at=datetime.now(timezone.utc),
        )
        session.add(row)
    await session.flush()


async def delete_health_cache_by_subject(
    session: AsyncSession,
    subject_id: str,
) -> None:
    """Delete health cache for a subject (used in subject deletion)."""
    stmt = delete(SubjectHealthCacheRow).where(SubjectHealthCacheRow.subject_id == subject_id)
    await session.execute(stmt)
