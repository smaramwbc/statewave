"""Data-access layer. All SQL lives here."""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.tables import EpisodeRow, MemoryRow


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
    limit: int = 100,
) -> Sequence[EpisodeRow]:
    stmt = (
        select(EpisodeRow)
        .where(EpisodeRow.subject_id == subject_id)
        .order_by(EpisodeRow.created_at.asc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_uncompiled_episodes(
    session: AsyncSession,
    subject_id: str,
    *,
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


async def delete_episodes_by_subject(session: AsyncSession, subject_id: str) -> int:
    stmt = delete(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
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
    kind: str | None = None,
    query: str | None = None,
    limit: int = 20,
) -> Sequence[MemoryRow]:
    stmt = (
        select(MemoryRow)
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
    )
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
    limit: int = 100,
) -> Sequence[MemoryRow]:
    stmt = (
        select(MemoryRow)
        .where(MemoryRow.subject_id == subject_id)
        .order_by(MemoryRow.created_at.asc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def delete_memories_by_subject(session: AsyncSession, subject_id: str) -> int:
    stmt = delete(MemoryRow).where(MemoryRow.subject_id == subject_id)
    result = await session.execute(stmt)
    return result.rowcount  # type: ignore[return-value]
