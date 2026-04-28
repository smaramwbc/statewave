"""Subject Snapshots service.

Provides admin/bootstrap capability for capturing and restoring subject state.
Used for demos, staging, migrations, onboarding — not the normal developer path.

Snapshots capture a subject's episodes + memories. On restore:
- New UUIDs for all rows
- Provenance remapped (episode IDs in memories → new episode IDs)
- Timestamps shifted (preserves relative offsets, anchors newest to now)
- Restored subject is indistinguishable from organically built one
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import delete, func, select

from server.db.engine import async_session_factory
from server.db.tables import EpisodeRow, MemoryRow, SubjectSnapshotRow

logger = structlog.stdlib.get_logger()

# Subjects with these prefixes are protected from cleanup
SNAPSHOT_SOURCE_PREFIX = "_snapshot/"
LIVE_PREFIX = "live_"
LIVE_TTL_HOURS = 24


# ─── Snapshot Creation ───


async def create_snapshot(
    name: str,
    source_subject_id: str,
    version: int = 1,
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Create a snapshot from an existing subject.

    Copies the subject's data into a protected source subject `_snapshot/{name}/v{version}`,
    then records snapshot metadata.
    """
    snapshot_subject = f"{SNAPSHOT_SOURCE_PREFIX}{name}/v{version}"

    async with async_session_factory() as session:
        # Check for existing snapshot with same name+version
        existing = await session.execute(
            select(SubjectSnapshotRow).where(
                SubjectSnapshotRow.name == name,
                SubjectSnapshotRow.version == version,
            )
        )
        if existing.scalar():
            return {"status": "exists", "name": name, "version": version}

        # Fetch source episodes
        eps = (
            (
                await session.execute(
                    select(EpisodeRow)
                    .where(EpisodeRow.subject_id == source_subject_id)
                    .order_by(EpisodeRow.created_at)
                )
            )
            .scalars()
            .all()
        )

        if not eps:
            raise ValueError(f"Source subject '{source_subject_id}' has no episodes")

        # Fetch source memories
        mems = (
            (
                await session.execute(
                    select(MemoryRow)
                    .where(MemoryRow.subject_id == source_subject_id)
                    .order_by(MemoryRow.created_at)
                )
            )
            .scalars()
            .all()
        )

        # Copy episodes into snapshot source subject with new IDs
        # We need an ID map so memories' source_episode_ids can be remapped
        snapshot_ep_id_map: dict[uuid.UUID, uuid.UUID] = {}
        for ep in eps:
            new_id = uuid.uuid4()
            snapshot_ep_id_map[ep.id] = new_id
            session.add(
                EpisodeRow(
                    id=new_id,
                    subject_id=snapshot_subject,
                    source=ep.source,
                    type=ep.type,
                    payload=ep.payload,
                    metadata_=ep.metadata_,
                    provenance={**(ep.provenance or {}), "original_id": str(ep.id)},
                    created_at=ep.created_at,
                    last_compiled_at=ep.last_compiled_at,
                )
            )

        # Copy memories into snapshot source subject
        for mem in mems:
            remapped_ids = [
                snapshot_ep_id_map.get(eid, eid) for eid in (mem.source_episode_ids or [])
            ]
            session.add(
                MemoryRow(
                    id=uuid.uuid4(),
                    subject_id=snapshot_subject,
                    kind=mem.kind,
                    content=mem.content,
                    summary=mem.summary,
                    confidence=mem.confidence,
                    valid_from=mem.valid_from,
                    valid_to=mem.valid_to,
                    source_episode_ids=remapped_ids,
                    metadata_=mem.metadata_,
                    status=mem.status,
                    embedding=mem.embedding,
                    created_at=mem.created_at,
                    updated_at=mem.updated_at,
                )
            )

        # Record snapshot metadata
        snapshot = SubjectSnapshotRow(
            name=name,
            version=version,
            source_subject_id=snapshot_subject,
            episode_count=len(eps),
            memory_count=len(mems),
            metadata_=metadata or {},
        )
        session.add(snapshot)
        await session.commit()

        logger.info(
            "snapshot_created", name=name, version=version, episodes=len(eps), memories=len(mems)
        )
        return {
            "status": "created",
            "id": str(snapshot.id),
            "name": name,
            "version": version,
            "episode_count": len(eps),
            "memory_count": len(mems),
        }


# ─── Snapshot Restore ───


async def restore_snapshot(
    snapshot_id: uuid.UUID,
    target_subject_id: str,
) -> dict:
    """Restore a snapshot into a new target subject.

    Copies all episodes and memories with:
    - New UUIDs
    - Remapped provenance (source_episode_ids)
    - Timestamps shifted so newest episode = now
    """
    async with async_session_factory() as session:
        # Get snapshot metadata
        snap = (
            await session.execute(
                select(SubjectSnapshotRow).where(SubjectSnapshotRow.id == snapshot_id)
            )
        ).scalar()

        if not snap:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")

        source_subject = snap.source_subject_id

        # Fetch source episodes
        eps = (
            (
                await session.execute(
                    select(EpisodeRow)
                    .where(EpisodeRow.subject_id == source_subject)
                    .order_by(EpisodeRow.created_at)
                )
            )
            .scalars()
            .all()
        )

        # Fetch source memories
        mems = (
            (
                await session.execute(
                    select(MemoryRow)
                    .where(MemoryRow.subject_id == source_subject)
                    .order_by(MemoryRow.created_at)
                )
            )
            .scalars()
            .all()
        )

        if not eps:
            raise ValueError(f"Snapshot source '{source_subject}' has no episodes")

        # ── Timestamp shifting ──
        now = datetime.now(timezone.utc)
        newest_ep_time = max(ep.created_at for ep in eps)
        # Ensure timezone-aware comparison
        if newest_ep_time.tzinfo is None:
            newest_ep_time = newest_ep_time.replace(tzinfo=timezone.utc)
        time_shift = now - newest_ep_time

        # ── Episode cloning with new IDs ──
        episode_id_map: dict[uuid.UUID, uuid.UUID] = {}

        for ep in eps:
            new_id = uuid.uuid4()
            episode_id_map[ep.id] = new_id
            ep_created = ep.created_at
            if ep_created.tzinfo is None:
                ep_created = ep_created.replace(tzinfo=timezone.utc)

            session.add(
                EpisodeRow(
                    id=new_id,
                    subject_id=target_subject_id,
                    source=ep.source,
                    type=ep.type,
                    payload=ep.payload,
                    metadata_=ep.metadata_,
                    provenance={
                        **(ep.provenance or {}),
                        "restored_from_snapshot": str(snapshot_id),
                        "original_episode_id": str(ep.id),
                    },
                    created_at=ep_created + time_shift,
                    last_compiled_at=(ep_created + time_shift) if ep.last_compiled_at else None,
                )
            )

        # ── Memory cloning with provenance remapping ──
        for mem in mems:
            new_id = uuid.uuid4()
            remapped_ids = [episode_id_map.get(eid, eid) for eid in (mem.source_episode_ids or [])]

            mem_created = mem.created_at
            if mem_created.tzinfo is None:
                mem_created = mem_created.replace(tzinfo=timezone.utc)
            mem_updated = mem.updated_at
            if mem_updated.tzinfo is None:
                mem_updated = mem_updated.replace(tzinfo=timezone.utc)
            mem_valid_from = mem.valid_from
            if mem_valid_from.tzinfo is None:
                mem_valid_from = mem_valid_from.replace(tzinfo=timezone.utc)
            mem_valid_to = mem.valid_to
            if mem_valid_to and mem_valid_to.tzinfo is None:
                mem_valid_to = mem_valid_to.replace(tzinfo=timezone.utc)

            session.add(
                MemoryRow(
                    id=new_id,
                    subject_id=target_subject_id,
                    kind=mem.kind,
                    content=mem.content,
                    summary=mem.summary,
                    confidence=mem.confidence,
                    valid_from=mem_valid_from + time_shift,
                    valid_to=(mem_valid_to + time_shift) if mem_valid_to else None,
                    source_episode_ids=remapped_ids,
                    metadata_={
                        **(mem.metadata_ or {}),
                        "restored_from_snapshot": str(snapshot_id),
                    },
                    status=mem.status,
                    embedding=mem.embedding,
                    created_at=mem_created + time_shift,
                    updated_at=mem_updated + time_shift,
                )
            )

        await session.commit()

        logger.info(
            "snapshot_restored",
            snapshot_id=str(snapshot_id),
            target=target_subject_id,
            episodes=len(eps),
            memories=len(mems),
        )
        return {
            "episodes_restored": len(eps),
            "memories_restored": len(mems),
            "subject_id": target_subject_id,
        }


# ─── Snapshot Queries ───


async def list_snapshots() -> list[dict]:
    """List all available snapshots."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(SubjectSnapshotRow).order_by(SubjectSnapshotRow.name, SubjectSnapshotRow.version)
        )
        return [
            {
                "id": str(s.id),
                "name": s.name,
                "version": s.version,
                "source_subject_id": s.source_subject_id,
                "episode_count": s.episode_count,
                "memory_count": s.memory_count,
                "metadata": s.metadata_,
                "created_at": s.created_at.isoformat(),
            }
            for s in result.scalars().all()
        ]


async def get_snapshot(snapshot_id: uuid.UUID) -> dict | None:
    """Get snapshot metadata by ID."""
    async with async_session_factory() as session:
        s = (
            await session.execute(
                select(SubjectSnapshotRow).where(SubjectSnapshotRow.id == snapshot_id)
            )
        ).scalar()
        if not s:
            return None
        return {
            "id": str(s.id),
            "name": s.name,
            "version": s.version,
            "source_subject_id": s.source_subject_id,
            "episode_count": s.episode_count,
            "memory_count": s.memory_count,
            "metadata": s.metadata_,
            "created_at": s.created_at.isoformat(),
        }


async def get_snapshot_by_name(name: str, version: int | None = None) -> dict | None:
    """Get snapshot by name (optionally specific version, else latest)."""
    async with async_session_factory() as session:
        query = select(SubjectSnapshotRow).where(SubjectSnapshotRow.name == name)
        if version is not None:
            query = query.where(SubjectSnapshotRow.version == version)
        else:
            query = query.order_by(SubjectSnapshotRow.version.desc())
        s = (await session.execute(query)).scalars().first()
        if not s:
            return None
        return {
            "id": str(s.id),
            "name": s.name,
            "version": s.version,
            "source_subject_id": s.source_subject_id,
            "episode_count": s.episode_count,
            "memory_count": s.memory_count,
            "metadata": s.metadata_,
            "created_at": s.created_at.isoformat(),
        }


async def delete_snapshot(snapshot_id: uuid.UUID) -> bool:
    """Delete a snapshot and its source data."""
    async with async_session_factory() as session:
        s = (
            await session.execute(
                select(SubjectSnapshotRow).where(SubjectSnapshotRow.id == snapshot_id)
            )
        ).scalar()
        if not s:
            return False

        source_subject = s.source_subject_id
        # Delete source data
        await session.execute(delete(MemoryRow).where(MemoryRow.subject_id == source_subject))
        await session.execute(delete(EpisodeRow).where(EpisodeRow.subject_id == source_subject))
        await session.execute(
            delete(SubjectSnapshotRow).where(SubjectSnapshotRow.id == snapshot_id)
        )
        await session.commit()
        return True


# ─── Cleanup ───


async def cleanup_ephemeral_subjects(
    prefix: str = LIVE_PREFIX,
    max_age_hours: int = LIVE_TTL_HOURS,
) -> int:
    """Delete episodes and memories for ephemeral subjects older than max_age_hours.

    NEVER deletes _snapshot/ subjects or any subject not matching prefix.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    async with async_session_factory() as session:
        # Find subjects with the given prefix whose newest episode is older than cutoff
        result = await session.execute(
            select(EpisodeRow.subject_id, func.max(EpisodeRow.created_at).label("latest"))
            .where(EpisodeRow.subject_id.like(f"{prefix}%"))
            .group_by(EpisodeRow.subject_id)
            .having(func.max(EpisodeRow.created_at) < cutoff)
        )
        stale_subjects = [row[0] for row in result.fetchall()]

        # Safety: never delete snapshot sources
        stale_subjects = [s for s in stale_subjects if not s.startswith(SNAPSHOT_SOURCE_PREFIX)]

        if not stale_subjects:
            return 0

        await session.execute(delete(MemoryRow).where(MemoryRow.subject_id.in_(stale_subjects)))
        await session.execute(delete(EpisodeRow).where(EpisodeRow.subject_id.in_(stale_subjects)))
        await session.commit()

        logger.info("ephemeral_cleanup", count=len(stale_subjects))
        return len(stale_subjects)
