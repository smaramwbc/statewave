"""Admin endpoints — operator introspection and advanced bootstrap capabilities."""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from server.core.config import settings
from server.services import webhooks

router = APIRouter(prefix="/admin", tags=["admin"])


# ─── Response Models ─────────────────────────────────────────────────────────


class SubjectListItem(BaseModel):
    subject_id: str
    tenant_id: str | None
    episode_count: int
    memory_count: int
    session_count: int
    last_episode_at: str | None
    health_state: str | None
    health_score: int | None
    open_sessions: int


class SubjectListResponse(BaseModel):
    subjects: list[SubjectListItem]
    total: int
    limit: int
    offset: int


class SubjectSummary(BaseModel):
    episode_count: int
    memory_count: int
    session_count: int
    first_seen_at: str | None
    last_activity_at: str | None


class SubjectHealthSummary(BaseModel):
    score: int
    state: str
    factors: list[dict]


class SubjectSLASummary(BaseModel):
    total_sessions: int
    resolved_sessions: int
    open_sessions: int
    avg_first_response_seconds: float | None
    avg_resolution_seconds: float | None
    first_response_breach_count: int
    resolution_breach_count: int


class SubjectDetailResponse(BaseModel):
    subject_id: str
    tenant_id: str | None
    summary: SubjectSummary
    health: SubjectHealthSummary | None
    sla: SubjectSLASummary | None


class MemoryListItem(BaseModel):
    id: str
    kind: str
    content: str
    summary: str
    confidence: float
    status: str
    source_episode_ids: list[str]
    valid_from: str
    valid_to: str | None
    created_at: str


class MemoryListResponse(BaseModel):
    memories: list[MemoryListItem]
    total: int
    limit: int
    offset: int


class EpisodeListItem(BaseModel):
    id: str
    session_id: str | None
    source: str
    type: str
    payload: dict
    metadata: dict
    provenance: dict
    created_at: str


class EpisodeListResponse(BaseModel):
    episodes: list[EpisodeListItem]
    total: int
    limit: int
    offset: int


def _require_snapshots():
    """Guard: raise 404 if snapshots feature is disabled."""
    if not settings.enable_snapshots:
        raise HTTPException(status_code=404, detail="Not found")


# ─── Dashboard Aggregation ───


@router.get("/dashboard")
async def dashboard_overview():
    """Single aggregation endpoint for the admin dashboard.

    Returns system health, migration status, job stats, webhook stats,
    data counts, and subject health distribution in one request.
    """
    import asyncio

    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import CompileJobRow, EpisodeRow, MemoryRow
    from server.services.migrations import check_migration_status
    from server.services.readiness import run_readiness_checks

    async def _get_counts():
        async with engine_module.get_session_factory()() as session:
            episodes = await session.scalar(select(func.count()).select_from(EpisodeRow)) or 0
            memories = await session.scalar(select(func.count()).select_from(MemoryRow)) or 0
            subjects = (
                await session.scalar(select(func.count(func.distinct(EpisodeRow.subject_id)))) or 0
            )
            return {"episodes": episodes, "memories": memories, "subjects": subjects}

    async def _get_job_stats():
        async with engine_module.get_session_factory()() as session:
            rows = await session.execute(
                select(CompileJobRow.status, func.count()).group_by(CompileJobRow.status)
            )
            stats = {row[0]: row[1] for row in rows}
            return stats

    async def _get_health_distribution():
        """Get subject health score distribution from cache table.

        Also includes subjects without health data as 'unknown'.
        """
        from server.db.tables import Base, EpisodeRow

        # Check if health cache table exists in metadata
        if "subject_health_cache" not in Base.metadata.tables:
            return None
        try:
            from sqlalchemy import text

            async with engine_module.get_session_factory()() as session:
                # Get health state distribution from cache
                rows = await session.execute(
                    text(
                        "SELECT last_state, COUNT(*) FROM subject_health_cache GROUP BY last_state"
                    )
                )
                dist = {row[0]: row[1] for row in rows}

                # Get total distinct subjects from episodes
                total_subjects = (
                    await session.scalar(select(func.count(func.distinct(EpisodeRow.subject_id))))
                    or 0
                )

                # Calculate subjects without health data
                subjects_with_health = sum(dist.values())
                unknown_count = total_subjects - subjects_with_health

                if unknown_count > 0:
                    dist["unknown"] = unknown_count

                return dist
        except Exception:
            return None

    async def _get_readiness():
        from server.db.engine import get_engine

        async with get_engine().connect() as conn:
            return await run_readiness_checks(conn)

    # Run all queries concurrently
    readiness, migration, counts, job_stats, webhook_stats, health_dist = await asyncio.gather(
        _get_readiness(),
        check_migration_status(),
        _get_counts(),
        _get_job_stats(),
        webhooks.get_delivery_stats(),
        _get_health_distribution(),
    )

    return {
        "readiness": {
            "status": readiness.status,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "latency_ms": c.latency_ms}
                for c in readiness.checks
            ],
        },
        "migration": {
            "current_revision": migration.current_revision,
            "expected_head": migration.expected_head,
            "is_compatible": migration.is_compatible,
            "pending_count": migration.pending_count,
        },
        "counts": counts,
        "jobs": job_stats,
        "webhooks": webhook_stats,
        "health_distribution": health_dist,
    }


@router.get("/tenants")
async def list_tenants():
    """List all distinct tenant IDs in the system."""
    from sqlalchemy import distinct, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow

    async with engine_module.get_session_factory()() as session:
        result = await session.execute(
            select(distinct(EpisodeRow.tenant_id))
            .where(EpisodeRow.tenant_id.isnot(None))
            .order_by(EpisodeRow.tenant_id)
        )
        tenants = [row[0] for row in result.all()]
        return {"tenants": tenants}


# ─── Subject Explorer ────────────────────────────────────────────────────────


@router.get("/subjects", response_model=SubjectListResponse)
async def list_subjects_admin(
    search: str | None = Query(None, description="Search in subject_id"),
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    health_state: str | None = Query(None, description="Filter by health state"),
    has_open_sessions: bool | None = Query(None, description="Filter by open sessions"),
    sort_by: Literal["subject_id", "last_activity", "episode_count", "memory_count"] = Query(
        "last_activity", description="Sort field"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List subjects with search, filtering, and aggregated stats for admin explorer."""

    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import (
        EpisodeRow,
        MemoryRow,
        ResolutionRow,
        SubjectHealthCacheRow,
    )

    async with engine_module.get_session_factory()() as session:
        # Build base subqueries for aggregation
        # Episode stats per subject
        ep_stats = (
            select(
                EpisodeRow.subject_id,
                EpisodeRow.tenant_id,
                func.count().label("episode_count"),
                func.max(EpisodeRow.created_at).label("last_episode_at"),
            )
            .group_by(EpisodeRow.subject_id, EpisodeRow.tenant_id)
            .subquery()
        )

        # Memory stats per subject
        mem_stats = (
            select(
                MemoryRow.subject_id,
                func.count().label("memory_count"),
            )
            .group_by(MemoryRow.subject_id)
            .subquery()
        )

        # Session count per subject (distinct non-null session_ids from episodes)
        session_stats = (
            select(
                EpisodeRow.subject_id,
                func.count(func.distinct(EpisodeRow.session_id)).label("session_count"),
            )
            .where(EpisodeRow.session_id.isnot(None))
            .group_by(EpisodeRow.subject_id)
            .subquery()
        )

        # Open sessions per subject
        open_sessions = (
            select(
                ResolutionRow.subject_id,
                func.count().label("open_count"),
            )
            .where(ResolutionRow.status == "open")
            .group_by(ResolutionRow.subject_id)
            .subquery()
        )

        # Health cache
        health_cache = select(SubjectHealthCacheRow).subquery()

        # Main query joining all
        stmt = select(
            ep_stats.c.subject_id,
            ep_stats.c.tenant_id,
            ep_stats.c.episode_count,
            func.coalesce(mem_stats.c.memory_count, 0).label("memory_count"),
            func.coalesce(session_stats.c.session_count, 0).label("session_count"),
            ep_stats.c.last_episode_at,
            health_cache.c.last_state.label("health_state"),
            health_cache.c.last_score.label("health_score"),
            func.coalesce(open_sessions.c.open_count, 0).label("open_sessions"),
        ).select_from(
            ep_stats.outerjoin(mem_stats, ep_stats.c.subject_id == mem_stats.c.subject_id)
            .outerjoin(session_stats, ep_stats.c.subject_id == session_stats.c.subject_id)
            .outerjoin(open_sessions, ep_stats.c.subject_id == open_sessions.c.subject_id)
            .outerjoin(health_cache, ep_stats.c.subject_id == health_cache.c.subject_id)
        )

        # Exclude internal subjects
        stmt = stmt.where(ep_stats.c.subject_id.not_like("_snapshot/%"))
        stmt = stmt.where(ep_stats.c.subject_id.not_like("_bootstrap_tmp/%"))

        # Apply filters
        if search:
            stmt = stmt.where(ep_stats.c.subject_id.ilike(f"%{search}%"))

        if tenant_id:
            stmt = stmt.where(ep_stats.c.tenant_id == tenant_id)

        if health_state:
            stmt = stmt.where(health_cache.c.last_state == health_state)

        if has_open_sessions is True:
            stmt = stmt.where(func.coalesce(open_sessions.c.open_count, 0) > 0)
        elif has_open_sessions is False:
            stmt = stmt.where(func.coalesce(open_sessions.c.open_count, 0) == 0)

        # Count total (before pagination)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await session.scalar(count_stmt) or 0

        # Apply sorting
        sort_column = {
            "subject_id": ep_stats.c.subject_id,
            "last_activity": ep_stats.c.last_episode_at,
            "episode_count": ep_stats.c.episode_count,
            "memory_count": func.coalesce(mem_stats.c.memory_count, 0),
        }.get(sort_by, ep_stats.c.last_episode_at)

        if sort_order == "desc":
            stmt = stmt.order_by(sort_column.desc().nulls_last())
        else:
            stmt = stmt.order_by(sort_column.asc().nulls_last())

        # Pagination
        stmt = stmt.limit(limit).offset(offset)

        result = await session.execute(stmt)
        rows = result.all()

        # Compute health for subjects that don't have cached values
        from server.db import repositories as repo
        from server.services.health import compute_health

        subjects = []
        for row in rows:
            health_state = row.health_state
            health_score = row.health_score

            # If no cached health, compute it now
            if health_state is None:
                try:
                    health_result = await compute_health(
                        session, row.subject_id, tenant_id=row.tenant_id
                    )
                    health_state = health_result.state
                    health_score = health_result.score
                    # Cache for future requests (best-effort, separate session)
                    try:
                        from server.db.engine import get_session_factory

                        async with get_session_factory()() as cache_session:
                            await repo.upsert_health_cache(
                                cache_session,
                                row.subject_id,
                                health_state,
                                health_score,
                                tenant_id=row.tenant_id,
                            )
                            await cache_session.commit()
                    except Exception:
                        pass
                except Exception:
                    pass

            subjects.append(
                SubjectListItem(
                    subject_id=row.subject_id,
                    tenant_id=row.tenant_id,
                    episode_count=row.episode_count,
                    memory_count=row.memory_count,
                    session_count=row.session_count,
                    last_episode_at=row.last_episode_at.isoformat()
                    if row.last_episode_at
                    else None,
                    health_state=health_state,
                    health_score=health_score,
                    open_sessions=row.open_sessions,
                )
            )

        return SubjectListResponse(
            subjects=subjects,
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get("/subjects/{subject_id}", response_model=SubjectDetailResponse)
async def get_subject_detail(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Get detailed information about a specific subject for admin inspection."""
    from datetime import timedelta

    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow, MemoryRow, ResolutionRow
    from server.services.health import compute_health
    from server.services.sla import compute_sla

    async with engine_module.get_session_factory()() as session:
        # Check subject exists
        ep_count_stmt = (
            select(func.count()).select_from(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
        )
        if tenant_id:
            ep_count_stmt = ep_count_stmt.where(EpisodeRow.tenant_id == tenant_id)
        ep_count = await session.scalar(ep_count_stmt) or 0

        mem_count_stmt = (
            select(func.count()).select_from(MemoryRow).where(MemoryRow.subject_id == subject_id)
        )
        if tenant_id:
            mem_count_stmt = mem_count_stmt.where(MemoryRow.tenant_id == tenant_id)
        mem_count = await session.scalar(mem_count_stmt) or 0

        if ep_count == 0 and mem_count == 0:
            raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found")

        # Get timestamps
        time_stmt = select(
            func.min(EpisodeRow.created_at).label("first_seen"),
            func.max(EpisodeRow.created_at).label("last_activity"),
        ).where(EpisodeRow.subject_id == subject_id)
        if tenant_id:
            time_stmt = time_stmt.where(EpisodeRow.tenant_id == tenant_id)
        time_result = await session.execute(time_stmt)
        time_row = time_result.one()

        # Session count — count distinct session_ids from both episodes and resolutions
        # First, from episodes (where session_id is not null)
        ep_session_stmt = select(func.count(func.distinct(EpisodeRow.session_id))).where(
            EpisodeRow.subject_id == subject_id,
            EpisodeRow.session_id.isnot(None),
        )
        if tenant_id:
            ep_session_stmt = ep_session_stmt.where(EpisodeRow.tenant_id == tenant_id)
        ep_session_count = await session.scalar(ep_session_stmt) or 0

        # Also from resolutions table
        res_session_stmt = select(func.count(func.distinct(ResolutionRow.session_id))).where(
            ResolutionRow.subject_id == subject_id
        )
        if tenant_id:
            res_session_stmt = res_session_stmt.where(ResolutionRow.tenant_id == tenant_id)
        res_session_count = await session.scalar(res_session_stmt) or 0

        # Use the higher of the two (they may overlap)
        session_count = max(ep_session_count, res_session_count)

        # Get tenant_id from the data if not specified
        actual_tenant_id = tenant_id
        if not actual_tenant_id:
            tenant_stmt = (
                select(EpisodeRow.tenant_id).where(EpisodeRow.subject_id == subject_id).limit(1)
            )
            actual_tenant_id = await session.scalar(tenant_stmt)

        summary = SubjectSummary(
            episode_count=ep_count,
            memory_count=mem_count,
            session_count=session_count,
            first_seen_at=time_row.first_seen.isoformat() if time_row.first_seen else None,
            last_activity_at=time_row.last_activity.isoformat() if time_row.last_activity else None,
        )

        # Health
        health_summary = None
        try:
            health_result = await compute_health(session, subject_id, tenant_id=tenant_id)
            health_summary = SubjectHealthSummary(
                score=health_result.score,
                state=health_result.state,
                factors=[
                    {"signal": f.signal, "impact": f.impact, "detail": f.detail}
                    for f in health_result.factors
                ],
            )
        except Exception:
            pass

        # Update health cache in background (separate session to avoid conflicts)
        if health_summary:
            try:
                from server.db.engine import get_session_factory
                from server.db import repositories as repo

                async with get_session_factory()() as cache_session:
                    await repo.upsert_health_cache(
                        cache_session,
                        subject_id,
                        health_summary.state,
                        health_summary.score,
                        tenant_id=tenant_id,
                    )
                    await cache_session.commit()
            except Exception:
                pass  # Cache update is best-effort

        # SLA
        sla_summary = None
        try:
            sla_result = await compute_sla(
                session,
                subject_id,
                tenant_id=tenant_id,
                first_response_threshold=timedelta(minutes=5),
                resolution_threshold=timedelta(hours=24),
            )
            sla_summary = SubjectSLASummary(
                total_sessions=sla_result.total_sessions,
                resolved_sessions=sla_result.resolved_sessions,
                open_sessions=sla_result.open_sessions,
                avg_first_response_seconds=sla_result.avg_first_response_seconds,
                avg_resolution_seconds=sla_result.avg_resolution_seconds,
                first_response_breach_count=sla_result.first_response_breach_count,
                resolution_breach_count=sla_result.resolution_breach_count,
            )
        except Exception:
            pass

        return SubjectDetailResponse(
            subject_id=subject_id,
            tenant_id=actual_tenant_id,
            summary=summary,
            health=health_summary,
            sla=sla_summary,
        )


@router.get("/subjects/{subject_id}/sla")
async def get_subject_sla(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Get SLA metrics and session list for a subject."""
    from datetime import timedelta

    from server.db import engine as engine_module
    from server.services.sla import compute_sla

    try:
        async with engine_module.get_session_factory()() as session:
            sla_result = await compute_sla(
                session,
                subject_id,
                tenant_id=tenant_id,
                first_response_threshold=timedelta(minutes=5),
                resolution_threshold=timedelta(hours=24),
            )
            return {
                "total_sessions": sla_result.total_sessions,
                "resolved_sessions": sla_result.resolved_sessions,
                "open_sessions": sla_result.open_sessions,
                "avg_first_response_seconds": sla_result.avg_first_response_seconds,
                "avg_resolution_seconds": sla_result.avg_resolution_seconds,
                "first_response_breach_count": sla_result.first_response_breach_count,
                "resolution_breach_count": sla_result.resolution_breach_count,
                "sessions": getattr(sla_result, "sessions", []),
            }
    except Exception:
        return {"total_sessions": 0, "resolved_sessions": 0, "open_sessions": 0, "sessions": []}


@router.get("/subjects/{subject_id}/memories", response_model=MemoryListResponse)
async def list_subject_memories(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    status: Literal["active", "superseded", "all"] = Query("all", description="Filter by status"),
    kind: str | None = Query(None, description="Filter by memory kind"),
    search: str | None = Query(None, description="Search in content, summary, and kind"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List memories for a subject with filtering, search, and pagination."""
    from sqlalchemy import func, or_, select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    async with engine_module.get_session_factory()() as session:
        base = select(MemoryRow).where(MemoryRow.subject_id == subject_id)
        if tenant_id:
            base = base.where(MemoryRow.tenant_id == tenant_id)
        if status != "all":
            base = base.where(MemoryRow.status == status)
        if kind:
            base = base.where(MemoryRow.kind == kind)
        if search:
            search_pattern = f"%{search}%"
            base = base.where(
                or_(
                    MemoryRow.content.ilike(search_pattern),
                    MemoryRow.summary.ilike(search_pattern),
                    MemoryRow.kind.ilike(search_pattern),
                )
            )

        # Count
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await session.scalar(count_stmt) or 0

        # Get data
        stmt = base.order_by(MemoryRow.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        memories = [
            MemoryListItem(
                id=str(m.id),
                kind=m.kind,
                content=m.content,
                summary=m.summary,
                confidence=m.confidence,
                status=m.status,
                source_episode_ids=[str(ep_id) for ep_id in (m.source_episode_ids or [])],
                valid_from=m.valid_from.isoformat(),
                valid_to=m.valid_to.isoformat() if m.valid_to else None,
                created_at=m.created_at.isoformat(),
            )
            for m in rows
        ]

        return MemoryListResponse(
            memories=memories,
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get("/subjects/{subject_id}/episodes", response_model=EpisodeListResponse)
async def list_subject_episodes(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    session_id: str | None = Query(None, description="Filter by session"),
    type: str | None = Query(None, description="Filter by episode type"),
    search: str | None = Query(None, description="Search in payload (JSON text), type, and source"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List episodes for a subject with filtering, search, and pagination."""
    from sqlalchemy import func, or_, select
    from sqlalchemy.dialects.postgresql import JSONB

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow

    async with engine_module.get_session_factory()() as session:
        base = select(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
        if tenant_id:
            base = base.where(EpisodeRow.tenant_id == tenant_id)
        if session_id:
            base = base.where(EpisodeRow.session_id == session_id)
        if type:
            base = base.where(EpisodeRow.type == type)
        if search:
            search_pattern = f"%{search}%"
            # Cast payload to text for searching
            base = base.where(
                or_(
                    EpisodeRow.payload.cast(JSONB).astext.ilike(search_pattern),
                    EpisodeRow.type.ilike(search_pattern),
                    EpisodeRow.source.ilike(search_pattern),
                    EpisodeRow.session_id.ilike(search_pattern),
                )
            )

        # Count
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await session.scalar(count_stmt) or 0

        # Get data
        stmt = base.order_by(EpisodeRow.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        episodes = [
            EpisodeListItem(
                id=str(e.id),
                session_id=e.session_id,
                source=e.source,
                type=e.type,
                payload=e.payload,
                metadata=e.metadata_,
                provenance=e.provenance,
                created_at=e.created_at.isoformat(),
            )
            for e in rows
        ]

        return EpisodeListResponse(
            episodes=episodes,
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get(
    "/subjects/{subject_id}/episodes/{episode_id}/citing-memories",
    response_model=MemoryListResponse,
)
async def list_citing_memories(
    subject_id: str,
    episode_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List memories that cite (were derived from) a specific episode.

    This enables reverse provenance lookup: from an episode, find all
    memories that list it in their source_episode_ids.
    """
    import uuid as uuid_module

    from sqlalchemy import any_, func, select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    # Validate episode_id is a valid UUID
    try:
        episode_uuid = uuid_module.UUID(episode_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid episode_id format")

    async with engine_module.get_session_factory()() as session:
        # Find memories where episode_id is in source_episode_ids array
        base = select(MemoryRow).where(
            MemoryRow.subject_id == subject_id,
            episode_uuid == any_(MemoryRow.source_episode_ids),
        )
        if tenant_id:
            base = base.where(MemoryRow.tenant_id == tenant_id)

        # Count
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await session.scalar(count_stmt) or 0

        # Get data ordered by created_at desc (newest first)
        stmt = base.order_by(MemoryRow.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        memories = [
            MemoryListItem(
                id=str(m.id),
                kind=m.kind,
                content=m.content,
                summary=m.summary,
                confidence=m.confidence,
                status=m.status,
                source_episode_ids=[str(eid) for eid in m.source_episode_ids],
                valid_from=m.valid_from.isoformat(),
                valid_to=m.valid_to.isoformat() if m.valid_to else None,
                created_at=m.created_at.isoformat(),
            )
            for m in rows
        ]

        return MemoryListResponse(
            memories=memories,
            total=total,
            limit=limit,
            offset=offset,
        )


# ─── Memory Evolution / Related Memories ─────────────────────────────────────


class RelatedMemoryItem(BaseModel):
    """A memory related to the target memory."""

    id: str
    kind: str
    content: str
    summary: str
    confidence: float
    status: str
    created_at: str
    relationship: str  # "supersedes" | "sibling" | "superseded_by"


class MemoryEvolutionResponse(BaseModel):
    """Response for memory evolution/related memories lookup."""

    memory_id: str
    status: str
    created_at: str
    superseding_memory: RelatedMemoryItem | None  # The memory that replaced this one
    superseded_memories: list[RelatedMemoryItem]  # Memories this one replaced
    sibling_memories: list[RelatedMemoryItem]  # Other memories from same sources
    source_episode_count: int


@router.get(
    "/subjects/{subject_id}/memories/{memory_id}/related",
    response_model=MemoryEvolutionResponse,
)
async def get_memory_related(
    subject_id: str,
    memory_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Get memory evolution and related memories.

    Returns:
    - superseding_memory: If this memory is superseded, the active memory that replaced it
    - superseded_memories: If this memory is active, older memories it superseded
    - sibling_memories: Other memories derived from the same source episodes
    """
    import uuid as uuid_module

    from sqlalchemy import any_, or_, select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    # Validate memory_id is a valid UUID
    try:
        memory_uuid = uuid_module.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")

    async with engine_module.get_session_factory()() as session:
        # Get the target memory
        stmt = select(MemoryRow).where(
            MemoryRow.id == memory_uuid,
            MemoryRow.subject_id == subject_id,
        )
        if tenant_id:
            stmt = stmt.where(MemoryRow.tenant_id == tenant_id)

        result = await session.execute(stmt)
        target = result.scalar_one_or_none()

        if not target:
            raise HTTPException(status_code=404, detail="Memory not found")

        superseding_memory = None
        superseded_memories: list[RelatedMemoryItem] = []
        sibling_memories: list[RelatedMemoryItem] = []

        # If this memory is superseded, find the active memory that replaced it
        if target.status == "superseded" and target.source_episode_ids:
            # Look for an active memory of the same kind with overlapping source episodes
            # that was created after this one
            superseder_stmt = (
                select(MemoryRow)
                .where(
                    MemoryRow.subject_id == subject_id,
                    MemoryRow.kind == target.kind,
                    MemoryRow.status == "active",
                    MemoryRow.created_at > target.created_at,
                    MemoryRow.id != target.id,
                )
                .order_by(MemoryRow.created_at.asc())
                .limit(1)
            )
            if tenant_id:
                superseder_stmt = superseder_stmt.where(MemoryRow.tenant_id == tenant_id)

            superseder_result = await session.execute(superseder_stmt)
            superseder = superseder_result.scalar_one_or_none()

            if superseder:
                superseding_memory = RelatedMemoryItem(
                    id=str(superseder.id),
                    kind=superseder.kind,
                    content=superseder.content,
                    summary=superseder.summary,
                    confidence=superseder.confidence,
                    status=superseder.status,
                    created_at=superseder.created_at.isoformat(),
                    relationship="supersedes",
                )

        # If this memory is active, find memories it superseded
        if target.status == "active":
            superseded_stmt = (
                select(MemoryRow)
                .where(
                    MemoryRow.subject_id == subject_id,
                    MemoryRow.kind == target.kind,
                    MemoryRow.status == "superseded",
                    MemoryRow.created_at < target.created_at,
                    MemoryRow.id != target.id,
                )
                .order_by(MemoryRow.created_at.desc())
                .limit(5)
            )
            if tenant_id:
                superseded_stmt = superseded_stmt.where(MemoryRow.tenant_id == tenant_id)

            superseded_result = await session.execute(superseded_stmt)
            for m in superseded_result.scalars().all():
                superseded_memories.append(
                    RelatedMemoryItem(
                        id=str(m.id),
                        kind=m.kind,
                        content=m.content,
                        summary=m.summary,
                        confidence=m.confidence,
                        status=m.status,
                        created_at=m.created_at.isoformat(),
                        relationship="superseded_by",
                    )
                )

        # Find sibling memories (same source episodes, different memory)
        if target.source_episode_ids:
            # Find memories that share at least one source episode
            sibling_conditions = [
                ep_id == any_(MemoryRow.source_episode_ids)
                for ep_id in target.source_episode_ids[:5]  # Limit to first 5 to avoid huge OR
            ]
            sibling_stmt = (
                select(MemoryRow)
                .where(
                    MemoryRow.subject_id == subject_id,
                    MemoryRow.id != target.id,
                    or_(*sibling_conditions),
                )
                .order_by(MemoryRow.created_at.desc())
                .limit(10)
            )
            if tenant_id:
                sibling_stmt = sibling_stmt.where(MemoryRow.tenant_id == tenant_id)

            sibling_result = await session.execute(sibling_stmt)
            for m in sibling_result.scalars().all():
                # Skip if already in superseding or superseded
                if superseding_memory and m.id == uuid_module.UUID(superseding_memory.id):
                    continue
                if any(s.id == str(m.id) for s in superseded_memories):
                    continue

                sibling_memories.append(
                    RelatedMemoryItem(
                        id=str(m.id),
                        kind=m.kind,
                        content=m.content,
                        summary=m.summary,
                        confidence=m.confidence,
                        status=m.status,
                        created_at=m.created_at.isoformat(),
                        relationship="sibling",
                    )
                )

        return MemoryEvolutionResponse(
            memory_id=str(target.id),
            status=target.status,
            created_at=target.created_at.isoformat(),
            superseding_memory=superseding_memory,
            superseded_memories=superseded_memories,
            sibling_memories=sibling_memories,
            source_episode_count=len(target.source_episode_ids),
        )


# ─── Session Timeline ─────────────────────────────────────────────────────────


class TimelineEpisodeEvent(BaseModel):
    """Episode event in a session timeline."""

    event_type: Literal["episode"] = "episode"
    id: str
    source: str
    type: str
    payload: dict
    metadata: dict
    provenance: dict
    created_at: str
    citing_memory_count: int


class TimelineResolutionEvent(BaseModel):
    """Resolution event in a session timeline."""

    event_type: Literal["resolution"] = "resolution"
    resolved_at: str
    status: str


class SessionTimelineResponse(BaseModel):
    """Session timeline with chronologically merged events."""

    session_id: str
    status: str
    first_message_at: str | None
    first_response_at: str | None
    resolved_at: str | None
    first_response_seconds: float | None
    resolution_seconds: float | None
    first_response_breached: bool
    resolution_breached: bool
    episode_count: int
    events: list[TimelineEpisodeEvent | TimelineResolutionEvent]


@router.get(
    "/subjects/{subject_id}/sessions/{session_id}/timeline",
    response_model=SessionTimelineResponse,
)
async def get_session_timeline(
    subject_id: str,
    session_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(200, ge=1, le=500, description="Max episodes to include"),
):
    """Get a chronological timeline of events for a session.

    Returns episodes in chronological order (oldest first), with resolution
    events interleaved at the correct timestamp. Each episode includes a
    citing_memory_count for quick provenance visibility.
    """
    from sqlalchemy import any_, func, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow, MemoryRow, ResolutionRow
    from server.services.sla import compute_sla

    async with engine_module.get_session_factory()() as session:
        # Get episodes for this session
        base = select(EpisodeRow).where(
            EpisodeRow.subject_id == subject_id,
            EpisodeRow.session_id == session_id,
        )
        if tenant_id:
            base = base.where(EpisodeRow.tenant_id == tenant_id)

        # Order chronologically (oldest first for timeline)
        stmt = base.order_by(EpisodeRow.created_at.asc()).limit(limit)
        result = await session.execute(stmt)
        episode_rows = result.scalars().all()

        # Get total count for this session
        count_stmt = select(func.count()).select_from(
            select(EpisodeRow)
            .where(
                EpisodeRow.subject_id == subject_id,
                EpisodeRow.session_id == session_id,
            )
            .subquery()
        )
        episode_count = await session.scalar(count_stmt) or 0

        # Get citing memory counts for all episode IDs in one query
        episode_ids = [e.id for e in episode_rows]
        citing_counts: dict[str, int] = {}

        if episode_ids:
            # Count memories that cite each episode
            for ep_id in episode_ids:
                count_q = select(func.count()).where(
                    MemoryRow.subject_id == subject_id,
                    ep_id == any_(MemoryRow.source_episode_ids),
                )
                citing_counts[str(ep_id)] = await session.scalar(count_q) or 0

        # Get resolution for this session
        resolution_stmt = select(ResolutionRow).where(
            ResolutionRow.subject_id == subject_id,
            ResolutionRow.session_id == session_id,
        )
        if tenant_id:
            resolution_stmt = resolution_stmt.where(ResolutionRow.tenant_id == tenant_id)
        resolution_result = await session.execute(resolution_stmt)
        resolution = resolution_result.scalar_one_or_none()

        # Compute SLA metrics for this session
        sla_result = await compute_sla(session, subject_id, tenant_id=tenant_id)
        session_sla = next((s for s in sla_result.sessions if s.session_id == session_id), None)

        # Build chronological event list
        events: list[TimelineEpisodeEvent | TimelineResolutionEvent] = []

        resolution_inserted = False
        resolved_at = resolution.resolved_at if resolution else None

        for ep in episode_rows:
            # Insert resolution event at the right position
            if not resolution_inserted and resolved_at and ep.created_at > resolved_at:
                events.append(
                    TimelineResolutionEvent(
                        resolved_at=resolved_at.isoformat(),
                        status=resolution.status if resolution else "resolved",
                    )
                )
                resolution_inserted = True

            events.append(
                TimelineEpisodeEvent(
                    id=str(ep.id),
                    source=ep.source,
                    type=ep.type,
                    payload=ep.payload,
                    metadata=ep.metadata_,
                    provenance=ep.provenance,
                    created_at=ep.created_at.isoformat(),
                    citing_memory_count=citing_counts.get(str(ep.id), 0),
                )
            )

        # If resolution is after all episodes, append at end
        if not resolution_inserted and resolved_at:
            events.append(
                TimelineResolutionEvent(
                    resolved_at=resolved_at.isoformat(),
                    status=resolution.status if resolution else "resolved",
                )
            )

        return SessionTimelineResponse(
            session_id=session_id,
            status=session_sla.status if session_sla else ("resolved" if resolution else "open"),
            first_message_at=(
                session_sla.first_message_at.isoformat()
                if session_sla and session_sla.first_message_at
                else None
            ),
            first_response_at=(
                session_sla.first_response_at.isoformat()
                if session_sla and session_sla.first_response_at
                else None
            ),
            resolved_at=resolved_at.isoformat() if resolved_at else None,
            first_response_seconds=session_sla.first_response_seconds if session_sla else None,
            resolution_seconds=session_sla.resolution_seconds if session_sla else None,
            first_response_breached=session_sla.first_response_breached if session_sla else False,
            resolution_breached=session_sla.resolution_breached if session_sla else False,
            episode_count=episode_count,
            events=events,
        )


# ─── Usage Metering ───


@router.get("/usage")
async def usage_metering(tenant_id: str | None = None):
    """On-demand usage metrics for operator capacity planning.

    Returns counts for key operations across time windows (today, 7d, 30d, all-time).
    Optionally filtered by tenant_id.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import and_, func, select

    from server.db import engine as engine_module
    from server.db.tables import CompileJobRow, EpisodeRow, MemoryRow, WebhookEventRow

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    async with engine_module.get_session_factory()() as session:

        async def _count(table, ts_col, *, since=None):
            stmt = select(func.count()).select_from(table)
            conditions = []
            if tenant_id and hasattr(table, "tenant_id"):
                conditions.append(table.tenant_id == tenant_id)
            if since:
                conditions.append(ts_col >= since)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            return await session.scalar(stmt) or 0

        # Episodes
        ep_today = await _count(EpisodeRow, EpisodeRow.created_at, since=today_start)
        ep_7d = await _count(EpisodeRow, EpisodeRow.created_at, since=seven_days_ago)
        ep_30d = await _count(EpisodeRow, EpisodeRow.created_at, since=thirty_days_ago)
        ep_total = await _count(EpisodeRow, EpisodeRow.created_at)

        # Memories compiled
        mem_today = await _count(MemoryRow, MemoryRow.created_at, since=today_start)
        mem_7d = await _count(MemoryRow, MemoryRow.created_at, since=seven_days_ago)
        mem_30d = await _count(MemoryRow, MemoryRow.created_at, since=thirty_days_ago)
        mem_total = await _count(MemoryRow, MemoryRow.created_at)

        # Compile jobs
        job_today = await _count(CompileJobRow, CompileJobRow.created_at, since=today_start)
        job_7d = await _count(CompileJobRow, CompileJobRow.created_at, since=seven_days_ago)
        job_30d = await _count(CompileJobRow, CompileJobRow.created_at, since=thirty_days_ago)
        job_total = await _count(CompileJobRow, CompileJobRow.created_at)

        # Webhooks
        wh_today = await _count(WebhookEventRow, WebhookEventRow.created_at, since=today_start)
        wh_7d = await _count(WebhookEventRow, WebhookEventRow.created_at, since=seven_days_ago)
        wh_30d = await _count(WebhookEventRow, WebhookEventRow.created_at, since=thirty_days_ago)
        wh_total = await _count(WebhookEventRow, WebhookEventRow.created_at)

        # Distinct subjects active in period
        async def _active_subjects(since=None):
            stmt = select(func.count(func.distinct(EpisodeRow.subject_id)))
            conditions = []
            if tenant_id:
                conditions.append(EpisodeRow.tenant_id == tenant_id)
            if since:
                conditions.append(EpisodeRow.created_at >= since)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            return await session.scalar(stmt) or 0

        subj_7d = await _active_subjects(since=seven_days_ago)
        subj_30d = await _active_subjects(since=thirty_days_ago)
        subj_total = await _active_subjects()

    return {
        "period_start": today_start.isoformat(),
        "generated_at": now.isoformat(),
        "tenant_id": tenant_id,
        "episodes": {"today": ep_today, "7d": ep_7d, "30d": ep_30d, "total": ep_total},
        "memories": {"today": mem_today, "7d": mem_7d, "30d": mem_30d, "total": mem_total},
        "compile_jobs": {"today": job_today, "7d": job_7d, "30d": job_30d, "total": job_total},
        "webhooks": {"today": wh_today, "7d": wh_7d, "30d": wh_30d, "total": wh_total},
        "active_subjects": {"7d": subj_7d, "30d": subj_30d, "total": subj_total},
    }


# ─── Compile Jobs (operator introspection) ───


@router.get("/jobs")
async def list_compile_jobs(
    status: str | None = Query(
        None, description="Filter by status: pending, running, completed, failed"
    ),
    subject_id: str | None = Query(None, description="Filter by subject"),
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List compile jobs for operator debugging.

    Returns recent jobs ordered by creation time (newest first).
    """
    from server.services.compile_jobs_durable import list_jobs

    jobs, total = await list_jobs(
        status=status, subject_id=subject_id, tenant_id=tenant_id, limit=limit, offset=offset
    )
    return {"jobs": jobs, "total": total, "limit": limit, "offset": offset}


# ─── Tenant Audit ───


@router.get("/tenant-audit")
async def tenant_audit():
    """Report rows with NULL tenant_id — helps operators backfill after enabling tenants."""
    from sqlalchemy import func, select

    from server.db.engine import get_session_factory
    from server.db.tables import CompileJobRow, EpisodeRow, MemoryRow

    async with get_session_factory()() as session:
        ep_null = await session.scalar(
            select(func.count()).select_from(EpisodeRow).where(EpisodeRow.tenant_id.is_(None))
        )
        mem_null = await session.scalar(
            select(func.count()).select_from(MemoryRow).where(MemoryRow.tenant_id.is_(None))
        )
        jobs_null = await session.scalar(
            select(func.count()).select_from(CompileJobRow).where(CompileJobRow.tenant_id.is_(None))
        )

    return {
        "null_tenant_rows": {
            "episodes": ep_null or 0,
            "memories": mem_null or 0,
            "compile_jobs": jobs_null or 0,
        },
        "guidance": "Backfill with UPDATE <table> SET tenant_id = 'your-tenant' WHERE tenant_id IS NULL",
    }


# ─── Backup / Restore ───


class ImportSubjectRequest(BaseModel):
    document: dict
    target_subject_id: str | None = None
    target_tenant_id: str | None = None
    preserve_ids: bool = True


@router.get("/export/{subject_id}")
async def export_subject_endpoint(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Scope export to tenant"),
):
    """Export all episodes and memories for a subject as a portable JSON document.

    The output includes a SHA-256 checksum for integrity verification.
    Use this to back up a subject before risky operations or to migrate
    between Statewave instances.
    """
    from server.services.backup import export_subject

    doc = await export_subject(subject_id, tenant_id=tenant_id)
    if doc["counts"]["episodes"] == 0 and doc["counts"]["memories"] == 0:
        raise HTTPException(status_code=404, detail=f"No data found for subject '{subject_id}'")
    return doc


@router.post("/import")
async def import_subject_endpoint(req: ImportSubjectRequest):
    """Import a previously exported subject document.

    Options:
    - target_subject_id: override subject_id (default: use original from export)
    - target_tenant_id: override tenant_id (default: use original from export)
    - preserve_ids: keep original UUIDs (default true; set false to generate new ones)

    Safety: validates format version and checksum before importing.
    """
    from server.services.backup import import_subject

    try:
        result = await import_subject(
            req.document,
            target_subject_id=req.target_subject_id,
            target_tenant_id=req.target_tenant_id,
            preserve_ids=req.preserve_ids,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Webhooks ───


@router.get("/webhooks")
async def list_webhook_events(
    status: str | None = Query(
        None, description="Filter by status: pending, delivered, dead_letter"
    ),
    event_type: str | None = Query(None, description="Filter by event type"),
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List webhook events for operator debugging.

    Returns recent events ordered by creation time (newest first).
    """
    events, total = await webhooks.list_events(
        status=status, event_type=event_type, tenant_id=tenant_id, limit=limit, offset=offset
    )
    return {"events": events, "total": total, "limit": limit, "offset": offset}


@router.get("/webhooks/stats")
async def webhook_stats():
    """Aggregate webhook delivery statistics."""
    return await webhooks.get_delivery_stats()


@router.get("/webhooks/{event_id}")
async def webhook_event_status(event_id: uuid.UUID):
    """Get delivery status of a specific webhook event."""
    result = await webhooks.get_event_status(event_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Webhook event not found")
    return result


# ─── Subject Snapshots (advanced bootstrap/admin, feature-flagged) ───


class RestoreSnapshotRequest(BaseModel):
    target_subject_id: str


class RestoreByNameRequest(BaseModel):
    name: str
    target_subject_id: str
    version: Optional[int] = None


class CreateSnapshotRequest(BaseModel):
    name: str
    source_subject_id: str
    version: int = 1
    metadata: Optional[dict] = None


@router.get("/snapshots")
async def list_snapshots_endpoint():
    """List available subject snapshots."""
    _require_snapshots()
    from server.services.snapshots import list_snapshots

    return {"snapshots": await list_snapshots()}


@router.post("/snapshots")
async def create_snapshot_endpoint(req: CreateSnapshotRequest):
    """Create a snapshot from an existing subject."""
    _require_snapshots()
    from server.services.snapshots import create_snapshot

    try:
        result = await create_snapshot(
            name=req.name,
            source_subject_id=req.source_subject_id,
            version=req.version,
            metadata=req.metadata or {},
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/snapshots/{snapshot_id}/restore")
async def restore_snapshot_endpoint(snapshot_id: uuid.UUID, req: RestoreSnapshotRequest):
    """Restore a snapshot into a new target subject.

    Creates copies of all episodes and memories with new IDs,
    remapped provenance, and shifted timestamps.
    """
    _require_snapshots()
    from server.services.snapshots import restore_snapshot

    try:
        result = await restore_snapshot(snapshot_id, req.target_subject_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/snapshots/restore-by-name")
async def restore_by_name_endpoint(req: RestoreByNameRequest):
    """Restore a snapshot by name (uses latest version if not specified).

    Convenience endpoint for demo/bootstrap flows.
    """
    _require_snapshots()
    from server.services.snapshots import get_snapshot_by_name, restore_snapshot

    snap = await get_snapshot_by_name(req.name, req.version)
    if not snap:
        raise HTTPException(status_code=404, detail=f"Snapshot '{req.name}' not found")

    try:
        result = await restore_snapshot(uuid.UUID(snap["id"]), req.target_subject_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/cleanup")
async def trigger_cleanup(
    prefix: str = Query(default="live_", description="Subject prefix to clean up"),
    max_age_hours: int = Query(default=24, description="Max age in hours"),
):
    """Manually trigger cleanup of stale ephemeral subjects."""
    _require_snapshots()
    from server.services.snapshots import cleanup_ephemeral_subjects

    count = await cleanup_ephemeral_subjects(prefix=prefix, max_age_hours=max_age_hours)
    return {"subjects_cleaned": count}
