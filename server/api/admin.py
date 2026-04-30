"""Admin endpoints — operator introspection and advanced bootstrap capabilities."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from server.core.config import settings
from server.services import webhooks

router = APIRouter(prefix="/admin", tags=["admin"])


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
        async with engine_module.async_session_factory() as session:
            episodes = await session.scalar(select(func.count()).select_from(EpisodeRow)) or 0
            memories = await session.scalar(select(func.count()).select_from(MemoryRow)) or 0
            subjects = (
                await session.scalar(select(func.count(func.distinct(EpisodeRow.subject_id)))) or 0
            )
            return {"episodes": episodes, "memories": memories, "subjects": subjects}

    async def _get_job_stats():
        async with engine_module.async_session_factory() as session:
            rows = await session.execute(
                select(CompileJobRow.status, func.count()).group_by(CompileJobRow.status)
            )
            stats = {row[0]: row[1] for row in rows}
            return stats

    async def _get_health_distribution():
        """Get subject health score distribution from cache table."""
        from server.db.tables import Base

        # Check if health cache table exists in metadata
        if "subject_health_cache" not in Base.metadata.tables:
            return None
        try:
            from sqlalchemy import text

            async with engine_module.async_session_factory() as session:
                rows = await session.execute(
                    text(
                        "SELECT last_state, COUNT(*) FROM subject_health_cache GROUP BY last_state"
                    )
                )
                return {row[0]: row[1] for row in rows}
        except Exception:
            return None

    async def _get_readiness():
        from server.db.engine import engine

        async with engine.connect() as conn:
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

    async with engine_module.async_session_factory() as session:

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
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List compile jobs for operator debugging.

    Returns recent jobs ordered by creation time (newest first).
    """
    from server.services.compile_jobs_durable import list_jobs

    jobs = await list_jobs(
        status=status, subject_id=subject_id, tenant_id=tenant_id, limit=limit, offset=offset
    )
    return {"jobs": jobs, "limit": limit, "offset": offset}


# ─── Tenant Audit ───


@router.get("/tenant-audit")
async def tenant_audit():
    """Report rows with NULL tenant_id — helps operators backfill after enabling tenants."""
    from sqlalchemy import func, select

    from server.db.engine import async_session_factory
    from server.db.tables import CompileJobRow, EpisodeRow, MemoryRow

    async with async_session_factory() as session:
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


@router.get("/snapshots")
async def list_snapshots_endpoint():
    """List available subject snapshots."""
    _require_snapshots()
    from server.services.snapshots import list_snapshots

    return {"snapshots": await list_snapshots()}


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
