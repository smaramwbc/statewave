"""Admin endpoints — operator introspection and advanced bootstrap capabilities."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from server.services import webhooks
from server.services.snapshots import (
    cleanup_ephemeral_subjects,
    create_snapshot,
    delete_snapshot,
    get_snapshot,
    get_snapshot_by_name,
    list_snapshots,
    restore_snapshot,
)

router = APIRouter(prefix="/admin", tags=["admin"])


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


# ─── Subject Snapshots (advanced bootstrap/admin) ───


class CreateSnapshotRequest(BaseModel):
    name: str
    source_subject_id: str
    version: int = 1
    metadata: dict = {}


class RestoreSnapshotRequest(BaseModel):
    target_subject_id: str


class RestoreByNameRequest(BaseModel):
    name: str
    target_subject_id: str
    version: Optional[int] = None


@router.post("/snapshots")
async def create_snapshot_endpoint(req: CreateSnapshotRequest):
    """Create a snapshot from an existing subject's state.

    Captures episodes + memories as a portable, restorable snapshot.
    """
    try:
        result = await create_snapshot(
            name=req.name,
            source_subject_id=req.source_subject_id,
            version=req.version,
            metadata=req.metadata,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/snapshots")
async def list_snapshots_endpoint():
    """List all available subject snapshots."""
    return {"snapshots": await list_snapshots()}


@router.get("/snapshots/{snapshot_id}")
async def get_snapshot_endpoint(snapshot_id: uuid.UUID):
    """Get snapshot metadata."""
    result = await get_snapshot(snapshot_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return result


@router.post("/snapshots/{snapshot_id}/restore")
async def restore_snapshot_endpoint(snapshot_id: uuid.UUID, req: RestoreSnapshotRequest):
    """Restore a snapshot into a new target subject.

    Creates copies of all episodes and memories with new IDs,
    remapped provenance, and shifted timestamps.
    """
    try:
        result = await restore_snapshot(snapshot_id, req.target_subject_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/snapshots/restore-by-name")
async def restore_by_name_endpoint(req: RestoreByNameRequest):
    """Restore a snapshot by name (uses latest version if version not specified).

    Convenience endpoint for demo/bootstrap flows where snapshot ID isn't known.
    """
    snap = await get_snapshot_by_name(req.name, req.version)
    if not snap:
        raise HTTPException(status_code=404, detail=f"Snapshot '{req.name}' not found")

    try:
        result = await restore_snapshot(uuid.UUID(snap["id"]), req.target_subject_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/snapshots/{snapshot_id}")
async def delete_snapshot_endpoint(snapshot_id: uuid.UUID):
    """Delete a snapshot and its stored source data."""
    deleted = await delete_snapshot(snapshot_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {"status": "deleted"}


@router.post("/cleanup")
async def trigger_cleanup(
    prefix: str = Query(default="live_", description="Subject prefix to clean up"),
    max_age_hours: int = Query(default=24, description="Max age in hours"),
):
    """Manually trigger cleanup of stale ephemeral subjects."""
    count = await cleanup_ephemeral_subjects(prefix=prefix, max_age_hours=max_age_hours)
    return {"subjects_cleaned": count}

