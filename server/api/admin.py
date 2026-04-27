"""Admin endpoints — operator introspection."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from server.services import webhooks

router = APIRouter(prefix="/admin", tags=["admin"])


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
