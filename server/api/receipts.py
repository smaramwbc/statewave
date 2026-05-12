"""State-assembly receipts read API.

Only two endpoints in v1:

  * `GET /v1/receipts/{receipt_id}` — point lookup.
  * `GET /v1/receipts` — list for a subject in a time window with
    cursor-based pagination.

Both are tenant-scoped via the existing `X-Statewave-Tenant` header.
Receipts are read-only on the wire — issue #49's accountability
guarantee depends on the audit log being immutable, so no
write/update/delete endpoints are exposed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.dependencies import get_tenant_id
from server.db import repositories as repo
from server.db.engine import get_session

router = APIRouter(tags=["receipts"])


@router.get(
    "/v1/receipts/{receipt_id}",
    summary="Fetch one state-assembly receipt",
)
async def get_receipt(
    receipt_id: str,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
) -> dict[str, Any]:
    """Return the full receipt body for a single id. 404 if the receipt
    does not exist OR if it belongs to a different tenant — the two
    cases are not distinguished on the wire so a tenant can't probe
    another tenant's id space."""
    row = await repo.get_receipt_by_id(session, receipt_id, tenant_id=tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="receipt not found")
    return row.body


@router.get(
    "/v1/receipts",
    summary="List state-assembly receipts for a subject",
)
async def list_receipts(
    subject_id: str = Query(..., description="Subject the receipts were written for"),
    since: datetime | None = Query(None, description="Lower bound on `created_at` (inclusive)"),
    until: datetime | None = Query(None, description="Upper bound on `created_at` (inclusive)"),
    cursor: str | None = Query(
        None,
        description=(
            "Pagination cursor — pass the last `receipt_id` from the previous "
            "page. ULIDs sort lexically by creation time, so this is a stable "
            "cursor even as new receipts are appended."
        ),
    ),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
) -> dict[str, Any]:
    """List receipts newest-first. Returns the receipt bodies plus a
    `next_cursor` that the caller passes back to fetch the next page;
    `next_cursor` is None when there are no more results."""
    rows = await repo.list_receipts(
        session,
        subject_id,
        tenant_id=tenant_id,
        since=since,
        until=until,
        cursor=cursor,
        limit=limit,
    )
    next_cursor = rows[-1].receipt_id if len(rows) == limit else None
    return {
        "receipts": [row.body for row in rows],
        "next_cursor": next_cursor,
    }
