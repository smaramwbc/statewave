"""State-assembly receipts read API + v0.9 replay.

  * `GET  /v1/receipts/{receipt_id}` — point lookup.
  * `GET  /v1/receipts/{receipt_id}/verify` — HMAC verification (v0.9 #157).
  * `GET  /v1/receipts` — list for a subject in a time window with
    cursor-based pagination.
  * `POST /v1/receipts/{receipt_id}/replay` — as-of-replay (v0.9 #159):
    re-runs the original assembly against current memories using the
    original policy bundle captured in the receipt's snapshot, emits a
    new ``mode="as_of_replay"`` receipt, and returns the diff envelope.

Tenant-scoped via the existing `X-Statewave-Tenant` header.

Receipts themselves remain immutable: replay does not modify the
original, it only emits a *new* receipt with `parent_receipt_id`
pointing back at the source. Issue #49's accountability guarantee is
preserved end-to-end.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
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
    return _row_to_response(row)


@router.get(
    "/v1/receipts/{receipt_id}/verify",
    summary="Verify the HMAC signature on a state-assembly receipt",
)
async def verify_receipt_endpoint(
    receipt_id: str,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
) -> dict[str, Any]:
    """Verify the HMAC signature stored on a receipt (v0.9, issue #157).

    Returns the verification verdict — `valid: true | false | null` —
    plus the `key_id` and `algorithm` used. Never echoes the signing
    key or any derivative on the response.

    `valid: null` distinguishes "we couldn't determine" cases
    (`no_signature`, `key_unavailable`, `unsupported_algorithm`) from
    `valid: false` ("we checked the math and the signature doesn't
    cover the body"). `key_unavailable` keeps historical receipts
    forensically inspectable when keys rotate out of operator config —
    never a 500.

    404 if the receipt doesn't exist (or belongs to a different tenant —
    indistinguishable on the wire, same as the detail endpoint)."""
    # Lazy import to keep this module's import graph shallow.
    from server.services.receipts import verify_receipt

    row = await repo.get_receipt_by_id(session, receipt_id, tenant_id=tenant_id)
    if row is None:
        raise HTTPException(status_code=404, detail="receipt not found")
    return verify_receipt(row)


def _row_to_response(row) -> dict[str, Any]:
    """Merge row-level lifecycle metadata into the receipt body for the wire.

    The body itself is the immutable audit artifact (never updated on
    disk after emission). `status` and `tombstoned_at` are row-level
    state — when the retention worker tombstones a receipt, those are
    the only columns that change. They surface as siblings of the body
    fields on the response so a client can tell "this audit record is
    still active" vs "this was retired by retention" without parsing a
    separate metadata blob.
    """
    out = dict(row.body)
    out["status"] = row.status
    if row.tombstoned_at is not None:
        out["tombstoned_at"] = row.tombstoned_at.isoformat()
    return out


class ReplayResponse(BaseModel):
    """Response shape for POST /v1/receipts/{id}/replay."""

    original_receipt_id: str
    replay_receipt_id: str | None
    diff: dict[str, Any]


@router.post(
    "/v1/receipts/{receipt_id}/replay",
    response_model=ReplayResponse,
    summary="Re-run the original assembly with the receipt's snapshot policy (v0.9 #159)",
    responses={
        404: {"description": "Receipt not found"},
        422: {
            "description": (
                "Receipt cannot be replayed. The standard error envelope "
                "carries `error.code = unreplayable.<reason>` where reason "
                "is one of `missing_policy_snapshot`, `nested_replay`, "
                "`invalid_snapshot`."
            )
        },
    },
)
async def replay_receipt_endpoint(
    receipt_id: str,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
) -> ReplayResponse:
    """Replay a v0.9+ receipt against the *current* memory state using
    the *original* policy bundle from the receipt's ``policy_snapshot``.

    Semantic: current code + original policy. Replay is NOT
    byte-for-byte reproduction — see docs/replay.md for the design
    rationale. The response carries a structural diff (added/removed
    selected_entries, added/removed policy filters, context_hash
    change) so an auditor can pinpoint what shifted between original
    emission and now.

    A fresh receipt is emitted with ``mode="as_of_replay"`` and
    ``parent_receipt_id`` set to the source ULID. The original receipt
    is never modified.

    Refusal codes (HTTP 422):

      * ``missing_policy_snapshot`` — pre-v0.9 receipt, no snapshot.
      * ``nested_replay`` — already a replay; replay the original parent.
      * ``invalid_snapshot`` — snapshot YAML corrupt / unparseable.
    """
    from server.services.replay import ReplayError, replay_receipt

    try:
        result = await replay_receipt(session, receipt_id=receipt_id, tenant_id=tenant_id)
    except ReplayError as exc:
        if exc.reason == "not_found":
            raise HTTPException(status_code=404, detail="receipt not found") from exc
        # The global error handler unwraps `code` + `message` from dict
        # details into the standard `{error: {...}}` envelope. We use
        # `unreplayable.<reason>` so machine consumers can switch on
        # the code without parsing the message body.
        raise HTTPException(
            status_code=422,
            detail={
                "code": f"unreplayable.{exc.reason}",
                "message": exc.detail or exc.reason,
            },
        ) from exc

    return ReplayResponse(
        original_receipt_id=result.original_receipt_id,
        replay_receipt_id=result.replay_receipt_id,
        diff=result.diff,
    )


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
    include_tombstoned: bool = Query(
        False,
        description=(
            "Include receipts that the retention worker has tombstoned. "
            "Default `false` — tombstoned rows persist for forensic lookup "
            "via `GET /v1/receipts/{id}` but are filtered out of the list "
            "view unless explicitly requested."
        ),
    ),
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
        include_tombstoned=include_tombstoned,
    )
    next_cursor = rows[-1].receipt_id if len(rows) == limit else None
    return {
        "receipts": [_row_to_response(row) for row in rows],
        "next_cursor": next_cursor,
    }
