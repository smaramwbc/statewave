"""Tenant dependency for FastAPI routes.

Extracts tenant_id from request.state (set by TenantMiddleware).
Returns None in single-tenant mode — repository functions treat None as
"no tenant filter" for backward compatibility.

Also exposes ``enforce_tenant_residency`` for routes that need the
v0.9 #161 region pin enforced before they hit any DB / receipt
emission code.
"""

from __future__ import annotations

import structlog
from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.config import settings
from server.db.engine import get_session

logger = structlog.stdlib.get_logger()


def get_tenant_id(request: Request) -> str | None:
    """FastAPI dependency — extract tenant_id from request state."""
    return getattr(request.state, "tenant_id", None)


async def enforce_tenant_residency(
    request: Request,
    tenant_id: str | None = None,
    session: AsyncSession | None = None,
) -> None:
    """Residency hard-check for tenant-scoped routes (v0.9 #161).

    Loads the tenant's pinned region from ``tenant_configs.config``
    and refuses the request with HTTP 403 ``residency.mismatch`` if
    the local ``settings.region`` does not match. No-ops when
    ``settings.region`` is unset (single-region mode) or the tenant
    is not pinned.

    Designed to be cheap on the happy path: one indexed SELECT
    against ``tenant_configs`` per request, then a string compare.
    Single-region deployments pay zero cost (early return).

    Wired as a FastAPI dependency so any route can opt in by adding
    ``Depends(enforce_tenant_residency)`` — the public surfaces
    (`/v1/context`, `/v1/handoff`, `/v1/episodes`, `/v1/memories/*`)
    do, and the admin surfaces enforce the same boundary at the
    middleware layer (see ``ResidencyMiddleware``).

    ``tenant_id`` and ``session`` are accepted as parameters so this
    function can be used both as a FastAPI dependency
    (FastAPI auto-resolves the standard deps) AND called directly
    from middleware/services that already hold a session.
    """
    server_region = settings.region
    if not server_region:
        return  # single-region mode, nothing to enforce

    if tenant_id is None:
        tenant_id = get_tenant_id(request)
    if not tenant_id:
        # Anonymous/single-tenant traffic falls through. The middleware
        # already 400'd if `require_tenant` is set and no tenant id
        # was supplied; we don't re-litigate that here.
        return

    if session is None:
        # Dependency-style invocation — FastAPI hasn't resolved a
        # session yet for us, so open a short-lived one. The cost is
        # one connection acquire; the read itself is a single row by
        # PK.
        session_gen = get_session()
        session = await session_gen.__anext__()
        try:
            await _do_check(request, tenant_id, server_region, session)
        finally:
            try:
                await session_gen.aclose()
            except Exception:
                pass
        return

    await _do_check(request, tenant_id, server_region, session)


async def _do_check(
    request: Request,
    tenant_id: str,
    server_region: str,
    session: AsyncSession,
) -> None:
    """The actual lookup + compare. Split out so middleware paths can
    reuse it after opening their own session."""
    from server.db import repositories as repo
    from server.services.residency import check_residency

    row = await repo.get_tenant_config(session, tenant_id)
    tenant_config = (row.config if row else {}) or {}
    mismatch = check_residency(tenant_config=tenant_config, server_region=server_region)
    if mismatch is None:
        return

    # Log with both regions for the operator. The wire response only
    # echoes the tenant's pinned region — a 403 message that named
    # the local server's region would leak topology details to a
    # caller probing the residency boundary.
    logger.warning(
        "residency_mismatch_refused",
        tenant_id=tenant_id,
        tenant_region=mismatch.tenant_region,
        server_region=mismatch.server_region,
        path=request.url.path,
    )
    raise HTTPException(
        status_code=403,
        detail={
            "code": "residency.mismatch",
            "message": (
                f"tenant {tenant_id!r} is pinned to region "
                f"{mismatch.tenant_region!r}; this request did not reach "
                "the right region. Retry against the regional endpoint."
            ),
        },
    )
