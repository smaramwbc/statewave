"""Residency enforcement middleware (v0.9, issue #161).

Runs after ``TenantMiddleware`` has resolved ``request.state.tenant_id``.
For any request carrying a tenant id, hard-checks the tenant's pinned
region against ``settings.region`` and refuses with HTTP 403
``residency.mismatch`` on conflict.

Why a middleware and not just a per-route dependency:

  * Single point of enforcement. Every endpoint that ever reads or
    writes tenant data passes through here automatically — there's
    no way to forget to add the dependency.
  * Catches admin paths too. Per the v0.9 #161 design decision,
    cross-region admin reads are not allowed; the same middleware
    refuses them.
  * Cost is bounded: short-circuits with zero DB work when
    ``settings.region`` is unset (the single-region default).

Paths exempted from the check:

  * The same ``_PUBLIC_PATHS`` set as ``TenantMiddleware`` — health,
    readiness, docs. These never carry tenant data.
  * Anonymous requests (no tenant id) — there's nothing to enforce
    against. ``TenantMiddleware`` already 400'd if a tenant header
    was required.
"""

from __future__ import annotations

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.stdlib.get_logger()

_EXEMPT_PATHS = {
    "/healthz",
    "/readyz",
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class ResidencyMiddleware(BaseHTTPMiddleware):
    """Refuses a request when the tenant is pinned to a region other
    than ``settings.region``. No-op in single-region mode (when
    ``settings.region`` is None)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Lazy import so the module is testable without booting Settings.
        from server.core.config import settings

        server_region = settings.region
        if not server_region:
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", None)
        if not tenant_id:
            # No tenant context — nothing to enforce. Anonymous traffic
            # is already gated by `require_tenant` if the operator
            # wanted it gated.
            return await call_next(request)

        # Open a short-lived session purely for the residency check.
        # The read is a single row by PK on `tenant_configs`; in the
        # happy path (matching region) the work is one round-trip
        # before the rest of the request proceeds with its own
        # session-via-dependency-injection.
        from server.db.engine import get_session_factory
        from server.db import repositories as repo
        from server.services.residency import check_residency

        try:
            async with get_session_factory()() as session:
                row = await repo.get_tenant_config(session, tenant_id)
        except Exception:
            # If the residency lookup itself blew up (DB transient,
            # schema mismatch during a migration window), we have a
            # choice: fail open (allow the request) or fail closed
            # (refuse). For residency we fail OPEN with a warning,
            # because failing closed here would take down every
            # tenant-scoped request on any DB blip — the data plane
            # is already separated per region by infra, and this
            # middleware is the second line of defence. The first
            # line is the deployment topology itself.
            logger.warning(
                "residency_check_failed_failing_open",
                tenant_id=tenant_id,
                path=request.url.path,
                exc_info=True,
            )
            return await call_next(request)

        tenant_config = (row.config if row else {}) or {}
        mismatch = check_residency(tenant_config=tenant_config, server_region=server_region)
        if mismatch is None:
            return await call_next(request)

        logger.warning(
            "residency_mismatch_refused",
            tenant_id=tenant_id,
            tenant_region=mismatch.tenant_region,
            server_region=mismatch.server_region,
            path=request.url.path,
        )
        # 403 with the structured `residency.mismatch` code. The
        # message names the tenant's pinned region (operator-set, safe
        # to surface) but NOT the server's local region — that would
        # leak topology to a caller probing the residency boundary.
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "code": "residency.mismatch",
                    "message": (
                        f"tenant {tenant_id!r} is pinned to region "
                        f"{mismatch.tenant_region!r}; this request did not "
                        "reach the right region. Retry against the regional "
                        "endpoint."
                    ),
                }
            },
        )
