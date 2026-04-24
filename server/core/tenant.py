"""Multi-tenant middleware — extracts tenant ID from request header.

EXPERIMENTAL: Tenant isolation is currently header-extraction only.
The tenant_id is NOT yet enforced in data access queries. Do not rely
on this for data isolation in production. Full tenant-scoped queries
are planned for a future release.

When STATEWAVE_REQUIRE_TENANT=true, all data-modifying requests must include
the tenant header (default: X-Tenant-ID). The tenant ID is attached to
request.state.tenant_id for downstream use.

When require_tenant is false (default), the middleware is a pass-through
that sets tenant_id to None — single-tenant mode.
"""

from __future__ import annotations

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.stdlib.get_logger()

_PUBLIC_PATHS = {"/healthz", "/readyz", "/docs", "/redoc", "/openapi.json"}


class TenantMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, header: str = "X-Tenant-ID", require: bool = False) -> None:
        super().__init__(app)
        self._header = header
        self._require = require

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        tenant_id = request.headers.get(self._header)
        request.state.tenant_id = tenant_id

        if self._require and not tenant_id and request.url.path not in _PUBLIC_PATHS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "missing_tenant",
                        "message": f"{self._header} header is required.",
                    }
                },
            )

        if tenant_id:
            structlog.contextvars.bind_contextvars(tenant_id=tenant_id)

        return await call_next(request)
