"""API key authentication middleware.

When STATEWAVE_API_KEY is set, all requests (except health checks) must
include a matching ``X-API-Key`` header.

When STATEWAVE_API_KEY is unset/empty, authentication is disabled
(open access — suitable for local dev).
"""

from __future__ import annotations

import hmac

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.stdlib.get_logger()

# Paths that never require authentication.
#
# The health/readiness pair and /v1/version are exempt because probes and
# version discovery have no credential to present. The schema trio — /docs,
# /redoc, /openapi.json — is a deliberate second category (#398): it is the
# self-describing surface the Swagger UI needs, and it stays readable on a
# keyed deployment. Gating it on `debug` is not a fix, because docker-compose
# ships the API with STATEWAVE_DEBUG=true, so such a gate would look like a
# hardening win while staying open on the default deployment. An operator who
# wants the schema private blocks those three paths at the proxy.
#
# ``server.core.tenant`` and ``server.core.residency_middleware`` carry their
# own copies of this set for the same paths; tests/test_middleware.py fails if
# the three drift apart.
_PUBLIC_PATHS = {"/healthz", "/readyz", "/health", "/ready", "/docs", "/redoc", "/openapi.json", "/v1/version"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None = None) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip auth if no key is configured (local dev mode)
        if not self._api_key:
            return await call_next(request)

        # Skip auth for public endpoints
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key")

        if not provided:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {"code": "missing_api_key", "message": "X-API-Key header is required."}
                },
            )

        if not hmac.compare_digest(provided.encode(), self._api_key.encode()):
            logger.warning("auth_failed", path=request.url.path)
            return JSONResponse(
                status_code=403,
                content={"error": {"code": "invalid_api_key", "message": "Invalid API key."}},
            )

        return await call_next(request)
