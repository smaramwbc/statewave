"""API key authentication middleware.

When STATEWAVE_API_KEY is set, all requests (except health checks) must
include a matching ``X-API-Key`` header or ``?api_key=`` query param.

When STATEWAVE_API_KEY is unset/empty, authentication is disabled
(open access — suitable for local dev).
"""

from __future__ import annotations

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.stdlib.get_logger()

# Paths that never require authentication
_PUBLIC_PATHS = {"/healthz", "/readyz", "/health", "/ready", "/docs", "/redoc", "/openapi.json"}


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

        # Check header first, then query param
        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")

        if not provided:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {"code": "missing_api_key", "message": "X-API-Key header is required."}
                },
            )

        if provided != self._api_key:
            logger.warning("auth_failed", path=request.url.path)
            return JSONResponse(
                status_code=403,
                content={"error": {"code": "invalid_api_key", "message": "Invalid API key."}},
            )

        return await call_next(request)
