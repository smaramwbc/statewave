"""Simple in-memory rate limiter middleware.

Uses a sliding-window counter per client IP. Configurable via:
- STATEWAVE_RATE_LIMIT_RPM: max requests per minute per IP (0 = disabled)

Not suitable for multi-process deployments — use an external rate limiter
(e.g. Redis-backed) in production if running multiple workers.
"""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths exempt from rate limiting
_EXEMPT_PATHS = {"/healthz", "/readyz"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rpm: int = 0) -> None:
        super().__init__(app)
        self._rpm = rpm  # 0 = disabled
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self._rpm <= 0:
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - 60.0

        # Prune old entries
        timestamps = self._hits[client_ip]
        self._hits[client_ip] = [t for t in timestamps if t > window_start]

        if len(self._hits[client_ip]) >= self._rpm:
            retry_after = int(60 - (now - self._hits[client_ip][0])) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": f"Rate limit exceeded. Max {self._rpm} requests per minute.",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        self._hits[client_ip].append(now)
        return await call_next(request)
