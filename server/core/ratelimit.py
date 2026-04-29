"""Rate-limit middleware — supports in-memory and distributed (Postgres) strategies.

Config:
- STATEWAVE_RATE_LIMIT_RPM: max requests/min per key (0 = disabled)
- STATEWAVE_RATE_LIMIT_STRATEGY: "distributed" | "memory" (default: "distributed")

The distributed strategy uses Postgres for shared state across workers/restarts.
The memory strategy uses a per-process sliding window (legacy, for development).
"""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths exempt from rate limiting
_EXEMPT_PATHS = {"/healthz", "/readyz", "/health", "/ready"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rpm: int = 0, strategy: str = "distributed") -> None:
        super().__init__(app)
        self._rpm = rpm  # 0 = disabled
        self._strategy = strategy
        # In-memory fallback store
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self._rpm <= 0:
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        if self._strategy == "distributed":
            allowed, retry_after = await self._check_distributed(client_ip)
        else:
            allowed, retry_after = self._check_memory(client_ip)

        if not allowed:
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

        return await call_next(request)

    async def _check_distributed(self, key: str) -> tuple[bool, int]:
        """Use Postgres-backed distributed rate limiter."""
        from server.services.ratelimit import check_rate_limit

        return await check_rate_limit(key, self._rpm)

    def _check_memory(self, key: str) -> tuple[bool, int]:
        """Legacy in-memory sliding window (single-process only)."""
        now = time.monotonic()
        window_start = now - 60.0

        timestamps = self._hits[key]
        self._hits[key] = [t for t in timestamps if t > window_start]

        if len(self._hits[key]) >= self._rpm:
            retry_after = int(60 - (now - self._hits[key][0])) + 1
            return False, retry_after

        self._hits[key].append(now)
        return True, 0
