"""Rate-limit middleware — supports in-memory and distributed (Postgres) strategies.

Config:
- STATEWAVE_RATE_LIMIT_RPM: max requests/min per key (0 = disabled)
- STATEWAVE_RATE_LIMIT_STRATEGY: "memory" (default) or "distributed"

"memory" keeps a per-process sliding window. "distributed" stores the window
in Postgres so replicas share one limit. Multi-replica deployments need
"distributed".
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Paths exempt from rate limiting
_EXEMPT_PATHS = {"/healthz", "/readyz", "/health", "/ready", "/v1/version"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rpm: int = 0, strategy: str = "memory") -> None:
        super().__init__(app)
        self._rpm = rpm  # 0 = disabled
        self._strategy = strategy
        # In-memory fallback store
        self._hits: dict[str, list[float]] = {}
        self._calls = 0

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

        self._calls += 1
        if self._calls >= 1000:
            self._calls = 0
            keys_to_drop = [
                k for k, timestamps in self._hits.items()
                if not timestamps or timestamps[-1] <= window_start
            ]
            for k in keys_to_drop:
                del self._hits[k]

        kept = [t for t in self._hits.get(key, ()) if t > window_start]

        if len(kept) >= self._rpm:
            self._hits[key] = kept
            retry_after = int(60 - (now - kept[0])) + 1
            return False, retry_after

        kept.append(now)
        self._hits[key] = kept
        return True, 0
