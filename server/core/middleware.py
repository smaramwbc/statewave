"""Request ID middleware — generates or propagates X-Request-ID."""

from __future__ import annotations

import re
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.stdlib.get_logger()

_HEADER = "X-Request-ID"
# A request id is a short correlation token; reject anything else a client sends.
_VALID_ID = re.compile(r"[A-Za-z0-9._:\-]{1,128}")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied = request.headers.get(_HEADER, "")
        request_id = supplied if _VALID_ID.fullmatch(supplied) else uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        # Bind to structlog context so all logs in this request include it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)
        response.headers[_HEADER] = request_id

        logger.info(
            "request_finished",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
        )

        structlog.contextvars.clear_contextvars()
        return response
