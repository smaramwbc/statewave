"""Structured error handling — consistent JSON error responses."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = structlog.stdlib.get_logger()


# ---------------------------------------------------------------------------
# Error response schema
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_json(code: str, message: str, request: Request, details: Any = None, status: int = 500):
    from fastapi.responses import JSONResponse

    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=details,
            request_id=_request_id(request),
        )
    )
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


# ---------------------------------------------------------------------------
# Register handlers
# ---------------------------------------------------------------------------

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "validation_error",
            request_id=_request_id(request),
            errors=exc.errors(),
        )
        return _error_json(
            code="validation_error",
            message="Request validation failed",
            details=exc.errors(),
            request=request,
            status=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(
            "http_error",
            request_id=_request_id(request),
            status=exc.status_code,
            detail=exc.detail,
        )
        return _error_json(
            code="http_error",
            message=str(exc.detail),
            request=request,
            status=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            request_id=_request_id(request),
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
            exc_info=True,
        )
        return _error_json(
            code="internal_error",
            message="An unexpected error occurred",
            request=request,
            status=500,
        )
