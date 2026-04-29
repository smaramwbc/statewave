"""Tenant dependency for FastAPI routes.

Extracts tenant_id from request.state (set by TenantMiddleware).
Returns None in single-tenant mode — repository functions treat None as
"no tenant filter" for backward compatibility.
"""

from __future__ import annotations

from fastapi import Request


def get_tenant_id(request: Request) -> str | None:
    """FastAPI dependency — extract tenant_id from request state."""
    return getattr(request.state, "tenant_id", None)
