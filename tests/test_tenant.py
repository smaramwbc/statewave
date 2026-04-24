"""Tests for tenant middleware."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from server.core.tenant import TenantMiddleware


async def _ok(request):
    return JSONResponse({"tenant_id": getattr(request.state, "tenant_id", None)})


async def _healthz(request):
    return JSONResponse({"ok": True})


def _make_app(require: bool = False):
    app = Starlette(routes=[
        Route("/test", _ok),
        Route("/healthz", _healthz),
    ])
    app.add_middleware(TenantMiddleware, header="X-Tenant-ID", require=require)
    return app


async def test_tenant_optional_no_header():
    app = _make_app(require=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/test")
    assert r.status_code == 200
    assert r.json()["tenant_id"] is None


async def test_tenant_optional_with_header():
    app = _make_app(require=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/test", headers={"X-Tenant-ID": "acme"})
    assert r.status_code == 200
    assert r.json()["tenant_id"] == "acme"


async def test_tenant_required_missing():
    app = _make_app(require=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/test")
    assert r.status_code == 400
    assert "missing_tenant" in r.json()["error"]["code"]


async def test_tenant_required_present():
    app = _make_app(require=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/test", headers={"X-Tenant-ID": "acme"})
    assert r.status_code == 200
    assert r.json()["tenant_id"] == "acme"


async def test_tenant_required_healthz_exempt():
    app = _make_app(require=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
