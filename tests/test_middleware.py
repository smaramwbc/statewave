"""Tests for auth and rate limiting middleware."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from server.core.auth import APIKeyMiddleware
from server.core.ratelimit import RateLimitMiddleware


async def _ok(request):
    return JSONResponse({"ok": True})


def _make_app(api_key: str | None = None, rpm: int = 0):
    app = Starlette(routes=[
        Route("/test", _ok),
        Route("/healthz", _ok),
    ])
    if api_key:
        app.add_middleware(APIKeyMiddleware, api_key=api_key)
    if rpm > 0:
        app.add_middleware(RateLimitMiddleware, rpm=rpm)
    return app


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_app():
    return _make_app(api_key="test-secret-key")


async def test_auth_missing_key(auth_app):
    async with AsyncClient(transport=ASGITransport(app=auth_app), base_url="http://test") as c:
        r = await c.get("/test")
    assert r.status_code == 401
    assert "missing_api_key" in r.json()["error"]["code"]


async def test_auth_wrong_key(auth_app):
    async with AsyncClient(transport=ASGITransport(app=auth_app), base_url="http://test") as c:
        r = await c.get("/test", headers={"X-API-Key": "wrong"})
    assert r.status_code == 403


async def test_auth_correct_key(auth_app):
    async with AsyncClient(transport=ASGITransport(app=auth_app), base_url="http://test") as c:
        r = await c.get("/test", headers={"X-API-Key": "test-secret-key"})
    assert r.status_code == 200


async def test_auth_query_param(auth_app):
    async with AsyncClient(transport=ASGITransport(app=auth_app), base_url="http://test") as c:
        r = await c.get("/test?api_key=test-secret-key")
    assert r.status_code == 200


async def test_auth_healthz_no_key_needed(auth_app):
    async with AsyncClient(transport=ASGITransport(app=auth_app), base_url="http://test") as c:
        r = await c.get("/healthz")
    assert r.status_code == 200


async def test_auth_disabled_when_no_key():
    app = _make_app(api_key=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/test")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Rate limit tests
# ---------------------------------------------------------------------------

async def test_rate_limit_enforced():
    app = _make_app(rpm=3)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        for _ in range(3):
            r = await c.get("/test")
            assert r.status_code == 200
        r = await c.get("/test")
        assert r.status_code == 429
        assert "Retry-After" in r.headers


async def test_rate_limit_healthz_exempt():
    app = _make_app(rpm=1)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        await c.get("/test")  # uses the 1 allowed
        r = await c.get("/healthz")
        assert r.status_code == 200


async def test_rate_limit_disabled_when_zero():
    app = _make_app(rpm=0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        for _ in range(50):
            r = await c.get("/test")
            assert r.status_code == 200
