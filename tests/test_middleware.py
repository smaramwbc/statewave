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
    app = Starlette(
        routes=[
            Route("/test", _ok),
            Route("/healthz", _ok),
        ]
    )
    if api_key:
        app.add_middleware(APIKeyMiddleware, api_key=api_key)
    if rpm > 0:
        app.add_middleware(RateLimitMiddleware, rpm=rpm, strategy="memory")
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


async def test_auth_same_length_wrong_key(auth_app):
    # Behavioral parity guard for the constant-time comparison: a wrong key of
    # the SAME length as the secret must still be rejected with 403. The
    # existing wrong-key test uses a shorter value; this one exercises the
    # equal-length path that hmac.compare_digest is specifically used for.
    wrong = "x" * len("test-secret-key")
    async with AsyncClient(transport=ASGITransport(app=auth_app), base_url="http://test") as c:
        r = await c.get("/test", headers={"X-API-Key": wrong})
    assert r.status_code == 403


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


async def test_debug_mode_does_not_disable_auth(monkeypatch):
    # docker-compose.yml runs the api with STATEWAVE_DEBUG=true, and people
    # read that as "auth is off in debug mode". It is not: debug only picks
    # the console log renderer. Whether a key is demanded depends solely on
    # STATEWAVE_API_KEY. This pins the real app's middleware wiring so a
    # future convenience shortcut cannot quietly open a keyed server.
    from server.app import create_app
    from server.core.config import settings
    from server.core.logging import setup_logging

    monkeypatch.setattr(settings, "debug", True)
    monkeypatch.setattr(settings, "api_key", "test-secret-key")

    try:
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            missing = await c.get("/v1/subjects")
            wrong = await c.get("/v1/subjects", headers={"X-API-Key": "dev-local-placeholder"})
    finally:
        setup_logging(debug=False)

    assert missing.status_code == 401
    assert wrong.status_code == 403


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
