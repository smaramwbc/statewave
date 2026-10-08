"""Tests for auth and rate limiting middleware."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from server.core.auth import APIKeyMiddleware
from server.core.middleware import RequestIDMiddleware
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


async def test_auth_query_param_rejected(auth_app):
    async with AsyncClient(transport=ASGITransport(app=auth_app), base_url="http://test") as c:
        r = await c.get("/test?api_key=test-secret-key")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "missing_api_key"


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


# ---------------------------------------------------------------------------
# Public schema endpoints (#398)
# ---------------------------------------------------------------------------


async def test_schema_endpoints_stay_public_when_api_key_set(monkeypatch):
    """/docs, /redoc and /openapi.json sit in the auth exemption set, so on a
    deployment with an API key they answer without one while /v1 stays 401.
    This is the behavior #398 asked about; the test records it as intended
    rather than leaving it to be discovered by reading auth.py."""
    from server.app import create_app
    from server.core.config import settings

    monkeypatch.setattr(settings, "api_key", "secret-key")
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        schema = await c.get("/openapi.json")
        swagger = await c.get("/docs")
        redoc = await c.get("/redoc")
        guarded = await c.get("/v1/subjects")

    assert schema.status_code == 200
    assert schema.json()["info"]["version"]
    assert swagger.status_code == 200
    assert redoc.status_code == 200
    assert guarded.status_code == 401


async def test_debug_mode_does_not_change_the_schema_exemption(monkeypatch):
    # debug=False is the setting that would have to matter for a
    # "gate /docs on settings.debug" fix to close anything, so this pins the
    # other half of the pairing that test_debug_mode_does_not_disable_auth
    # already asserts for the key itself. The exemption is keyed on
    # STATEWAVE_API_KEY alone, in both directions: debug changes only the log
    # renderer, and with debug off the schema trio is still exempt. A gate on
    # debug would be dead code on the compose deployment, which sets it true.
    from server.app import create_app
    from server.core.config import settings

    monkeypatch.setattr(settings, "debug", False)
    monkeypatch.setattr(settings, "api_key", "secret-key")
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        schema = await c.get("/openapi.json")
        guarded = await c.get("/v1/subjects")

    assert schema.status_code == 200
    assert guarded.status_code == 401


def test_exemption_sets_agree_across_middlewares():
    """auth, tenant and residency each carry their own exempt-path literal.

    residency_middleware.py states the rule out loud — "the same _PUBLIC_PATHS
    set as TenantMiddleware" — and a request has to clear all three, so a path
    added to one set and not the others becomes a route that authenticates
    fine but answers 400 missing_tenant, or a schema path that 403s on a
    regional pin. Nothing but this test keeps the three copies together.
    """
    from server.core.auth import _PUBLIC_PATHS as auth_paths
    from server.core.residency_middleware import _EXEMPT_PATHS as residency_paths
    from server.core.tenant import _PUBLIC_PATHS as tenant_paths

    assert auth_paths == tenant_paths == residency_paths


# ---------------------------------------------------------------------------
# Request ID tests (#501)
# ---------------------------------------------------------------------------


@pytest.fixture
def request_id_app():
    app = Starlette(routes=[Route("/test", _ok)])
    app.add_middleware(RequestIDMiddleware)
    return app


async def test_request_id_invalid_charset_rejected(request_id_app):
    for bad_id in [
        "has spaces",
        "newline\nin\nid",
        "trailing\n",
        "trailing\r\n",
        "<script>alert(1)</script>",
        "invalid@char!",
        "",
    ]:
        async with AsyncClient(
            transport=ASGITransport(app=request_id_app), base_url="http://test"
        ) as c:
            r = await c.get("/test", headers={"X-Request-ID": bad_id})
        assert r.status_code == 200
        req_id = r.headers["x-request-id"]
        assert req_id != bad_id
        assert len(req_id) == 16
        assert all(ch in "0123456789abcdef" for ch in req_id)
