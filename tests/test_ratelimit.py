"""Tests for distributed rate limiting.

Unit tests use mocked DB; integration tests need Postgres.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from server.core.ratelimit import RateLimitMiddleware


async def _ok(request):
    return JSONResponse({"ok": True})


def _make_app(rpm: int, strategy: str = "memory"):
    app = Starlette(routes=[Route("/test", _ok), Route("/healthz", _ok)])
    app.add_middleware(RateLimitMiddleware, rpm=rpm, strategy=strategy)
    return app


# ---------------------------------------------------------------------------
# In-memory strategy (unit tests, no DB)
# ---------------------------------------------------------------------------


class TestMemoryStrategy:
    @pytest.mark.anyio
    async def test_allows_under_limit(self):
        app = _make_app(rpm=5, strategy="memory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for _ in range(5):
                r = await c.get("/test")
                assert r.status_code == 200

    @pytest.mark.anyio
    async def test_blocks_over_limit(self):
        app = _make_app(rpm=3, strategy="memory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for _ in range(3):
                await c.get("/test")
            r = await c.get("/test")
            assert r.status_code == 429
            assert "Retry-After" in r.headers

    @pytest.mark.anyio
    async def test_healthz_exempt(self):
        app = _make_app(rpm=1, strategy="memory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.get("/test")  # uses the 1 allowed
            r = await c.get("/healthz")
            assert r.status_code == 200

    @pytest.mark.anyio
    async def test_disabled_when_zero(self):
        app = _make_app(rpm=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for _ in range(100):
                r = await c.get("/test")
                assert r.status_code == 200


# ---------------------------------------------------------------------------
# Distributed strategy (mocked DB)
# ---------------------------------------------------------------------------


class TestDistributedStrategy:
    @pytest.mark.anyio
    async def test_allows_when_under_limit(self):
        """When DB returns count <= rpm, request is allowed."""
        with patch("server.services.ratelimit.check_rate_limit", new_callable=AsyncMock) as mock:
            mock.return_value = (True, 0)
            app = _make_app(rpm=10, strategy="distributed")
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/test")
                assert r.status_code == 200
                mock.assert_called_once()

    @pytest.mark.anyio
    async def test_blocks_when_over_limit(self):
        """When DB returns count > rpm, request is blocked with 429."""
        with patch("server.services.ratelimit.check_rate_limit", new_callable=AsyncMock) as mock:
            mock.return_value = (False, 42)
            app = _make_app(rpm=10, strategy="distributed")
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/test")
                assert r.status_code == 429
                assert r.headers["Retry-After"] == "42"

    @pytest.mark.anyio
    async def test_graceful_degradation_on_db_error(self):
        """When DB is unreachable, the service catches the error and allows (fail-open)."""
        # The service itself catches exceptions and returns (True, 0)
        # so we test that behavior directly
        from server.services.ratelimit import check_rate_limit

        with patch("server.services.ratelimit.async_session_factory") as mock_factory:
            mock_factory.side_effect = Exception("DB down")
            allowed, retry = await check_rate_limit("test-ip", 10)
            assert allowed is True
            assert retry == 0

    @pytest.mark.anyio
    async def test_healthz_exempt_distributed(self):
        """Health paths bypass rate limiting even in distributed mode."""
        with patch("server.services.ratelimit.check_rate_limit", new_callable=AsyncMock) as mock:
            mock.return_value = (False, 60)  # would block if checked
            app = _make_app(rpm=1, strategy="distributed")
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.get("/healthz")
                assert r.status_code == 200
                mock.assert_not_called()


# ---------------------------------------------------------------------------
# Config backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    @pytest.mark.anyio
    async def test_rpm_zero_disables_all_strategies(self):
        """When rpm=0, no rate limiting regardless of strategy."""
        app = _make_app(rpm=0, strategy="distributed")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            for _ in range(50):
                r = await c.get("/test")
                assert r.status_code == 200

    @pytest.mark.anyio
    async def test_error_response_format(self):
        """429 response has correct JSON structure."""
        app = _make_app(rpm=1, strategy="memory")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.get("/test")
            r = await c.get("/test")
            assert r.status_code == 429
            body = r.json()
            assert body["error"]["code"] == "rate_limited"
            assert "120" not in body["error"]["message"]  # uses actual rpm
            assert "1" in body["error"]["message"]
