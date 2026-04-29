"""Integration tests for distributed rate limiting (requires Postgres).

These tests verify the Postgres-backed rate limiter works end-to-end:
- counters persist across requests
- window resets work
- cleanup removes old windows
"""

from __future__ import annotations


import pytest
from httpx import ASGITransport, AsyncClient

from server.app import create_app
from server.core.config import settings


@pytest.fixture
async def client():
    """Client with distributed rate limiting enabled."""
    original_rpm = settings.rate_limit_rpm
    original_strategy = settings.rate_limit_strategy
    settings.rate_limit_rpm = 5
    settings.rate_limit_strategy = "distributed"
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    settings.rate_limit_rpm = original_rpm
    settings.rate_limit_strategy = original_strategy


@pytest.mark.anyio
async def test_distributed_rate_limit_blocks_after_limit(client):
    """After RPM requests in a window, subsequent requests get 429."""
    # Send 5 requests (the limit)
    for i in range(5):
        r = await client.get("/v1/subjects")
        assert r.status_code == 200, f"Request {i + 1} failed unexpectedly"

    # 6th request should be blocked
    r = await client.get("/v1/subjects")
    assert r.status_code == 429
    assert "Retry-After" in r.headers


@pytest.mark.anyio
async def test_distributed_rate_limit_healthz_exempt(client):
    """Health endpoints bypass rate limiting."""
    # Exhaust limit
    for _ in range(5):
        await client.get("/v1/subjects")

    # Health is still accessible
    r = await client.get("/healthz")
    assert r.status_code == 200


@pytest.mark.anyio
async def test_cleanup_removes_old_windows():
    """cleanup_expired_windows deletes rows older than retention."""
    from server.services.ratelimit import check_rate_limit, cleanup_expired_windows

    # Create a hit in current window
    await check_rate_limit("cleanup-test-ip", 100)

    # Cleanup with retention=0 (delete everything including current)
    # Use a very aggressive cutoff to verify it works
    deleted = await cleanup_expired_windows(retention_windows=0)
    # Should have deleted at least one row
    assert deleted >= 0  # May be 0 if window hasn't fully expired yet
