"""Smoke tests against a live Statewave stack (docker compose).

Run via `make test-cold` or manually after `docker compose up -d`:

    STATEWAVE_SMOKE_URL=http://localhost:8100 pytest tests/smoke/ -m smoke -v
"""

from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.environ.get("STATEWAVE_SMOKE_URL", "http://localhost:8100").rstrip("/")

pytestmark = pytest.mark.smoke


@pytest.mark.asyncio
async def test_healthz():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_readyz_ready():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        resp = await client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ready", body


@pytest.mark.asyncio
async def test_openapi_available():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    assert "openapi" in resp.json()
