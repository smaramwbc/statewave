"""Tests for GET /admin/usage metering endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from server.app import create_app


@pytest.mark.asyncio
async def test_usage_endpoint():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Basic shape test
        resp = await client.get("/admin/usage")
        assert resp.status_code == 200
        data = resp.json()

        assert "episodes" in data
        assert "memories" in data
        assert "compile_jobs" in data
        assert "webhooks" in data
        assert "active_subjects" in data
        assert "generated_at" in data
        assert "period_start" in data
        assert data["tenant_id"] is None

        for key in ("episodes", "memories", "compile_jobs", "webhooks"):
            for window in ("today", "7d", "30d", "total"):
                assert isinstance(data[key][window], int)

        assert "7d" in data["active_subjects"]
        assert "30d" in data["active_subjects"]
        assert "total" in data["active_subjects"]

        # Tenant filter — nonexistent returns zeros
        resp2 = await client.get("/admin/usage", params={"tenant_id": "nonexistent"})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["tenant_id"] == "nonexistent"
        assert data2["episodes"]["total"] == 0
