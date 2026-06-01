"""Unit guard: upsert_health_cache looks up the existing row scoped to the
caller's tenant, so it can't find-and-overwrite another tenant's row for the
same subject_id.

Full SQL-level isolation (read/overwrite/delete across tenants) is proven
against a real database in tests/integration/test_health_cache_isolation.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from server.db import repositories as repo


async def test_upsert_scopes_existing_lookup_by_tenant():
    with patch.object(repo, "get_health_cache", new=AsyncMock(return_value=None)) as mock_get:
        session = AsyncMock()
        await repo.upsert_health_cache(session, "user-1", "at_risk", 30, tenant_id="tenant-a")

    mock_get.assert_awaited_once_with(session, "user-1", tenant_id="tenant-a")


async def test_upsert_single_tenant_lookup_passes_none():
    with patch.object(repo, "get_health_cache", new=AsyncMock(return_value=None)) as mock_get:
        session = AsyncMock()
        await repo.upsert_health_cache(session, "user-1", "healthy", 100)

    mock_get.assert_awaited_once_with(session, "user-1", tenant_id=None)
