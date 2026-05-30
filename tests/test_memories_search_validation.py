"""Validation tests for memory search parameters."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1])
async def test_search_memories_rejects_non_positive_limit(client: AsyncClient, limit: int):
    response = await client.get(
        "/v1/memories/search",
        params={"subject_id": "user-1", "limit": limit},
    )

    assert response.status_code == 422
