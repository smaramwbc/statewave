"""Tests for public subject management routes."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_list_subjects_returns_unpaginated_total(monkeypatch):
    from server.api import subjects as api_subjects

    session = AsyncMock()

    async def fake_list_subjects(_session, *, tenant_id, limit, offset):
        assert _session is session
        assert tenant_id == "tenant-a"
        assert limit == 1
        assert offset == 1
        return [
            {
                "subject_id": "user-b",
                "episode_count": 2,
                "memory_count": 1,
            }
        ]

    async def fake_count_subjects(_session, *, tenant_id):
        assert _session is session
        assert tenant_id == "tenant-a"
        return 3

    monkeypatch.setattr(api_subjects.repo, "list_subjects", fake_list_subjects)
    monkeypatch.setattr(api_subjects.repo, "count_subjects", fake_count_subjects, raising=False)

    response = await api_subjects.list_subjects(
        limit=1,
        offset=1,
        session=session,
        tenant_id="tenant-a",
    )

    assert response.total == 3
    assert len(response.subjects) == 1
