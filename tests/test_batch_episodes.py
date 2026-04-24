"""Tests for batch episode ingestion — validation and routing."""

from __future__ import annotations

from httpx import AsyncClient


class TestBatchEpisodesValidation:
    async def test_batch_empty_list_rejected(self, client: AsyncClient):
        resp = await client.post("/v1/episodes/batch", json={"episodes": []})
        assert resp.status_code == 422

    async def test_batch_over_100_rejected(self, client: AsyncClient):
        episodes = [
            {
                "subject_id": "user-flood",
                "source": "test",
                "type": "message",
                "payload": {"text": "x"},
            }
            for _ in range(101)
        ]
        resp = await client.post("/v1/episodes/batch", json={"episodes": episodes})
        assert resp.status_code == 422

    async def test_batch_missing_required_fields_rejected(self, client: AsyncClient):
        resp = await client.post("/v1/episodes/batch", json={"episodes": [{"subject_id": "x"}]})
        assert resp.status_code == 422
