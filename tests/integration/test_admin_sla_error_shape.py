"""Regression: GET /admin/subjects/{id}/sla must return the same response
shape on its error path as on success.

The bare-except fallback returned only 4 of the success branch's 8 keys (with
HTTP 200), so a typed admin client reading avg_*/breach_count got missing keys
on any transient DB error — and the failure was indistinguishable from a
genuinely empty subject.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

_EXPECTED_KEYS = {
    "total_sessions",
    "resolved_sessions",
    "open_sessions",
    "avg_first_response_seconds",
    "avg_resolution_seconds",
    "first_response_breach_count",
    "resolution_breach_count",
    "sessions",
}


@pytest.mark.anyio
async def test_sla_error_path_matches_success_shape(
    client: AsyncClient, subject_id: str, monkeypatch
):
    # Success path (empty subject → all-zero metrics) carries the full shape.
    r = await client.get(f"/admin/subjects/{subject_id}/sla")
    assert r.status_code == 200
    assert set(r.json().keys()) == _EXPECTED_KEYS

    # Error path must carry the identical shape, not a truncated one.
    async def boom(*args, **kwargs):
        raise RuntimeError("db exploded")

    monkeypatch.setattr("server.services.sla.compute_sla", boom)
    r = await client.get(f"/admin/subjects/{subject_id}/sla")
    assert r.status_code == 200
    assert set(r.json().keys()) == _EXPECTED_KEYS
