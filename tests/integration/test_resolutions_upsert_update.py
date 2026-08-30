"""Regression: the UPDATE path of POST /v1/resolutions returned 500.

`create_resolution` upserts and then serialises the row it gets back.
`session.commit()` expires every instance in the session, so reading a column
off that row afterwards is lazy IO — which raises `MissingGreenlet` on the
async engine and reaches the client as `500 internal_error`.

It only ever bit the second and later writes to a logical key. The INSERT path
returns the instance the request just constructed, whose attributes are still
populated in Python; the UPDATE path returns the row `upsert_resolution` loaded
from the database, which the commit expired.

The write itself always landed, so the endpoint persisted the caller's value and
then told them it had failed — the worst of the two possible wrong answers for
the only upsert-by-logical-key on the consumer surface.
"""

from __future__ import annotations

import pytest


@pytest.mark.anyio
async def test_repeated_writes_to_one_key_all_succeed(client, subject_id):
    """Four writes to one (subject_id, session_id) — every one a 200."""
    statuses = ["open", "resolved", "open", "resolved"]
    for status in statuses:
        response = await client.post(
            "/v1/resolutions",
            json={
                "subject_id": subject_id,
                "session_id": "session-1",
                "status": status,
                "resolution_summary": f"summary for {status}",
            },
        )
        assert response.status_code == 200, f"{status} write returned {response.status_code}"


@pytest.mark.anyio
async def test_the_update_response_carries_the_new_value(client, subject_id):
    """The response body must describe the write that just happened.

    Asserting only on the status code would pass against a route that returned
    a stale first-write body.
    """
    await client.post(
        "/v1/resolutions",
        json={"subject_id": subject_id, "session_id": "s", "status": "open"},
    )
    second = await client.post(
        "/v1/resolutions",
        json={
            "subject_id": subject_id,
            "session_id": "s",
            "status": "resolved",
            "resolution_summary": "closed it",
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "resolved"
    assert body["resolution_summary"] == "closed it"
    assert body["resolved_at"] is not None


@pytest.mark.anyio
async def test_the_upsert_still_collapses_to_one_row(client, subject_id):
    """The point of the endpoint: one logical key, one row, holding the latest."""
    for status in ["open", "resolved", "open"]:
        await client.post(
            "/v1/resolutions",
            json={"subject_id": subject_id, "session_id": "s", "status": status},
        )

    listed = await client.get("/v1/resolutions", params={"subject_id": subject_id})
    rows = listed.json()
    assert len(rows) == 1
    assert rows[0]["status"] == "open"
    # An update must not be mistaken for a resolve: `resolved_at` is cleared
    # when the status moves back off `resolved`.
    assert rows[0]["resolved_at"] is None
