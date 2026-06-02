"""Integration: optional tenant filter on the operator webhook read endpoints.

The /admin/webhooks endpoints are operator-global; `list`/`purge` already accept
an optional `tenant_id` filter, and these tests cover the same filter on
`/admin/webhooks/stats` and `/admin/webhooks/{event_id}` (added so an operator
can scope debugging to one tenant). Unique tenant ids keep the assertions
independent of other rows in the shared test database.
"""

from __future__ import annotations

import uuid

from server.db.tables import WebhookEventRow


def _tenant() -> str:
    return f"wt-{uuid.uuid4().hex[:10]}"


async def _seed(session_factory, tenant_a: str, tenant_b: str) -> WebhookEventRow:
    """Two events for tenant_a (pending, delivered) and one for tenant_b
    (dead_letter). Returns the tenant_a pending row."""
    a_pending = WebhookEventRow(
        tenant_id=tenant_a, event="episode.created", payload={"event": "x", "data": {}},
        status="pending",
    )
    rows = [
        a_pending,
        WebhookEventRow(
            tenant_id=tenant_a, event="episode.created", payload={"event": "x", "data": {}},
            status="delivered",
        ),
        WebhookEventRow(
            tenant_id=tenant_b, event="episode.created", payload={"event": "x", "data": {}},
            status="dead_letter",
        ),
    ]
    async with session_factory() as s:
        s.add_all(rows)
        await s.commit()
        await s.refresh(a_pending)
    return a_pending


async def test_stats_filtered_by_tenant(client, session_factory):
    tenant_a, tenant_b = _tenant(), _tenant()
    await _seed(session_factory, tenant_a, tenant_b)

    resp = await client.get("/admin/webhooks/stats", params={"tenant_id": tenant_a})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pending"] == 1
    assert body["delivered"] == 1
    assert body["dead_letter"] == 0  # tenant_b's dead_letter must not leak in
    assert body["total"] == 2


async def test_event_status_tenant_filter(client, session_factory):
    tenant_a, tenant_b = _tenant(), _tenant()
    row = await _seed(session_factory, tenant_a, tenant_b)

    # Matching tenant → found.
    ok = await client.get(f"/admin/webhooks/{row.id}", params={"tenant_id": tenant_a})
    assert ok.status_code == 200
    assert ok.json()["id"] == str(row.id)

    # Wrong tenant → not-found (consistent with the filter on list/purge).
    miss = await client.get(f"/admin/webhooks/{row.id}", params={"tenant_id": tenant_b})
    assert miss.status_code == 404

    # No filter → operator global view still returns it.
    glob = await client.get(f"/admin/webhooks/{row.id}")
    assert glob.status_code == 200
