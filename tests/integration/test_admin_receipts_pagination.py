"""Regression: /admin/receipts cursor pagination must not drop or duplicate rows.

admin_list_receipts ordered by created_at but paginated with `receipt_id < cursor`
— the same keyset-pagination mismatch fixed for repo.list_receipts in #223, here
on the operator audit endpoint that compliance reviewers walk page by page.
created_at (DB commit time) and receipt_id (app-side ULID) come from different
clocks, so ordering by one while cursoring on the other silently skips/re-shows
receipts across pages.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import delete

from server.db.tables import ReceiptRow


@pytest.mark.anyio
async def test_admin_receipts_pagination_returns_every_row_once(
    client, session_factory, subject_id
):
    n = 5
    base = dt.datetime(2020, 6, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    # receipt_id increases lexically with i, but created_at DECREASES with i, so
    # the two orderings maximally disagree. Namespaced by the unique subject so
    # they don't collide with other tests sharing the session DB.
    prefix = (subject_id.replace("-", "") + "0" * 24)[:24]
    ids = [prefix + f"{i:02d}" for i in range(n)]
    async with session_factory() as session:
        for i, rid in enumerate(ids):
            session.add(
                ReceiptRow(
                    receipt_id=rid,
                    mode="retrieval",
                    subject_id=subject_id,
                    context_hash="0" * 64,
                    context_size_bytes=0,
                    body={"rid": rid},
                    as_of=base,
                    created_at=base - dt.timedelta(days=i),
                    status="active",
                )
            )
        await session.commit()

    # Walk every page exactly as a client does: next_cursor = last receipt_id,
    # stop when the server reports no further cursor.
    seen: list[str] = []
    cursor: str | None = None
    limit = 2
    for _ in range(n + 3):  # safety bound against an infinite loop
        params = {"subject_id": subject_id, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = await client.get("/admin/receipts", params=params)
        assert resp.status_code == 200
        data = resp.json()
        page = data["receipts"]
        if not page:
            break
        seen.extend(r["rid"] for r in page)
        if data["next_cursor"] is None:
            break
        cursor = data["next_cursor"]

    assert sorted(seen) == sorted(ids), "every receipt must be returned across pages"
    assert len(seen) == len(set(seen)), "no receipt may be returned twice"

    async with session_factory() as session:
        await session.execute(delete(ReceiptRow).where(ReceiptRow.subject_id == subject_id))
        await session.commit()
