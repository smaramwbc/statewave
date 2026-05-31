"""Regression: cursor pagination over the receipts audit log must not drop or
duplicate rows.

list_receipts ordered by created_at but paginated with `receipt_id < cursor`.
Keyset pagination is only sound when the ORDER BY and the cursor predicate use
the same key; created_at (DB commit time) and receipt_id (app-side ULID) come
from different clocks and can disagree (clock skew, multi-replica, same-ms
inserts), silently skipping or re-showing receipts across pages — data loss in
an append-only audit trail that compliance reviewers walk page by page.
"""

from __future__ import annotations

import datetime as dt

import pytest

from server.db import repositories as repo
from server.db.tables import ReceiptRow


@pytest.mark.anyio
async def test_pagination_returns_every_receipt_exactly_once(session_factory, subject_id):
    n = 5
    base = dt.datetime(2020, 6, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
    # receipt_id increases lexically with i, but created_at DECREASES with i,
    # so the created_at order and the receipt_id order maximally disagree.
    # IDs are namespaced by the unique subject so they neither collide with the
    # all-zeros "unknown id" probe other tests use nor leak across the shared
    # session DB.
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
                    body={},
                    as_of=base,
                    created_at=base - dt.timedelta(days=i),
                    status="active",
                )
            )
        await session.commit()

    # Walk every page exactly as the API does (next_cursor = last receipt_id,
    # stop when a page is under-full).
    seen: list[str] = []
    cursor: str | None = None
    limit = 2
    for _ in range(n + 3):  # safety bound against an infinite loop
        async with session_factory() as session:
            rows = await repo.list_receipts(
                session, subject_id, tenant_id=None, limit=limit, cursor=cursor
            )
        if not rows:
            break
        seen.extend(r.receipt_id for r in rows)
        if len(rows) < limit:
            break
        cursor = rows[-1].receipt_id

    assert sorted(seen) == sorted(ids), "every receipt must be returned across pages"
    assert len(seen) == len(set(seen)), "no receipt may be returned twice"

    # Clean up this test's receipts from the shared session DB.
    from sqlalchemy import delete

    async with session_factory() as session:
        await session.execute(delete(ReceiptRow).where(ReceiptRow.subject_id == subject_id))
        await session.commit()
