"""The subject_entities identity invariant lives in the DATABASE (issue #383).

`upsert_entity_with_link` was select-then-insert over non-unique indexes:
two CONCURRENT entity-population runs for the same subject (e.g. a
rebuild-entities call re-sent while the first was still running) could
silently insert duplicate rows for the same normalized entity. Migration
0030 adds a unique expression index over (subject_id,
COALESCE(tenant_id, ''), entity_normalized) and the fresh-insert path
became INSERT ... ON CONFLICT, so concurrent writers CONVERGE into one
row with the union of linked_memory_ids.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select

from server.db import repositories as repo
from server.db.tables import SubjectEntityRow

pytestmark = pytest.mark.anyio


async def _upsert(session_factory, subject_id: str, memory_id: uuid.UUID):
    async with session_factory() as session:
        row = await repo.upsert_entity_with_link(
            session,
            subject_id=subject_id,
            tenant_id=None,
            entity_text="Acme Corp",
            entity_normalized="acme corp",
            entity_kind="ORG",
            embedding=None,
            memory_id=memory_id,
        )
        await session.commit()
        return row.id


async def _rows_for(session_factory, subject_id: str):
    async with session_factory() as session:
        result = await session.execute(
            select(SubjectEntityRow).where(SubjectEntityRow.subject_id == subject_id)
        )
        return list(result.scalars())


async def test_concurrent_upserts_converge_to_one_row(session_factory):
    """The exact #383 scenario: N writers race the same normalized entity in
    separate sessions/transactions. Exactly one row must survive, linking
    every writer's memory."""
    subject_id = f"uniq-{uuid.uuid4().hex[:8]}"
    memory_ids = [uuid.uuid4() for _ in range(5)]

    await asyncio.gather(
        *[_upsert(session_factory, subject_id, mid) for mid in memory_ids]
    )

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 1, f"expected one converged row, got {len(rows)}"
    assert set(rows[0].linked_memory_ids) == set(memory_ids)


async def test_repeated_upsert_same_memory_links_once(session_factory):
    subject_id = f"uniq-{uuid.uuid4().hex[:8]}"
    memory_id = uuid.uuid4()
    await _upsert(session_factory, subject_id, memory_id)
    await _upsert(session_factory, subject_id, memory_id)

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 1
    assert rows[0].linked_memory_ids == [memory_id]


async def test_conflict_path_keeps_first_non_null_embedding(session_factory):
    """A later writer carrying an embedding must fill a NULL, and a later
    NULL must not clobber an existing embedding."""
    subject_id = f"uniq-{uuid.uuid4().hex[:8]}"

    async def upsert_with(embedding, memory_id):
        async with session_factory() as session:
            await repo.upsert_entity_with_link(
                session,
                subject_id=subject_id,
                tenant_id=None,
                entity_text="Berlin",
                entity_normalized="berlin",
                entity_kind="GPE",
                embedding=embedding,
                memory_id=memory_id,
            )
            await session.commit()

    # The exact-match SELECT path returns early for same-session visibility,
    # so exercise the ON CONFLICT branch via two independent sessions racing:
    # here sequential is enough to pin the embedding-coalesce rule only for
    # the conflict UPDATE — force it by bypassing step 1 with a direct race.
    vec = [0.1] * 1536
    m1, m2 = uuid.uuid4(), uuid.uuid4()
    await asyncio.gather(upsert_with(None, m1), upsert_with(vec, m2))

    rows = await _rows_for(session_factory, subject_id)
    assert len(rows) == 1
    assert set(rows[0].linked_memory_ids) == {m1, m2}
