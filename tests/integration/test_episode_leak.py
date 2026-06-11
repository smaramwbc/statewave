"""Phase-1 episode-leak fix.

A fact that has been correctly *superseded* must not resurface verbatim via its
originating episode in the "Recent interactions" section of an assembled
context — otherwise the stale value leaks straight back into the prompt even
though its memory is no longer active.

`repo.superseded_only_episode_ids` reports episodes referenced *solely* by
non-active memories; `assemble_context` skips them. Episodes that still back an
active memory, or back no memory at all, are preserved unchanged.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

from server.db import repositories as repo
from server.db.tables import EpisodeRow, MemoryRow
from server.services.context import assemble_context

_BASE = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _episode(subject_id: str, text: str, *, tenant_id: str | None = None) -> EpisodeRow:
    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id=subject_id,
        tenant_id=tenant_id,
        source="test",
        type="conversation",
        payload={"text": text},
        occurred_at=_BASE,
    )


def _memory(
    subject_id: str,
    content: str,
    *,
    status: str,
    episode_ids: list[uuid.UUID],
    kind: str = "profile_fact",
    tenant_id: str | None = None,
) -> MemoryRow:
    return MemoryRow(
        id=uuid.uuid4(),
        subject_id=subject_id,
        tenant_id=tenant_id,
        kind=kind,
        content=content,
        summary=content[:200],
        confidence=0.9,
        valid_from=_BASE,
        source_episode_ids=episode_ids,
        metadata_={},
        status=status,
    )


# ---------------------------------------------------------------------------
# repo.superseded_only_episode_ids — deterministic set logic
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_episode_backed_only_by_superseded_is_obsolete(session_factory, subject_id):
    ep = _episode(subject_id, "Stripe charges 3.5%")
    async with session_factory() as session:
        session.add(ep)
        session.add(_memory(subject_id, "Stripe 3.5%", status="superseded", episode_ids=[ep.id]))
        await session.commit()

    async with session_factory() as session:
        obsolete = await repo.superseded_only_episode_ids(session, subject_id)

    assert obsolete == {ep.id}


@pytest.mark.anyio
async def test_episode_with_an_active_backing_is_kept(session_factory, subject_id):
    """Partial supersession: an episode backing BOTH a superseded and an active
    memory must be preserved (the active fact still depends on it)."""
    ep = _episode(subject_id, "shared episode")
    async with session_factory() as session:
        session.add(ep)
        session.add(_memory(subject_id, "old fact", status="superseded", episode_ids=[ep.id]))
        session.add(_memory(subject_id, "live fact", status="active", episode_ids=[ep.id]))
        await session.commit()

    async with session_factory() as session:
        obsolete = await repo.superseded_only_episode_ids(session, subject_id)

    assert obsolete == set()


@pytest.mark.anyio
async def test_episode_with_no_backing_memory_is_kept(session_factory, subject_id):
    """A raw / uncompiled episode (no memory references it) is never obsolete."""
    ep = _episode(subject_id, "uncompiled raw episode")
    async with session_factory() as session:
        session.add(ep)
        await session.commit()

    async with session_factory() as session:
        obsolete = await repo.superseded_only_episode_ids(session, subject_id)

    assert obsolete == set()


@pytest.mark.anyio
async def test_tombstoned_backing_also_counts_as_dead(session_factory, subject_id):
    ep = _episode(subject_id, "tombstoned-backed")
    async with session_factory() as session:
        session.add(ep)
        session.add(_memory(subject_id, "gone", status="tombstoned", episode_ids=[ep.id]))
        await session.commit()

    async with session_factory() as session:
        obsolete = await repo.superseded_only_episode_ids(session, subject_id)

    assert obsolete == {ep.id}


@pytest.mark.anyio
async def test_obsolete_lookup_is_tenant_scoped(session_factory, subject_id):
    """A superseded memory under tenant B must not mark tenant A's episode
    obsolete (and vice-versa), even if they share a subject id."""
    ep_a = _episode(subject_id, "tenant a episode", tenant_id="tenant-a")
    async with session_factory() as session:
        session.add(ep_a)
        session.add(
            _memory(subject_id, "a superseded", status="superseded",
                    episode_ids=[ep_a.id], tenant_id="tenant-a")
        )
        await session.commit()

    async with session_factory() as session:
        obsolete_a = await repo.superseded_only_episode_ids(session, subject_id, tenant_id="tenant-a")
        obsolete_b = await repo.superseded_only_episode_ids(session, subject_id, tenant_id="tenant-b")

    assert obsolete_a == {ep_a.id}
    assert obsolete_b == set()


# ---------------------------------------------------------------------------
# assemble_context — the leak is closed end to end
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_assembled_context_drops_superseded_backed_episode(session_factory, subject_id):
    stale_ep = _episode(subject_id, "Stripe charges 3.5% plus 35 cents per transaction")
    live_ep = _episode(subject_id, "Stripe charges 2.9% plus 30 cents per transaction")
    async with session_factory() as session:
        session.add(stale_ep)
        session.add(live_ep)
        session.add(
            _memory(subject_id, "Stripe pricing is 3.5% plus 35 cents",
                    status="superseded", episode_ids=[stale_ep.id])
        )
        session.add(
            _memory(subject_id, "Stripe pricing is 2.9% plus 30 cents",
                    status="active", episode_ids=[live_ep.id])
        )
        await session.commit()

    async with session_factory() as session:
        bundle = await assemble_context(
            session, subject_id, task="What is Stripe's current processing fee?", max_tokens=4000
        )

    episode_ids = {str(e.id) for e in bundle.episodes}
    # The live episode survives; the superseded-backed one is dropped.
    assert str(live_ep.id) in episode_ids
    assert str(stale_ep.id) not in episode_ids
    # And the stale value never reaches the assembled prompt text.
    assert "2.9" in bundle.assembled_context
    assert "3.5" not in bundle.assembled_context
