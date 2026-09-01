"""#376's acceptance experiment, end to end against a real database.

An operator registers `guide.walkthrough` as single-valued in
`tenant_configs.config.claim_keys`; a consumer then observes the same logical
fact four times with alternating values through the structured path. The
registered key must behave exactly like a built-in: deterministic supersession
down to ONE active row carrying the latest value. Without the registration the
same four writes leave four active rows — the unbounded growth the issue
reports.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete

from server.db import repositories as repo
from server.db.tables import EpisodeRow, MemoryRow, TenantConfigRow
from server.services.claims import load_tenant_claim_keys
from server.services.compilers.heuristic import HeuristicCompiler
from server.services.conflicts import resolve_conflicts

pytestmark = pytest.mark.anyio

_TENANT = f"t-claimkeys-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def _cleanup(session_factory):
    yield
    async with session_factory() as session:
        await session.execute(delete(MemoryRow).where(MemoryRow.tenant_id == _TENANT))
        await session.execute(delete(EpisodeRow).where(EpisodeRow.tenant_id == _TENANT))
        await session.execute(
            delete(TenantConfigRow).where(TenantConfigRow.tenant_id == _TENANT)
        )
        await session.commit()


def _episode(subject_id: str, value: str, when: str, text: str) -> EpisodeRow:
    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id=subject_id,
        tenant_id=_TENANT,
        source="test",
        type="message",
        payload={
            "event_time": when,
            "statewave": {
                "memory_candidates": [
                    {
                        "kind": "domain_fact",
                        "text": text,
                        "claim": {
                            "schema_version": 1,
                            "key": "guide.walkthrough",
                            "value": value,
                        },
                    }
                ]
            },
        },
        metadata_={},
        provenance={},
    )


async def _run(session_factory, subject_id: str, registered: bool) -> list[MemoryRow]:
    async with session_factory() as session:
        if registered:
            session.add(
                TenantConfigRow(
                    tenant_id=_TENANT,
                    config={"claim_keys": {"guide.walkthrough": "single"}},
                )
            )
            await session.commit()

        claim_keys = await load_tenant_claim_keys(session, _TENANT)
        texts = [
            "kicked off the walkthrough",
            "wrapped it up end to end",
            "back into the tour again",
            "finished every last step",
        ]
        episodes = [
            _episode(subject_id, v, f"2026-0{i + 1}-01T00:00:00+00:00", texts[i])
            for i, v in enumerate(["started", "done", "started", "done"])
        ]
        rows = HeuristicCompiler().compile(episodes, claim_keys=claim_keys)
        for r in rows:
            r.tenant_id = _TENANT
            session.add(r)
        await session.commit()

        await resolve_conflicts(session, subject_id, tenant_id=_TENANT)
        await session.commit()
        active = await repo.list_active_memories_by_subject(
            session, subject_id, tenant_id=_TENANT
        )
        return list(active)


async def test_registered_key_bounds_state_to_one_active_row(session_factory):
    active = await _run(session_factory, f"s-{uuid.uuid4().hex[:8]}", registered=True)
    assert len(active) == 1
    assert active[0].metadata_["claim"]["value"] == "done"  # the latest event wins


async def test_without_registration_the_same_writes_grow_unbounded(session_factory):
    """The pre-#376 behaviour, pinned as the contrast: no registration, no
    claim attached, four active rows."""
    active = await _run(session_factory, f"s-{uuid.uuid4().hex[:8]}", registered=False)
    assert len(active) == 4
    assert all("claim" not in (m.metadata_ or {}) for m in active)
