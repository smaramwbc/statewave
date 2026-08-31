"""Claims on structured candidates must carry a temporal anchor (#369 follow-up).

Without ``valid_from``, ``_claim_cmp`` orders same-bucket claims by
``created_at`` — which ties for rows compiled in one transaction and then
falls to random UUIDs, so contradiction resolution was non-deterministic on
the v1 structured-candidate path (measured: 8 identical runs → 5/3 split).
v2 envelopes already carried a producer-supplied ``valid_from``; v1 dropped
it, and neither defaulted. Both now anchor to ``episode_valid_from`` (which
honours ``payload.event_time``) whenever the claim doesn't supply its own.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from server.db.tables import EpisodeRow
from server.services.compilers.heuristic import episode_valid_from
from server.services.structured import compile_candidates

# A valid v2 claim shape for the registry's v2-capable key (kept local — see
# tests/test_claims_v2.py for the canonical matrix).
_BASE_Q = {
    "payment_method": "card",
    "rate_type": "standard",
    "currency": "USD",
    "charge_unit": "transaction",
}
_V350 = {"percentage_basis_points": 350, "fixed_minor_units": 35}

_EVENT = datetime(2023, 5, 4, 12, 0, tzinfo=timezone.utc)


def _episode(claim: dict, event_time: str | None = "2023-05-04T12:00:00+00:00") -> EpisodeRow:
    payload: dict = {
        "statewave": {
            "memory_candidates": [
                {"kind": "domain_fact", "text": "the user's name is ada", "claim": claim}
            ]
        }
    }
    if event_time is not None:
        payload["event_time"] = event_time
    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id="s-1",
        source="test",
        type="message",
        payload=payload,
        metadata_={},
        provenance={},
    )


def _claim_md(ep: EpisodeRow) -> dict:
    rows = compile_candidates(ep)
    assert rows is not None and len(rows) == 1
    return rows[0].metadata_["claim"]


def test_v1_claim_without_valid_from_anchors_to_the_episode():
    ep = _episode({"schema_version": 1, "key": "identity.name", "value": "Ada"})
    md = _claim_md(ep)
    assert md["valid_from"] == episode_valid_from(ep).isoformat()
    assert md["valid_from"] == _EVENT.isoformat()


def test_v1_producer_supplied_valid_from_wins():
    ep = _episode(
        {
            "schema_version": 1,
            "key": "identity.name",
            "value": "Ada",
            "valid_from": "2020-01-01T00:00:00+00:00",
            "valid_to": "2021-01-01T00:00:00+00:00",
        }
    )
    md = _claim_md(ep)
    assert md["valid_from"] == "2020-01-01T00:00:00+00:00"
    assert md["valid_to"] == "2021-01-01T00:00:00+00:00"


def _v2_claim(**extra) -> dict:
    return {
        "schema_version": 2,
        "key": "pricing.processing_rate",
        "entity_key": "acme gmbh",
        "qualifiers": dict(_BASE_Q),
        "value": dict(_V350),
        **extra,
    }


def test_v2_claim_without_valid_from_anchors_to_the_episode():
    ep = _episode(_v2_claim())
    md = _claim_md(ep)
    assert md.get("valid_from") == episode_valid_from(ep).isoformat()


def test_v2_producer_supplied_valid_from_untouched():
    ep = _episode(_v2_claim(valid_from="2019-06-01T00:00:00+00:00"))
    md = _claim_md(ep)
    assert md.get("valid_from") == "2019-06-01T00:00:00+00:00"


def test_two_compiled_episodes_order_by_event_time_not_row_ids():
    """The end the fix serves: same key, different values, compiled in one
    transaction — the claim envelopes alone must give a deterministic order."""
    early = _episode(
        {"schema_version": 1, "key": "identity.name", "value": "Ada"},
        event_time="2023-01-01T00:00:00+00:00",
    )
    late = _episode(
        {"schema_version": 1, "key": "identity.name", "value": "Grace"},
        event_time="2023-09-01T00:00:00+00:00",
    )
    md_early, md_late = _claim_md(early), _claim_md(late)
    assert md_early["valid_from"] < md_late["valid_from"]


def test_valid_to_before_the_anchor_leaves_valid_from_absent():
    """A fact that ENDED before the episode must not get the episode anchor as
    its start — that would persist an inverted window and make the ended fact
    look newest to the resolver."""
    ep = _episode(
        {
            "schema_version": 1,
            "key": "identity.name",
            "value": "Ada",
            "valid_to": "2019-01-01T00:00:00+00:00",
        }
    )
    md = _claim_md(ep)
    assert "valid_from" not in md
    assert md["valid_to"] == "2019-01-01T00:00:00+00:00"


def test_v2_valid_to_before_the_anchor_leaves_valid_from_absent():
    ep = _episode(_v2_claim(valid_to="2019-01-01T00:00:00+00:00"))
    md = _claim_md(ep)
    assert md.get("valid_from") is None
    assert md.get("valid_to") == "2019-01-01T00:00:00+00:00"


def test_compile_then_resolve_is_deterministic_end_to_end():
    """The bug that motivated all of this: two contradictory facts compiled in
    one transaction (tied created_at, random UUIDs) resolved to different
    winners across runs. Compiled envelopes now carry event-time anchors, so
    the later EVENT must win on every run, in either input order."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from server.services.conflicts import resolve_conflicts

    async def _one_round(order_reversed: bool) -> str:
        early = _episode(
            {"schema_version": 1, "key": "identity.name", "value": "Ada"},
            event_time="2023-01-01T00:00:00+00:00",
        )
        late = _episode(
            {"schema_version": 1, "key": "identity.name", "value": "Grace"},
            event_time="2023-09-01T00:00:00+00:00",
        )
        rows = [compile_candidates(e)[0] for e in (early, late)]
        tied = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for r in rows:
            r.created_at = tied  # the pathological shape: same-transaction ties
            r.status = "active"
        if order_reversed:
            rows.reverse()
        with patch("server.services.conflicts.repo") as mock_repo:
            mock_repo.list_active_memories_by_subject = AsyncMock(return_value=rows)
            mock_repo.mark_memories_superseded = AsyncMock()
            await resolve_conflicts(AsyncMock(), "s-1")
        (winner,) = [r for r in rows if r.status == "active"]
        return winner.metadata_["claim"]["value"]

    winners = {asyncio.run(_one_round(order_reversed=i % 2 == 1)) for i in range(8)}
    assert winners == {"grace"}


def test_naive_valid_to_does_not_crash_the_guard_v1():
    """_parse_dt returns naive datetimes for offset-less ISO strings; the guard
    must normalize before comparing or one poison episode 500s every compile
    of its subject (review finding)."""
    ep = _episode(
        {
            "schema_version": 1,
            "key": "identity.name",
            "value": "Ada",
            "valid_to": "2019-01-01T00:00:00",  # offset-less → naive
        }
    )
    md = _claim_md(ep)
    assert "valid_from" not in md
    assert md["valid_to"] == "2019-01-01T00:00:00"


def test_naive_valid_to_does_not_crash_the_guard_v2():
    ep = _episode(_v2_claim(valid_to="2019-01-01T00:00:00"))
    md = _claim_md(ep)
    assert md.get("valid_from") is None
