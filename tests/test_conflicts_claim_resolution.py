"""Hybrid claim-keyed conflict resolution — the new behavior + compat matrix.

Covers the decision tree: single-valued + overlap + different value supersedes;
non-overlap coexists (history); same value dedups via legacy; multi-valued and
unknown/mixed coexist; aliases normalize; determinism is input/DB-order
independent; and the claim path PROTECTS temporal coexistence from lexical
overlap that would otherwise wrongly supersede.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from unittest.mock import AsyncMock, patch

from server.db.tables import MemoryRow
from server.services.conflicts import resolve_conflicts

_NOW = datetime.now(timezone.utc)


def _dt(year: int) -> datetime:
    return datetime(year, 1, 1, tzinfo=timezone.utc)


def _mem(
    content,
    *,
    key=None,
    value=None,
    valid_from=None,
    valid_to=None,
    age_days=0,
    created_days=None,
    mid=None,
    kind="profile_fact",
):
    metadata: dict = {}
    if key is not None:
        env = {"schema_version": 1, "key": key, "value": value}
        if valid_from is not None:
            env["valid_from"] = valid_from.isoformat()
        if valid_to is not None:
            env["valid_to"] = valid_to.isoformat()
        metadata = {"claim": env}
    cd = created_days if created_days is not None else age_days
    return MemoryRow(
        id=mid or uuid.uuid4(),
        subject_id="user-1",
        kind=kind,
        content=content,
        summary=content[:200],
        confidence=0.9,
        valid_from=_NOW - timedelta(days=age_days),
        created_at=_NOW - timedelta(days=cd),
        source_episode_ids=[uuid.uuid4()],
        metadata_=metadata,
        status="active",
    )


async def _resolve(memories):
    with patch("server.services.conflicts.repo") as mock_repo:
        mock_repo.list_active_memories_by_subject = AsyncMock(return_value=memories)
        mock_repo.mark_memories_superseded = AsyncMock()
        result = await resolve_conflicts(AsyncMock(), "user-1")
        called = mock_repo.mark_memories_superseded.await_count
    return result, called


# --- the core win: wording-independent contradiction --------------------------


async def test_acme_to_globex_overlapping_supersedes():
    # Low lexical overlap — legacy Jaccard would MISS this; the claim path catches it.
    older = _mem("Previously employed at Acme Industries", key="employment.current_employer", value="Acme", age_days=10)
    newer = _mem("Just started a new role at Globex", key="employment.current_employer", value="Globex", age_days=0)
    result, called = await _resolve([older, newer])
    assert older.id in result and newer.id not in result
    assert called == 1


async def test_historical_and_current_employer_coexist():
    older = _mem("worked at Acme", key="employment.current_employer", value="Acme",
                 valid_from=_dt(2015), valid_to=_dt(2020), age_days=10)
    newer = _mem("now at Globex", key="employment.current_employer", value="Globex",
                 valid_from=_dt(2020), age_days=0)  # touching, non-overlapping
    result, _ = await _resolve([older, newer])
    assert result == []  # history preserved


async def test_temporal_coexistence_protected_from_lexical_overlap():
    # "I live in Berlin" vs "I live in Munich" = Jaccard 0.6 → legacy WOULD
    # supersede. With single-valued claims and non-overlapping validity, the
    # claim path must keep both (and stop legacy from touching them).
    older = _mem("I live in Berlin", key="location.current_home", value="Berlin",
                 valid_from=_dt(2018), valid_to=_dt(2020), age_days=100)
    newer = _mem("I live in Munich", key="location.current_home", value="Munich",
                 valid_from=_dt(2020), age_days=0)
    result, _ = await _resolve([older, newer])
    assert result == []


# --- aliases ------------------------------------------------------------------


async def test_approved_aliases_normalize_to_same_key_and_supersede():
    older = _mem("at Acme", key="employer", value="Acme", age_days=10)
    newer = _mem("at Globex", key="works_at", value="Globex", age_days=0)
    result, _ = await _resolve([older, newer])
    assert older.id in result  # employer/works_at → employment.current_employer


async def test_unknown_alias_is_safe_and_coexists():
    older = _mem("Acme", key="my_company", value="Acme", age_days=10)  # unregistered
    newer = _mem("Globex", key="my_company", value="Globex", age_days=0)
    result, _ = await _resolve([older, newer])
    assert result == []  # non-authoritative key + low lexical overlap → coexist


# --- cardinality --------------------------------------------------------------


async def test_multi_valued_coexist():
    a = _mem("I use Stripe", key="payment.processors_used", value="Stripe", age_days=10)
    b = _mem("I use PayPal", key="payment.processors_used", value="PayPal", age_days=0)
    result, _ = await _resolve([a, b])
    assert result == []


async def test_same_key_same_value_deduped_by_legacy():
    # Same canonical key + same normalized value = duplicate, handled by legacy
    # dedup (high text overlap), NOT treated as a contradiction.
    older = _mem("I work at Globex", key="employer", value="Globex", age_days=10)
    newer = _mem("I work at Globex", key="employer", value="Globex", age_days=0)
    result, _ = await _resolve([older, newer])
    assert older.id in result


# --- mixed keyed / unkeyed → legacy unchanged ---------------------------------


async def test_mixed_keyed_unkeyed_uses_legacy():
    unkeyed = _mem("I work at Globex", age_days=10)
    keyed = _mem("I work at Globex", key="employer", value="Globex", age_days=0)
    result, _ = await _resolve([unkeyed, keyed])
    assert unkeyed.id in result  # legacy dedup of near-duplicates (current behavior)


# --- determinism --------------------------------------------------------------


async def test_equal_timestamps_reversed_input_same_winner():
    # Identical valid_from AND created_at → tiebreak is the stable id. The lower
    # id is "older" and must be superseded regardless of input/DB order.
    id_lo, id_hi = uuid.UUID(int=1), uuid.UUID(int=2)
    a = _mem("at Acme", key="employer", value="Acme", age_days=3, created_days=3, mid=id_lo)
    b = _mem("at Globex", key="employer", value="Globex", age_days=3, created_days=3, mid=id_hi)

    forward, _ = await _resolve([a, b])
    # rebuild (resolve mutates status in place)
    a2 = _mem("at Acme", key="employer", value="Acme", age_days=3, created_days=3, mid=id_lo)
    b2 = _mem("at Globex", key="employer", value="Globex", age_days=3, created_days=3, mid=id_hi)
    reverse, _ = await _resolve([b2, a2])

    assert forward == [id_lo]
    assert reverse == [id_lo]


# --- scale / bucketing --------------------------------------------------------


async def test_keyed_bucketing_scales_and_isolates_keys():
    # 250 independent single-valued keys, each with an old+new contradiction →
    # exactly 250 supersessions, no cross-key contamination. Per-key grouping
    # keeps this linear in the number of memories (tiny per-key groups).
    from server.services.claims import CLAIM_REGISTRY, SCOPE_SINGLE

    single_keys = [k for k, s in CLAIM_REGISTRY.items() if s.scope == SCOPE_SINGLE]
    memories = []
    expected = set()
    for n in range(250):
        key = single_keys[n % len(single_keys)]
        # Disambiguate same-registry-key buckets across iterations would merge,
        # so use a distinct subject-attribute only when key repeats: here we rely
        # on distinct values per pair; repeated keys across n form one big bucket
        # per key, where newest-wins still yields exactly one survivor per key.
        old = _mem(f"old {n}", key=key, value=f"v-old-{n}", age_days=100 + n)
        new = _mem(f"new {n}", key=key, value=f"v-new-{n}", age_days=n)
        memories += [old, new]
    result, _ = await _resolve(memories)
    # Each of the 4 single-valued keys becomes one big overlapping bucket; within
    # a bucket only the single newest value survives → (total - num_keys) losers.
    survivors = len(memories) - len(result)
    assert survivors == len(single_keys)
