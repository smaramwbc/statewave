"""Characterization of conflict-resolution behavior that MUST survive the
hybrid claim-keyed resolver unchanged.

These pin the legacy lexical (Jaccard) path for every case the resolver is NOT
allowed to alter: unkeyed memories, arbitrary non-claim metadata, malformed /
unknown-key / unsupported claims, and multi-valued or mixed pairings. They pass
on the pre-resolver code (which ignores metadata entirely) and must keep passing
after — proving the new path is strictly additive.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from server.db.tables import MemoryRow
from server.services.conflicts import resolve_conflicts

_NOW = datetime.now(timezone.utc)


def _mem(content, *, kind="profile_fact", metadata=None, age_days=0, status="active"):
    return MemoryRow(
        id=uuid.uuid4(),
        subject_id="user-1",
        kind=kind,
        content=content,
        summary=content[:200],
        confidence=0.8,
        valid_from=_NOW - timedelta(days=age_days),
        created_at=_NOW - timedelta(days=age_days),
        source_episode_ids=[uuid.uuid4()],
        metadata_=metadata or {},
        status=status,
    )


async def _resolve(memories):
    with patch("server.services.conflicts.repo") as mock_repo:
        mock_repo.list_active_memories_by_subject = AsyncMock(return_value=memories)
        mock_repo.mark_memories_superseded = AsyncMock()
        result = await resolve_conflicts(AsyncMock(), "user-1")
    return result


async def test_arbitrary_non_claim_metadata_does_not_change_resolution():
    older = _mem("my name is Alice", metadata={"foo": "bar", "n": 1}, age_days=5)
    newer = _mem("my name is Alice Chen", metadata={"foo": "baz"}, age_days=0)
    result = await _resolve([older, newer])
    assert older.id in result and newer.id not in result


async def test_malformed_claim_metadata_falls_back_to_legacy():
    # A garbage claim must behave exactly like no claim → legacy Jaccard wins.
    older = _mem("my name is Alice", metadata={"claim": {"garbage": True}}, age_days=5)
    newer = _mem("my name is Alice Chen", metadata={"claim": 42}, age_days=0)
    result = await _resolve([older, newer])
    assert older.id in result  # superseded by legacy overlap, not by claim logic


async def test_unknown_claim_key_falls_back_to_legacy_no_supersession():
    # Unknown key is non-authoritative; distinct facts must coexist (legacy).
    a = _mem("I use Stripe", metadata={"claim": {"schema_version": 1, "key": "shoe_size", "value": "10"}}, age_days=5)
    b = _mem("I use PayPal", metadata={"claim": {"schema_version": 1, "key": "shoe_size", "value": "11"}}, age_days=0)
    result = await _resolve([a, b])
    assert result == []  # below Jaccard threshold, no claim authority → coexist


async def test_distinct_facts_still_coexist():
    a = _mem("my name is Alice", age_days=5)
    b = _mem("I work at Globex Corporation", age_days=0)
    assert await _resolve([a, b]) == []


async def test_near_duplicate_still_deduped():
    older = _mem("I use Stripe", age_days=5)
    newer = _mem("I use Stripe.", age_days=0)
    result = await _resolve([older, newer])
    assert older.id in result


async def test_stripe_paypal_coexist_even_with_multi_valued_claim():
    # Multi-valued registry key → coexistence preserved (legacy low overlap).
    a = _mem("I use Stripe", metadata={"claim": {"schema_version": 1, "key": "payment.processors_used", "value": "Stripe"}}, age_days=5)
    b = _mem("I use PayPal", metadata={"claim": {"schema_version": 1, "key": "payment.processors_used", "value": "PayPal"}}, age_days=0)
    assert await _resolve([a, b]) == []


async def test_per_kind_grouping_preserved():
    # Conflicts only within a kind; cross-kind never merges.
    fact_old = _mem("my name is Alice", kind="profile_fact", age_days=5)
    fact_new = _mem("my name is Alice Chen", kind="profile_fact", age_days=0)
    summ = _mem("my name is Alice", kind="episode_summary", age_days=3)
    result = await _resolve([fact_old, fact_new, summ])
    assert fact_old.id in result
    assert summ.id not in result  # different kind, untouched


async def test_single_memory_no_conflict():
    assert await _resolve([_mem("solo")]) == []
