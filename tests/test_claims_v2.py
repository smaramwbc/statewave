"""v2 external-entity, qualifier-keyed claims: identity, canonicalization, and
generic resolver bucketing. Distinct (entity, qualifier) identities must never
share a contradiction bucket; same identity + different value supersedes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from server.db.tables import MemoryRow
from server.services.claims import build_v2_envelope, resolve_claim
from server.services.conflicts import resolve_conflicts

_NOW = datetime.now(timezone.utc)
_BASE_Q = {
    "payment_method": "card",
    "rate_type": "standard",
    "currency": "USD",
    "charge_unit": "transaction",
}
_V350 = {"percentage_basis_points": 350, "fixed_minor_units": 35}
_V290 = {"percentage_basis_points": 290, "fixed_minor_units": 30}


def _env(*, entity="organization:stripe", quals=None, value=None):
    return {
        "claim": {
            "schema_version": 2,
            "key": "pricing.processing_rate",
            "entity_key": entity,
            "qualifiers": quals if quals is not None else dict(_BASE_Q),
            "value": value if value is not None else dict(_V350),
        }
    }


def _rc(**kw):
    return resolve_claim(_env(**kw))


# --- identity / bucketing -----------------------------------------------------


def test_v2_resolves_with_bucket_and_scope():
    rc = _rc()
    assert rc is not None and rc.bucket[0] == "v2" and rc.scope == "single"
    assert rc.canonical_key == "pricing.processing_rate"


def test_same_identity_different_value_same_bucket():
    a, b = _rc(value=_V350), _rc(value=_V290)
    assert a.bucket == b.bucket and a.value != b.value


def test_distinct_identities_have_distinct_buckets():
    base = _rc().bucket
    assert _rc(entity="organization:square").bucket != base  # Stripe vs Square
    assert _rc(quals={**_BASE_Q, "payment_method": "ach"}).bucket != base  # card vs ACH
    assert _rc(quals={**_BASE_Q, "rate_type": "international"}).bucket != base  # standard vs intl
    assert _rc(quals={**_BASE_Q, "currency": "EUR"}).bucket != base  # USD vs EUR
    assert _rc(quals={**_BASE_Q, "charge_unit": "monthly"}).bucket != base  # txn vs monthly


def test_optional_qualifier_is_not_a_wildcard():
    with_region = _rc(quals={**_BASE_Q, "region": "us"})
    without = _rc()
    assert with_region.bucket != without.bucket  # presence of region changes identity


def test_qualifier_object_ordering_irrelevant():
    q1 = {
        "payment_method": "card",
        "rate_type": "standard",
        "currency": "USD",
        "charge_unit": "transaction",
    }
    q2 = {
        "charge_unit": "transaction",
        "currency": "USD",
        "rate_type": "standard",
        "payment_method": "card",
    }
    assert _rc(quals=q1).bucket == _rc(quals=q2).bucket


def test_value_object_ordering_irrelevant_for_equality():
    a = _rc(value={"percentage_basis_points": 350, "fixed_minor_units": 35})
    b = _rc(value={"fixed_minor_units": 35, "percentage_basis_points": 350})
    assert a.value == b.value


# --- safety: malformed / missing / unsupported -> None ------------------------


def test_missing_required_qualifier_non_authoritative():
    q = dict(_BASE_Q)
    q.pop("currency")
    assert _rc(quals=q) is None


def test_unapproved_qualifier_rejected():
    assert _rc(quals={**_BASE_Q, "bogus": "x"}) is None


def test_malformed_entity_non_authoritative():
    assert _rc(entity="") is None
    assert _rc(entity=123) is None


def test_malformed_value_non_authoritative():
    assert _rc(value=float("nan")) is None
    assert _rc(value=object()) is None  # arbitrary object rejected


def test_v2_for_v1_key_unsupported():
    # identity.name is a v1 (string) key; a v2 envelope for it is non-authoritative.
    assert (
        resolve_claim(
            {
                "claim": {
                    "schema_version": 2,
                    "key": "identity.name",
                    "entity_key": "x",
                    "qualifiers": {},
                    "value": {},
                }
            }
        )
        is None
    )


def test_unknown_key_v2_non_authoritative():
    assert (
        resolve_claim(
            {
                "claim": {
                    "schema_version": 2,
                    "key": "made.up",
                    "entity_key": "x",
                    "qualifiers": {},
                    "value": {},
                }
            }
        )
        is None
    )


def test_build_v2_envelope_roundtrips():
    env = build_v2_envelope(_env()["claim"])
    assert env is not None
    rc = resolve_claim(env)
    assert rc is not None and rc.canonical_key == "pricing.processing_rate"
    assert env["claim"]["entity_key"] == "organization:stripe"  # canonicalized


# --- generic resolver behavior (mocked repo) ----------------------------------


def _mem(content, env, *, age):
    return MemoryRow(
        id=uuid.uuid4(),
        subject_id="s",
        kind="profile_fact",
        content=content,
        summary=content,
        confidence=0.9,
        valid_from=_NOW - timedelta(days=age),
        created_at=_NOW - timedelta(days=age),
        source_episode_ids=[uuid.uuid4()],
        metadata_=env,
        status="active",
    )


async def _resolve(mems):
    with patch("server.services.conflicts.repo") as r:
        r.list_active_memories_by_subject = AsyncMock(return_value=mems)
        r.mark_memories_superseded = AsyncMock()
        return await resolve_conflicts(AsyncMock(), "s")


async def test_stripe_rate_supersedes_same_identity():
    old = _mem("stripe 3.5% + 35c", _env(value=_V350), age=10)
    new = _mem("stripe 2.9% + 30c", _env(value=_V290), age=0)
    res = await _resolve([old, new])
    assert old.id in res and new.id not in res


async def test_negative_controls_coexist():
    base = _mem("stripe std card usd txn 3.5", _env(value=_V350), age=10)
    others = [
        _mem("square card", _env(entity="organization:square", value=_V290), age=0),
        _mem("stripe ach", _env(quals={**_BASE_Q, "payment_method": "ach"}, value=_V290), age=0),
        _mem(
            "stripe intl", _env(quals={**_BASE_Q, "rate_type": "international"}, value=_V290), age=0
        ),
        _mem("stripe eur", _env(quals={**_BASE_Q, "currency": "EUR"}, value=_V290), age=0),
        _mem(
            "stripe monthly", _env(quals={**_BASE_Q, "charge_unit": "monthly"}, value=_V290), age=0
        ),
        _mem(
            "stripe enterprise region", _env(quals={**_BASE_Q, "region": "eu"}, value=_V290), age=0
        ),
    ]
    res = await _resolve([base, *others])
    assert res == []  # every distinct identity coexists; nothing superseded


async def test_same_identity_same_value_not_a_contradiction():
    a = _mem("stripe a", _env(value=_V350), age=10)
    b = _mem("stripe b", _env(value=_V350), age=0)  # identical value, distinct text
    assert await _resolve([a, b]) == []  # same value -> not a contradiction (coexist)
