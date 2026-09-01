"""Consumers can register their own single-valued claim keys (#376).

Before this, `CLAIM_REGISTRY` was a hardcoded dict of nine keys and
`canonicalize_key` returned None for anything else — so a consumer's own
vocabulary was not merely non-authoritative, the claim was dropped entirely
and its memories grew without bound.

Cardinality is declared by an operator through
`PATCH /admin/tenants/{id}/config`, never by the episode payload: a producer
that could label its own key `single` could collapse rows that must coexist,
which is the invariant #236 established.
"""

from __future__ import annotations

import pytest

from server.services.claims import (
    SCOPE_MULTI,
    SCOPE_SINGLE,
    ClaimDefinition,
    build_claim_envelope,
    canonicalize_key,
    resolve_claim,
)

_GUIDE = {"guide.walkthrough": ClaimDefinition("guide.walkthrough", SCOPE_SINGLE)}


def test_unregistered_key_is_still_dropped_without_registration():
    """The default is unchanged: opt-in only."""
    assert canonicalize_key("guide.walkthrough") is None
    assert build_claim_envelope("guide.walkthrough", "done") is None


def test_registered_key_resolves_and_carries_its_declared_cardinality():
    assert canonicalize_key("guide.walkthrough", _GUIDE) == "guide.walkthrough"
    env = build_claim_envelope("guide.walkthrough", "feature-x@build-y", extra_keys=_GUIDE)
    assert env is not None
    claim = env["claim"]
    assert claim["key"] == "guide.walkthrough"
    assert claim["scope"] == SCOPE_SINGLE  # from the registration, not the payload

    resolved = resolve_claim(env, _GUIDE)
    assert resolved is not None
    assert resolved.canonical_key == "guide.walkthrough"
    assert resolved.scope == SCOPE_SINGLE


def test_a_tenant_cannot_shadow_a_built_in_key():
    """Shipping a new built-in must never be silently overridden by a tenant
    that happened to pick the same name — the built-in always wins."""
    shadow = {"identity.name": ClaimDefinition("identity.name", SCOPE_MULTI)}
    resolved = resolve_claim(
        {"claim": {"schema_version": 1, "key": "identity.name", "value": "ada"}}, shadow
    )
    assert resolved is not None
    assert resolved.scope == SCOPE_SINGLE  # the built-in's cardinality, not the tenant's


def test_multi_registration_does_not_supersede():
    """A key registered `multi` must not enter the single-valued path."""
    keys = {"guide.tag": ClaimDefinition("guide.tag", SCOPE_MULTI)}
    resolved = resolve_claim(
        {"claim": {"schema_version": 1, "key": "guide.tag", "value": "beta"}}, keys
    )
    assert resolved is not None
    assert resolved.scope == SCOPE_MULTI


def test_one_tenants_vocabulary_cannot_leak_into_another():
    """Resolution is a pure function of its inputs — no global registry
    mutation — so a key registered for tenant A is unknown to tenant B."""
    env = build_claim_envelope("guide.walkthrough", "done", extra_keys=_GUIDE)
    assert resolve_claim(env, _GUIDE) is not None
    assert resolve_claim(env, {}) is None
    assert resolve_claim(env, None) is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"claim_keys": {"guide.walkthrough": "single"}}, {"guide.walkthrough": SCOPE_SINGLE}),
        ({"claim_keys": {"guide.x": "nonsense"}}, {}),          # bad cardinality dropped
        ({"claim_keys": {"identity.name": "multi"}}, {}),        # built-in never shadowed
        ({"claim_keys": "not-a-mapping"}, {}),
        ({}, {}),
    ],
)
@pytest.mark.asyncio
async def test_loader_revalidates_whatever_is_in_jsonb(raw, expected, monkeypatch):
    """The row could predate a schema change or have been written by direct
    SQL, so one bad entry must never break compilation for the whole tenant."""
    from server.services import claims as claims_mod

    class _Row:
        config = raw

    async def fake_get(_session, _tenant):
        return _Row()

    monkeypatch.setattr("server.db.repositories.get_tenant_config", fake_get)
    out = await claims_mod.load_tenant_claim_keys(object(), "tenant-a")
    assert {k: v.scope for k, v in out.items()} == expected


@pytest.mark.asyncio
async def test_loader_is_inert_without_a_tenant():
    """Untenanted servers do no config read at all."""
    from server.services import claims as claims_mod

    assert await claims_mod.load_tenant_claim_keys(None, None) == {}
