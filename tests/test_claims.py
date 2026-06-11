"""Claim envelope: parsing, registry authority, and fail-safe validation.

Every unsafe input must degrade to ``None`` (→ legacy behavior), never raise.
"""

from __future__ import annotations

from datetime import datetime, timezone

from server.services.claims import (
    CLAIM_SCHEMA_VERSION,
    SCOPE_MULTI,
    SCOPE_SINGLE,
    ResolvedClaim,
    build_claim_envelope,
    canonicalize_key,
    intervals_overlap,
    normalize_value,
    resolve_claim,
)


def _env(**kw) -> dict:
    base = {"schema_version": CLAIM_SCHEMA_VERSION, "key": "employment.current_employer", "value": "Globex"}
    base.update(kw)
    return {"claim": base}


# --- canonicalize_key -------------------------------------------------------


def test_registered_key_passes_through():
    assert canonicalize_key("employment.current_employer") == "employment.current_employer"


def test_approved_aliases_normalize():
    for alias in ("employer", "works_at", "company", "current_employer"):
        assert canonicalize_key(alias) == "employment.current_employer"


def test_unknown_key_is_not_authoritative():
    assert canonicalize_key("favorite_color") is None
    assert canonicalize_key("") is None
    assert canonicalize_key(None) is None


# --- normalize_value --------------------------------------------------------


def test_normalize_value_collapses_and_lowercases():
    assert normalize_value("  Globex   Corp ") == "globex corp"


def test_normalize_value_rejects_non_string_and_empty():
    assert normalize_value(123) is None
    assert normalize_value("   ") is None
    assert normalize_value(None) is None


# --- resolve_claim happy paths ----------------------------------------------


def test_resolve_single_valued_claim():
    rc = resolve_claim(_env()["claim"] and _env())
    assert isinstance(rc, ResolvedClaim)
    assert rc.canonical_key == "employment.current_employer"
    assert rc.value == "globex"
    assert rc.scope == SCOPE_SINGLE


def test_resolve_multi_valued_claim():
    rc = resolve_claim({"claim": {"schema_version": 1, "key": "payment_processors", "value": "Stripe"}})
    assert rc is not None
    assert rc.canonical_key == "payment.processors_used"
    assert rc.scope == SCOPE_MULTI


def test_resolve_uses_registry_scope_not_envelope_scope():
    # Envelope lies and claims 'single'; registry says payment.processors_used is multi.
    rc = resolve_claim({"claim": {"schema_version": 1, "key": "payment.processors_used",
                                  "value": "Stripe", "scope": "single"}})
    assert rc is not None and rc.scope == SCOPE_MULTI


def test_resolve_alias_to_canonical():
    rc = resolve_claim({"claim": {"schema_version": 1, "key": "company", "value": "Acme"}})
    assert rc is not None and rc.canonical_key == "employment.current_employer"


def test_resolve_parses_temporal():
    rc = resolve_claim(_env(valid_from="2025-01-01T00:00:00Z"))
    assert rc is not None and rc.valid_from == datetime(2025, 1, 1, tzinfo=timezone.utc)


# --- resolve_claim fail-safe cases (all -> None, never raise) ----------------


def test_missing_or_nondict_metadata():
    assert resolve_claim(None) is None
    assert resolve_claim({}) is None
    assert resolve_claim("not-a-dict") is None  # type: ignore[arg-type]
    assert resolve_claim({"claim": "not-a-dict"}) is None


def test_unsupported_schema_version_falls_back():
    assert resolve_claim(_env(schema_version=2)) is None
    assert resolve_claim(_env(schema_version=999)) is None


def test_unknown_key_falls_back():
    assert resolve_claim({"claim": {"schema_version": 1, "key": "shoe_size", "value": "10"}}) is None


def test_malformed_envelope_falls_back():
    # missing required 'value'
    assert resolve_claim({"claim": {"schema_version": 1, "key": "employer"}}) is None
    # value wrong type
    assert resolve_claim({"claim": {"schema_version": 1, "key": "employer", "value": 5}}) is None
    # bad temporal
    assert resolve_claim(_env(valid_from="not-a-date")) is None


def test_extra_envelope_fields_are_ignored_not_rejected():
    rc = resolve_claim(_env(future_field="whatever", another=123))
    assert rc is not None and rc.canonical_key == "employment.current_employer"


def test_arbitrary_user_metadata_alongside_claim_is_ignored():
    md = {"starter_pack_id": "x", "custom": True, **_env()}
    rc = resolve_claim(md)
    assert rc is not None  # other keys don't interfere


# --- build_claim_envelope ---------------------------------------------------


def test_build_envelope_for_registered_key():
    env = build_claim_envelope("employer", "Globex", source="heuristic")
    assert env is not None
    claim = env["claim"]
    assert claim["key"] == "employment.current_employer"
    assert claim["scope"] == SCOPE_SINGLE
    assert claim["schema_version"] == CLAIM_SCHEMA_VERSION
    assert claim["source"] == "heuristic"


def test_build_envelope_returns_none_for_unknown_key():
    assert build_claim_envelope("favorite_color", "red") is None


def test_build_envelope_roundtrips_through_resolve():
    env = build_claim_envelope("name", "Alice Chen")
    assert env is not None
    rc = resolve_claim(env)
    assert rc is not None and rc.canonical_key == "identity.name" and rc.value == "alice chen"


def test_build_envelope_serializes_temporal_isoformat():
    env = build_claim_envelope("employer", "Globex", valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert env is not None and env["claim"]["valid_from"].startswith("2025-01-01")


# --- intervals_overlap ------------------------------------------------------

_T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
_T2 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_two_open_ended_current_facts_overlap():
    assert intervals_overlap(_T0, None, _T1, None) is True


def test_touching_intervals_do_not_overlap():
    # Acme [_T0,_T1)  Globex [_T1, open) — past vs current, coexist as history.
    assert intervals_overlap(_T0, _T1, _T1, None) is False


def test_disjoint_intervals_do_not_overlap():
    assert intervals_overlap(_T0, _T1, _T2, None) is False


def test_overlapping_intervals_overlap():
    assert intervals_overlap(_T0, _T2, _T1, None) is True


def test_missing_from_is_open_start():
    assert intervals_overlap(None, _T1, _T0, None) is True
