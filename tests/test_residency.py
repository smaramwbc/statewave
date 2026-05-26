"""Unit tests for residency check + region-pin validation (v0.9 #161).

Pure tests — exercise the decision matrix directly without booting
the app. End-to-end enforcement (middleware + admin patch) is in
tests/integration/test_residency.py.
"""

from __future__ import annotations

from server.services.residency import (
    ResidencyMismatch,
    check_residency,
    validate_region_pin,
)


# ---------------------------------------------------------------------------
# check_residency — the decision matrix
# ---------------------------------------------------------------------------


def test_single_region_mode_always_allows():
    """server_region None = single-region deployment, residency
    disabled entirely. The check must be a no-op regardless of what
    the tenant config says."""
    assert check_residency(tenant_config={"region": "us"}, server_region=None) is None
    assert check_residency(tenant_config={}, server_region=None) is None
    assert check_residency(tenant_config=None, server_region=None) is None


def test_unpinned_tenant_allowed_anywhere():
    """A tenant without a pinned region is global by default."""
    assert check_residency(tenant_config={}, server_region="eu") is None
    assert check_residency(tenant_config=None, server_region="eu") is None
    # Empty string is treated as "no pin" (same as missing key).
    assert check_residency(tenant_config={"region": ""}, server_region="eu") is None


def test_matching_region_allowed():
    assert check_residency(tenant_config={"region": "eu"}, server_region="eu") is None


def test_mismatched_region_refused():
    result = check_residency(tenant_config={"region": "us"}, server_region="eu")
    assert isinstance(result, ResidencyMismatch)
    assert result.tenant_region == "us"
    assert result.server_region == "eu"


def test_case_sensitive_region_compare():
    """`EU` and `eu` are distinct regions — we fail loudly rather
    than risk a typo silently allowing cross-region traffic."""
    result = check_residency(tenant_config={"region": "EU"}, server_region="eu")
    assert isinstance(result, ResidencyMismatch)


def test_non_string_region_refused_defensively():
    """A malformed JSONB value (e.g. an int written via direct SQL)
    must NOT silently allow the request."""
    result = check_residency(tenant_config={"region": 42}, server_region="eu")
    assert isinstance(result, ResidencyMismatch)


def test_tenant_config_missing_region_key_is_allowed():
    """A pre-v0.9 tenant config that pre-dates the region key passes
    through as if unpinned — backwards compatibility."""
    legacy = {"receipts": "always", "policy_mode": "log_only"}
    assert check_residency(tenant_config=legacy, server_region="eu") is None


# ---------------------------------------------------------------------------
# validate_region_pin — operator-facing safety check
# ---------------------------------------------------------------------------


def test_validate_pin_to_matching_region_allowed():
    assert validate_region_pin(proposed_region="eu", server_region="eu") is None


def test_validate_pin_to_different_region_refused():
    msg = validate_region_pin(proposed_region="us", server_region="eu")
    assert msg is not None
    assert "us" in msg
    assert "eu" in msg


def test_validate_pin_allowed_in_single_region_mode():
    """Dev / single-region mode: server_region is None, so the safety
    check is meaningless. Operator authoring multi-region configs
    from a dev box must be allowed to pin to any region."""
    assert validate_region_pin(proposed_region="us", server_region=None) is None
    assert validate_region_pin(proposed_region="eu", server_region=None) is None
