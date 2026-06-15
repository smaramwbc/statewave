"""Unit tests for the settings catalogue + validation + redaction.

No DB, no FastAPI client — just pure-Python checks. The matching DB-backed
tests live in `test_admin_settings.py`.
"""

from __future__ import annotations

import pytest

from server.core.config import settings as env_settings
from server.core.dynamic_settings import (
    SettingValidationError,
    _validate_value,
)
from server.core.settings_catalogue import (
    CATALOGUE,
    get_spec,
    is_secret,
    redact,
)


def test_every_catalogue_key_matches_a_pydantic_attribute():
    """The catalogue is a contract — every key MUST resolve to a real
    attribute on the Settings class, otherwise reads through
    ``get_setting`` would fall through to the silent ``None`` branch
    instead of returning the env default."""
    missing = [k for k in CATALOGUE if not hasattr(env_settings, k)]
    assert missing == [], f"catalogue keys missing from Settings: {missing}"


def test_secrets_have_string_or_null_kind():
    """Redaction only makes sense for string-shaped secrets. If a future
    setting marks itself as a secret but uses a non-string kind, that's a
    bug — there's no sensible way to redact a number to a `•••` preview."""
    for spec in CATALOGUE.values():
        if spec.is_secret:
            assert spec.kind in (
                "string",
                "string_or_null",
                "json",
            ), f"secret {spec.key} has un-redactable kind {spec.kind}"


def test_non_editable_settings_carry_a_reason():
    """If we forbid editing a setting, the description must explain why —
    otherwise the admin UI will show a locked field with no rationale and
    operators will file confused bug reports."""
    for spec in CATALOGUE.values():
        if not spec.editable:
            assert spec.description, f"non-editable {spec.key} needs a description"


def test_tenant_overridable_keys_are_in_expected_scope():
    """User confirmed (chat: 'yes and yes') the tenant-override scope is
    deliberately narrow: LLM + webhook + rate-limit. Lock that down here
    so a future PR can't quietly widen it without an explicit decision."""
    overridable = {k for k, spec in CATALOGUE.items() if spec.tenant_overridable}
    expected_categories = {"llm", "embeddings", "webhooks", "rate_limits"}
    for key in overridable:
        spec = CATALOGUE[key]
        assert (
            spec.category in expected_categories
        ), f"{key} is tenant-overridable but in unexpected category {spec.category}"


# ─── redact() ────────────────────────────────────────────────────────────


def test_redact_short_value():
    assert redact("ab", kind="string") == "•••"


def test_redact_string_keeps_last_three():
    assert redact("sk-supersecret123", kind="string") == "•••123"


def test_redact_empty_or_none_is_untouched():
    """The UI uses empty/null to render 'not set' — redacting it would lie."""
    assert redact(None, kind="string") is None
    assert redact("", kind="string") == ""


def test_is_secret_defaults_to_true_on_unknown_key():
    """Defence-in-depth: if a key isn't catalogued, treat it as secret. A
    missing catalogue entry should never accidentally leak the raw value."""
    assert is_secret("brand-new-thing-not-yet-catalogued") is True


# ─── _validate_value ─────────────────────────────────────────────────────


def test_validate_rejects_bool_as_int():
    """Python's `True` is technically int(1). Without an explicit guard a
    JSON `true` would silently land as `1` for an int setting — a
    confusing footgun worth refusing."""
    spec = get_spec("rate_limit_rpm")
    assert spec is not None
    with pytest.raises(SettingValidationError):
        _validate_value(spec, True)


def test_validate_accepts_int():
    spec = get_spec("rate_limit_rpm")
    assert spec is not None
    assert _validate_value(spec, 60) == 60


def test_validate_string_or_null_accepts_none():
    spec = get_spec("api_key")
    assert spec is not None
    assert _validate_value(spec, None) is None


def test_validate_string_rejects_null():
    """Non-nullable string settings should reject explicit nulls."""
    spec = get_spec("litellm_model")
    assert spec is not None
    with pytest.raises(SettingValidationError):
        _validate_value(spec, None)


def test_validate_float_coerces_int_to_float():
    spec = get_spec("litellm_timeout_seconds")
    assert spec is not None
    result = _validate_value(spec, 30)
    assert isinstance(result, float) and result == 30.0


def test_validate_json_accepts_dict():
    spec = get_spec("kind_ttl_days")
    assert spec is not None
    assert _validate_value(spec, {"episode_summary": 30}) == {"episode_summary": 30}


# ─── allowed_values (enum) ───────────────────────────────────────────────


def test_allowed_values_rejects_off_list():
    """The user's exact example: typing 'lllm' when 'llm' is expected
    must fail loudly, not silently land in the DB to confuse the
    compiler later."""
    spec = get_spec("compiler_type")
    assert spec is not None
    with pytest.raises(SettingValidationError) as exc:
        _validate_value(spec, "lllm")
    assert "lllm" in str(exc.value)
    # The error must SUGGEST the fix — a bare 'not allowed' message
    # leaves the operator guessing whether the right value is 'llm' or
    # 'LLM' or 'large_language_model'.
    assert "did you mean 'llm'" in str(exc.value).lower()


def test_allowed_values_accepts_valid():
    spec = get_spec("compiler_type")
    assert spec is not None
    assert _validate_value(spec, "llm") == "llm"
    assert _validate_value(spec, "heuristic") == "heuristic"


def test_allowed_values_suggests_for_each_known_enum():
    """Sanity-check the Levenshtein-2 cutoff for every enum we have."""
    cases = [
        ("compiler_type", "huristic", "heuristic"),
        ("embedding_provider", "litellmm", "litellm"),
        ("rate_limit_strategy", "memry", "memory"),
    ]
    for key, bad, expected_suggestion in cases:
        spec = get_spec(key)
        assert spec is not None
        with pytest.raises(SettingValidationError) as exc:
            _validate_value(spec, bad)
        assert expected_suggestion in str(exc.value).lower()


# ─── numeric bounds ──────────────────────────────────────────────────────


def test_int_below_min_rejected():
    spec = get_spec("compile_batch_size")
    assert spec is not None and spec.min_value == 1
    with pytest.raises(SettingValidationError):
        _validate_value(spec, 0)


def test_int_above_max_rejected():
    spec = get_spec("litellm_max_retries")
    assert spec is not None and spec.max_value == 10
    with pytest.raises(SettingValidationError):
        _validate_value(spec, 99)


def test_float_temperature_bounds():
    spec = get_spec("litellm_temperature")
    assert spec is not None
    _validate_value(spec, 0.0)
    _validate_value(spec, 2.0)
    with pytest.raises(SettingValidationError):
        _validate_value(spec, -0.1)
    with pytest.raises(SettingValidationError):
        _validate_value(spec, 2.5)


# ─── URL format ──────────────────────────────────────────────────────────


def test_url_must_be_http_or_https():
    spec = get_spec("webhook_url")
    assert spec is not None and spec.format == "url"
    with pytest.raises(SettingValidationError):
        _validate_value(spec, "example.com/hook")
    _validate_value(spec, "https://example.com/hook")
    # Empty string is allowed — that's the documented "disable" sentinel.
    _validate_value(spec, "")


# ─── import count caps ───────────────────────────────────────────────────


def test_import_count_limits_reject_non_positive():
    """Setting any per-import count cap to 0 or negative blocks every import
    because memory_packs enforces ``len(items) > limit``.  These three specs
    must mirror the ``min_value=1`` already present on ``memory_import_max_bytes``
    so the admin API rejects a misconfigured value before it silently kills
    all imports."""
    count_cap_keys = (
        "memory_import_max_episodes",
        "memory_import_max_memories",
        "memory_import_max_subjects",
    )
    for key in count_cap_keys:
        spec = get_spec(key)
        assert spec is not None, f"missing catalogue entry for {key}"
        with pytest.raises(SettingValidationError, match=key):
            _validate_value(spec, 0)
        with pytest.raises(SettingValidationError, match=key):
            _validate_value(spec, -1)


def test_import_count_limits_accept_positive():
    count_cap_keys = (
        "memory_import_max_episodes",
        "memory_import_max_memories",
        "memory_import_max_subjects",
    )
    for key in count_cap_keys:
        spec = get_spec(key)
        assert spec is not None
        assert _validate_value(spec, 1) == 1
        assert _validate_value(spec, 50_000) == 50_000
