"""Unit tests for memory TTL — config parsing, valid_to computation, compiler stamping.

DB-backed cleanup + retrieval-filter behaviour live in
`tests/integration/test_memory_ttl.py`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from server.core.config import Settings
from server.services.compilers.heuristic import HeuristicCompiler
from server.services.memory_ttl import compute_valid_to


# ---------------------------------------------------------------------------
# compute_valid_to
# ---------------------------------------------------------------------------


def test_compute_valid_to_returns_none_when_kind_not_in_dict():
    assert compute_valid_to("episode_summary", datetime.now(timezone.utc), {}) is None


def test_compute_valid_to_returns_none_when_dict_value_is_zero():
    # Defence in depth — the config validator rejects 0/negative, but the
    # compute function should also be safe if someone bypasses the validator.
    assert (
        compute_valid_to("episode_summary", datetime.now(timezone.utc), {"episode_summary": 0})
        is None
    )


def test_compute_valid_to_adds_days_when_kind_configured():
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = compute_valid_to("episode_summary", base, {"episode_summary": 30})
    assert result == base + timedelta(days=30)


def test_compute_valid_to_upgrades_naive_datetime_to_utc():
    naive = datetime(2026, 1, 1, 12, 0, 0)  # no tzinfo
    result = compute_valid_to("episode_summary", naive, {"episode_summary": 7})
    assert result is not None
    assert result.tzinfo is timezone.utc


def test_compute_valid_to_only_affects_listed_kind():
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ttl = {"episode_summary": 7}
    assert compute_valid_to("episode_summary", base, ttl) == base + timedelta(days=7)
    assert compute_valid_to("profile_fact", base, ttl) is None


# ---------------------------------------------------------------------------
# Settings.kind_ttl_days validator
# ---------------------------------------------------------------------------


def test_settings_kind_ttl_days_defaults_to_empty_dict():
    s = Settings(_env_file=None)
    assert s.kind_ttl_days == {}


def test_settings_kind_ttl_days_accepts_dict_value():
    s = Settings(_env_file=None, kind_ttl_days={"episode_summary": 30})
    assert s.kind_ttl_days == {"episode_summary": 30}


def test_settings_kind_ttl_days_parses_json_string():
    # The env-var path delivers strings; the validator must JSON-decode.
    s = Settings(_env_file=None, kind_ttl_days='{"episode_summary": 30, "profile_fact": 365}')
    assert s.kind_ttl_days == {"episode_summary": 30, "profile_fact": 365}


def test_settings_kind_ttl_days_empty_string_is_empty_dict():
    s = Settings(_env_file=None, kind_ttl_days="")
    assert s.kind_ttl_days == {}


def test_settings_kind_ttl_days_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        Settings(_env_file=None, kind_ttl_days="{not json")


def test_settings_kind_ttl_days_rejects_non_object_json():
    with pytest.raises(ValueError, match="must decode to a JSON object"):
        Settings(_env_file=None, kind_ttl_days="[1, 2, 3]")


def test_settings_kind_ttl_days_rejects_zero_or_negative_days():
    # Disabling expiry for a kind = leave it out of the dict, not 0.
    with pytest.raises(ValueError, match="must be > 0"):
        Settings(_env_file=None, kind_ttl_days={"episode_summary": 0})
    with pytest.raises(ValueError, match="must be > 0"):
        Settings(_env_file=None, kind_ttl_days={"episode_summary": -1})


def test_settings_kind_ttl_days_rejects_non_integer_days():
    with pytest.raises(ValueError, match="must be an integer"):
        Settings(_env_file=None, kind_ttl_days={"episode_summary": "30"})
    with pytest.raises(ValueError, match="must be an integer"):
        Settings(_env_file=None, kind_ttl_days={"episode_summary": 30.5})


# ---------------------------------------------------------------------------
# Heuristic compiler — TTL stamping at memory creation
# ---------------------------------------------------------------------------


def _make_episode(payload: dict, subject_id: str = "user-ttl"):
    from server.db.tables import EpisodeRow

    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id=subject_id,
        source="test",
        type="conversation",
        payload=payload,
        metadata_={},
        provenance={},
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_heuristic_compiler_stamps_valid_to_when_kind_has_ttl(monkeypatch):
    # The compiler reads settings.kind_ttl_days at invocation time. Monkey-
    # patch the live singleton — Settings() doesn't have a setter for
    # kind_ttl_days at runtime, but the attribute is mutable.
    from server.core.config import settings

    monkeypatch.setattr(settings, "kind_ttl_days", {"episode_summary": 30})

    ep = _make_episode({"text": "Hello, this is a test message."})
    memories = HeuristicCompiler().compile([ep])

    summaries = [m for m in memories if m.kind == "episode_summary"]
    assert summaries, "expected at least one episode_summary memory"
    expected = ep.created_at + timedelta(days=30)
    for m in summaries:
        assert m.valid_to == expected


def test_heuristic_compiler_leaves_valid_to_none_when_kind_not_in_ttl(monkeypatch):
    from server.core.config import settings

    # Configure TTL only for episode_summary; profile_fact must stay None.
    monkeypatch.setattr(settings, "kind_ttl_days", {"episode_summary": 30})

    ep = _make_episode(
        {"messages": [{"role": "user", "content": "My name is Alice and I work at Acme."}]}
    )
    memories = HeuristicCompiler().compile([ep])
    facts = [m for m in memories if m.kind == "profile_fact"]
    assert facts, "expected at least one profile_fact memory"
    for m in facts:
        assert m.valid_to is None


def test_heuristic_compiler_no_ttl_config_leaves_valid_to_none(monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "kind_ttl_days", {})

    ep = _make_episode({"text": "Plain content."})
    memories = HeuristicCompiler().compile([ep])
    assert memories, "expected at least one memory"
    for m in memories:
        assert m.valid_to is None
