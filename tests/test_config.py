"""Tests for Settings enum-like field validators (startup config validation)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from server.core.config import Settings


def test_env_example_documents_every_setting():
    """Keep the operator-facing environment inventory in sync with Settings."""
    env_example = (Path(__file__).parents[1] / ".env.example").read_text()
    documented = set(re.findall(r"^#?\s*(STATEWAVE_[A-Z0-9_]+)\s*=", env_example, re.MULTILINE))
    expected = {f"STATEWAVE_{name.upper()}" for name in Settings.model_fields}

    assert expected <= documented, "Settings missing from .env.example: " + ", ".join(
        sorted(expected - documented)
    )


# ── rate_limit_strategy ─────────────────────────────────────────────────────


def test_rate_limit_strategy_accepts_valid_values():
    for value in ("memory", "distributed"):
        assert Settings(_env_file=None, rate_limit_strategy=value).rate_limit_strategy == value


def test_rate_limit_strategy_default_is_valid():
    assert Settings(_env_file=None).rate_limit_strategy == "memory"


def test_rate_limit_strategy_rejects_typo():
    # A typo here silently degrades to per-process in-memory limiting
    # (effective global limit becomes rpm*workers) and skips distributed
    # cleanup, so it must fail at startup rather than be accepted.
    with pytest.raises(ValueError, match="STATEWAVE_RATE_LIMIT_STRATEGY must be one of"):
        Settings(_env_file=None, rate_limit_strategy="distrubuted")


# ── compiler_type ───────────────────────────────────────────────────────────


def test_compiler_type_accepts_valid_values():
    for value in ("heuristic", "llm"):
        assert Settings(_env_file=None, compiler_type=value).compiler_type == value


def test_compiler_type_default_is_valid():
    assert Settings(_env_file=None).compiler_type == "heuristic"


def test_compiler_type_rejects_typo():
    with pytest.raises(ValueError, match="STATEWAVE_COMPILER_TYPE must be one of"):
        Settings(_env_file=None, compiler_type="heruistic")


# ── embedding_provider ──────────────────────────────────────────────────────


def test_embedding_provider_accepts_valid_values():
    for value in ("stub", "litellm", "none"):
        assert Settings(_env_file=None, embedding_provider=value).embedding_provider == value


def test_embedding_provider_default_is_valid():
    assert Settings(_env_file=None).embedding_provider == "stub"


def test_embedding_provider_rejects_typo():
    with pytest.raises(ValueError, match="STATEWAVE_EMBEDDING_PROVIDER must be one of"):
        Settings(_env_file=None, embedding_provider="littelm")
