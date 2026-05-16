"""Tests for LLM configuration warnings and helpers."""

from __future__ import annotations

import logging

from server.core.config import Settings
import server.services.llm as llm_module
from server.services.llm import (
    litellm_api_key_configured,
    llm_requires_api_key,
    warn_if_llm_compiler_missing_api_key,
)


def test_litellm_api_key_configured_rejects_empty_string(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "settings",
        Settings(_env_file=None, litellm_api_key=""),
    )
    assert litellm_api_key_configured() is False


def test_litellm_api_key_configured_accepts_non_empty(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "settings",
        Settings(_env_file=None, litellm_api_key="sk-test"),
    )
    assert litellm_api_key_configured() is True


def test_warn_if_llm_compiler_missing_api_key_logs_when_compiler_llm(monkeypatch, caplog):
    monkeypatch.setattr(
        llm_module,
        "settings",
        Settings(_env_file=None, compiler_type="llm", litellm_api_key=None),
    )
    with caplog.at_level(logging.WARNING):
        warn_if_llm_compiler_missing_api_key()

    warning_messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("llm_compiler_missing_api_key" in m for m in warning_messages), (
        f"expected llm_compiler_missing_api_key warning, got: {warning_messages}"
    )


def test_warn_if_llm_compiler_missing_api_key_silent_when_heuristic(monkeypatch, caplog):
    monkeypatch.setattr(
        llm_module,
        "settings",
        Settings(_env_file=None, compiler_type="heuristic", litellm_api_key=None),
    )
    with caplog.at_level(logging.WARNING):
        warn_if_llm_compiler_missing_api_key()

    assert not any("llm_compiler_missing_api_key" in r.getMessage() for r in caplog.records)


def test_warn_if_llm_compiler_missing_api_key_silent_when_key_set(monkeypatch, caplog):
    monkeypatch.setattr(
        llm_module,
        "settings",
        Settings(_env_file=None, compiler_type="llm", litellm_api_key="sk-test"),
    )
    with caplog.at_level(logging.WARNING):
        warn_if_llm_compiler_missing_api_key()

    assert not any("llm_compiler_missing_api_key" in r.getMessage() for r in caplog.records)


# ── Ollama exemption (#122 follow-up; credit @LPHuynh) ──────────────────────


def test_llm_requires_api_key_false_for_ollama(monkeypatch):
    for model in ("ollama/llama3", "ollama_chat/llama3"):
        monkeypatch.setattr(
            llm_module, "settings", Settings(_env_file=None, litellm_model=model)
        )
        assert llm_requires_api_key() is False


def test_llm_requires_api_key_true_for_hosted_default(monkeypatch):
    monkeypatch.setattr(
        llm_module, "settings", Settings(_env_file=None, litellm_model="gpt-4o-mini")
    )
    assert llm_requires_api_key() is True


def test_warn_silent_for_ollama_model_without_key(monkeypatch, caplog):
    """LLM compiler + local Ollama + no key is a *valid* config — no warning."""
    monkeypatch.setattr(
        llm_module,
        "settings",
        Settings(
            _env_file=None,
            compiler_type="llm",
            litellm_api_key=None,
            litellm_model="ollama/llama3",
        ),
    )
    with caplog.at_level(logging.WARNING):
        warn_if_llm_compiler_missing_api_key()

    assert not any(
        "llm_compiler_missing_api_key" in r.getMessage() for r in caplog.records
    )
