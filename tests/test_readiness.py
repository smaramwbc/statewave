"""Tests for deep readiness checks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import server.services.readiness as readiness_module
from server.services.readiness import (
    ReadinessResult,
    _check_db,
    _check_llm,
    _check_queue,
    database_url_status,
    run_readiness_checks,
)


class TestCheckResult:
    def test_ready_http_status(self):
        r = ReadinessResult(status="ready")
        assert r.http_status == 200

    def test_degraded_http_status(self):
        r = ReadinessResult(status="degraded")
        assert r.http_status == 200

    def test_not_ready_http_status(self):
        r = ReadinessResult(status="not_ready")
        assert r.http_status == 503


@pytest.mark.asyncio
async def test_check_db_success():
    conn = AsyncMock()
    result = await _check_db(conn)
    assert result.status == "ok"
    assert result.name == "database"
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_check_db_failure():
    conn = AsyncMock()
    conn.execute = AsyncMock(side_effect=Exception("connection refused"))
    result = await _check_db(conn)
    assert result.status == "fail"
    assert "connection refused" in result.detail


def _mock_conn_with_scalar(value):
    conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = value
    conn.execute = AsyncMock(return_value=mock_result)
    return conn


@pytest.mark.asyncio
async def test_check_queue_healthy():
    conn = _mock_conn_with_scalar(0)
    result = await _check_queue(conn)
    assert result.status == "ok"


@pytest.mark.asyncio
async def test_check_queue_stuck_jobs():
    conn = _mock_conn_with_scalar(3)
    result = await _check_queue(conn)
    assert result.status == "degraded"
    assert "3 stuck" in result.detail


@pytest.mark.asyncio
async def test_check_llm_not_configured():
    with patch("server.services.readiness.litellm_api_key_configured", return_value=False):
        result = await _check_llm()
        assert result.status == "ok"
        assert result.detail == "STATEWAVE_LITELLM_API_KEY is not set"


@pytest.mark.asyncio
async def test_check_llm_empty_api_key_treated_as_not_set():
    with patch("server.services.readiness.litellm_api_key_configured", return_value=False):
        result = await _check_llm()
        assert result.detail == "STATEWAVE_LITELLM_API_KEY is not set"


@pytest.mark.asyncio
async def test_check_llm_ollama_without_key_probes_instead_of_skipping():
    """Ollama needs no key — don't short-circuit with 'key not set'; actually
    probe the local server (#122 follow-up, credit @LPHuynh)."""
    with (
        patch("server.services.readiness.litellm_api_key_configured", return_value=False),
        patch("server.services.readiness.llm_requires_api_key", return_value=False),
        patch("server.services.llm.aping", new=AsyncMock(return_value=None)) as aping,
    ):
        result = await _check_llm()

    aping.assert_awaited_once()
    assert result.status == "ok"
    assert result.detail != "STATEWAVE_LITELLM_API_KEY is not set"


@pytest.mark.asyncio
async def test_run_readiness_all_ok():
    conn = _mock_conn_with_scalar(0)

    with patch("server.services.readiness.litellm_api_key_configured", return_value=False):
        result = await run_readiness_checks(conn)

    assert result.status == "ready"
    assert len(result.checks) == 3


@pytest.mark.asyncio
async def test_run_readiness_db_fail():
    conn = AsyncMock()
    conn.execute = AsyncMock(side_effect=Exception("down"))

    with patch("server.services.readiness.litellm_api_key_configured", return_value=False):
        result = await run_readiness_checks(conn)

    assert result.status == "not_ready"
    assert result.http_status == 503


# ── #66: /readyz distinguishes DB not-set / unparseable / unreachable ───────


def test_database_url_status_missing(monkeypatch):
    monkeypatch.setattr(readiness_module.settings, "database_url", "")
    state, detail = database_url_status()
    assert state == "missing"
    assert detail == "DATABASE_URL is not set"


def test_database_url_status_unparseable(monkeypatch):
    monkeypatch.setattr(readiness_module.settings, "database_url", "this is not a url")
    state, detail = database_url_status()
    assert state == "unparseable"
    assert detail.startswith("DATABASE_URL is set but couldn't be parsed:")


def test_database_url_status_ok(monkeypatch):
    monkeypatch.setattr(
        readiness_module.settings,
        "database_url",
        "postgresql+asyncpg://statewave:statewave@localhost:5432/statewave",
    )
    state, detail = database_url_status()
    assert state == "ok"
    assert detail is None


@pytest.mark.asyncio
async def test_run_readiness_conn_none_classifies_db_and_keeps_llm():
    """conn=None → DB fails with the supplied detail, queue degraded,
    LLM still evaluated (it's DB-independent)."""
    with patch("server.services.readiness.litellm_api_key_configured", return_value=False):
        result = await run_readiness_checks(
            None, db_unavailable_detail="DATABASE_URL is not set"
        )

    by_name = {c.name: c for c in result.checks}
    assert by_name["database"].status == "fail"
    assert by_name["database"].detail == "DATABASE_URL is not set"
    assert by_name["queue"].status == "degraded"
    assert "llm" in by_name  # LLM check still ran
    assert result.status == "not_ready"
    assert result.http_status == 503


@pytest.mark.asyncio
async def test_readyz_http_reports_database_url_not_set(client, monkeypatch):
    """End-to-end: an unset DB URL yields a clear, actionable 503 — and
    does so deterministically regardless of whether CI has Postgres up
    (we short-circuit before any engine/connection)."""
    monkeypatch.setattr(readiness_module.settings, "database_url", "")
    resp = await client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    db = next(c for c in body["checks"] if c["name"] == "database")
    assert db["status"] == "fail"
    assert db["detail"] == "DATABASE_URL is not set"
