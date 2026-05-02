"""Tests for deep readiness checks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.services.readiness import (
    ReadinessResult,
    _check_db,
    _check_llm,
    _check_queue,
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
    with patch("server.services.readiness.settings") as mock_settings:
        mock_settings.litellm_api_key = None
        result = await _check_llm()
        assert result.status == "ok"
        assert "not configured" in result.detail


@pytest.mark.asyncio
async def test_run_readiness_all_ok():
    conn = _mock_conn_with_scalar(0)

    with patch("server.services.readiness.settings") as mock_settings:
        mock_settings.litellm_api_key = None
        result = await run_readiness_checks(conn)

    assert result.status == "ready"
    assert len(result.checks) == 3


@pytest.mark.asyncio
async def test_run_readiness_db_fail():
    conn = AsyncMock()
    conn.execute = AsyncMock(side_effect=Exception("down"))

    with patch("server.services.readiness.settings") as mock_settings:
        mock_settings.litellm_api_key = None
        result = await run_readiness_checks(conn)

    assert result.status == "not_ready"
    assert result.http_status == 503
