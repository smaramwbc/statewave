"""Tests for the bootstrap script's retry-on-transient-failure behavior.

Pinned because the failure mode the retry recovers from — a transient
HTTP failure (5xx from the platform's reverse proxy, or a mid-stream
connection drop) landing on an in-flight call during a rolling
deployment — is hard to exercise by accident in CI but causes silent
data loss in prod (the support-docs subject ends up empty until the
next refresh). The unit tests below exercise the retry helper with a
mocked httpx call covering each transient-failure shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from scripts.bootstrap_docs_pack import _request_with_retry


def _resp(status_code: int) -> httpx.Response:
    return httpx.Response(status_code=status_code)


@pytest.fixture(autouse=True)
def _no_sleep():
    """Skip real sleeps in retry backoff so tests run instantly."""
    with patch("scripts.bootstrap_docs_pack.asyncio.sleep", new=AsyncMock()):
        yield


@pytest.mark.asyncio
async def test_returns_immediately_on_first_success():
    """Happy path: a 200 on the first try returns without retry."""
    fn = AsyncMock(return_value=_resp(200))
    resp = await _request_with_retry("op", fn, attempts=5)
    assert resp.status_code == 200
    assert fn.await_count == 1


@pytest.mark.asyncio
async def test_retries_on_502_and_succeeds():
    """A 502 (the exact symptom of a Fly rolling deploy hitting the
    in-flight request) must trigger a retry."""
    fn = AsyncMock(side_effect=[_resp(502), _resp(502), _resp(200)])
    resp = await _request_with_retry("op", fn, attempts=5, initial_delay_s=0.01)
    assert resp.status_code == 200
    assert fn.await_count == 3


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
@pytest.mark.asyncio
async def test_retries_on_each_retryable_status(status):
    fn = AsyncMock(side_effect=[_resp(status), _resp(200)])
    resp = await _request_with_retry("op", fn, attempts=3, initial_delay_s=0.01)
    assert resp.status_code == 200
    assert fn.await_count == 2


@pytest.mark.asyncio
async def test_does_not_retry_on_4xx_other_than_429():
    """A 404 is a real "not found" — retrying would just hammer the
    API. Same for 400-class auth/validation errors."""
    fn = AsyncMock(return_value=_resp(404))
    resp = await _request_with_retry("op", fn, attempts=5)
    assert resp.status_code == 404
    assert fn.await_count == 1


@pytest.mark.asyncio
async def test_retries_on_network_error():
    """Mid-stream TCP drop during a deploy surfaces as a NetworkError /
    RemoteProtocolError. The retry must catch and recover."""
    fn = AsyncMock(
        side_effect=[
            httpx.RemoteProtocolError("connection broken"),
            _resp(200),
        ]
    )
    resp = await _request_with_retry("op", fn, attempts=5, initial_delay_s=0.01)
    assert resp.status_code == 200
    assert fn.await_count == 2


@pytest.mark.asyncio
async def test_retries_on_timeout():
    fn = AsyncMock(
        side_effect=[
            httpx.ReadTimeout("timed out"),
            _resp(200),
        ]
    )
    resp = await _request_with_retry("op", fn, attempts=5, initial_delay_s=0.01)
    assert resp.status_code == 200
    assert fn.await_count == 2


@pytest.mark.asyncio
async def test_raises_after_exhausting_attempts_on_network_error():
    """After max attempts the original exception propagates so the
    caller can fail loudly rather than getting a silently bad response."""
    fn = AsyncMock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(httpx.ConnectError):
        await _request_with_retry("op", fn, attempts=3, initial_delay_s=0.01)
    assert fn.await_count == 3


@pytest.mark.asyncio
async def test_returns_last_response_after_exhausting_attempts_on_5xx():
    """When every attempt returns a retryable status, the final response
    is returned (not raised) so the caller's existing error handling
    runs and prints the body for debugging."""
    fn = AsyncMock(side_effect=[_resp(502)] * 5)
    resp = await _request_with_retry("op", fn, attempts=5, initial_delay_s=0.01)
    assert resp.status_code == 502
    assert fn.await_count == 5


@pytest.mark.asyncio
async def test_backoff_is_exponential_and_capped():
    """Sleep delays should grow 2 → 4 → 8 → 16 → 30 (capped at 30)."""
    fn = AsyncMock(side_effect=[_resp(502)] * 6)
    sleep_mock = AsyncMock()
    with patch("scripts.bootstrap_docs_pack.asyncio.sleep", new=sleep_mock):
        await _request_with_retry(
            "op", fn, attempts=6, initial_delay_s=2.0, max_delay_s=30.0
        )
    delays = [call.args[0] for call in sleep_mock.await_args_list]
    assert delays == [2.0, 4.0, 8.0, 16.0, 30.0]
