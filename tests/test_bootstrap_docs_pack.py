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

import scripts.bootstrap_docs_pack as bootstrap
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


# --- _compile_async resubmission -------------------------------------------
#
# Pinned because the recurring refresh failure is a core deploy restarting
# the server mid-compile: the async job is orphaned (poll timeout) or lands
# in a durable `failed` row, and every occurrence recovered on exactly one
# manual workflow rerun. _compile_async folds that rerun into the script —
# compile-start is idempotent over uncompiled episodes, so a resubmission
# resumes where the dead job stopped.


class _FakeClient:
    """Queued responses for the compile start (post) and poll (get) calls."""

    def __init__(self, posts, gets):
        self.post_calls = 0
        self._posts = list(posts)
        self._gets = list(gets)

    async def post(self, *_a, **_kw):
        self.post_calls += 1
        return self._posts.pop(0)

    async def get(self, *_a, **_kw):
        return self._gets.pop(0)


def _job_start(job_id: str) -> httpx.Response:
    return httpx.Response(status_code=202, json={"job_id": job_id})


def _job_poll(status: str, **extra) -> httpx.Response:
    return httpx.Response(status_code=200, json={"status": status, **extra})


@pytest.mark.asyncio
async def test_compile_returns_without_resubmit_on_success():
    client = _FakeClient(
        posts=[_job_start("j1")],
        gets=[_job_poll("running"), _job_poll("completed", memories_created=7)],
    )
    result = await bootstrap._compile_async(client, "http://x", "subj")
    assert result["memories_created"] == 7
    assert client.post_calls == 1


@pytest.mark.asyncio
async def test_compile_returns_inline_result_when_server_lacks_async():
    """No job_id in the start response = older server did the work inline."""
    client = _FakeClient(
        posts=[httpx.Response(status_code=200, json={"memories_created": 3})],
        gets=[],
    )
    result = await bootstrap._compile_async(client, "http://x", "subj")
    assert result["memories_created"] == 3
    assert client.post_calls == 1


@pytest.mark.asyncio
async def test_compile_resubmits_once_after_failed_job_status():
    """A deploy-killed job lands in `failed`; the resubmission must resume."""
    client = _FakeClient(
        posts=[_job_start("j1"), _job_start("j2")],
        gets=[
            _job_poll("failed"),
            _job_poll("completed", memories_created=5),
        ],
    )
    result = await bootstrap._compile_async(client, "http://x", "subj")
    assert result["memories_created"] == 5
    assert client.post_calls == 2


@pytest.mark.asyncio
async def test_compile_resubmits_once_after_poll_timeout():
    """An orphaned job (server restarted, row stuck `running`) times out;
    the resubmission must resume rather than the script exiting 1."""
    # Two poll intervals reach the ceiling — attempt 1 sees only `running`.
    with patch.object(bootstrap, "_COMPILE_MAX_WAIT_S", 2 * bootstrap._COMPILE_POLL_INTERVAL_S):
        client = _FakeClient(
            posts=[_job_start("j1"), _job_start("j2")],
            gets=[
                _job_poll("running"),
                _job_poll("running"),
                _job_poll("completed", memories_created=9),
            ],
        )
        result = await bootstrap._compile_async(client, "http://x", "subj")
    assert result["memories_created"] == 9
    assert client.post_calls == 2


@pytest.mark.asyncio
async def test_compile_exits_nonzero_after_second_death():
    client = _FakeClient(
        posts=[_job_start("j1"), _job_start("j2")],
        gets=[_job_poll("failed"), _job_poll("failed")],
    )
    with pytest.raises(SystemExit) as exc:
        await bootstrap._compile_async(client, "http://x", "subj")
    assert exc.value.code == 1
    assert client.post_calls == 2


# --- entity rebuild after the swap import (issue #380) ----------------------
#
# Pinned because the rebuild is best-effort BY DESIGN: the swap already
# succeeded, so neither a network failure (attempts=1, no re-POST that
# would race a still-running rebuild) nor a non-200 (incl. a 404 from a
# server predating the endpoint) may turn the refresh red.


class _ImportClient:
    def __init__(self, rebuild_outcome):
        self._rebuild_outcome = rebuild_outcome
        self.rebuild_posts = 0

    async def post(self, url, **_kw):
        if url.endswith("/rebuild-entities"):
            self.rebuild_posts += 1
            if isinstance(self._rebuild_outcome, Exception):
                raise self._rebuild_outcome
            return self._rebuild_outcome
        return httpx.Response(status_code=200, json={"memories_imported": 5})


@pytest.mark.asyncio
async def test_rebuild_network_failure_is_nonfatal_and_single_attempt(capsys):
    client = _ImportClient(httpx.ConnectTimeout("slow rebuild"))
    result = await bootstrap._import_into(client, "http://x", {"doc": 1}, "live-subj")
    assert result["memories_imported"] == 5, "swap result survives a rebuild failure"
    assert client.rebuild_posts == 1, "must never re-POST a possibly-running rebuild"
    assert "WARN entity rebuild" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_rebuild_404_from_old_server_is_nonfatal(capsys):
    client = _ImportClient(httpx.Response(status_code=404))
    result = await bootstrap._import_into(client, "http://x", {"doc": 1}, "live-subj")
    assert result["memories_imported"] == 5
    err = capsys.readouterr().err
    assert err.count("WARN entity rebuild") == 1


@pytest.mark.asyncio
async def test_rebuild_success_reports_count(capsys):
    client = _ImportClient(httpx.Response(status_code=200, json={"entities_rebuilt": 42}))
    result = await bootstrap._import_into(client, "http://x", {"doc": 1}, "live-subj")
    assert result["memories_imported"] == 5
    assert "rebuilt 42 entity rows" in capsys.readouterr().out
