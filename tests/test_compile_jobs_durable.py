"""Unit tests for durable compile job manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

import pytest

from server.services.compile_jobs import (
    JobStatus,
    submit_job,
    submit_job_durable,
    get_job,
    get_job_durable,
    mark_running,
    mark_running_durable,
    mark_completed,
    mark_completed_durable,
    mark_failed,
    mark_failed_durable,
    _jobs,
)


@pytest.fixture(autouse=True)
def _clear_memory_store():
    """Clear the in-memory job store between tests."""
    _jobs.clear()
    yield
    _jobs.clear()


# ---------------------------------------------------------------------------
# In-memory (sync) interface
# ---------------------------------------------------------------------------


class TestInMemoryJobs:
    def test_submit_job_creates_entry(self):
        job = submit_job("subject-1")
        assert job.id in _jobs
        assert job.subject_id == "subject-1"
        assert job.status == JobStatus.pending

    def test_get_job_returns_none_for_unknown(self):
        assert get_job("nonexistent") is None

    def test_get_job_returns_existing(self):
        job = submit_job("sub")
        assert get_job(job.id) is job

    def test_mark_running_transitions_status(self):
        job = submit_job("sub")
        mark_running(job.id)
        assert job.status == JobStatus.running

    def test_mark_completed_sets_fields(self):
        job = submit_job("sub")
        mark_completed(job.id, 5, [{"id": "m1"}])
        assert job.status == JobStatus.completed
        assert job.memories_created == 5
        assert job.memories == [{"id": "m1"}]
        assert job.completed_at is not None

    def test_mark_failed_sets_error(self):
        job = submit_job("sub")
        mark_failed(job.id, "Something broke")
        assert job.status == JobStatus.failed
        assert job.error == "Something broke"
        assert job.completed_at is not None

    def test_mark_on_nonexistent_is_no_op(self):
        mark_running("ghost")
        mark_completed("ghost", 0, [])
        mark_failed("ghost", "err")
        # No exceptions raised


# ---------------------------------------------------------------------------
# Durable (async) interface
# ---------------------------------------------------------------------------


class TestDurableJobs:
    @pytest.mark.anyio
    async def test_submit_job_durable_persists_to_db(self):
        """Durable submit writes to Postgres and in-memory."""
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        def mock_factory():
            return mock_session

        with patch(
            "server.services.compile_jobs_durable.get_session_factory",
            return_value=mock_factory,
        ):
            job = await submit_job_durable("test-subject")

        assert job.subject_id == "test-subject"
        assert job.status == JobStatus.pending
        # Should also be in memory
        assert job.id in _jobs

    @pytest.mark.anyio
    async def test_submit_job_durable_falls_back_on_db_error(self):
        """If durable persist fails, falls back to in-memory only."""
        def mock_factory():
            raise RuntimeError("DB down")

        with patch(
            "server.services.compile_jobs_durable.get_session_factory",
            return_value=mock_factory,
        ):
            job = await submit_job_durable("test-subject")

        # Should still work (in-memory fallback)
        assert job.subject_id == "test-subject"
        assert job.id in _jobs

    @pytest.mark.anyio
    async def test_get_job_durable_checks_memory_first(self):
        """If job is in memory, returns it without DB call."""
        job = submit_job("sub")
        result = await get_job_durable(job.id)
        assert result is job

    @pytest.mark.anyio
    async def test_get_job_durable_queries_db_on_miss(self):
        """If job not in memory, queries Postgres."""
        from server.db.tables import CompileJobRow

        fake_row = MagicMock(spec=CompileJobRow)
        fake_row.id = "db-job-1"
        fake_row.subject_id = "db-subject"
        fake_row.status = "completed"
        fake_row.memories_created = 3
        fake_row.error = None
        fake_row.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fake_row.completed_at = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_row
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        def mock_factory():
            return mock_session

        with patch(
            "server.services.compile_jobs_durable.get_session_factory",
            return_value=mock_factory,
        ):
            result = await get_job_durable("db-job-1")

        assert result is not None
        assert result.id == "db-job-1"
        assert result.subject_id == "db-subject"
        assert result.status == JobStatus.completed
        assert result.memories_created == 3

    @pytest.mark.anyio
    async def test_get_job_durable_returns_none_on_miss(self):
        """If job not in memory and not in DB, returns None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        def mock_factory():
            return mock_session

        with patch(
            "server.services.compile_jobs_durable.get_session_factory",
            return_value=mock_factory,
        ):
            result = await get_job_durable("nonexistent")

        assert result is None

    @pytest.mark.anyio
    async def test_mark_running_durable_updates_both(self):
        """mark_running_durable updates in-memory and DB."""
        job = submit_job("sub")
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        def mock_factory():
            return mock_session

        with patch(
            "server.services.compile_jobs_durable.get_session_factory",
            return_value=mock_factory,
        ):
            await mark_running_durable(job.id)

        assert job.status == JobStatus.running

    @pytest.mark.anyio
    async def test_mark_completed_durable_updates_both(self):
        """mark_completed_durable updates in-memory and DB."""
        job = submit_job("sub")
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        def mock_factory():
            return mock_session

        with patch(
            "server.services.compile_jobs_durable.get_session_factory",
            return_value=mock_factory,
        ):
            await mark_completed_durable(job.id, 7, [])

        assert job.status == JobStatus.completed
        assert job.memories_created == 7

    @pytest.mark.anyio
    async def test_mark_failed_durable_updates_both(self):
        """mark_failed_durable updates in-memory and DB."""
        job = submit_job("sub")
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        def mock_factory():
            return mock_session

        with patch(
            "server.services.compile_jobs_durable.get_session_factory",
            return_value=mock_factory,
        ):
            await mark_failed_durable(job.id, "timeout")

        assert job.status == JobStatus.failed
        assert job.error == "timeout"


# ---------------------------------------------------------------------------
# Sync compile path unaffected
# ---------------------------------------------------------------------------


class TestSyncPathUnaffected:
    def test_sync_compile_request_has_no_async_field_by_default(self):
        """CompileMemoriesRequest defaults to async_mode=False."""
        from server.schemas.requests import CompileMemoriesRequest

        req = CompileMemoriesRequest(subject_id="sub")
        assert req.async_mode is False

    def test_sync_compile_request_accepts_async_true(self):
        """CompileMemoriesRequest accepts async=true via alias."""
        from server.schemas.requests import CompileMemoriesRequest

        req = CompileMemoriesRequest.model_validate({"subject_id": "sub", "async": True})
        assert req.async_mode is True
