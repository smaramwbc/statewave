"""Tests for resolution tracking API schemas and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.schemas.requests import CreateResolutionRequest
from server.schemas.responses import ResolutionResponse


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_create_resolution_valid():
    req = CreateResolutionRequest(
        subject_id="user-1",
        session_id="sess-abc",
        status="resolved",
        resolution_summary="Issued refund",
    )
    assert req.status == "resolved"
    assert req.resolution_summary == "Issued refund"


def test_create_resolution_defaults():
    req = CreateResolutionRequest(subject_id="u1", session_id="s1")
    assert req.status == "open"
    assert req.resolution_summary is None
    assert req.metadata == {}


def test_create_resolution_invalid_status():
    with pytest.raises(ValidationError):
        CreateResolutionRequest(subject_id="u1", session_id="s1", status="invalid_status")


def test_create_resolution_missing_session():
    with pytest.raises(ValidationError):
        CreateResolutionRequest(subject_id="u1")


def test_create_resolution_empty_subject():
    with pytest.raises(ValidationError):
        CreateResolutionRequest(subject_id="", session_id="s1")


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


def test_resolution_response_fields():
    import uuid
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    resp = ResolutionResponse(
        id=uuid.uuid4(),
        subject_id="user-1",
        session_id="sess-1",
        status="resolved",
        resolution_summary="Fixed by restart",
        resolved_at=now,
        metadata={"category": "billing"},
        created_at=now,
        updated_at=now,
    )
    assert resp.status == "resolved"
    assert resp.metadata["category"] == "billing"
