"""Regression tests for issue #124.

`session_id` is used as a URL path segment (e.g.
`GET /admin/subjects/{subject_id}/sessions/{session_id}/timeline`), so a
`/` made the route unmatchable and the session unreachable — the same
class of bug as #121/#123, just for session_id. Fix: the canonical
`SessionId` contract enforced at every session_id write/query ingress,
optional-friendly (None still allowed where the field is optional).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.core.identifiers import SESSION_ID_RE, SUBJECT_ID_RE, is_valid_session_id
from server.schemas.requests import (
    CreateEpisodeRequest,
    CreateResolutionRequest,
    GetContextRequest,
    HandoffRequest,
)

_VALID = ["sess-1", "tenant-a:sess.7", "abc_123", "x" * 256]
_INVALID = ["sess/1", "a b", "a?b", "a#b", "a%2Fb", "", "x" * 257]


def test_124_session_id_shares_the_subject_id_contract():
    assert SESSION_ID_RE is SUBJECT_ID_RE


@pytest.mark.parametrize("sid", _VALID)
def test_124_valid_session_ids_accepted(sid):
    assert CreateEpisodeRequest(
        subject_id="s", source="x", type="t", payload={}, session_id=sid
    ).session_id == sid
    assert CreateResolutionRequest(subject_id="s", session_id=sid).session_id == sid
    assert HandoffRequest(subject_id="s", session_id=sid).session_id == sid
    assert is_valid_session_id(sid)


@pytest.mark.parametrize("sid", _INVALID)
def test_124_invalid_session_ids_rejected_everywhere(sid):
    # Optional field (CreateEpisodeRequest / GetContextRequest): a bad
    # *value* is still rejected — only None is exempt.
    with pytest.raises(ValidationError):
        CreateEpisodeRequest(
            subject_id="s", source="x", type="t", payload={}, session_id=sid
        )
    with pytest.raises(ValidationError):
        GetContextRequest(subject_id="s", task="t", session_id=sid)
    # Required field (CreateResolutionRequest / HandoffRequest).
    with pytest.raises(ValidationError):
        CreateResolutionRequest(subject_id="s", session_id=sid)
    with pytest.raises(ValidationError):
        HandoffRequest(subject_id="s", session_id=sid)
    assert not is_valid_session_id(sid)


def test_124_session_id_stays_optional_when_omitted_or_none():
    """The optional fields must still accept absence / explicit None."""
    assert CreateEpisodeRequest(
        subject_id="s", source="x", type="t", payload={}
    ).session_id is None
    assert CreateEpisodeRequest(
        subject_id="s", source="x", type="t", payload={}, session_id=None
    ).session_id is None
    assert GetContextRequest(subject_id="s", task="t").session_id is None


@pytest.mark.asyncio
async def test_124_http_post_episode_with_slash_session_returns_422(client):
    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": "user-1",
            "source": "support-chat",
            "type": "conversation",
            "payload": {"messages": [{"role": "user", "content": "hi"}]},
            "session_id": "sess/1",
        },
    )
    assert resp.status_code == 422
    assert "session_id" in resp.text
