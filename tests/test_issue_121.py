"""Regression tests for issue #121.

A subject id is used directly as a URL path segment by many admin/v1
routes. A `/` in the id made FastAPI's single-segment `{subject_id}`
converter 404, so a subject could be created (the write only length-checked)
and then be unreachable/undeletable via every path-param endpoint.

Fix: one canonical subject-id contract, enforced at every write/query
ingress. These tests lock in (a) the contract, (b) that it rejects the
URL-breaking characters, and (c) that memory_packs and the request schemas
can never drift to *different* contracts again (the root cause).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.core.identifiers import SUBJECT_ID_RE, is_valid_subject_id
from server.schemas.requests import (
    CompileMemoriesRequest,
    CreateEpisodeRequest,
    GetContextRequest,
)
from server.services import memory_packs as mp

_VALID = ["user-1", "tenant-a:agent-7", "support_v2.1", "test:user-42", "x" * 256]
_INVALID = [
    "user-1/slash_test",          # the reported bug — path separator
    "test:user-42/slash_test",    # with the tenant prefix, as in the issue log
    "a b",                        # whitespace
    "a?b", "a#b", "a%2Fb",        # other URL-significant characters
    "",                           # empty
    "x" * 257,                    # exceeds the 256 cap
]


@pytest.mark.parametrize("sid", _VALID)
def test_121_valid_subject_ids_accepted(sid):
    assert CreateEpisodeRequest(subject_id=sid, source="s", type="t", payload={}).subject_id == sid
    assert CompileMemoriesRequest(subject_id=sid).subject_id == sid
    assert GetContextRequest(subject_id=sid, task="t").subject_id == sid
    assert is_valid_subject_id(sid)


@pytest.mark.parametrize("sid", _INVALID)
def test_121_invalid_subject_ids_rejected_at_write_and_query(sid):
    for model, kwargs in (
        (CreateEpisodeRequest, dict(source="s", type="t", payload={})),
        (CompileMemoriesRequest, {}),
        (GetContextRequest, dict(task="t")),
    ):
        with pytest.raises(ValidationError):
            model(subject_id=sid, **kwargs)
    assert not is_valid_subject_id(sid)


def test_121_single_source_of_truth():
    """memory_packs must reuse the exact canonical regex — the root cause of
    #121 was two divergent subject-id contracts in one codebase."""
    assert mp._SUBJECT_ID_RE is SUBJECT_ID_RE


@pytest.mark.asyncio
async def test_121_http_post_episode_with_slash_returns_422(client):
    """The slash id is rejected at request validation, before any DB access."""
    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": "user-1/slash_test",
            "source": "support-chat",
            "type": "conversation",
            "payload": {"messages": [{"role": "user", "content": "hi"}]},
        },
    )
    assert resp.status_code == 422
    assert "subject_id" in resp.text
