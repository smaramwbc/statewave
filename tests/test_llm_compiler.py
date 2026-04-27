"""Tests for LLM compiler."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

try:
    import litellm  # noqa: F401
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

pytestmark = pytest.mark.skipif(not HAS_LITELLM, reason="litellm not installed")

from server.db.tables import EpisodeRow  # noqa: E402
from server.services.compilers.llm import LLMCompiler  # noqa: E402


def _make_episode(**kw) -> EpisodeRow:
    defaults = dict(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="test",
        type="chat",
        payload={"messages": [{"role": "user", "content": "My name is Alice. I work at Acme Corp."}]},
        metadata_={},
        provenance={},
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(kw)
    return EpisodeRow(**defaults)


def _fake_response(content: str) -> MagicMock:
    """Create a fake litellm.completion() response."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    return resp


def _make_compiler() -> LLMCompiler:
    """Create a compiler without calling __init__ (avoids import check in CI)."""
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    from concurrent.futures import ThreadPoolExecutor
    compiler._executor = ThreadPoolExecutor(max_workers=1)
    return compiler


@patch("server.services.compilers.llm.litellm")
def test_llm_compile_basic(mock_litellm):
    compiler = _make_compiler()

    llm_response = json.dumps([
        {"kind": "profile_fact", "content": "Name is Alice", "summary": "Name is Alice", "confidence": 0.9},
        {"kind": "profile_fact", "content": "Works at Acme Corp", "summary": "Works at Acme Corp", "confidence": 0.85},
        {"kind": "episode_summary", "content": "User introduced themselves", "summary": "Introduction", "confidence": 0.8},
    ])
    mock_litellm.completion.return_value = _fake_response(llm_response)

    ep = _make_episode()
    memories = compiler.compile([ep])

    assert len(memories) == 3
    assert memories[0].kind == "profile_fact"
    assert memories[0].source_episode_ids == [ep.id]
    assert memories[0].metadata_.get("compiler") == "llm"


@patch("server.services.compilers.llm.litellm")
def test_llm_compile_empty_payload(mock_litellm):
    compiler = _make_compiler()

    ep = _make_episode(payload={})
    memories = compiler.compile([ep])
    assert memories == []
    mock_litellm.completion.assert_not_called()


@patch("server.services.compilers.llm.litellm")
def test_llm_compile_api_failure_returns_empty(mock_litellm):
    compiler = _make_compiler()
    mock_litellm.completion.side_effect = RuntimeError("API down")

    ep = _make_episode()
    memories = compiler.compile([ep])
    assert memories == []


@patch("server.services.compilers.llm.litellm")
def test_llm_compile_invalid_json_returns_empty(mock_litellm):
    compiler = _make_compiler()
    mock_litellm.completion.return_value = _fake_response("not json at all")

    ep = _make_episode()
    memories = compiler.compile([ep])
    assert memories == []


@patch("server.services.compilers.llm.litellm")
def test_llm_compile_strips_markdown_fences(mock_litellm):
    compiler = _make_compiler()

    fenced = '```json\n[{"kind":"profile_fact","content":"Test","summary":"Test","confidence":0.9}]\n```'
    mock_litellm.completion.return_value = _fake_response(fenced)

    memories = compiler.compile([_make_episode()])
    assert len(memories) == 1


@patch("server.services.compilers.llm.litellm")
def test_llm_compile_clamps_confidence(mock_litellm):
    compiler = _make_compiler()

    llm_response = json.dumps([
        {"kind": "profile_fact", "content": "A", "summary": "A", "confidence": 5.0},
        {"kind": "profile_fact", "content": "B", "summary": "B", "confidence": -1.0},
    ])
    mock_litellm.completion.return_value = _fake_response(llm_response)

    memories = compiler.compile([_make_episode()])
    assert memories[0].confidence == 1.0
    assert memories[1].confidence == 0.0
