"""Tests for LLM compiler."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from server.db.tables import EpisodeRow
from server.services.compilers.llm import LLMCompiler


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


class _FakeChatCompletion:
    """Minimal stand-in for OpenAI chat completion response."""

    def __init__(self, content: str):
        self.choices = [MagicMock(message=MagicMock(content=content))]


@patch("server.services.compilers.llm.LLMCompiler.__init__", return_value=None)
def test_llm_compile_basic(mock_init):
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    compiler._client = MagicMock()

    llm_response = json.dumps([
        {"kind": "profile_fact", "content": "Name is Alice", "summary": "Name is Alice", "confidence": 0.9},
        {"kind": "profile_fact", "content": "Works at Acme Corp", "summary": "Works at Acme Corp", "confidence": 0.85},
        {"kind": "episode_summary", "content": "User introduced themselves", "summary": "Introduction", "confidence": 0.8},
    ])
    compiler._client.chat.completions.create.return_value = _FakeChatCompletion(llm_response)

    ep = _make_episode()
    memories = compiler.compile([ep])

    assert len(memories) == 3
    assert memories[0].kind == "profile_fact"
    assert memories[0].source_episode_ids == [ep.id]
    assert memories[0].metadata_.get("compiler") == "llm"


@patch("server.services.compilers.llm.LLMCompiler.__init__", return_value=None)
def test_llm_compile_empty_payload(mock_init):
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    compiler._client = MagicMock()

    ep = _make_episode(payload={})
    memories = compiler.compile([ep])
    assert memories == []
    compiler._client.chat.completions.create.assert_not_called()


@patch("server.services.compilers.llm.LLMCompiler.__init__", return_value=None)
def test_llm_compile_api_failure_returns_empty(mock_init):
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    compiler._client = MagicMock()
    compiler._client.chat.completions.create.side_effect = RuntimeError("API down")

    ep = _make_episode()
    memories = compiler.compile([ep])
    assert memories == []


@patch("server.services.compilers.llm.LLMCompiler.__init__", return_value=None)
def test_llm_compile_invalid_json_returns_empty(mock_init):
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    compiler._client = MagicMock()
    compiler._client.chat.completions.create.return_value = _FakeChatCompletion("not json at all")

    ep = _make_episode()
    memories = compiler.compile([ep])
    assert memories == []


@patch("server.services.compilers.llm.LLMCompiler.__init__", return_value=None)
def test_llm_compile_strips_markdown_fences(mock_init):
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    compiler._client = MagicMock()

    fenced = '```json\n[{"kind":"profile_fact","content":"Test","summary":"Test","confidence":0.9}]\n```'
    compiler._client.chat.completions.create.return_value = _FakeChatCompletion(fenced)

    memories = compiler.compile([_make_episode()])
    assert len(memories) == 1


@patch("server.services.compilers.llm.LLMCompiler.__init__", return_value=None)
def test_llm_compile_clamps_confidence(mock_init):
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    compiler._client = MagicMock()

    llm_response = json.dumps([
        {"kind": "profile_fact", "content": "A", "summary": "A", "confidence": 5.0},
        {"kind": "profile_fact", "content": "B", "summary": "B", "confidence": -1.0},
    ])
    compiler._client.chat.completions.create.return_value = _FakeChatCompletion(llm_response)

    memories = compiler.compile([_make_episode()])
    assert memories[0].confidence == 1.0
    assert memories[1].confidence == 0.0
