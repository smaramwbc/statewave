"""Compiler-version provenance metadata."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.core.config import settings
from server.db.tables import EpisodeRow
from server.services.compilers.heuristic import HeuristicCompiler
from server.services.compilers.llm import (
    LLMCompiler,
    _RECALL_SWEEP_PROMPT,
    _llm_compiler_metadata,
    _prompt_version,
    _system_prompt,
)
from server.services.structured import compile_candidates


_NOW = datetime.now(timezone.utc)


def _episode(
    text: str = "My name is Alice",
    *,
    payload: dict | None = None,
) -> EpisodeRow:
    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="test",
        type="conversation",
        payload=payload if payload is not None else {"text": text},
        metadata_={},
        provenance={},
        created_at=_NOW,
    )


def _llm_compiler() -> LLMCompiler:
    compiler = LLMCompiler.__new__(LLMCompiler)
    compiler._model = "gpt-4o-mini"
    compiler._api_key = "test-key"
    compiler._client = None
    return compiler


def test_llm_metadata_records_model_prompt_and_compiler_version():
    metadata = _llm_compiler_metadata("gpt-4o-mini")

    assert metadata["compiler"] == "llm"
    assert metadata["model"] == "gpt-4o-mini"
    assert metadata["prompt_version"] == _prompt_version(_system_prompt())
    assert metadata["compiler_version"].startswith("sha256:")
    assert "pass" not in metadata


def test_llm_compiler_version_changes_with_model():
    first = _llm_compiler_metadata("gpt-4o-mini")
    second = _llm_compiler_metadata("gpt-5")

    assert first["compiler_version"] != second["compiler_version"]


def test_llm_compiler_version_changes_with_active_prompt(monkeypatch):
    monkeypatch.setattr(settings, "compile_compose_unit", False)
    normal = _llm_compiler_metadata("gpt-4o-mini")

    monkeypatch.setattr(settings, "compile_compose_unit", True)
    compose = _llm_compiler_metadata("gpt-4o-mini")

    assert normal["prompt_version"] != compose["prompt_version"]
    assert normal["compiler_version"] != compose["compiler_version"]


def test_recall_sweep_version_covers_both_prompt_templates():
    metadata = _llm_compiler_metadata(
        "gpt-4o-mini",
        recall_sweep=True,
    )

    assert metadata["compiler"] == "llm"
    assert metadata["pass"] == "recall_sweep"
    assert metadata["prompt_version"] == _prompt_version(
        _system_prompt(),
        _RECALL_SWEEP_PROMPT,
    )


def test_recall_sweep_has_different_compiler_version_from_first_pass():
    first_pass = _llm_compiler_metadata("gpt-4o-mini")
    recall = _llm_compiler_metadata(
        "gpt-4o-mini",
        recall_sweep=True,
    )

    assert first_pass["compiler_version"] != recall["compiler_version"]


@pytest.mark.asyncio
async def test_llm_compile_stamps_first_and_recall_passes(monkeypatch):
    compiler = _llm_compiler()
    monkeypatch.setattr(settings, "compile_recall_sweep", True)

    first_pass = [
        {
            "kind": "profile_fact",
            "content": "Alice prefers email",
            "summary": "Alice prefers email",
            "confidence": 0.9,
            "episode_index": 0,
        }
    ]
    recall_pass = [
        {
            "kind": "profile_fact",
            "content": "Alice uses dark mode",
            "summary": "Alice uses dark mode",
            "confidence": 0.9,
            "episode_index": 0,
        }
    ]

    with (
        patch.object(
            compiler,
            "_call_llm_async",
            new_callable=AsyncMock,
            return_value=first_pass,
        ),
        patch.object(
            compiler,
            "_call_recall_sweep",
            new_callable=AsyncMock,
            return_value=recall_pass,
        ),
    ):
        memories = await compiler.compile_async([_episode()])

    by_content = {memory.content: memory for memory in memories}

    first_metadata = by_content["Alice prefers email"].metadata_
    recall_metadata = by_content["Alice uses dark mode"].metadata_

    assert first_metadata["compiler"] == "llm"
    assert "pass" not in first_metadata

    assert recall_metadata["compiler"] == "llm"
    assert recall_metadata["pass"] == "recall_sweep"

    assert (
        first_metadata["prompt_version"]
        != recall_metadata["prompt_version"]
    )
    assert (
        first_metadata["compiler_version"]
        != recall_metadata["compiler_version"]
    )


def test_heuristic_memories_record_compiler_version():
    memories = HeuristicCompiler().compile(
        [_episode("I prefer email")]
    )

    assert memories
    assert all(
        memory.metadata_["compiler"] == "heuristic"
        for memory in memories
    )
    assert all(
        memory.metadata_["compiler_version"] == "heuristic:1"
        for memory in memories
    )


def test_heuristic_compiler_metadata_preserves_claim():
    memories = HeuristicCompiler().compile(
        [_episode("My name is Alice")]
    )

    claimed = [
        memory
        for memory in memories
        if memory.metadata_.get("claim")
    ]

    assert claimed
    assert claimed[0].metadata_["compiler"] == "heuristic"
    assert claimed[0].metadata_["compiler_version"] == "heuristic:1"
    assert claimed[0].metadata_["claim"]["key"] == "identity.name"


def test_structured_candidates_record_compiler_version():
    episode = _episode(
        payload={
            "statewave": {
                "memory_candidates": [
                    {
                        "kind": "profile_fact",
                        "text": "Alice prefers email",
                    }
                ]
            }
        }
    )

    memories = compile_candidates(episode)

    assert memories is not None
    assert len(memories) == 1
    assert memories[0].metadata_["compiler"] == "structured"
    assert memories[0].metadata_["compiler_version"] == "structured:1"


def test_structured_compiler_provenance_is_reserved():
    episode = _episode(
        payload={
            "statewave": {
                "memory_candidates": [
                    {
                        "kind": "profile_fact",
                        "text": "Alice prefers email",
                        "metadata": {
                            "source": "connector",
                            "compiler": "spoofed",
                            "compiler_version": "spoofed",
                            "model": "gpt-5",
                            "prompt_version": "sha256:deadbeef",
                        },
                    }
                ]
            }
        }
    )

    memories = compile_candidates(episode)

    assert memories is not None
    metadata = memories[0].metadata_

    assert metadata["source"] == "connector"
    assert metadata["compiler"] == "structured"
    assert metadata["compiler_version"] == "structured:1"
    # `model` and `prompt_version` are stamped only by the compilers that have one. The
    # structured path has neither, so a candidate must not be able to supply them: the admin
    # compiler-trace endpoint reads `model` straight out of this metadata.
    assert "model" not in metadata
    assert "prompt_version" not in metadata
