"""Tests for the compiler abstraction and HeuristicCompiler."""

import uuid
from datetime import datetime, timezone

from server.db.tables import EpisodeRow
from server.services.compilers import get_compiler
from server.services.compilers.heuristic import HeuristicCompiler, extract_payload_text


def _ep(payload: dict, subject_id: str = "user-1") -> EpisodeRow:
    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id=subject_id,
        source="test",
        type="conversation",
        payload=payload,
        metadata_={},
        provenance={},
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_heuristic_compiler_satisfies_protocol():
    compiler = HeuristicCompiler()
    # Structural check — it has the right method signature
    assert hasattr(compiler, "compile")
    assert callable(compiler.compile)


def test_get_compiler_returns_heuristic_by_default():
    compiler = get_compiler()
    assert isinstance(compiler, HeuristicCompiler)


# ---------------------------------------------------------------------------
# HeuristicCompiler behavior
# ---------------------------------------------------------------------------


def test_compile_chat_produces_summary_and_facts():
    compiler = HeuristicCompiler()
    ep = _ep(
        {"messages": [{"role": "user", "content": "My name is Alice and I work at Acme Corp."}]}
    )
    memories = compiler.compile([ep])
    kinds = {m.kind for m in memories}
    assert "episode_summary" in kinds
    assert "profile_fact" in kinds


def test_compile_empty_payload():
    compiler = HeuristicCompiler()
    memories = compiler.compile([_ep({})])
    assert memories == []


def test_compile_preserves_provenance():
    compiler = HeuristicCompiler()
    ep = _ep({"text": "Hello world"})
    memories = compiler.compile([ep])
    for m in memories:
        assert ep.id in m.source_episode_ids


def test_compile_text_payload():
    compiler = HeuristicCompiler()
    memories = compiler.compile([_ep({"text": "Some plain text"})])
    assert len(memories) == 1
    assert memories[0].kind == "episode_summary"


def test_compile_content_payload():
    compiler = HeuristicCompiler()
    memories = compiler.compile([_ep({"content": "Some content"})])
    assert len(memories) == 1
    assert memories[0].kind == "episode_summary"


# ---------------------------------------------------------------------------
# Payload text extraction
# ---------------------------------------------------------------------------


def test_extract_payload_text_messages():
    text = extract_payload_text({"messages": [{"role": "user", "content": "hi"}]})
    assert "user: hi" in text


def test_extract_payload_text_empty():
    assert extract_payload_text({}) == ""
    assert extract_payload_text({"foo": "bar"}) == ""
