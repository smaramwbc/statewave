"""Tests for the compiler abstraction and HeuristicCompiler."""

import uuid
from datetime import datetime, timezone

from server.db.tables import EpisodeRow
from server.services.compilers import get_compiler
from server.services.compilers.heuristic import (
    HeuristicCompiler,
    episode_valid_from,
    extract_payload_text,
)


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


def test_get_compiler_returns_heuristic_by_default(monkeypatch):
    # Pin the compiler choice so a developer's local `.env` (which the
    # docker-compose quickstart writes with compiler=llm) doesn't flip
    # this test. The factory reads `settings.compiler_type` lazily, so
    # patching the global Settings instance is enough.
    import server.core.config as config_module
    from server.core.config import Settings

    monkeypatch.setattr(
        config_module, "settings", Settings(_env_file=None, compiler_type="heuristic")
    )
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


# ---------------------------------------------------------------------------
# Temporal anchor (episode_valid_from)
# ---------------------------------------------------------------------------


def test_episode_valid_from_uses_explicit_event_time_iso():
    ep = _ep({"event_time": "2023-05-08T13:56:00+00:00", "text": "x"})
    out = episode_valid_from(ep)
    assert out == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)


def test_episode_valid_from_uses_first_message_timestamp_locomo():
    # LoCoMo's idiomatic format — driver test case for the bench.
    ep = _ep(
        {
            "messages": [
                {"role": "user", "content": "hi", "timestamp": "1:56 pm on 8 May, 2023"},
                {"role": "user", "content": "later", "timestamp": "2:00 pm on 8 May, 2023"},
            ]
        }
    )
    out = episode_valid_from(ep)
    assert out == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)


def test_episode_valid_from_explicit_overrides_first_message():
    # Explicit override wins even if messages also carry timestamps.
    ep = _ep(
        {
            "event_time": "2024-01-01T00:00:00+00:00",
            "messages": [{"role": "user", "content": "x", "timestamp": "1:56 pm on 8 May, 2023"}],
        }
    )
    assert episode_valid_from(ep) == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_episode_valid_from_falls_back_to_created_at():
    fixed = datetime(2026, 5, 11, tzinfo=timezone.utc)
    ep = EpisodeRow(
        id=uuid.uuid4(),
        subject_id="s",
        source="test",
        type="conversation",
        payload={"text": "no timestamps anywhere"},
        metadata_={},
        provenance={},
        created_at=fixed,
    )
    assert episode_valid_from(ep) == fixed


def test_episode_valid_from_falls_back_on_unparseable_timestamp():
    # Bad timestamp shouldn't crash — fall through to next candidate.
    fixed = datetime(2026, 5, 11, tzinfo=timezone.utc)
    ep = EpisodeRow(
        id=uuid.uuid4(),
        subject_id="s",
        source="test",
        type="conversation",
        payload={
            "messages": [{"role": "user", "content": "x", "timestamp": "definitely not a date"}]
        },
        metadata_={},
        provenance={},
        created_at=fixed,
    )
    assert episode_valid_from(ep) == fixed
