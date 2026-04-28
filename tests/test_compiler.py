"""Unit tests for the heuristic memory compiler."""

import uuid
from datetime import datetime

from server.db.tables import EpisodeRow
from server.services.compiler import compile_memories_from_episodes


def _make_episode(payload: dict, subject_id: str = "user-1") -> EpisodeRow:
    row = EpisodeRow(
        id=uuid.uuid4(),
        subject_id=subject_id,
        source="test",
        type="conversation",
        payload=payload,
        metadata_={},
        provenance={},
        created_at=datetime.utcnow(),
    )
    return row


def test_episode_summary_created():
    ep = _make_episode({"text": "Hello, how can I help you today?"})
    memories = compile_memories_from_episodes([ep])
    kinds = [m.kind for m in memories]
    assert "episode_summary" in kinds


def test_profile_fact_extraction():
    ep = _make_episode(
        {"messages": [{"role": "user", "content": "My name is Alice and I work at Acme Corp."}]}
    )
    memories = compile_memories_from_episodes([ep])
    facts = [m for m in memories if m.kind == "profile_fact"]
    assert len(facts) >= 1
    contents = " ".join(f.content for f in facts)
    assert "Alice" in contents


def test_provenance_preserved():
    ep = _make_episode({"text": "Some interaction"})
    memories = compile_memories_from_episodes([ep])
    for m in memories:
        assert ep.id in m.source_episode_ids


def test_empty_payload_produces_no_memories():
    ep = _make_episode({})
    memories = compile_memories_from_episodes([ep])
    assert len(memories) == 0


def test_unknown_payload_shape_produces_no_memories():
    ep = _make_episode({"foo": "bar", "baz": 42})
    memories = compile_memories_from_episodes([ep])
    assert len(memories) == 0


def test_compile_is_deterministic():
    """Same input produces same number and kind of memories."""
    ep = _make_episode({"messages": [{"role": "user", "content": "My name is Bob"}]})
    first = compile_memories_from_episodes([ep])
    second = compile_memories_from_episodes([ep])
    assert len(first) == len(second)
    assert [m.kind for m in first] == [m.kind for m in second]
