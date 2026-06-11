"""Characterization of compiler OUTPUT invariants that claim emission must not
disturb: same memory texts, kinds, count, ordering, summaries, and source
episode links. Claim metadata is purely additive on top of these.

Passes on the pre-claim compiler and must keep passing after.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from server.db.tables import EpisodeRow
from server.services.compilers.heuristic import HeuristicCompiler, _extract_profile_facts


def _ep(text: str) -> EpisodeRow:
    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="test",
        type="conversation",
        payload={"text": text},
        metadata_={},
        provenance={},
        created_at=datetime.now(timezone.utc),
    )


def test_heuristic_text_kind_count_ordering_invariant():
    ep = _ep("My name is Alice Chen. I work at Globex. I live in Berlin.")
    mems = HeuristicCompiler().compile([ep])

    # one summary + three facts, in stable order
    assert [m.kind for m in mems] == [
        "episode_summary",
        "profile_fact",
        "profile_fact",
        "profile_fact",
    ]
    facts = [m.content for m in mems if m.kind == "profile_fact"]
    assert facts == ["My name is Alice Chen", "I work at Globex", "I live in Berlin"]
    # source links + summaries preserved
    for m in mems:
        assert m.source_episode_ids == [ep.id]
    assert mems[1].summary == "My name is Alice Chen"


def test_extract_profile_facts_public_signature_unchanged():
    # Still returns plain strings (group(0)), order preserved.
    facts = _extract_profile_facts("My name is Alice. I work at Globex.")
    assert facts == ["My name is Alice", "I work at Globex"]


def test_negative_statements_still_produce_no_profile_fact():
    for text in [
        "My employer is not Acme.",
        "I might work at Acme.",
        "Alice works at Acme.",
        "I previously worked at Acme.",
        "I am considering moving to Berlin.",
    ]:
        mems = HeuristicCompiler().compile([_ep(text)])
        assert all(m.kind != "profile_fact" for m in mems), text
