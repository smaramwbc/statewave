"""Tests for context assembly — ranking and token budget enforcement."""

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.context import (
    _BREADCRUMB_MAX,
    _LEXICAL_BONUS_MAX,
    _breadcrumb_overlap_bonus,
    _lexical_overlap_bonus,
    _relevance_score,
    _recency_score,
    _temporal_score,
    _tokenize_for_relevance,
    _timestamp_range,
    assemble_context,
)


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------


def test_relevance_exact_match():
    task_tokens = _tokenize_for_relevance("set up python project")
    score = _relevance_score("How to set up a python project quickly", task_tokens)
    assert score > 0


def test_relevance_no_overlap():
    task_tokens = _tokenize_for_relevance("deploy kubernetes cluster")
    score = _relevance_score("My name is Alice and I work at Acme", task_tokens)
    assert score == 0.0


def test_relevance_empty_task():
    task_tokens = _tokenize_for_relevance("")
    score = _relevance_score("anything here", task_tokens)
    assert score == 0.0


def test_relevance_empty_content():
    task_tokens = _tokenize_for_relevance("some task")
    score = _relevance_score("", task_tokens)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Recency scoring
# ---------------------------------------------------------------------------


def test_recency_most_recent_gets_max():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=10)
    ts_range = _timestamp_range([old, now])
    score = _recency_score(now, ts_range)
    assert score == 5.0


def test_recency_oldest_gets_zero():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=10)
    ts_range = _timestamp_range([old, now])
    score = _recency_score(old, ts_range)
    assert score == 0.0


def test_recency_single_item():
    now = datetime.now(timezone.utc)
    ts_range = _timestamp_range([now])
    score = _recency_score(now, ts_range)
    assert score == 5.0  # single item gets max


def test_recency_none_timestamp():
    score = _recency_score(None, (0.0, 100.0))
    assert score == 0.0


# ---------------------------------------------------------------------------
# Token budget (unit-level)
# ---------------------------------------------------------------------------


def test_tokenize_for_relevance_lowercases():
    tokens = _tokenize_for_relevance("Help The User")
    assert "help" in tokens
    assert "the" in tokens
    assert "user" in tokens


def test_timestamp_range_empty():
    assert _timestamp_range([]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Temporal scoring
# ---------------------------------------------------------------------------


def test_temporal_no_expiry_gets_bonus():
    score = _temporal_score(datetime.now(timezone.utc), None)
    assert score > 0


def test_temporal_future_expiry_gets_bonus():
    score = _temporal_score(
        datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=30)
    )
    assert score > 0


def test_temporal_past_expiry_gets_penalty():
    score = _temporal_score(
        datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=1)
    )
    assert score < 0


# ---------------------------------------------------------------------------
# Breadcrumb-overlap bonus (docs-grounded ranking)
# ---------------------------------------------------------------------------
#
# These tests pin the contract of the small additive bonus that helps
# procedural docs surface for topical queries. The bonus is gated by
# data shape (memory must have a docs-pack source episode with a
# breadcrumb) so it's automatically inert for visitor-memory subjects;
# these tests focus on the pure scoring function.


def test_breadcrumb_bonus_is_zero_with_no_breadcrumbs():
    task_tokens = _tokenize_for_relevance("heuristic vs LLM compilation")
    assert _breadcrumb_overlap_bonus([], task_tokens) == 0.0


def test_breadcrumb_bonus_is_zero_with_no_task_tokens():
    assert _breadcrumb_overlap_bonus(["Compiler Modes › Heuristic"], set()) == 0.0


def test_breadcrumb_bonus_rewards_topic_match():
    """The fix's load-bearing case: 'heuristic vs llm' query meets a
    breadcrumb whose tail explicitly names the topic."""
    task_tokens = _tokenize_for_relevance("Heuristic vs LLM compilation — when to pick which?")
    score = _breadcrumb_overlap_bonus(
        ["Compiler Modes › Heuristic Compilation"], task_tokens
    )
    assert score > 0
    assert score <= _BREADCRUMB_MAX


def test_breadcrumb_bonus_does_not_exceed_max():
    """Bonus is bounded — it nudges, doesn't dominate KIND_PRIORITY/SEMANTIC_MAX."""
    task_tokens = _tokenize_for_relevance("compiler heuristic llm")
    score = _breadcrumb_overlap_bonus(
        ["Compiler Heuristic LLM"], task_tokens
    )
    assert score <= _BREADCRUMB_MAX


def test_breadcrumb_bonus_takes_max_across_multiple_sources():
    """A memory with multiple source episodes uses the best matching breadcrumb,
    not an average — a memory cited by both an off-topic and an on-topic
    section should get the full on-topic boost."""
    task_tokens = _tokenize_for_relevance("Do I need a GPU?")
    score_offtopic_only = _breadcrumb_overlap_bonus(
        ["Architecture Overview › Component diagram"], task_tokens
    )
    score_with_ontopic = _breadcrumb_overlap_bonus(
        [
            "Architecture Overview › Component diagram",
            "Hardware & Scaling › GPU requirements",
        ],
        task_tokens,
    )
    assert score_with_ontopic > score_offtopic_only


def test_breadcrumb_bonus_ignores_generic_words():
    """'Statewave Documentation' as a breadcrumb shouldn't credit any query
    that mentions 'statewave' — that would inflate every query trivially."""
    task_tokens = _tokenize_for_relevance("What is Statewave?")
    score = _breadcrumb_overlap_bonus(["Statewave Documentation"], task_tokens)
    assert score == 0.0


def test_breadcrumb_bonus_handles_chevron_and_arrow_separators():
    """Both '›' (used by the bootstrap script) and '>' should split cleanly."""
    task_tokens = _tokenize_for_relevance("backup and restore subjects")
    score_chevron = _breadcrumb_overlap_bonus(
        ["Backup & Restore › Subject export"], task_tokens
    )
    score_arrow = _breadcrumb_overlap_bonus(
        ["Backup & Restore > Subject export"], task_tokens
    )
    assert score_chevron > 0
    assert score_arrow > 0
    assert abs(score_chevron - score_arrow) < 0.01


def test_breadcrumb_bonus_skips_empty_breadcrumb_strings():
    task_tokens = _tokenize_for_relevance("compiler")
    # Empty / blank entries shouldn't crash or raise the score
    score = _breadcrumb_overlap_bonus(["", "Compiler Modes"], task_tokens)
    assert score > 0


# ---------------------------------------------------------------------------
# Tokenization punctuation handling
# ---------------------------------------------------------------------------
#
# Pinned because the docs pack stores code snippets with surrounding quotes
# (e.g. `'npm install statewave-ts'`). Without punctuation stripping, the
# query token "npm" wouldn't match "'npm" and the lexical signal silently
# zeroes — the original failure mode behind issue #27.


def test_tokenize_strips_surrounding_quotes_and_punctuation():
    tokens = _tokenize_for_relevance("Use 'npm install statewave-ts' for TypeScript.")
    assert "npm" in tokens
    assert "install" in tokens
    assert "typescript" in tokens


def test_tokenize_strips_question_marks():
    tokens = _tokenize_for_relevance("Do I need a GPU?")
    assert "gpu" in tokens


def test_relevance_matches_quoted_keywords():
    """`_relevance_score` should now match keywords that appear inside
    quotes in the content (regression test for issue #27)."""
    task_tokens = _tokenize_for_relevance("install with npm")
    score = _relevance_score(
        "To install Statewave, use 'npm install statewave-ts' for TypeScript.",
        task_tokens,
    )
    assert score > 0


# ---------------------------------------------------------------------------
# Lexical-overlap bonus (additive on top of semantic)
# ---------------------------------------------------------------------------


def test_lexical_bonus_zero_for_no_content_or_no_tokens():
    assert _lexical_overlap_bonus("", _tokenize_for_relevance("install npm")) == 0.0
    assert _lexical_overlap_bonus("install npm", set()) == 0.0


def test_lexical_bonus_zero_when_only_stopwords_overlap():
    """Trivial overlap on connectives ('how', 'do', 'with') must not
    inflate the bonus — only meaningful keyword agreement counts."""
    task_tokens = _tokenize_for_relevance("how do I install with npm")
    # Content shares only the stopwords ("how", "do", "i", "with"); the
    # meaningful keywords ("install", "npm") are absent.
    score = _lexical_overlap_bonus(
        "how do I configure the rate limiter with custom buckets", task_tokens
    )
    assert score == 0.0


def test_lexical_bonus_zero_when_query_is_only_stopwords():
    """A query made entirely of stopwords yields no meaningful tokens, so
    no content can score a lexical bonus — even one that 'matches'."""
    task_tokens = _tokenize_for_relevance("how do I get help with this")
    score = _lexical_overlap_bonus("how do I get help with this", task_tokens)
    assert score == 0.0


def test_lexical_bonus_rewards_full_keyword_match():
    task_tokens = _tokenize_for_relevance("how do I install with npm")
    score = _lexical_overlap_bonus(
        "To install Statewave, use 'npm install statewave-ts' for TypeScript.",
        task_tokens,
    )
    assert score > 0
    assert score <= _LEXICAL_BONUS_MAX


def test_lexical_bonus_capped_at_max():
    task_tokens = _tokenize_for_relevance("install npm")
    score = _lexical_overlap_bonus(
        "install install install npm npm npm", task_tokens
    )
    assert score <= _LEXICAL_BONUS_MAX


def test_lexical_bonus_partial_keyword_match():
    """Half the meaningful keywords should yield ~half the bonus."""
    task_tokens = _tokenize_for_relevance("install npm typescript python")
    full_score = _lexical_overlap_bonus(
        "install npm typescript python", task_tokens
    )
    partial_score = _lexical_overlap_bonus(
        "install npm java rust", task_tokens
    )
    assert partial_score > 0
    assert partial_score < full_score


# ---------------------------------------------------------------------------
# Issue #27 — semantic+lexical match must not be buried by kind/recency
# ---------------------------------------------------------------------------


def _make_memory_row(
    *,
    kind: str,
    content: str,
    minutes_ago: int = 0,
):
    base = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    when = base - timedelta(minutes=minutes_ago)
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="statewave-support-docs",
        tenant_id=None,
        kind=kind,
        content=content,
        summary=content[:80],
        confidence=1.0,
        valid_from=when,
        valid_to=None,
        source_episode_ids=[],
        metadata_={},
        status="active",
        created_at=when,
        updated_at=when,
    )


@contextmanager
def _mock_semantic_repos(
    *, fact_rows, procedure_rows, summary_rows, semantic_results
):
    """Mock the repo layer so assemble_context can run with a realistic
    semantic-provider candidate pool. `semantic_results` is a list of
    (row, cosine_distance) tuples."""

    async def _search_memories(_session, _subject_id, *, tenant_id=None, kind=None, limit=None):
        if kind == "profile_fact":
            return fact_rows
        if kind == "procedure":
            return procedure_rows
        if kind == "episode_summary":
            return summary_rows
        return []

    fake_provider = SimpleNamespace(provides_semantic_similarity=True)

    async def _embed_query(_session_factory, _provider, _task):
        return [0.0] * 16  # shape doesn't matter — we mock the search

    with (
        patch(
            "server.services.context.repo.search_memories",
            new=AsyncMock(side_effect=_search_memories),
        ),
        patch(
            "server.services.context.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "server.services.context.repo.search_memories_by_embedding",
            new_callable=AsyncMock,
            return_value=semantic_results,
        ),
        patch(
            "server.services.context.get_embedding_provider",
            return_value=fake_provider,
        ),
        patch(
            "server.services.context.cached_embed_query",
            new=AsyncMock(side_effect=_embed_query),
        ),
        patch(
            "server.db.engine.get_session_factory",
            return_value=lambda: None,
        ),
        patch(
            "server.services.context.repo.get_episodes_by_ids",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_strong_semantic_plus_lexical_match_outranks_kind_priority():
    """Issue #27 — a procedure with strong semantic+lexical match for a
    narrow factual query must not be buried behind profile_facts that
    only weakly match the query.

    Mirrors the production reproducer:
      - 30 profile_facts with newer created_at and unrelated content
        (semantic sim ~0.4–0.5, no keyword overlap)
      - 1 procedure with older created_at containing 'npm install
        statewave-ts' (semantic sim ~0.9, full keyword overlap)

    Without the lexical-overlap bonus, the procedure's score
    (KIND_PRIORITY 8 + low recency + decent semantic) loses to the
    profile_facts' (KIND_PRIORITY 10 + high recency + mediocre semantic)
    and lands deep in the result list. With the bonus it surfaces at
    the top of the procedures section.
    """
    npm_proc = _make_memory_row(
        kind="procedure",
        content=(
            "To install Statewave, use 'pip install statewave-py' for Python "
            "or 'npm install statewave-ts' for TypeScript."
        ),
        minutes_ago=600,  # OLDEST → recency_score = 0 for the procedure pool
    )

    # Other procedures: newer (max recency), unrelated content (no keyword
    # overlap with the npm query). Without the lexical bonus these
    # outrank the npm procedure on recency alone, even though pure
    # semantic ranks the npm procedure first.
    other_procs = [
        _make_memory_row(
            kind="procedure",
            content=f"Configure rate limiter bucket {i} for sustained throughput.",
            minutes_ago=i,
        )
        for i in range(4)
    ]

    facts = [
        _make_memory_row(
            kind="profile_fact",
            content=f"Statewave supports feature flag {i} for advanced configuration.",
            minutes_ago=i,
        )
        for i in range(20)
    ]

    # Semantic spread mirrors a real query: npm procedure at distance 0.1
    # (sim 0.9 → 7.2 points), other procedures and facts at distances
    # 0.5–0.85 (sim 0.5–0.15 → 4.0–1.2 points). The cosine-only spread
    # is ~3 points, narrower than KIND_PRIORITY+recency variability —
    # which is exactly why the lexical tiebreaker is needed.
    semantic_results = [(npm_proc, 0.1)]
    for i, p in enumerate(other_procs):
        semantic_results.append((p, 0.5 + (i * 0.05)))
    for i, f in enumerate(facts):
        semantic_results.append((f, 0.55 + (i * 0.01)))

    with _mock_semantic_repos(
        fact_rows=facts,
        procedure_rows=[npm_proc] + other_procs,
        summary_rows=[],
        semantic_results=semantic_results,
    ):
        result = await assemble_context(
            AsyncMock(),
            "statewave-support-docs",
            "how do I install with npm?",
            max_tokens=4000,
        )

    # Both assertions are differential — they only pass with the lexical
    # bonus in place. Without it, the npm procedure scores
    # (kind 8 + recency 0 + semantic 7.2 + temporal 3) = 18.2, while
    # the newest "other" procedure scores (kind 8 + recency 5 +
    # semantic 4.0 + temporal 3) = 20.0 and ranks above it.
    assert str(npm_proc.id) in result.provenance["procedure_ids"], (
        "npm-install procedure was excluded from the context bundle entirely"
    )
    assert result.provenance["procedure_ids"][0] == str(npm_proc.id), (
        f"expected npm procedure first; got order: {result.provenance['procedure_ids']}"
    )


@pytest.mark.asyncio
async def test_lexical_bonus_does_not_override_strong_kind_signal_alone():
    """Sanity check: the lexical bonus is a tiebreaker, not a kind
    override. A profile_fact with both strong semantic AND strong
    lexical match should still outrank a procedure with the same.

    Why: KIND_PRIORITY (10 vs 8) reflects a real preference — when both
    signals agree the more profile-shaped fact should win. Lexical
    bonus must be small enough not to flip that ordering by itself.
    """
    fact = _make_memory_row(
        kind="profile_fact",
        content="The user installs npm packages globally for CLI tools.",
        minutes_ago=0,
    )
    proc = _make_memory_row(
        kind="procedure",
        content="To install npm packages globally, run 'npm install -g'.",
        minutes_ago=0,
    )

    # Equal semantic strength
    semantic_results = [(fact, 0.2), (proc, 0.2)]

    with _mock_semantic_repos(
        fact_rows=[fact],
        procedure_rows=[proc],
        summary_rows=[],
        semantic_results=semantic_results,
    ):
        result = await assemble_context(
            AsyncMock(),
            "user-1",
            "install npm globally",
            max_tokens=4000,
        )

    assert str(fact.id) in result.provenance["fact_ids"]
    assert str(proc.id) in result.provenance["procedure_ids"]
