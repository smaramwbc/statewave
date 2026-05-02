"""Tests for context assembly — ranking and token budget enforcement."""

from server.services.context import (
    _BREADCRUMB_MAX,
    _breadcrumb_overlap_bonus,
    _relevance_score,
    _recency_score,
    _temporal_score,
    _tokenize_for_relevance,
    _timestamp_range,
)
from datetime import datetime, timezone, timedelta


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
