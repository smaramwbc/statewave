"""Tests for context assembly — ranking and token budget enforcement."""

from server.services.context import (
    _relevance_score,
    _recency_score,
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
