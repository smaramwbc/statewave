"""The lexical tokenizer is a single source of truth (server.services.tokenization).

Relevance scoring and conflict detection must share it. When they drifted, a
trailing "." silently zeroed word-overlap (the "Stripe." != "Stripe" class of
bug, #198/#199). These tests fail if the two ever diverge again.
"""

from __future__ import annotations

from server.services import conflicts, context
from server.services.tokenization import EDGE_PUNCT, tokenize


def test_relevance_and_conflict_share_one_constant():
    assert context._RELEVANCE_PUNCT is EDGE_PUNCT
    assert conflicts._TOKEN_EDGE_PUNCT is EDGE_PUNCT


def test_relevance_and_conflict_share_one_tokenizer():
    assert context._tokenize_for_relevance is tokenize
    assert conflicts._tokenize is tokenize


def test_tokenize_strips_edge_punctuation():
    assert tokenize("Production outage!") == {"production", "outage"}
    assert tokenize("'npm install'") == {"npm", "install"}
    assert tokenize("Stripe.") == {"stripe"}


def test_tokenize_handles_empty_and_blank():
    assert tokenize("") == set()
    assert tokenize("   ") == set()
