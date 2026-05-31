"""Shared lexical tokenization — the single source of truth for the
edge-punctuation stripping used by relevance scoring (``context.py``) and
conflict detection (``conflicts.py``).

Keeping it in one place is a correctness guard. When the two copies drifted
apart, a trailing ``"."`` welded onto a word (``"Stripe." != "Stripe"``) and a
sentence-final ``"outage!"`` failed to match ``"outage"``, silently zeroing
word-overlap (the bug class fixed in #198/#199). A CI fitness function
(``tests/test_no_raw_tokenization.py``) keeps raw ``.lower().split()`` from
creeping back into other modules.
"""

from __future__ import annotations

# Punctuation stripped from the edges of each token before comparison.
EDGE_PUNCT = "?.,:;()[]{}'\"!"


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens with surrounding punctuation stripped.

    Without stripping, fragments like ``'npm install'`` (literal quotes from a
    markdown code snippet) tokenize as ``'npm`` and won't intersect a clean
    query token ``npm``, silently zeroing the lexical signal.
    """
    if not text:
        return set()
    return {
        stripped
        for stripped in (token.strip(EDGE_PUNCT) for token in text.lower().split())
        if stripped
    }
