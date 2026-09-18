"""Shared lexical tokenization — the single source of truth for the token
views that relevance scoring (``context.py``), conflict detection
(``conflicts.py``) and compile-time dedup (``dedup.py``) compare on.

Keeping it in one place is a correctness guard. When the two copies drifted
apart, a trailing ``"."`` welded onto a word (``"Stripe." != "Stripe"``) and a
sentence-final ``"outage!"`` failed to match ``"outage"``, silently zeroing
word-overlap (the bug class fixed in #198/#199). A CI fitness function
(``tests/test_no_raw_tokenization.py``) keeps raw ``.lower().split()`` from
creeping back into other modules.

Three views, deliberately distinct:

* :func:`tokenize` — every word, punctuation-stripped. Jaccard overlap.
* :func:`sig_tokens` — content words only (>=3 chars, no stopwords). "Do these
  two statements assert the same things?"
* :func:`number_set` — numbers and month names. "Does a value or a date
  differ?" — the one signal a content-word comparison cannot see, because
  ``"30"`` is two characters and drops out of :func:`sig_tokens`.

The last two are used by dedup's merge gates and by the widening guard in
``conflicts.py`` (#414). They answer the same question in both places, so they
must be the same function in both places: a dedup gate that merged what the
widening guard considered a different statement would be the #198 drift again,
one layer up.
"""

from __future__ import annotations

import re

# Punctuation stripped from the edges of each token before comparison.
EDGE_PUNCT = "?.,:;()[]{}'\"!"

_NUM = re.compile(r"\d+")
_TOK = re.compile(r"[a-z0-9]+")
_MONTHS = frozenset(
    "january february march april may june july august september october "
    "november december".split()
)
# Function words ignored when comparing the "significant content" of two facts.
# Differences confined to these (or to word order / punctuation) carry no new
# assertion; a difference in ANY non-stopword token (a name, place, noun, verb)
# is treated as potentially-distinct content.
_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with from by as is are was were be "
    "been being has have had do does did this that these those it its their his "
    "her my your our we you they he she i me us them about into over under than "
    "then now during while which who whom whose what when where why how also".split()
)


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


def number_set(content: str) -> frozenset[str]:
    """Numbers + month names — the signal that keeps knowledge-update / date
    pairs distinct, so dedup never collapses a changed value into its
    predecessor and the widening guard can see a dropped ``"30 days"``."""
    low = (content or "").casefold()
    nums = set(_NUM.findall(low))
    months = {m for m in _MONTHS if m in low}
    return frozenset(nums | months)


def sig_tokens(content: str) -> frozenset[str]:
    """Significant content words: tokens >=3 chars, minus stopwords.

    Two facts may only merge when these are IDENTICAL — so any differing
    name/place/noun/verb (e.g. "Mia" vs "Mary", "hiking" vs "biking") blocks
    the merge, while word-order / stopword / punctuation differences (the
    windowing-reformat case) do not.
    """
    return frozenset(
        t for t in _TOK.findall((content or "").casefold())
        if len(t) >= 3 and t not in _STOPWORDS
    )
