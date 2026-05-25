"""Heuristic detector registry for auto-labeling (v0.9, issue #158).

Each detector is a small dataclass pairing a label string with a pure
``detect(text) -> bool`` predicate. The detectors here aim for high
precision over high recall: a false positive surfaces a noisy
suggestion in the admin UI (annoying), while a false promotion into
``sensitivity_labels`` requires an explicit operator click (safe).

Detector inventory (v0.9 first wave):

  * ``pii.email``   — RFC-5322-ish email regex
  * ``pii.phone``   — E.164 + common US/EU loose forms
  * ``financial.card`` — 13–19 digit groups, validated by Luhn checksum
  * ``secret.token`` — common high-entropy / known-prefix tokens
                       (aws, google, github, openai, slack, bearer JWT)

Each detector's ``detect`` is a pure function — no I/O, no global
state — so they're trivially unit-testable.

Adding a new detector: define a dataclass instance below, append to
``DETECTORS``, and add a unit test in
``tests/test_auto_labeling.py``. The label MUST follow the
``<category>.<specific>`` schema; the admin endpoint's filter UI
groups suggestions by category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:
    """A named, label-emitting predicate over memory text."""

    label: str
    description: str
    detect: "callable[[str], bool]"  # type: ignore[valid-type]


# ─── pii.email ────────────────────────────────────────────────────────
# Conservative: requires a TLD with ≥2 letters, no spaces inside the
# local-part. Matches the common "name@host.tld" shape that operators
# would expect to be flagged. We accept the occasional false negative
# on weird-but-valid RFC-5322 addresses; the goal is precision.

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


def _detect_email(text: str) -> bool:
    return bool(_EMAIL_RE.search(text))


# ─── pii.phone ────────────────────────────────────────────────────────
# Two shapes:
#   - E.164 / international:    +<digits> (10–15 digits total, optional
#                               spaces/dashes/dots between groups).
#   - Loose national:           10 digits in a row, possibly grouped as
#                               (NNN) NNN-NNNN or NNN-NNN-NNNN.
# The regexes are anchored on word boundaries to keep them from firing
# on bare 10-digit identifiers like timestamps. We additionally require
# at least one separator OR the leading `+` so a raw `1234567890` in a
# UUID-ish slug doesn't trip the detector.

_PHONE_INTL_RE = re.compile(r"\+\d[\d\s().\-]{8,18}\d")
_PHONE_US_RE = re.compile(r"\b\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b")


def _detect_phone(text: str) -> bool:
    if _PHONE_INTL_RE.search(text):
        return True
    return bool(_PHONE_US_RE.search(text))


# ─── financial.card ───────────────────────────────────────────────────
# We require:
#   1. A run of 13–19 digits (allowing single spaces or single dashes
#      between groups of digits).
#   2. The digit-only form passes the Luhn checksum.
# Luhn alone gives a ~10% false-positive rate on random digits; the
# length window + group-separator pattern keeps random IDs from
# tripping. We still bias toward false-negatives over false-positives.

_CARD_RE = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")


def _luhn_ok(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if d < 0 or d > 9:
            return False
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _detect_card(text: str) -> bool:
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"[ \-]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


# ─── secret.token ─────────────────────────────────────────────────────
# Pattern-based: known prefixes from major providers, plus a generic
# bearer-JWT shape. We deliberately do NOT do entropy-based detection
# in v0.9 — it produces too many false positives on hashes / UUIDs /
# git SHAs that an operator would not consider secrets. The
# precision-first approach means we miss bespoke tokens, but operators
# can manually label those via the SDK.

_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # AWS access key ID
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # GitHub fine-grained / classic PATs and OAuth tokens
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    # OpenAI API keys (sk- prefix, ≥40 chars)
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    # Google API keys
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    # Slack tokens (xoxa/b/p/r/s + dash form)
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),
    # Bearer JWT (three base64url segments). The middle/last segments
    # require ≥16 chars so a random "a.b.c" doesn't trip the pattern.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b"),
)


def _detect_token(text: str) -> bool:
    return any(p.search(text) for p in _TOKEN_PATTERNS)


# ─── Registry ─────────────────────────────────────────────────────────
# Order is stable but irrelevant for correctness — the pipeline
# de-duplicates and sorts. Tests assert on this tuple directly to
# guard against accidental detector deletion in refactors.

DETECTORS: tuple[Detector, ...] = (
    Detector(
        label="pii.email",
        description="Email address (RFC-5322-ish pattern).",
        detect=_detect_email,
    ),
    Detector(
        label="pii.phone",
        description="Phone number (E.164 or grouped national form).",
        detect=_detect_phone,
    ),
    Detector(
        label="financial.card",
        description="Credit-card-shaped 13–19 digit run passing Luhn.",
        detect=_detect_card,
    ),
    Detector(
        label="secret.token",
        description="Known-provider API key or bearer JWT.",
        detect=_detect_token,
    ),
)


# Public utility — used by docs / admin UI as the source of truth for
# the v0.9 label catalogue. Kept as a function so callers can't mutate
# the tuple.
def label_catalogue() -> list[dict[str, str]]:
    return [{"label": d.label, "description": d.description} for d in DETECTORS]


__all__ = ["DETECTORS", "Detector", "label_catalogue"]
