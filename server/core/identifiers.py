"""Canonical identifier contracts shared across the API surface.

`subject_id` is an opaque-but-bounded token. Many admin/v1 routes address a
subject by putting it directly in the URL path
(`/admin/subjects/{subject_id}`, `/admin/subjects/{subject_id}/memories`,
`DELETE /v1/subjects/{subject_id}`, …) using FastAPI's default
single-segment `{subject_id}` converter. A `/` in the id makes that path
ambiguous, so the route never matches and the subject becomes
**created-but-unreachable** (issue #121: the episode write succeeded, then
the admin panel and every path-param endpoint 404'd).

`server.services.memory_packs` already defined exactly this charset for the
starter-pack path; the bug was that the primary write ingress (episodes,
batch, compile, …) only length-checked. This module is the single source of
truth — every request schema and `memory_packs` import from here so the
contract can never drift apart again.

Pre-existing subjects that already contain `/` remain remediable: the admin
bulk-delete path matches by `subject_id_prefix` (an unconstrained string),
so an operator can still purge them — see the issue/PR for the steps.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import StringConstraints

# Length cap matches the historical `Field(max_length=256)` on the write
# schemas (the more permissive of the two prior limits — memory_packs used
# 128 — so unifying upward is not an additional breaking change).
SUBJECT_ID_MAX_LEN = 256

# Letters, digits, underscore, dot, dash, colon. Notably excludes `/`,
# whitespace, and other URL-significant characters (`?`, `#`, `%`, …). The
# colon is intentionally kept — the tenant convention uses it (`tenant:id`).
SUBJECT_ID_CHARSET = r"A-Za-z0-9_.\-:"
SUBJECT_ID_PATTERN = rf"^[{SUBJECT_ID_CHARSET}]{{1,{SUBJECT_ID_MAX_LEN}}}$"
SUBJECT_ID_RE = re.compile(SUBJECT_ID_PATTERN)

SUBJECT_ID_CHARSET_DESC = (
    "letters, digits, underscore, dot, dash, or colon — no '/', whitespace, "
    f"or other URL-significant characters (1–{SUBJECT_ID_MAX_LEN} chars)"
)

# Reusable pydantic type. Using StringConstraints (not a separate regex on
# top of a `str` field) means FastAPI returns a 422 with a clear,
# self-documenting message and the constraint shows up in the OpenAPI schema.
SubjectId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=SUBJECT_ID_MAX_LEN,
        pattern=rf"^[{SUBJECT_ID_CHARSET}]+$",
    ),
]


def is_valid_subject_id(value: str) -> bool:
    """True iff `value` satisfies the canonical subject-id contract."""
    return bool(value) and SUBJECT_ID_RE.match(value) is not None


# ── Session ids ────────────────────────────────────────────────────────────
# A session_id has the *exact same* problem as a subject_id: it is used as a
# URL path segment (e.g. GET /admin/subjects/{subject_id}/sessions/
# {session_id}/timeline), so a `/` makes the route unmatchable and the
# session unreachable — the #121 failure mode (tracked as #124). Same charset,
# same cap. Kept as its own name so error messages say "session_id" and the
# two can diverge later without churn.
SESSION_ID_MAX_LEN = SUBJECT_ID_MAX_LEN
SESSION_ID_RE = SUBJECT_ID_RE
SESSION_ID_CHARSET_DESC = SUBJECT_ID_CHARSET_DESC

SessionId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=SESSION_ID_MAX_LEN,
        pattern=rf"^[{SUBJECT_ID_CHARSET}]+$",
    ),
]


def is_valid_session_id(value: str) -> bool:
    """True iff `value` satisfies the canonical session-id contract."""
    return bool(value) and SESSION_ID_RE.match(value) is not None
