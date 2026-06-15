"""Unit tests for the LIKE-pattern escape helper used by admin search endpoints.

SQL LIKE treats ``%`` (any sequence) and ``_`` (any single character) as
metacharacters.  User-supplied search strings must have those characters
escaped before they are interpolated into a LIKE pattern, otherwise a search
for ``user_123`` also matches ``userX123``, and ``100%`` matches any string
starting with ``100``.  Likewise, the hardcoded internal-subject prefix
filters (``_snapshot/``, ``_bootstrap_tmp/``) must escape the leading ``_``
so they do not accidentally exclude user subjects whose IDs happen to match
the wildcard (e.g. ``Xsnapshot/foo``).

The helper is a pure-Python function — no DB, no HTTP client required.
"""

from __future__ import annotations

from server.api.admin import _like_escape


def test_percent_is_escaped():
    assert _like_escape("100%") == "100\\%"


def test_underscore_is_escaped():
    assert _like_escape("user_123") == "user\\_123"


def test_backslash_is_escaped_first():
    """Backslash must be escaped before the others so that a literal
    backslash in the input does not turn into an escape prefix for the
    subsequent ``%`` / ``_`` substitution."""
    assert _like_escape("path\\file") == "path\\\\file"


def test_combined_metacharacters():
    assert _like_escape("a%b_c\\d") == "a\\%b\\_c\\\\d"


def test_normal_text_is_unchanged():
    assert _like_escape("hello world") == "hello world"


def test_empty_string_is_unchanged():
    assert _like_escape("") == ""
