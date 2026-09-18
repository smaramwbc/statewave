"""Fitness function: the token views used for lexical comparison are defined in
``server.services.tokenization`` and nowhere else.

Raw splitting leaves punctuation welded onto tokens (``'npm`` won't match
``npm``; ``"Stripe."`` won't match ``"Stripe"``) — the silent-overlap bug class
of #198/#199. A second copy of a token view is the same failure one layer up:
dedup's merge gates and the widening guard in ``conflicts.py`` ask the same
question about the same two views (#414), and a private copy in either module
lets them answer differently without anything failing.

Only the shared tokenizer module may contain these definitions; any new
occurrence elsewhere fails CI and should import from it instead.
"""

from __future__ import annotations

import ast
import pathlib

from server.services import conflicts, dedup, tokenization

_SERVER = pathlib.Path(__file__).resolve().parents[1] / "server"
_SHARED = _SERVER / "services" / "tokenization.py"
_ALLOWED = {_SHARED}

# Token views that must have exactly one definition. Names, not bodies: a
# re-implementation under the same name is the drift this catches, and a
# re-implementation under a different name still has to be reviewed in.
_SHARED_VIEWS = {"tokenize", "sig_tokens", "number_set"}


def test_no_raw_lower_split_outside_shared_tokenizer():
    offenders: list[str] = []
    for path in _SERVER.rglob("*.py"):
        if path in _ALLOWED:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if ".lower().split()" in line:
                offenders.append(f"{path.relative_to(_SERVER.parent)}:{lineno}")
    assert not offenders, (
        "Use server.services.tokenization.tokenize() instead of raw "
        ".lower().split() for lexical comparison: " + "; ".join(offenders)
    )


def test_the_shared_token_views_are_defined_only_once():
    offenders: list[str] = []
    for path in _SERVER.rglob("*.py"):
        if path in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.lstrip("_") in _SHARED_VIEWS:
                    offenders.append(f"{path.relative_to(_SERVER.parent)}:{node.lineno}")
    assert not offenders, (
        "Import these from server.services.tokenization instead of redefining "
        "them: " + "; ".join(offenders)
    )


def test_dedup_and_the_widening_guard_share_one_definition():
    """Not just "defined once" — actually the same object in both callers, so
    a future edit to one view cannot change dedup's gates and leave the
    resolver's guard behind (or the reverse)."""
    assert dedup.sig_tokens is tokenization.sig_tokens
    assert dedup.number_set is tokenization.number_set
    assert conflicts.sig_tokens is tokenization.sig_tokens
    assert conflicts.number_set is tokenization.number_set
