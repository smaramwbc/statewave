"""Fitness function: pagination parameters must be bounded.

An unbounded ``limit`` lets a caller request the whole table (or, with
``limit=0`` / a negative value, get surprising empty results) — the boundary
class fixed in #208/#209. This test scans every route module and asserts each
``limit`` Query has both ``ge`` and ``le``, and each ``offset`` Query has ``ge``.
"""

from __future__ import annotations

import ast
import pathlib

_API_DIR = pathlib.Path(__file__).resolve().parents[1] / "server" / "api"

_REQUIRED = {"limit": {"ge", "le"}, "offset": {"ge"}}


def _query_keywords(default: ast.expr | None) -> set[str] | None:
    """Return the keyword-arg names of a ``Query(...)`` default, else None."""
    if isinstance(default, ast.Call) and isinstance(default.func, ast.Name):
        if default.func.id == "Query":
            return {kw.arg for kw in default.keywords if kw.arg is not None}
    return None


def _iter_param_defaults(func: ast.AsyncFunctionDef | ast.FunctionDef):
    a = func.args
    positional = a.args
    pos_defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    yield from zip(positional, pos_defaults)
    yield from zip(a.kwonlyargs, a.kw_defaults)


def _violations() -> list[str]:
    bad: list[str] = []
    for path in sorted(_API_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            for arg, default in _iter_param_defaults(node):
                required = _REQUIRED.get(arg.arg)
                if required is None:
                    continue
                kws = _query_keywords(default)
                if kws is None:
                    continue
                missing = required - kws
                if missing:
                    bad.append(f"{path.name}:{node.name}:{arg.arg} missing {sorted(missing)}")
    return bad


def test_pagination_params_are_bounded():
    bad = _violations()
    assert not bad, "Unbounded pagination params (add ge/le to Query): " + "; ".join(bad)
