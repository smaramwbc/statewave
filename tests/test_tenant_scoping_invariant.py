"""Fitness function: every subject-scoped repository query must also accept a
``tenant_id``.

A query keyed only by ``subject_id`` silently reads across tenants — the bug
class fixed in #206 (conflict resolution) and #207 (compile-job polling). This
test fails CI when a *new* repository helper takes ``subject_id`` without a
``tenant_id`` parameter, so the next one can't merge unscoped.

The allowlist is a shrinking register of *known* gaps, not a license to add
more. Do not append to it to silence a finding — add the ``tenant_id`` and apply
``_tenant_filter`` instead.
"""

from __future__ import annotations

import ast
import pathlib

_REPOSITORIES = (
    pathlib.Path(__file__).resolve().parents[1] / "server" / "db" / "repositories.py"
)

# Known, accepted exceptions. The subject-health cache is keyed by a bare
# ``subject_id`` PRIMARY KEY (server/db/tables.py: SubjectHealthCacheRow), so
# isolating it needs a composite-key schema migration, not a query filter — it
# is tracked separately. Multi-tenancy is experimental and does not yet enforce
# full isolation.
_ALLOWED_UNSCOPED = {
    "get_health_cache",
    "delete_health_cache_by_subject",
}


def _subject_scoped_without_tenant() -> set[str]:
    tree = ast.parse(_REPOSITORIES.read_text())
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            args = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            if "subject_id" in args and "tenant_id" not in args:
                out.add(node.name)
    return out


def test_subject_queries_are_tenant_scoped():
    offenders = _subject_scoped_without_tenant() - _ALLOWED_UNSCOPED
    assert not offenders, (
        "These repository functions take subject_id but not tenant_id — add a "
        "tenant_id parameter and apply _tenant_filter so they cannot read across "
        f"tenants: {sorted(offenders)}"
    )


def test_allowlist_has_no_stale_entries():
    # Keep the register honest: once a function is fixed it must be removed from
    # the allowlist, or the list rots into a place new gaps hide.
    stale = _ALLOWED_UNSCOPED - _subject_scoped_without_tenant()
    assert not stale, f"Remove now-fixed entries from the allowlist: {sorted(stale)}"
