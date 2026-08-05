"""Every module `server/` imports at runtime must actually be installed.

Imports written inside a function body are only executed when that code path
runs, so an undeclared dependency there survives startup, the test suite and
the container smoke test, and only fails when a user hits the endpoint. That is
exactly how `numpy` broke: it reached us transitively through pgvector, and
pgvector 0.5.0 dropped it, leaving two admin endpoints raising
ModuleNotFoundError in a released image.

This walks the AST of `server/` and asserts every third-party module imported
inside a function is importable, so the next dependency that silently
disappears fails here instead of in production.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVER = _REPO_ROOT / "server"

# Modules that are genuinely optional: they ship in an extra and every call
# site guards the import (a `_HAS_*` flag or try/except ImportError), so the
# server runs correctly without them.
_OPTIONAL = {
    "litellm",        # `llm` extra
    "opentelemetry",  # `otel` extra, guarded by _HAS_OTEL in server/core/tracing.py
}


def _function_local_imports() -> set[tuple[str, str, int]]:
    """Return (module, path, lineno) for every import inside a function body."""
    found: set[tuple[str, str, int]] = set()

    for path in sorted(_SERVER.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    names = [alias.name for alias in inner.names]
                elif isinstance(inner, ast.ImportFrom):
                    # Relative imports resolve within the package, not to a
                    # distribution, so they cannot be a missing dependency.
                    if inner.level or not inner.module:
                        continue
                    names = [inner.module]
                else:
                    continue

                for name in names:
                    top = name.split(".")[0]
                    if top in sys.stdlib_module_names or top == "server":
                        continue
                    rel = path.relative_to(_REPO_ROOT).as_posix()
                    found.add((top, rel, inner.lineno))

    return found


def test_function_local_imports_are_installed():
    missing = sorted(
        (module, path, lineno)
        for module, path, lineno in _function_local_imports()
        if module not in _OPTIONAL and importlib.util.find_spec(module) is None
    )

    assert not missing, "Undeclared runtime dependencies:\n" + "\n".join(
        f"  {path}:{lineno} imports {module!r}, which is not installed"
        for module, path, lineno in missing
    )


def test_directly_imported_packages_are_declared():
    """Guard the two dependencies that were previously only transitive."""
    for module in ("numpy", "httpx"):
        assert importlib.util.find_spec(module) is not None, (
            f"{module} is imported by server/ but is not installed; "
            "it must stay declared in pyproject.toml"
        )
