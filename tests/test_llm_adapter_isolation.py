"""Static guardrail: only `server.services.llm` may import LiteLLM or
provider SDKs directly.

All other Statewave modules must route LLM/embedding calls through the
central adapter. This protects the abstraction so:

  * Provider swaps stay single-edit (config only, no scattered SDK calls).
  * Timeouts, retries, and error mapping stay uniform.
  * Tests can mock the adapter rather than the SDK.

If this test fails, you've added an `import litellm`, `import openai`,
`from anthropic import …`, etc. to a file that should call
`server.services.llm` instead.

Detection is AST-based — text-grep would false-positive on docstrings.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Modules whose direct import outside the adapter is forbidden.
# LiteLLM is the in-tree abstraction; provider SDKs would bypass it.
FORBIDDEN_IMPORT_ROOTS = {
    "litellm",
    "openai",
    "anthropic",
    "google.generativeai",
    "cohere",
    "mistralai",
    "voyageai",
}

# Files allowed to import from the forbidden list. The adapter itself
# obviously needs LiteLLM; pyproject and the like don't ship runtime code.
ADAPTER_FILE = "server/services/llm.py"
ALLOWED_RELATIVE_PATHS = {ADAPTER_FILE}

# Roots to scan — only Statewave runtime code under `server/`. Tests are
# allowed to import litellm directly (they may need to construct a stub
# response object or skip on absence).
SCAN_ROOTS = ("server",)


def _module_root(name: str) -> str:
    """First dotted segment of a module path."""
    return name.split(".", 1)[0]


def _matches_forbidden(module: str) -> bool:
    if module in FORBIDDEN_IMPORT_ROOTS:
        return True
    # google.generativeai — match parent path too
    return _module_root(module) in {_module_root(f) for f in FORBIDDEN_IMPORT_ROOTS} and any(
        module == f or module.startswith(f + ".") for f in FORBIDDEN_IMPORT_ROOTS
    )


def _collect_imports(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.append(node.module)
    return found


def test_only_central_adapter_imports_provider_sdks():
    repo_root = Path(__file__).resolve().parent.parent
    violations: list[str] = []

    for root in SCAN_ROOTS:
        for py_file in (repo_root / root).rglob("*.py"):
            rel = py_file.relative_to(repo_root).as_posix()
            if rel in ALLOWED_RELATIVE_PATHS:
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover
                continue
            for module_name in _collect_imports(tree):
                if _matches_forbidden(module_name):
                    violations.append(f"{rel}: imports `{module_name}`")

    assert not violations, (
        "Direct provider SDK / LiteLLM imports outside "
        f"`{ADAPTER_FILE}` are forbidden — route through the central "
        "adapter instead. Violations:\n  " + "\n  ".join(violations)
    )
