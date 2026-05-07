#!/usr/bin/env python
"""Bump or verify the project version across all user-facing surfaces.

`pyproject.toml` is the source of truth. The server reads its version
dynamically via `importlib.metadata`, so it never needs editing. Markdown
and YAML cannot read package metadata at render time, so this script keeps
the README status lines and issue-template examples in lockstep.

Usage:
    python scripts/bump_version.py 0.8.0     # rewrite pyproject + docs
    python scripts/bump_version.py --check   # exit 1 if anything drifted

Exit codes:
    0 — bumped successfully, or all targets already match pyproject
    1 — drift detected (--check) or invalid arguments
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
PYPROJECT_VERSION_RE = re.compile(r'^version = "(?P<version>[^"]+)"', re.MULTILINE)


@dataclass
class Target:
    """A single anchored substitution in a tracked file."""

    path: Path
    # Format strings substituted with `version=...`. The literal `{version}`
    # placeholder is the only field interpolated; everything else is matched
    # verbatim, so historical refs like `(v0.5)` and demo content are safe.
    pattern: str
    template: str

    def expected(self, version: str) -> str:
        return self.template.format(version=version)

    def stale(self, version: str) -> bool:
        return self.expected(version) not in self.path.read_text(encoding="utf-8")


def _readme_targets() -> list[Target]:
    return [
        Target(
            path=ROOT / "README.md",
            pattern=r"> \*\*Status:\*\* v(?P<version>\S+) — actively developed\.",
            template="> **Status:** v{version} — actively developed.",
        ),
        Target(
            path=ROOT / "README.md",
            pattern=r"Statewave is in active development \(v(?P<version>[^)]+)\)\. Honest status:",
            template="Statewave is in active development (v{version}). Honest status:",
        ),
    ]


def _issue_template_targets() -> list[Target]:
    templates = ROOT / ".github" / "ISSUE_TEMPLATE"
    return [
        Target(
            path=templates / "operator_issue.yml",
            pattern=r"Statewave version: \[e\.g\., (?P<version>[^\]]+)\]",
            template="Statewave version: [e.g., {version}]",
        ),
        Target(
            path=templates / "sdk_issue.yml",
            pattern=r"SDK version: \[e\.g\., (?P<version>[^\]]+)\]",
            template="SDK version: [e.g., {version}]",
        ),
        Target(
            path=templates / "bug_report.yml",
            pattern=r"Statewave version: \[e\.g\., (?P<version>[^\]]+)\]",
            template="Statewave version: [e.g., {version}]",
        ),
    ]


def all_targets() -> list[Target]:
    return _readme_targets() + _issue_template_targets()


def read_pyproject_version() -> str:
    match = PYPROJECT_VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError('could not locate `version = "…"` in pyproject.toml')
    return match["version"]


def write_pyproject_version(new: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    updated, count = PYPROJECT_VERSION_RE.subn(f'version = "{new}"', text, count=1)
    if count != 1:
        raise RuntimeError('could not locate `version = "…"` in pyproject.toml')
    PYPROJECT.write_text(updated, encoding="utf-8")


def apply_target(target: Target, new: str) -> bool:
    text = target.path.read_text(encoding="utf-8")
    pattern = re.compile(target.pattern)
    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            f"pattern not found in {target.path.relative_to(ROOT)}: {target.pattern!r}"
        )
    if match.group("version") == new:
        return False
    updated = pattern.sub(target.expected(new), text, count=1)
    target.path.write_text(updated, encoding="utf-8")
    return True


def cmd_bump(new: str) -> int:
    if not SEMVER_RE.match(new):
        print(f"error: {new!r} is not a valid version (expected X.Y.Z)", file=sys.stderr)
        return 1

    current = read_pyproject_version()
    if current == new:
        print(f"pyproject.toml already at {new}; checking docs are in sync…")
    else:
        write_pyproject_version(new)
        print(f"pyproject.toml: {current} → {new}")

    changed = 0
    for target in all_targets():
        if apply_target(target, new):
            print(f"  updated {target.path.relative_to(ROOT)}")
            changed += 1
    if changed == 0:
        print("  (docs already in sync)")
    return 0


def cmd_check() -> int:
    version = read_pyproject_version()
    drifted = [t for t in all_targets() if t.stale(version)]
    if not drifted:
        print(f"all version references match pyproject.toml ({version})")
        return 0
    print(f"version drift detected — pyproject.toml is {version}, but:", file=sys.stderr)
    for t in drifted:
        print(f"  - {t.path.relative_to(ROOT)} does not contain expected text", file=sys.stderr)
    print(f"\nfix: python scripts/bump_version.py {version}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("version", nargs="?", help="new version, e.g. 0.8.0")
    group.add_argument("--check", action="store_true", help="verify only, do not edit")
    args = parser.parse_args()

    if args.check:
        return cmd_check()
    return cmd_bump(args.version)


if __name__ == "__main__":
    raise SystemExit(main())
