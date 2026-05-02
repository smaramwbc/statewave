"""Statewave docs → episodes loader.

Pure (no I/O outside of file reads) — turns the curated `statewave-docs`
markdown corpus into a list of section-level units that
`bootstrap_docs_pack.py` ingests as episodes.

Design: split each markdown file at H1/H2/H3 boundaries. Each section
becomes one episode with `source="statewave-docs"`, `type="doc_section"`.
Provenance carries a deterministic content hash so re-runs can detect
unchanged sections.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Curated allowlist — see statewave-docs/default-support-docs-pack.md
# for the rationale behind each inclusion/exclusion.
MANIFEST: tuple[str, ...] = (
    "README.md",
    "getting-started.md",
    "product.md",
    "why-statewave.md",
    "SECURITY.md",
    "SUPPORT.md",
    "api/v1-contract.md",
    "architecture/overview.md",
    "architecture/compiler-modes.md",
    "architecture/privacy-and-data-flow.md",
    "architecture/ranking.md",
    "architecture/repo-map.md",
    "deployment/guide.md",
    "deployment/hardware-and-scaling.md",
    "deployment/troubleshooting.md",
    "deployment/migrations.md",
    "dev/backup-restore.md",
)

SUBJECT_ID = "statewave-support-docs"
PACK_VERSION = 1
DEFAULT_BASE_URL = "https://github.com/smaramwbc/statewave-docs/blob/main"

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*```")
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9\s-]")
_SLUG_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DocSection:
    """One ingestible unit. Maps 1:1 to an episode."""

    doc_path: str
    title: str
    heading_path: tuple[str, ...]
    body: str
    content_hash: str
    url: str

    def to_episode_payload(self) -> dict:
        """Episode payload — what the heuristic compiler sees."""
        breadcrumb = " › ".join(self.heading_path) if self.heading_path else self.title
        return {
            "title": self.title,
            "heading_path": list(self.heading_path),
            "breadcrumb": breadcrumb,
            "doc_path": self.doc_path,
            "url": self.url,
            "text": self.body,
        }

    def to_episode_provenance(self, pack_version: int = PACK_VERSION) -> dict:
        return {
            "doc_path": self.doc_path,
            "content_hash": self.content_hash,
            "pack_version": pack_version,
        }


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = _SLUG_STRIP_RE.sub("", s)
    s = _SLUG_SPACE_RE.sub("-", s)
    return s


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scan_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return [(line_idx, level, title)] for every H1/H2/H3.

    Skips headings inside fenced code blocks.
    """
    out: list[tuple[int, int, str]] = []
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


def _heading_path(headings: list[tuple[int, int, str]], idx: int) -> tuple[str, ...]:
    """Walk backward from idx to build the breadcrumb of ancestor headings."""
    _, level, _ = headings[idx]
    path: list[str] = []
    needed_level = level
    for back in range(idx, -1, -1):
        _, bl_level, bl_title = headings[back]
        if bl_level <= needed_level:
            path.append(bl_title)
            needed_level = bl_level - 1
            if needed_level <= 0:
                break
    path.reverse()
    return tuple(path)


def chunk_markdown(
    text: str,
    doc_path: str,
    base_url: str = DEFAULT_BASE_URL,
) -> list[DocSection]:
    """Split markdown into sections at H1/H2/H3.

    Heading-only sections (no body before the next heading) are dropped,
    but their titles still appear in descendants' `heading_path`.
    A document with no headings becomes a single section keyed by its path.
    """
    lines = text.splitlines()
    headings = _scan_headings(lines)

    if not headings:
        body = text.strip()
        if not body:
            return []
        return [
            DocSection(
                doc_path=doc_path,
                title=doc_path,
                heading_path=(doc_path,),
                body=body,
                content_hash=_content_hash(body),
                url=f"{base_url}/{doc_path}",
            )
        ]

    sections: list[DocSection] = []
    for idx, (line_idx, _level, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_idx + 1 : end]).strip()
        if not body:
            continue

        path = _heading_path(headings, idx)
        anchor = _slugify(title)
        sections.append(
            DocSection(
                doc_path=doc_path,
                title=title,
                heading_path=path,
                body=body,
                content_hash=_content_hash(body),
                url=f"{base_url}/{doc_path}#{anchor}",
            )
        )

    return sections


def load_docs(
    docs_root: Path,
    manifest: tuple[str, ...] = MANIFEST,
    base_url: str = DEFAULT_BASE_URL,
) -> list[DocSection]:
    """Load every doc in `manifest` from `docs_root` and chunk it.

    Raises FileNotFoundError if a manifest entry is missing — callers
    surface this as a clear error so operators don't ship a partial pack.
    """
    sections: list[DocSection] = []
    missing: list[str] = []
    for rel in manifest:
        full = docs_root / rel
        if not full.exists():
            missing.append(rel)
            continue
        text = full.read_text(encoding="utf-8")
        sections.extend(chunk_markdown(text, rel, base_url=base_url))
    if missing:
        raise FileNotFoundError(
            "Manifest references docs that are not present at "
            f"{docs_root}: {missing}"
        )
    return sections
