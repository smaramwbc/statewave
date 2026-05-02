"""Tests for the docs → episodes chunker.

The chunker is pure (no I/O on the hot path) so these are fast unit tests.
We assert on:
  * heading-level splits and breadcrumbs
  * idempotent content hashing
  * fenced-code-block heading masking
  * dropping of heading-only sections
  * manifest loader error on missing files
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ isn't auto-importable from tests/, so put the repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.docs_loader import (  # noqa: E402
    MANIFEST,
    DocSection,
    chunk_markdown,
    load_docs,
)


def test_splits_at_h2_boundaries():
    md = """# Title

intro paragraph

## Alpha

alpha body line 1
alpha body line 2

## Beta

beta body
"""
    sections = chunk_markdown(md, "x.md")
    titles = [s.title for s in sections]
    assert titles == ["Title", "Alpha", "Beta"]
    assert "alpha body line 1" in sections[1].body
    assert "beta body" in sections[2].body
    # Title section gets only its own intro, not children
    assert "alpha" not in sections[0].body.lower()


def test_heading_path_breadcrumb():
    md = """# Top

intro

## Middle

mid body

### Leaf

leaf body
"""
    sections = chunk_markdown(md, "x.md")
    by_title = {s.title: s for s in sections}
    assert by_title["Top"].heading_path == ("Top",)
    assert by_title["Middle"].heading_path == ("Top", "Middle")
    assert by_title["Leaf"].heading_path == ("Top", "Middle", "Leaf")


def test_heading_only_sections_are_dropped():
    md = """# Doc

## Empty Parent

### Child

real content here
"""
    sections = chunk_markdown(md, "x.md")
    titles = [s.title for s in sections]
    # "Empty Parent" has no body of its own — must be skipped
    assert "Empty Parent" not in titles
    assert "Child" in titles
    # But the breadcrumb for Child still includes Empty Parent
    child = next(s for s in sections if s.title == "Child")
    assert child.heading_path == ("Doc", "Empty Parent", "Child")


def test_fenced_code_blocks_do_not_create_sections():
    md = """## Real Section

before

```python
# This looks like a heading but is code
## Also code
```

after
"""
    sections = chunk_markdown(md, "x.md")
    assert len(sections) == 1
    assert sections[0].title == "Real Section"
    assert "## Also code" in sections[0].body


def test_content_hash_is_deterministic_and_changes_with_body():
    md1 = "# A\n\nhello\n"
    md2 = "# A\n\nhello\n"
    md3 = "# A\n\nhello world\n"

    h1 = chunk_markdown(md1, "x.md")[0].content_hash
    h2 = chunk_markdown(md2, "x.md")[0].content_hash
    h3 = chunk_markdown(md3, "x.md")[0].content_hash

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # sha256 hex


def test_url_uses_doc_path_and_anchor():
    md = "## My Section Name\n\nbody\n"
    s = chunk_markdown(md, "deployment/guide.md", base_url="https://example.com/repo")[0]
    assert s.url == "https://example.com/repo/deployment/guide.md#my-section-name"


def test_doc_with_no_headings_becomes_one_section():
    md = "Just a single paragraph of text with no headings."
    sections = chunk_markdown(md, "x.md")
    assert len(sections) == 1
    assert sections[0].body == md


def test_empty_doc_yields_no_sections():
    assert chunk_markdown("", "x.md") == []
    assert chunk_markdown("\n\n   \n", "x.md") == []


def test_to_episode_payload_carries_breadcrumb_and_url():
    md = "## Foo\n\nbar\n"
    s = chunk_markdown(md, "x.md")[0]
    payload = s.to_episode_payload()
    assert payload["title"] == "Foo"
    assert payload["heading_path"] == ["Foo"]
    assert payload["breadcrumb"] == "Foo"
    assert payload["text"] == "bar"
    assert payload["doc_path"] == "x.md"
    assert payload["url"].endswith("x.md#foo")


def test_to_episode_provenance_has_hash_and_pack_version():
    md = "## Foo\n\nbar\n"
    s = chunk_markdown(md, "x.md")[0]
    prov = s.to_episode_provenance(pack_version=7)
    assert prov["doc_path"] == "x.md"
    assert prov["content_hash"] == s.content_hash
    assert prov["pack_version"] == 7


def test_load_docs_raises_on_missing_manifest_entry(tmp_path: Path):
    # Only place one of the manifest files
    (tmp_path / "README.md").write_text("# Hi\n\nbody\n")
    with pytest.raises(FileNotFoundError) as exc_info:
        load_docs(tmp_path, manifest=("README.md", "does-not-exist.md"))
    assert "does-not-exist.md" in str(exc_info.value)


def test_load_docs_loads_full_manifest(tmp_path: Path):
    # Stub every manifest entry with a tiny doc
    for rel in MANIFEST:
        full = tmp_path / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(f"# {rel}\n\nbody for {rel}\n")
    sections = load_docs(tmp_path)
    # One section per stub doc (each has exactly one heading with body)
    assert len(sections) == len(MANIFEST)
    paths = {s.doc_path for s in sections}
    assert paths == set(MANIFEST)


def test_real_statewave_docs_corpus_loads_cleanly():
    """Smoke test against the actual sibling statewave-docs/ if present.

    Skipped when running from a context where the docs aren't checked out.
    """
    docs_root = _REPO_ROOT.parent / "statewave-docs"
    if not docs_root.is_dir():
        pytest.skip("statewave-docs sibling dir not present")
    sections = load_docs(docs_root)
    assert len(sections) > len(MANIFEST), "expected multiple sections per doc"
    # No duplicate (doc_path, heading_path) pairs
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for s in sections:
        key = (s.doc_path, s.heading_path)
        assert key not in seen, f"duplicate section key: {key}"
        seen.add(key)
    # Every section has non-empty body and a 64-char hash
    for s in sections:
        assert s.body.strip()
        assert len(s.content_hash) == 64
