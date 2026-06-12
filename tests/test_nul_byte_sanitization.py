"""A NUL byte (U+0000) in generated content cannot be stored in a Postgres text
column — it raises CharacterNotInRepertoireError and 500s the whole compile
batch. MemoryRow strips NUL (and only NUL) at the ORM boundary so one stray byte
an LLM emits can't sink every memory in the batch.
"""

from __future__ import annotations

from server.db.tables import MemoryRow


def test_memory_content_strips_nul_byte():
    m = MemoryRow(content="before\x00after", summary="ok\x00")
    assert m.content == "beforeafter"
    assert m.summary == "ok"


def test_clean_content_is_untouched():
    text = "On 2026-06-12, added a Helm chart — fixed hook ordering. 日本語 ✓"
    m = MemoryRow(content=text, summary=text)
    assert m.content == text
    assert m.summary == text


def test_only_nul_is_removed_other_control_chars_kept():
    # Tabs/newlines are valid in Postgres text; only NUL is illegal.
    m = MemoryRow(content="line1\n\tline2\x00", summary="")
    assert m.content == "line1\n\tline2"
