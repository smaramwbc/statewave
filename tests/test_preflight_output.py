from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.preflight import _glyph, _status_prefix, main  # noqa: E402
from server.services import migrations  # noqa: E402

_ERROR = migrations.MigrationStatus(error="could not connect to database")
_UP_TO_DATE = migrations.MigrationStatus(current_revision="0030_head", is_compatible=True)
_FRESH_DB = migrations.MigrationStatus(
    current_revision=None, pending_count=2, pending_revisions=["0029_one", "0030_two"]
)
_PENDING = migrations.MigrationStatus(
    current_revision="0029_one", pending_count=1, pending_revisions=["0030_two"]
)


def _run(monkeypatch, capsys, status, *argv: str) -> tuple[int, str]:
    async def fake_check(*args, **kwargs):
        return status

    monkeypatch.setattr(migrations, "check_migration_status", fake_check)
    code = asyncio.run(main(list(argv)))
    return code, capsys.readouterr().out


def test_plain_status_prefixes_are_ascii():
    assert _status_prefix("error", plain=True) == "ERROR:"
    assert _status_prefix("success", plain=True) == "OK:"
    assert _status_prefix("warning", plain=True) == "WARN:"


def test_default_status_prefixes_remain_unicode_symbols():
    assert _status_prefix("error", plain=False) == "❌"
    assert _status_prefix("success", plain=False) == "✅"
    assert _status_prefix("warning", plain=False) == "⚠️"


def test_plain_glyphs_are_ascii():
    assert _glyph("arrow", plain=True) == "->"
    assert _glyph("dash", plain=True) == "-"


def test_default_glyphs_remain_unicode_symbols():
    assert _glyph("arrow", plain=False) == "→"
    assert _glyph("dash", plain=False) == "—"


@pytest.mark.parametrize(
    "status", [_ERROR, _UP_TO_DATE, _FRESH_DB, _PENDING], ids=["error", "current", "fresh", "pending"]
)
def test_plain_output_is_ascii_on_every_branch(monkeypatch, capsys, status):
    # An operator on a cp1252 console reads this before a migration: one
    # stray symbol anywhere in the run is an encode error, not a cosmetic.
    _, out = _run(monkeypatch, capsys, status, "--plain")

    assert out.isascii(), out


def test_plain_error_line_carries_its_label_once(monkeypatch, capsys):
    code, out = _run(monkeypatch, capsys, _ERROR, "--plain")

    assert code == 1
    assert "ERROR: could not connect to database" in out
    assert out.count("ERROR:") == 1


def test_default_output_keeps_unicode_decoration(monkeypatch, capsys):
    _, out = _run(monkeypatch, capsys, _FRESH_DB)

    assert "→ 0029_one" in out
    assert "(none — fresh DB)" in out
