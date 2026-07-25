from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.preflight import _status_prefix  # noqa: E402


def test_plain_status_prefixes_are_ascii():
    assert _status_prefix("error", plain=True) == "ERROR:"
    assert _status_prefix("success", plain=True) == "OK:"
    assert _status_prefix("warning", plain=True) == "WARN:"


def test_default_status_prefixes_remain_unicode_symbols():
    assert _status_prefix("error", plain=False) == "\u274c"
    assert _status_prefix("success", plain=False) == "\u2705"
    assert _status_prefix("warning", plain=False) == "\u26a0\ufe0f"