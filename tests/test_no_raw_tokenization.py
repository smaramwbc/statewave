"""Fitness function: raw ``.lower().split()`` whitespace tokenization for lexical
comparison must go through ``server.services.tokenization``.

Raw splitting leaves punctuation welded onto tokens (``'npm`` won't match
``npm``; ``"Stripe."`` won't match ``"Stripe"``) — the silent-overlap bug class
of #198/#199. Only the shared tokenizer module may contain the pattern; any new
occurrence elsewhere fails CI and should call ``tokenize()`` instead.
"""

from __future__ import annotations

import pathlib

_SERVER = pathlib.Path(__file__).resolve().parents[1] / "server"
_ALLOWED = {_SERVER / "services" / "tokenization.py"}


def test_no_raw_lower_split_outside_shared_tokenizer():
    offenders: list[str] = []
    for path in _SERVER.rglob("*.py"):
        if path in _ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if ".lower().split()" in line:
                offenders.append(f"{path.relative_to(_SERVER.parent)}:{lineno}")
    assert not offenders, (
        "Use server.services.tokenization.tokenize() instead of raw "
        ".lower().split() for lexical comparison: " + "; ".join(offenders)
    )
