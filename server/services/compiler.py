"""Backward-compatibility shim — delegates to compilers.heuristic.

Import from server.services.compilers instead for new code.
"""

from __future__ import annotations

from typing import Sequence

from server.db.tables import EpisodeRow, MemoryRow
from server.services.compilers.heuristic import HeuristicCompiler

_compiler = HeuristicCompiler()


def compile_memories_from_episodes(episodes: list[EpisodeRow]) -> list[MemoryRow]:
    """Shim — calls HeuristicCompiler.compile()."""
    return _compiler.compile(episodes)
