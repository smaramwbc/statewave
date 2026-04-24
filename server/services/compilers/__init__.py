"""Compiler interface and registry.

All memory compilers implement the BaseCompiler protocol.
The get_compiler() factory returns the active compiler based on config.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from server.db.tables import EpisodeRow, MemoryRow


class BaseCompiler(Protocol):
    """Protocol that all memory compilers must satisfy."""

    def compile(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Derive memory rows from a batch of episodes.

        Contract:
        - Must be deterministic for the same input.
        - Must set source_episode_ids on every produced memory.
        - Must not mutate the input episodes.
        - May return an empty list.
        """
        ...


def get_compiler() -> BaseCompiler:
    """Return the active compiler based on configuration."""
    from server.core.config import settings

    if settings.compiler_type == "heuristic":
        from server.services.compilers.heuristic import HeuristicCompiler
        return HeuristicCompiler()
    elif settings.compiler_type == "llm":
        from server.services.compilers.llm import LLMCompiler
        return LLMCompiler(
            api_key=settings.openai_api_key,
            model=settings.llm_compiler_model,
        )
    else:
        raise ValueError(f"Unknown compiler type: {settings.compiler_type}")
