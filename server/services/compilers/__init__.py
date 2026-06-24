"""Compiler interface and registry.

All memory compilers implement the BaseCompiler protocol.
The get_compiler() factory returns the active compiler based on config.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from server.db.tables import EpisodeRow, MemoryRow
from server.services.compilers.errors import CompilationError

__all__ = ["BaseCompiler", "CompilationError", "get_compiler"]


class BaseCompiler(Protocol):
    """Protocol that all memory compilers must satisfy."""

    def compile(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Derive memory rows from a batch of episodes.

        Contract:
        - Must be deterministic for the same input.
        - Must set source_episode_ids on every produced memory.
        - Must not mutate the input episodes.
        - May return an empty list when the input legitimately yields no
          memories. An empty list means "extracted nothing", which lets the
          caller mark the episodes compiled.
        - Must raise `CompilationError` when extraction could NOT run (config
          or provider failure). Never swallow such a failure into an empty
          list — that would let the caller consume episodes for a run that
          produced nothing. See `errors.CompilationError`.
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

        # Extraction model = the dedicated compile override when set, else the
        # general model. Extraction-model strength is the dominant memory-quality
        # lever (see config.litellm_compile_model), so this is the one knob to
        # raise for cloud-parity memory.
        return LLMCompiler(
            model=settings.litellm_compile_model or settings.litellm_model
        )
    else:
        raise ValueError(f"Unknown compiler type: {settings.compiler_type}")
