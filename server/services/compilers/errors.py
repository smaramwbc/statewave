"""Compiler error types.

Kept in their own module so both the compilers and the API layer can import
them without pulling in a concrete compiler (and the heavy, optional LLM
dependency that comes with it).
"""

from __future__ import annotations


class CompilationError(Exception):
    """A compiler could not RUN extraction (config or provider failure).

    This is deliberately distinct from a compiler legitimately returning an
    empty list — "extracted zero memories" and "could not extract" are
    different outcomes and must be handled differently.

    Contract for callers (see server/api/memories.py):
    - When this is raised, the episodes were NOT processed. Callers MUST NOT
      mark them compiled — they must stay uncompiled so a later, correctly
      configured recompile can reprocess them. Marking them compiled would
      silently consume the source episode for a run that produced nothing.
    - The failure must be surfaced to the client (a 5xx / failed job), not
      reported as a clean ``memories_created: 0``.

    The canonical trigger is selecting the LLM compiler with no reachable
    provider credential: every provider round-trip fails, and the old
    behaviour swallowed that into an empty result that still advanced the
    episodes' compiled state.
    """
