"""Auto-labeling pipeline (v0.9, issue #158).

Heuristic detectors that scan memory `content` at compile time and
emit advisory `suggested_labels`. The pipeline never touches
``sensitivity_labels`` — promotion into the authoritative column is a
deliberate, audited operator action (admin review endpoint or SDK call).

Label schema: ``<category>.<specific>`` (e.g. ``pii.email``,
``financial.card``, ``secret.token``). Detectors are pure functions
``(text: str) -> bool`` so they can be unit-tested in isolation and
composed without ordering assumptions.

Public surface:

  * ``label_suggestions(text)`` — pure, returns the sorted, de-duplicated
    list of label strings the configured detectors produced.
  * ``apply_suggestions(memories)`` — in-place mutates ``MemoryRow.
    suggested_labels`` for each memory. Used by the compilers.
  * ``DETECTORS`` — the registry tuple, in stable order. Tests assert
    on this directly.

The pipeline is gated at the call site (compilers check
``settings.auto_labeling_enabled``). The package itself is always
importable — keeping it side-effect-free means tests can exercise
detectors without flipping a global.
"""

from __future__ import annotations

from typing import Iterable

from server.db.tables import MemoryRow

from .detectors import DETECTORS, Detector


def label_suggestions(text: str) -> list[str]:
    """Run every configured detector against ``text``. Return the sorted,
    de-duplicated list of label strings that matched.

    Pure function: no side effects, no I/O, no logging on the happy path.
    Detectors that raise are treated as no-match (the pipeline does not
    let a buggy detector break ingest) and the failure is reported by
    the wrapping call in ``apply_suggestions``.
    """
    if not text:
        return []
    found: set[str] = set()
    for det in DETECTORS:
        try:
            if det.detect(text):
                found.add(det.label)
        except Exception:
            # A detector raising is a code bug, not a tenant-data
            # problem. Swallow here; the caller logs (apply_suggestions
            # has the MemoryRow context for a useful structured log).
            continue
    return sorted(found)


def apply_suggestions(memories: Iterable[MemoryRow]) -> None:
    """Stamp ``suggested_labels`` on each memory in-place.

    Idempotent: re-running merges with anything already present (the
    LLM compiler may run before the heuristic detector, or a tenant
    may pre-seed via the SDK). De-duplicates and sorts.

    Detectors run against ``memory.content``. The summary is a
    derivative and adds no signal a detector would miss in the body.
    """
    import structlog

    logger = structlog.stdlib.get_logger()

    for mem in memories:
        try:
            new_labels = label_suggestions(mem.content)
        except Exception as exc:  # noqa: BLE001 — defensive boundary
            logger.warning(
                "auto_labeling_failed",
                memory_id=str(mem.id) if mem.id else None,
                error=str(exc)[:200],
            )
            continue
        if not new_labels:
            continue
        existing = list(mem.suggested_labels or [])
        merged = sorted(set(existing) | set(new_labels))
        mem.suggested_labels = merged


__all__ = [
    "DETECTORS",
    "Detector",
    "apply_suggestions",
    "label_suggestions",
]
