"""Canonical webhook event-type vocabulary.

Dependency-free on purpose: both the config layer (``server.core.config``)
and the delivery service (``server.services.webhooks``) import it, and the
delivery service pulls in the database layer — so the vocabulary has to
live somewhere neither side's import graph routes back through.

Every string in ``KNOWN_WEBHOOK_EVENTS`` is an event passed to
``webhooks.fire()`` somewhere in the server. Adding a new event type
means adding it here too, otherwise ``STATEWAVE_WEBHOOK_EVENTS`` will
reject any filter that names it.
"""

from __future__ import annotations

#: Every event type the server emits via ``webhooks.fire()``.
#:
#: Keep this in sync with the ``webhooks.fire(...)`` call sites. The
#: ``test_webhook_events`` suite asserts the set is non-empty and the
#: config validator checks operator-supplied filters against it.
KNOWN_WEBHOOK_EVENTS: frozenset[str] = frozenset(
    {
        "episode.created",
        "episodes.batch_created",
        "memories.compiled",
        "subject.deleted",
        "subject.health_degraded",
        "subject.health_improved",
    }
)


def parse_webhook_event_filter(value: object) -> list[str]:
    """Normalise and validate a webhook event-type allowlist.

    Accepts any of:

    - ``None`` or ``""`` — no filter; every event is delivered.
    - a comma-separated string — the operator-facing env-var form,
      e.g. ``STATEWAVE_WEBHOOK_EVENTS=memories.compiled,subject.deleted``.
    - a list / tuple / set of event types — the programmatic form used
      by tests and embedders.

    Unknown event types are rejected eagerly so a typo surfaces at
    startup, not as silently-dropped webhooks weeks later. The result is
    sorted and de-duplicated; an empty list means "no filter".
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = [str(part).strip() for part in value]
    else:
        raise ValueError(
            "STATEWAVE_WEBHOOK_EVENTS must be a comma-separated string or a "
            f"list of event types, got {type(value).__name__}"
        )

    cleaned = sorted({item for item in items if item})
    unknown = [item for item in cleaned if item not in KNOWN_WEBHOOK_EVENTS]
    if unknown:
        raise ValueError(
            "STATEWAVE_WEBHOOK_EVENTS names unknown event type(s): "
            f"{', '.join(unknown)}. Known event types: "
            f"{', '.join(sorted(KNOWN_WEBHOOK_EVENTS))}."
        )
    return cleaned
