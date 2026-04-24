"""Webhook / event hook service.

Fires HTTP callbacks when key events occur:
- episode.created
- memories.compiled
- subject.deleted

Webhooks are configured via STATEWAVE_WEBHOOK_URL. When empty, webhooks
are disabled (no-op). Delivery is fire-and-forget with a short timeout.
Failures are logged but never block the main request.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.stdlib.get_logger()

_webhook_url: str | None = None
_timeout: float = 5.0


def configure(url: str | None, timeout: float = 5.0) -> None:
    """Set the global webhook URL. Called at app startup."""
    global _webhook_url, _timeout
    _webhook_url = url
    _timeout = timeout


async def fire(event: str, payload: dict[str, Any]) -> None:
    """Fire a webhook event. Non-blocking, best-effort delivery."""
    if not _webhook_url:
        return

    body = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }

    # Fire-and-forget — don't await in the request path
    asyncio.create_task(_deliver(body))


async def _deliver(body: dict[str, Any]) -> None:
    """Actually POST the webhook payload."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=_timeout) as client:
            resp = await client.post(_webhook_url, json=body)  # type: ignore[arg-type]
            logger.debug(
                "webhook_delivered",
                event=body["event"],
                status=resp.status_code,
                url=_webhook_url,
            )
    except Exception:
        logger.warning(
            "webhook_delivery_failed",
            event=body["event"],
            url=_webhook_url,
            exc_info=True,
        )
