"""Reliable webhook delivery service.

Events are persisted to the `webhook_events` table before delivery is attempted.
A background worker processes pending events with exponential backoff.
After max_attempts (default 5), events are marked as dead_letter.

Delivery statuses:
- pending: awaiting delivery attempt
- delivered: successfully delivered (2xx response)
- dead_letter: all attempts exhausted

Retry schedule (exponential backoff with jitter):
  attempt 1: immediate
  attempt 2: ~30s
  attempt 3: ~2min
  attempt 4: ~8min
  attempt 5: ~30min
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.engine import async_session_factory
from server.db.tables import WebhookEventRow

logger = structlog.stdlib.get_logger()

_webhook_url: str | None = None
_timeout: float = 5.0
_worker_task: asyncio.Task | None = None
_poll_interval: float = 5.0  # seconds between queue checks


def configure(url: str | None, timeout: float = 5.0) -> None:
    """Set the global webhook URL. Called at app startup."""
    global _webhook_url, _timeout
    _webhook_url = url
    _timeout = timeout


async def start_worker() -> None:
    """Start the background delivery worker."""
    global _worker_task
    if not _webhook_url:
        logger.info("webhook_worker_disabled", reason="no webhook URL configured")
        return
    _worker_task = asyncio.create_task(_delivery_loop())
    logger.info("webhook_worker_started", url=_webhook_url)


async def stop_worker() -> None:
    """Stop the background delivery worker."""
    global _worker_task
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
        logger.info("webhook_worker_stopped")


async def fire(event: str, payload: dict[str, Any], db: AsyncSession | None = None) -> uuid.UUID | None:
    """Enqueue a webhook event for delivery.

    If no webhook URL is configured, returns None (no-op).
    The event is persisted immediately and delivered asynchronously.
    """
    if not _webhook_url:
        return None

    event_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    row = WebhookEventRow(
        id=event_id,
        event=event,
        payload={
            "event": event,
            "timestamp": now.isoformat(),
            "data": payload,
        },
        status="pending",
        attempts=0,
        next_attempt_at=now,  # deliver immediately
        created_at=now,
    )

    if db:
        db.add(row)
        # Will be committed with the caller's transaction
    else:
        async with async_session_factory() as session:
            session.add(row)
            await session.commit()

    return event_id


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter: ~30s, ~2m, ~8m, ~30m."""
    base = 30 * (4 ** (attempt - 1))  # 30, 120, 480, 1920
    jitter = random.uniform(0.5, 1.5)
    return base * jitter


async def _delivery_loop() -> None:
    """Background loop that processes pending webhook events."""
    while True:
        try:
            await _process_pending_batch()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("webhook_worker_error", exc_info=True)
        await asyncio.sleep(_poll_interval)


async def _process_pending_batch(batch_size: int = 20) -> int:
    """Process a batch of pending events. Returns count processed."""
    now = datetime.now(timezone.utc)
    processed = 0

    async with async_session_factory() as session:
        # Fetch pending events that are due for delivery
        stmt = (
            select(WebhookEventRow)
            .where(WebhookEventRow.status == "pending")
            .where(WebhookEventRow.next_attempt_at <= now)
            .order_by(WebhookEventRow.next_attempt_at)
            .limit(batch_size)
        )
        result = await session.execute(stmt)
        events = result.scalars().all()

        for event in events:
            await _attempt_delivery(event, session)
            processed += 1

        await session.commit()

    return processed


async def _attempt_delivery(event: WebhookEventRow, session: AsyncSession) -> None:
    """Attempt to deliver a single webhook event."""
    import httpx

    now = datetime.now(timezone.utc)
    event.attempts += 1
    event.last_attempt_at = now

    try:
        async with httpx.AsyncClient(timeout=_timeout) as client:
            resp = await client.post(_webhook_url, json=event.payload)  # type: ignore[arg-type]

        event.http_status = resp.status_code

        if 200 <= resp.status_code < 300:
            event.status = "delivered"
            event.delivered_at = now
            logger.debug(
                "webhook_delivered",
                event_id=str(event.id),
                event_type=event.event,
                attempt=event.attempts,
                status=resp.status_code,
            )
        else:
            _handle_failure(event, f"HTTP {resp.status_code}: {resp.text[:200]}")

    except Exception as exc:
        event.http_status = None
        _handle_failure(event, f"{type(exc).__name__}: {str(exc)[:200]}")


def _handle_failure(event: WebhookEventRow, error: str) -> None:
    """Handle a failed delivery attempt — retry or dead-letter."""
    event.last_error = error

    if event.attempts >= event.max_attempts:
        event.status = "dead_letter"
        logger.warning(
            "webhook_dead_letter",
            event_id=str(event.id),
            event_type=event.event,
            attempts=event.attempts,
            last_error=error,
        )
    else:
        # Schedule retry with backoff
        backoff = _backoff_seconds(event.attempts)
        event.next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)
        logger.info(
            "webhook_retry_scheduled",
            event_id=str(event.id),
            event_type=event.event,
            attempt=event.attempts,
            next_in_seconds=round(backoff),
        )


# ── Query helpers (for admin endpoints) ───────────────────────────────────


async def get_event_status(event_id: uuid.UUID) -> dict[str, Any] | None:
    """Get the delivery status of a single webhook event."""
    async with async_session_factory() as session:
        row = await session.get(WebhookEventRow, event_id)
        if not row:
            return None
        return {
            "id": str(row.id),
            "event": row.event,
            "status": row.status,
            "attempts": row.attempts,
            "max_attempts": row.max_attempts,
            "last_attempt_at": row.last_attempt_at.isoformat() if row.last_attempt_at else None,
            "next_attempt_at": row.next_attempt_at.isoformat() if row.next_attempt_at else None,
            "last_error": row.last_error,
            "http_status": row.http_status,
            "created_at": row.created_at.isoformat(),
            "delivered_at": row.delivered_at.isoformat() if row.delivered_at else None,
        }


async def get_delivery_stats() -> dict[str, Any]:
    """Get aggregate webhook delivery statistics."""
    from sqlalchemy import func as sqlfunc

    async with async_session_factory() as session:
        stmt = (
            select(
                WebhookEventRow.status,
                sqlfunc.count(WebhookEventRow.id).label("count"),
            )
            .group_by(WebhookEventRow.status)
        )
        result = await session.execute(stmt)
        counts = {row.status: row.count for row in result}

    return {
        "pending": counts.get("pending", 0),
        "delivered": counts.get("delivered", 0),
        "dead_letter": counts.get("dead_letter", 0),
        "total": sum(counts.values()),
    }
