"""Handoff context pack service — generates compact escalation/transfer briefs.

A handoff pack answers:
- Who is the customer?
- What key profile/account facts matter?
- What is the current active issue?
- What has already been tried?
- What related issue history matters?
- What is resolved and should be deprioritized?
- What should the next agent know immediately?
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.config import settings
from server.db import repositories as repo
from server.schemas.responses import HandoffResponse, HealthFactorResponse, ResolutionSummaryItem
from server.services.compilers.heuristic import extract_payload_text
from server.services.health import compute_health
from server.services.health_alerts import check_and_alert
from server.services.sla import compute_sla


async def assemble_handoff(
    session: AsyncSession,
    subject_id: str,
    session_id: str,
    reason: str = "escalation",
    max_tokens: int | None = None,
    tenant_id: str | None = None,
) -> HandoffResponse:
    """Build a compact handoff context pack for agent escalation/transfer."""
    budget = max_tokens or 4000
    enc = tiktoken.get_encoding(settings.tiktoken_model)

    # -- Fetch data ---------------------------------------------------------
    fact_rows = await repo.search_memories(
        session, subject_id, tenant_id=tenant_id, kind="profile_fact", limit=20
    )
    episode_rows = await repo.list_episodes_by_subject(
        session, subject_id, tenant_id=tenant_id, limit=30
    )
    resolution_rows = await repo.list_resolutions(
        session, subject_id, tenant_id=tenant_id, limit=20
    )

    # -- Key facts ----------------------------------------------------------
    key_facts = [row.content for row in fact_rows if row.status == "active"]

    # -- Active issue (current session episodes) ----------------------------
    current_session_eps = [
        row for row in episode_rows if getattr(row, "session_id", None) == session_id
    ]
    other_eps = [row for row in episode_rows if getattr(row, "session_id", None) != session_id]

    # Build active issue description from current session
    active_lines: list[str] = []
    for ep in current_session_eps[:10]:  # cap for compactness
        text = extract_payload_text(ep.payload)
        if text:
            active_lines.append(f"[{ep.source}/{ep.type}] {text[:200]}")

    active_issue = ""
    if active_lines:
        active_issue = active_lines[0]  # First message is usually the issue statement
        if len(active_lines) > 1:
            active_issue += f" (+ {len(active_lines) - 1} more messages in session)"

    # -- Attempted steps (procedure-like actions in current session) ---------
    attempted_steps: list[str] = []
    for ep in current_session_eps:
        text = extract_payload_text(ep.payload)
        if not text:
            continue
        # Heuristic: agent/system messages that look like actions
        if ep.source in ("agent", "system", "tool") or ep.type in (
            "action",
            "tool_call",
            "resolution_attempt",
        ):
            attempted_steps.append(text[:150])
        elif ep.type == "message" and ep.source == "assistant":
            # Assistant messages often contain actions taken
            attempted_steps.append(text[:150])

    # -- Resolution history -------------------------------------------------
    resolution_history: list[ResolutionSummaryItem] = []
    for r in resolution_rows:
        resolution_history.append(
            ResolutionSummaryItem(
                session_id=r.session_id,
                status=r.status,
                summary=r.resolution_summary,
                resolved_at=r.resolved_at,
            )
        )

    # -- Recent context (non-current session, most recent first) ------------
    recent_context: list[str] = []
    for ep in other_eps[:5]:
        text = extract_payload_text(ep.payload)
        if text:
            sid_label = f" [{ep.session_id}]" if getattr(ep, "session_id", None) else ""
            recent_context.append(f"[{ep.source}/{ep.type}]{sid_label} {text[:120]}")

    # -- Customer summary ---------------------------------------------------
    customer_summary = _build_customer_summary(key_facts, subject_id)

    # -- Health scoring -----------------------------------------------------
    health = await compute_health(session, subject_id, tenant_id=tenant_id)
    await check_and_alert(session, health, tenant_id=tenant_id)
    health_factors_top = health.factors[:3]  # Top 3 for compactness

    # -- Assemble handoff notes (readable text for next agent) --------------
    parts: list[str] = []
    parts.append(f"# Handoff Brief — {subject_id}")
    parts.append(f"Reason: {reason}")
    parts.append(f"Session: {session_id}")
    parts.append("")

    if customer_summary:
        parts.append(f"## Customer\n{customer_summary}")
        parts.append("")

    # Health section — immediately after customer, before issue details
    _state_icon = {"at_risk": "🔴", "watch": "🟡", "healthy": "🟢"}.get(health.state, "⚪")
    health_line = f"{_state_icon} Health: {health.state.upper()} (score: {health.score})"
    if health_factors_top:
        factor_strs = [f.detail for f in health_factors_top]
        health_line += " — " + "; ".join(factor_strs)
    parts.append(f"## Customer Health\n{health_line}")
    parts.append("")

    # -- SLA summary (compact, only if meaningful) --------------------------
    sla_result = await compute_sla(session, subject_id, tenant_id=tenant_id)
    sla_lines: list[str] = []
    if sla_result.total_sessions > 0:
        if sla_result.first_response_breach_count > 0:
            sla_lines.append(f"⚠️ First-response SLA breached in {sla_result.first_response_breach_count} session(s)")
        if sla_result.resolution_breach_count > 0:
            sla_lines.append(f"⚠️ Resolution SLA breached in {sla_result.resolution_breach_count} session(s)")
        if sla_result.open_sessions > 0:
            # Find longest open
            open_slas = [s for s in sla_result.sessions if s.status == "open"]
            if open_slas:
                max_open = max(s.open_duration_seconds or 0 for s in open_slas)
                hours = max_open / 3600
                sla_lines.append(f"Open issue age: {hours:.1f}h")
        if sla_result.avg_first_response_seconds is not None:
            avg_min = sla_result.avg_first_response_seconds / 60
            sla_lines.append(f"Avg first response: {avg_min:.1f} min")
    if sla_lines:
        parts.append("## SLA Status")
        for line in sla_lines:
            parts.append(f"- {line}")
        parts.append("")

    if active_issue:
        parts.append(f"## Active Issue\n{active_issue}")
        parts.append("")

    if attempted_steps:
        parts.append("## What Has Been Tried")
        for step in attempted_steps[:5]:
            parts.append(f"- {step}")
        parts.append("")

    if resolution_history:
        resolved = [r for r in resolution_history if r.status == "resolved"]
        unresolved = [r for r in resolution_history if r.status != "resolved"]
        if unresolved:
            parts.append("## Open Issues")
            for r in unresolved:
                parts.append(f"- [{r.session_id}] {r.summary or 'No summary'}")
        if resolved:
            parts.append("## Previously Resolved")
            for r in resolved[:3]:
                parts.append(f"- [{r.session_id}] {r.summary or 'Resolved'}")
        parts.append("")

    if key_facts:
        parts.append("## Key Facts")
        for fact in key_facts[:8]:
            parts.append(f"- {fact}")
        parts.append("")

    if recent_context:
        parts.append("## Recent Context (other sessions)")
        for line in recent_context:
            parts.append(f"- {line}")
        parts.append("")

    handoff_notes = "\n".join(parts)

    # Trim to budget if needed
    tokens = enc.encode(handoff_notes)
    if len(tokens) > budget:
        handoff_notes = enc.decode(tokens[:budget])

    token_estimate = min(len(tokens), budget)

    # -- Provenance ---------------------------------------------------------
    provenance: dict[str, Any] = {
        "fact_ids": [str(row.id) for row in fact_rows if row.status == "active"],
        "episode_ids": [str(ep.id) for ep in current_session_eps[:10]],
        "resolution_ids": [str(r.id) for r in resolution_rows],
        "context_episode_ids": [str(ep.id) for ep in other_eps[:5]],
    }

    return HandoffResponse(
        subject_id=subject_id,
        session_id=session_id,
        reason=reason,
        generated_at=datetime.now(timezone.utc),
        customer_summary=customer_summary,
        active_issue=active_issue,
        attempted_steps=attempted_steps[:5],
        key_facts=key_facts[:8],
        resolution_history=resolution_history,
        recent_context=recent_context,
        health_score=health.score,
        health_state=health.state,
        health_factors=[
            HealthFactorResponse(signal=f.signal, impact=f.impact, detail=f.detail)
            for f in health_factors_top
        ],
        handoff_notes=handoff_notes,
        token_estimate=token_estimate,
        provenance=provenance,
    )


def _build_customer_summary(facts: list[str], subject_id: str) -> str:
    """Build a 1-2 line customer summary from profile facts."""
    if not facts:
        return f"Subject: {subject_id}"
    # Take first 2-3 facts as the summary
    summary_parts = facts[:3]
    return f"{subject_id} — " + "; ".join(summary_parts)
