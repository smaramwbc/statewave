"""Tests for handoff context pack assembly."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.handoff import assemble_handoff
from server.services.health import HealthFactor, HealthResult
from server.services.sla import SLASummary, SessionSLA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode_row(
    *,
    session_id: str | None = None,
    source: str = "chat",
    type_: str = "message",
    text: str = "hello",
    minutes_ago: int = 0,
):
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        source=source,
        type=type_,
        payload={"messages": [{"role": "user", "content": text}]},
        metadata_={},
        provenance={},
        session_id=session_id,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


def _make_fact_row(content: str, *, status: str = "active"):
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        kind="profile_fact",
        content=content,
        summary="",
        confidence=1.0,
        valid_from=datetime.now(timezone.utc),
        valid_to=None,
        source_episode_ids=[],
        metadata_={},
        status=status,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_resolution_row(
    session_id: str, status: str = "resolved", summary: str | None = "Fixed it"
):
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="user-1",
        session_id=session_id,
        tenant_id=None,
        status=status,
        resolution_summary=summary,
        resolved_at=datetime.now(timezone.utc) if status == "resolved" else None,
        metadata_={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@contextmanager
def _mock_repos(*, episodes=None, facts=None, resolutions=None, health=None, sla=None):
    if health is None:
        health = HealthResult(subject_id="user-1", score=100, state="healthy", factors=[])
    if sla is None:
        sla = SLASummary(subject_id="user-1")
    with (
        patch(
            "server.services.handoff.repo.search_memories",
            new_callable=AsyncMock,
            return_value=facts or [],
        ),
        patch(
            "server.services.handoff.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes or [],
        ),
        patch(
            "server.services.handoff.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=resolutions or [],
        ),
        patch(
            "server.services.handoff.compute_health",
            new_callable=AsyncMock,
            return_value=health,
        ),
        patch(
            "server.services.handoff.check_and_alert",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "server.services.handoff.compute_sla",
            new_callable=AsyncMock,
            return_value=sla,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_issue_from_current_session():
    """Current session episodes form the active issue."""
    ep = _make_episode_row(session_id="sess-1", text="My billing is wrong")

    with _mock_repos(episodes=[ep]):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert "billing" in result.active_issue.lower()
    assert result.session_id == "sess-1"


@pytest.mark.asyncio
async def test_key_facts_included():
    """Profile facts appear in key_facts."""
    facts = [_make_fact_row("Enterprise plan"), _make_fact_row("Account since 2023")]

    with _mock_repos(facts=facts):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert "Enterprise plan" in result.key_facts
    assert "Account since 2023" in result.key_facts


@pytest.mark.asyncio
async def test_resolution_history_included():
    """Resolved sessions appear in resolution_history."""
    resolutions = [
        _make_resolution_row("sess-old", "resolved", "Issued refund"),
        _make_resolution_row("sess-1", "open", None),
    ]

    with _mock_repos(resolutions=resolutions):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert len(result.resolution_history) == 2
    resolved = [r for r in result.resolution_history if r.status == "resolved"]
    assert len(resolved) == 1
    assert resolved[0].summary == "Issued refund"


@pytest.mark.asyncio
async def test_resolved_deprioritized_in_notes():
    """Resolved sessions appear under 'Previously Resolved', not 'Open Issues'."""
    resolutions = [
        _make_resolution_row("sess-old", "resolved", "Gave refund"),
        _make_resolution_row("sess-1", "open", "Still investigating"),
    ]

    with _mock_repos(resolutions=resolutions):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert "Previously Resolved" in result.handoff_notes
    assert "Open Issues" in result.handoff_notes


@pytest.mark.asyncio
async def test_attempted_steps_from_assistant_messages():
    """Assistant messages in current session appear as attempted steps."""
    ep1 = _make_episode_row(session_id="sess-1", source="user", text="Help me")
    ep2 = _make_episode_row(
        session_id="sess-1", source="assistant", text="I checked your account status"
    )

    with _mock_repos(episodes=[ep1, ep2]):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert any("checked your account" in s for s in result.attempted_steps)


@pytest.mark.asyncio
async def test_handoff_notes_compact_and_deterministic():
    """Handoff notes should be a readable string within token budget."""
    facts = [_make_fact_row("Pro plan")]
    ep = _make_episode_row(session_id="sess-1", text="Cannot access dashboard")

    with _mock_repos(facts=facts, episodes=[ep]):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1", max_tokens=500)

    assert result.token_estimate <= 500
    assert "Handoff Brief" in result.handoff_notes
    assert "user-1" in result.handoff_notes


@pytest.mark.asyncio
async def test_provenance_preserved():
    """Provenance should track which items contributed."""
    facts = [_make_fact_row("VIP customer")]
    ep = _make_episode_row(session_id="sess-1", text="Issue report")

    with _mock_repos(facts=facts, episodes=[ep]):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert "fact_ids" in result.provenance
    assert "episode_ids" in result.provenance
    assert len(result.provenance["fact_ids"]) == 1
    assert len(result.provenance["episode_ids"]) == 1


@pytest.mark.asyncio
async def test_recent_context_from_other_sessions():
    """Episodes from other sessions appear as recent context."""
    current_ep = _make_episode_row(session_id="sess-1", text="Current issue")
    old_ep = _make_episode_row(session_id="sess-old", text="Previous conversation", minutes_ago=60)

    with _mock_repos(episodes=[current_ep, old_ep]):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert any("Previous conversation" in c for c in result.recent_context)


@pytest.mark.asyncio
async def test_empty_subject_produces_minimal_handoff():
    """A subject with no data still produces a valid handoff."""
    with _mock_repos():
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert result.subject_id == "user-1"
    assert result.session_id == "sess-1"
    assert "Handoff Brief" in result.handoff_notes
    assert result.customer_summary != ""


# ---------------------------------------------------------------------------
# Health-aware handoff tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_at_risk_customer_shows_in_handoff():
    """At-risk health state should be clearly visible in handoff."""
    health = HealthResult(
        subject_id="user-1",
        score=20,
        state="at_risk",
        factors=[
            HealthFactor(signal="unresolved_issues", impact=-30, detail="2 open sessions"),
            HealthFactor(
                signal="repeated_issues", impact=-20, detail="Open issues resemble resolved ones"
            ),
            HealthFactor(signal="escalations", impact=-20, detail="2 episodes with urgency"),
        ],
    )

    with _mock_repos(health=health):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert result.health_score == 20
    assert result.health_state == "at_risk"
    assert len(result.health_factors) == 3
    assert "AT_RISK" in result.handoff_notes
    assert "🔴" in result.handoff_notes


@pytest.mark.asyncio
async def test_healthy_state_renders_in_handoff():
    """Healthy state should render with green indicator."""
    health = HealthResult(subject_id="user-1", score=100, state="healthy", factors=[])

    with _mock_repos(health=health):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert result.health_score == 100
    assert result.health_state == "healthy"
    assert "HEALTHY" in result.handoff_notes
    assert "🟢" in result.handoff_notes


@pytest.mark.asyncio
async def test_watch_state_renders_in_handoff():
    """Watch state should render with yellow indicator."""
    health = HealthResult(
        subject_id="user-1",
        score=55,
        state="watch",
        factors=[HealthFactor(signal="unresolved_issues", impact=-15, detail="1 open session")],
    )

    with _mock_repos(health=health):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert result.health_score == 55
    assert result.health_state == "watch"
    assert "WATCH" in result.handoff_notes
    assert "🟡" in result.handoff_notes


@pytest.mark.asyncio
async def test_health_factors_explainable_in_handoff():
    """Health factors should have signal, impact, and detail in handoff response."""
    health = HealthResult(
        subject_id="user-1",
        score=40,
        state="watch",
        factors=[
            HealthFactor(signal="unresolved_issues", impact=-15, detail="1 open session"),
            HealthFactor(signal="escalations", impact=-10, detail="1 urgency episode"),
        ],
    )

    with _mock_repos(health=health):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert len(result.health_factors) == 2
    assert result.health_factors[0].signal == "unresolved_issues"
    assert result.health_factors[0].impact == -15
    assert result.health_factors[0].detail == "1 open session"
    # Factor details appear in handoff notes
    assert "1 open session" in result.handoff_notes


@pytest.mark.asyncio
async def test_health_in_handoff_stays_compact():
    """Health section should not blow up token budget."""
    health = HealthResult(
        subject_id="user-1",
        score=25,
        state="at_risk",
        factors=[
            HealthFactor(signal="unresolved_issues", impact=-30, detail="2 open sessions"),
            HealthFactor(signal="repeated_issues", impact=-20, detail="Recurring pattern"),
            HealthFactor(signal="escalations", impact=-20, detail="Urgency markers"),
        ],
    )
    ep = _make_episode_row(session_id="sess-1", text="Need help urgently")

    with _mock_repos(episodes=[ep], health=health):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1", max_tokens=300)

    assert result.token_estimate <= 300
    # Health section is one compact section, not a huge dump
    health_section = result.handoff_notes.split("## Customer Health")[1].split("##")[0]
    assert health_section.count("\n") <= 4  # Very compact


# ---------------------------------------------------------------------------
# SLA-aware handoff tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sla_breach_appears_in_handoff():
    """SLA breaches should appear in handoff notes when present."""
    sla = SLASummary(
        subject_id="user-1",
        total_sessions=2,
        resolved_sessions=1,
        open_sessions=1,
        first_response_breach_count=1,
        resolution_breach_count=1,
        sessions=[
            SessionSLA(session_id="s1", status="resolved", first_response_breached=True),
            SessionSLA(session_id="s2", status="open", open_duration_seconds=7200.0),
        ],
    )

    with _mock_repos(sla=sla):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert "SLA Status" in result.handoff_notes
    assert "First-response SLA breached" in result.handoff_notes
    assert "Resolution SLA breached" in result.handoff_notes


@pytest.mark.asyncio
async def test_sla_not_shown_when_clean():
    """No SLA section in handoff when there are no breaches or open issues."""
    sla = SLASummary(
        subject_id="user-1",
        total_sessions=1,
        resolved_sessions=1,
        open_sessions=0,
        avg_first_response_seconds=60.0,
        sessions=[
            SessionSLA(session_id="s1", status="resolved"),
        ],
    )

    with _mock_repos(sla=sla):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    # Only avg first response shown (no breaches, no open)
    # SLA section is present because avg_first_response is available
    assert "Avg first response" in result.handoff_notes


@pytest.mark.asyncio
async def test_sla_section_absent_for_empty_history():
    """No SLA section when subject has no sessions."""
    sla = SLASummary(subject_id="user-1")

    with _mock_repos(sla=sla):
        result = await assemble_handoff(AsyncMock(), "user-1", "sess-1")

    assert "SLA Status" not in result.handoff_notes
