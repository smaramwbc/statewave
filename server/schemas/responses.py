"""API response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from server.domain.models import MemoryKind, MemoryStatus


class EpisodeResponse(BaseModel):
    id: uuid.UUID
    subject_id: str
    source: str
    type: str
    payload: dict[str, Any]
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    session_id: str | None = None
    occurred_at: datetime
    created_at: datetime

    @classmethod
    def from_row(cls, row: Any) -> "EpisodeResponse":
        """Build a response from an `EpisodeRow` ORM instance.

        Single source of truth for the ORM->schema field mapping (#295) —
        every router that returns an episode should call this instead of
        re-listing the fields inline.
        """
        return cls(
            id=row.id,
            subject_id=row.subject_id,
            source=row.source,
            type=row.type,
            payload=row.payload,
            metadata=row.metadata_,
            provenance=row.provenance,
            session_id=row.session_id,
            occurred_at=row.occurred_at,
            created_at=row.created_at,
        )


class BatchCreateEpisodesResponse(BaseModel):
    episodes_created: int
    episodes: list[EpisodeResponse]


class MemoryResponse(BaseModel):
    id: uuid.UUID
    subject_id: str
    kind: MemoryKind
    content: str
    summary: str
    confidence: float
    valid_from: datetime
    valid_to: datetime | None
    source_episode_ids: list[uuid.UUID]
    metadata: dict[str, Any]
    status: MemoryStatus
    sensitivity_labels: list[str] = Field(
        default_factory=list,
        description=(
            "Per-memory capability tags consumed by the policy layer (#50). "
            "Empty list = untagged = default-allow under any policy."
        ),
    )
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Any) -> "MemoryResponse":
        """Build a response from a `MemoryRow` ORM instance.

        Single source of truth for the ORM->schema field mapping (#295) —
        every router that returns a memory should call this instead of
        re-listing the fields inline.
        """
        return cls(
            id=row.id,
            subject_id=row.subject_id,
            kind=row.kind,
            content=row.content,
            summary=row.summary,
            confidence=row.confidence,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            source_episode_ids=row.source_episode_ids or [],
            metadata=row.metadata_,
            status=row.status,
            sensitivity_labels=list(getattr(row, "sensitivity_labels", None) or []),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class CompileMemoriesResponse(BaseModel):
    subject_id: str
    memories_created: int
    memories: list[MemoryResponse]
    # Drain signal (issue #134). `has_more` is True when more uncompiled
    # episodes remain after this call returned; `remaining_episodes` is
    # the count of those episodes. Sync callers loop on `has_more`; async
    # mode (`async: true`) drains internally and always returns
    # `has_more=False` once the job reaches `completed`.
    has_more: bool = False
    remaining_episodes: int = 0


class SearchMemoriesResponse(BaseModel):
    memories: list[MemoryResponse]
    search_mode: Literal["semantic", "text", "text_fallback"] = Field(
        "text",
        description=(
            "Which search path actually ran: 'semantic' (embedding/hybrid "
            "search executed), 'text' (plain text search — semantic was not "
            "requested, or requested without a q), or 'text_fallback' "
            "(semantic was requested with a q but could not run — no embedding "
            "provider configured, or the provider errored — so text search ran "
            "instead)."
        ),
    )


class SessionInfo(BaseModel):
    session_id: str
    episode_count: int
    first_at: datetime | None = None
    last_at: datetime | None = None


class ContextBundleResponse(BaseModel):
    subject_id: str
    task: str
    facts: list[MemoryResponse] = Field(default_factory=list)
    episodes: list[EpisodeResponse] = Field(default_factory=list)
    procedures: list[MemoryResponse] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    assembled_context: str = ""
    token_estimate: int = 0
    sessions: list[SessionInfo] = Field(
        default_factory=list, description="Sessions represented in the context bundle"
    )
    receipt_id: str | None = Field(
        None,
        description=(
            "ULID of the state-assembly receipt, when emitted. None when no "
            "receipt was emitted for this call (caller didn't request one, "
            "tenant config is `never`, or emission failed — check "
            "`receipt_emitted` to distinguish the last case)."
        ),
    )
    receipt_emitted: bool = Field(
        False,
        description=(
            "True iff a receipt was successfully written for this call. "
            "When False and `receipt_id` is also None, the call requested no "
            "receipt; when False but emission was attempted, the failure was "
            "logged server-side and the assembly result is still authoritative."
        ),
    )


class TimelineResponse(BaseModel):
    subject_id: str
    episodes: list[EpisodeResponse]
    memories: list[MemoryResponse]
    episodes_has_more: bool = False
    memories_has_more: bool = False


class DeleteSubjectResponse(BaseModel):
    subject_id: str
    episodes_deleted: int
    memories_deleted: int


class SubjectSummary(BaseModel):
    subject_id: str
    episode_count: int
    memory_count: int


class ListSubjectsResponse(BaseModel):
    subjects: list[SubjectSummary]
    total: int


class ResolutionResponse(BaseModel):
    id: uuid.UUID
    subject_id: str
    session_id: str
    status: str
    resolution_summary: str | None = None
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Any) -> "ResolutionResponse":
        """Build a response from a `ResolutionRow` ORM instance.

        Single source of truth for the ORM->schema field mapping (#295) —
        every router that returns a resolution should call this instead of
        re-listing the fields inline.
        """
        return cls(
            id=row.id,
            subject_id=row.subject_id,
            session_id=row.session_id,
            status=row.status,
            resolution_summary=row.resolution_summary,
            resolved_at=row.resolved_at,
            metadata=row.metadata_,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ResolutionSummaryItem(BaseModel):
    session_id: str
    status: str
    summary: str | None = None
    resolved_at: datetime | None = None


class HealthFactorResponse(BaseModel):
    signal: str
    impact: int
    detail: str


class HandoffResponse(BaseModel):
    subject_id: str
    session_id: str
    reason: str
    generated_at: datetime
    customer_summary: str = ""
    active_issue: str = ""
    attempted_steps: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    resolution_history: list[ResolutionSummaryItem] = Field(default_factory=list)
    recent_context: list[str] = Field(default_factory=list)
    health_score: int | None = None
    health_state: str | None = None  # healthy | watch | at_risk
    health_factors: list[HealthFactorResponse] = Field(default_factory=list)
    handoff_notes: str = ""
    token_estimate: int = 0
    provenance: dict[str, Any] = Field(default_factory=dict)
    receipt_id: str | None = Field(
        None,
        description="ULID of the state-assembly receipt, when emitted.",
    )
    receipt_emitted: bool = Field(
        False,
        description="True iff a receipt was successfully written for this handoff.",
    )


class HealthResponse(BaseModel):
    subject_id: str
    score: int
    state: str  # healthy | watch | at_risk
    factors: list[HealthFactorResponse] = Field(default_factory=list)


class SessionSLAResponse(BaseModel):
    session_id: str
    status: str  # resolved | open
    first_message_at: str | None = None
    first_response_at: str | None = None
    resolved_at: str | None = None
    first_response_seconds: float | None = None
    resolution_seconds: float | None = None
    open_duration_seconds: float | None = None
    first_response_breached: bool = False
    resolution_breached: bool = False


class SLASummaryResponse(BaseModel):
    subject_id: str
    total_sessions: int
    resolved_sessions: int
    open_sessions: int
    avg_first_response_seconds: float | None = None
    avg_resolution_seconds: float | None = None
    first_response_breach_count: int = 0
    resolution_breach_count: int = 0
    sessions: list[SessionSLAResponse] = Field(default_factory=list)


class LLMUsage(BaseModel):
    """Token usage for one completion, OpenAI-shape. Present when the
    provider reports it; omitted (the whole object is ``None``) for
    providers that don't surface usage — e.g. some local/Ollama setups."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMCompleteResponse(BaseModel):
    """Response body for `POST /v1/llm/complete`.

    `reply` is the assistant text. `usage` and `model` are optional metadata
    so first-party consoles (the admin "Chat with Memory", which delegates
    its LLM call here rather than holding its own provider key) can surface
    token + model stats. Both are ``None`` when unavailable. Unlike the error
    path — which deliberately never echoes the model identifier — returning
    the configured model on the success path is safe: this endpoint sits
    behind the same ``X-API-Key`` trust boundary as the rest of ``/v1/*``."""

    reply: str
    usage: LLMUsage | None = None
    model: str | None = None


class TenantConfigResponse(BaseModel):
    """Read-side projection of a `tenant_configs` row. `version` is the
    optimistic-concurrency counter — pass it back as `expected_version`
    on the next PATCH to fail-fast on lost updates."""

    tenant_id: str
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The full config document. Known keys: `receipts`, "
            "`receipt_retention_days`, `policy_mode`, "
            "`require_caller_identity`. Unknown keys are preserved "
            "across writes for forward-compatibility."
        ),
    )
    version: int = Field(
        ...,
        ge=0,
        description=(
            "Incremented on every PATCH. `0` is returned when the tenant "
            "has no row in `tenant_configs` yet (the default state)."
        ),
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None