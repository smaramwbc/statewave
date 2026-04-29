"""API response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

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
    created_at: datetime


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
    created_at: datetime
    updated_at: datetime


class CompileMemoriesResponse(BaseModel):
    subject_id: str
    memories_created: int
    memories: list[MemoryResponse]


class SearchMemoriesResponse(BaseModel):
    memories: list[MemoryResponse]


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


class TimelineResponse(BaseModel):
    subject_id: str
    episodes: list[EpisodeResponse]
    memories: list[MemoryResponse]


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


class HealthResponse(BaseModel):
    subject_id: str
    score: int
    state: str  # healthy | watch | at_risk
    factors: list[HealthFactorResponse] = Field(default_factory=list)


class SessionSLAResponse(BaseModel):
    session_id: str
    status: str  # resolved | open
    first_message_at: str
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
