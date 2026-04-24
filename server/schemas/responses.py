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
    created_at: datetime


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


class ContextBundleResponse(BaseModel):
    subject_id: str
    task: str
    facts: list[MemoryResponse] = Field(default_factory=list)
    episodes: list[EpisodeResponse] = Field(default_factory=list)
    procedures: list[MemoryResponse] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    assembled_context: str = ""
    token_estimate: int = 0


class TimelineResponse(BaseModel):
    subject_id: str
    episodes: list[EpisodeResponse]
    memories: list[MemoryResponse]


class DeleteSubjectResponse(BaseModel):
    subject_id: str
    episodes_deleted: int
    memories_deleted: int
