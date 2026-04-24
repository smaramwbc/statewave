"""Pure domain value objects (Pydantic). No ORM coupling."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryKind(str, Enum):
    profile_fact = "profile_fact"
    episode_summary = "episode_summary"
    procedure = "procedure"
    artifact_ref = "artifact_ref"


class MemoryStatus(str, Enum):
    active = "active"
    superseded = "superseded"
    deleted = "deleted"


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

class Episode(BaseModel):
    """Immutable raw event record."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    subject_id: str
    source: str
    type: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class Memory(BaseModel):
    """Derived memory object with provenance."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    subject_id: str
    kind: MemoryKind
    content: str
    summary: str = ""
    confidence: float = 1.0
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_to: datetime | None = None
    source_episode_ids: list[uuid.UUID] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.active
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# ContextBundle
# ---------------------------------------------------------------------------

class ContextBundle(BaseModel):
    """Runtime output for AI applications."""

    subject_id: str
    task: str
    facts: list[Memory] = Field(default_factory=list)
    episodes: list[Episode] = Field(default_factory=list)
    procedures: list[Memory] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    assembled_context: str = ""
    token_estimate: int = 0
