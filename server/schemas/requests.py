"""API request schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateEpisodeRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    source: str = Field(..., min_length=1, max_length=256)
    type: str = Field(..., min_length=1, max_length=128)
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = Field(None, max_length=256)


class BatchCreateEpisodesRequest(BaseModel):
    episodes: list[CreateEpisodeRequest] = Field(..., min_length=1, max_length=100)


class CompileMemoriesRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    async_mode: bool = Field(default=False, alias="async")


class SearchMemoriesRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    kind: str | None = None
    query: str | None = None
    limit: int = Field(20, ge=1, le=100)


class GetContextRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    task: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int | None = Field(None, ge=1, le=128000)
    session_id: str | None = Field(
        None,
        max_length=256,
        description="Current session ID — episodes in this session receive a relevance boost",
    )


class CreateResolutionRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    session_id: str = Field(..., min_length=1, max_length=256)
    status: str = Field("open", pattern=r"^(open|resolved|unresolved)$")
    resolution_summary: str | None = Field(None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    session_id: str = Field(
        ..., min_length=1, max_length=256, description="Session being handed off"
    )
    reason: str = Field("escalation", max_length=256, description="Why the handoff is happening")
    max_tokens: int | None = Field(None, ge=1, le=16000)
