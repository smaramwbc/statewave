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


class BatchCreateEpisodesRequest(BaseModel):
    episodes: list[CreateEpisodeRequest] = Field(..., min_length=1, max_length=100)


class CompileMemoriesRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)


class SearchMemoriesRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    kind: str | None = None
    query: str | None = None
    limit: int = Field(20, ge=1, le=100)


class GetContextRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    task: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int | None = Field(None, ge=1, le=128000)
