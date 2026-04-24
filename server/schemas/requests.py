"""API request schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateEpisodeRequest(BaseModel):
    subject_id: str
    source: str
    type: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class CompileMemoriesRequest(BaseModel):
    subject_id: str


class SearchMemoriesRequest(BaseModel):
    subject_id: str
    kind: str | None = None
    query: str | None = None
    limit: int = 20


class GetContextRequest(BaseModel):
    subject_id: str
    task: str
    max_tokens: int | None = None
