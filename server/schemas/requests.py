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


class LLMChatMessage(BaseModel):
    """Single chat-completion message. Mirrors the OpenAI/LiteLLM wire shape."""

    role: str = Field(..., pattern=r"^(system|user|assistant|tool)$")
    content: str = Field(..., max_length=16000)


class LLMCompleteRequest(BaseModel):
    """Request body for `POST /v1/llm/complete`.

    Intentionally narrow: callers (the website widget, internal demo flows)
    pass messages and optional generation knobs; **provider/model selection
    lives entirely in server config** (`STATEWAVE_LITELLM_MODEL` and
    friends). This is not a generic public LLM API.
    """

    messages: list[LLMChatMessage] = Field(..., min_length=1, max_length=50)
    max_tokens: int | None = Field(None, ge=1, le=4096)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
