"""LLM-backed memory compiler — uses httpx for direct OpenAI API calls.

Extracts structured memories (profile facts, preferences, episode summaries,
procedures) from episode payloads using OpenAI-compatible APIs.

Falls back gracefully on parse errors or API failures.

Requires:
- STATEWAVE_COMPILER_TYPE=llm
- OPENAI_API_KEY or STATEWAVE_OPENAI_API_KEY
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx
import structlog

from server.db.tables import EpisodeRow, MemoryRow
from server.services.compilers.heuristic import extract_payload_text

logger = structlog.stdlib.get_logger()

_SYSTEM_PROMPT = """\
You are a memory extraction engine for an AI context system called Statewave.

Given a raw episode (a recorded interaction), extract structured memories.

Each memory must be one of these kinds:
- profile_fact: a concrete fact about the subject (name, role, location, preference, etc.)
- episode_summary: a concise summary of what happened in this interaction
- procedure: a step-by-step process, workflow, or instruction that was discussed

Return a JSON array of memory objects. Each object must have:
- "kind": one of the kinds above
- "content": the full memory text
- "summary": a one-sentence summary (max 200 chars)
- "confidence": a float 0.0–1.0 indicating extraction confidence

Rules:
- Extract ALL distinct facts; do not merge unrelated facts into one memory.
- Be precise and factual — never invent information not in the episode.
- If the episode contains no extractable memories, return an empty array [].
- Return ONLY the JSON array, no markdown fences or extra text.
"""

# OpenAI-compatible base URL (can be overridden for Azure, etc.)
_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


class LLMCompiler:
    """Async LLM memory compiler using direct httpx calls. Implements BaseCompiler protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        # Set API key for backward compat (STATEWAVE_OPENAI_API_KEY → OPENAI_API_KEY)
        if api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        self._api_key = os.environ.get("OPENAI_API_KEY", api_key or "")
        self._model = model
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=_OPENAI_BASE_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
        return self._client

    def compile(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Sync compile — falls back to heuristic. Use compile_async instead."""
        from server.services.compilers.heuristic import HeuristicCompiler
        logger.warning("llm_compiler_sync_fallback", msg="Using heuristic fallback in sync context")
        return HeuristicCompiler().compile(episodes)

    async def compile_async(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Async compile — processes episodes concurrently."""
        memories: list[MemoryRow] = []
        for ep in episodes:
            result = await self._compile_episode_async(ep)
            memories.extend(result)
        return memories

    async def _compile_episode_async(self, ep: EpisodeRow) -> list[MemoryRow]:
        text = extract_payload_text(ep.payload)
        if not text:
            return []

        try:
            raw_memories = await self._call_llm_async(text)
        except Exception:
            logger.warning("llm_compile_failed", episode_id=str(ep.id), exc_info=True)
            return []

        results: list[MemoryRow] = []
        for mem in raw_memories:
            kind = mem.get("kind", "episode_summary")
            if kind not in ("profile_fact", "episode_summary", "procedure"):
                kind = "episode_summary"

            content = mem.get("content", "")
            if not content:
                continue

            results.append(
                MemoryRow(
                    id=uuid.uuid4(),
                    subject_id=ep.subject_id,
                    kind=kind,
                    content=content,
                    summary=mem.get("summary", content[:200]),
                    confidence=min(max(float(mem.get("confidence", 0.7)), 0.0), 1.0),
                    valid_from=ep.created_at or datetime.now(timezone.utc),
                    source_episode_ids=[ep.id],
                    metadata_={"compiler": "llm", "model": self._model},
                    status="active",
                )
            )

        logger.info(
            "llm_compile_done",
            episode_id=str(ep.id),
            memories_extracted=len(results),
        )
        return results

    async def _call_llm_async(self, text: str) -> list[dict[str, Any]]:
        """Async LLM call via direct httpx to OpenAI-compatible endpoint."""
        client = self._get_client()
        resp = await client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract memories from this episode:\n\n{text[:8000]}"},
                ],
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"] or "[]"
        # Strip markdown fences if the model wraps them
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        return parsed
