"""LLM-backed memory compiler — uses httpx for direct OpenAI API calls.

Extracts structured memories (profile facts, preferences, episode summaries,
procedures) from episode payloads using OpenAI-compatible APIs.

Optimized for speed:
- Batches small episodes into a single LLM call
- Runs multiple batches in parallel with concurrency control
- Falls back gracefully on parse errors or API failures

Requires:
- STATEWAVE_COMPILER_TYPE=llm
- OPENAI_API_KEY or STATEWAVE_OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
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

# ─── Configuration ───

_MAX_BATCH_CHARS = 6000  # Max total chars per LLM call (leaves room for prompt + response)
_MAX_CONCURRENCY = 4  # Max parallel LLM calls
_MAX_TOKENS = 3000  # Response token limit per batch

_SYSTEM_PROMPT = """\
You are a memory extraction engine for an AI context system called Statewave.

Given one or more raw episodes (recorded interactions), extract structured memories.

Each memory must be one of these kinds:
- profile_fact: a concrete fact about the subject (name, role, location, preference, etc.)
- episode_summary: a concise summary of what happened in this interaction
- procedure: a step-by-step process, workflow, or instruction that was discussed

Return a JSON array of memory objects. Each object must have:
- "kind": one of the kinds above
- "content": the full memory text
- "summary": a one-sentence summary (max 200 chars)
- "confidence": a float 0.0–1.0 indicating extraction confidence
- "episode_index": the 0-based index of the episode this memory came from

Rules:
- Extract ALL distinct facts; do not merge unrelated facts into one memory.
- Be precise and factual — never invent information not in the episode.
- If an episode contains no extractable memories, skip it.
- Return ONLY the JSON array, no markdown fences or extra text.
"""

# OpenAI-compatible base URL (can be overridden for Azure, etc.)
_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


class LLMCompiler:
    """Async LLM memory compiler with batching + parallelism. Implements BaseCompiler protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
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
                timeout=60.0,
            )
        return self._client

    def compile(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Sync fallback — uses heuristic compiler."""
        from server.services.compilers.heuristic import HeuristicCompiler
        logger.warning("llm_compiler_sync_fallback")
        return HeuristicCompiler().compile(episodes)

    async def compile_async(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Async compile — batches episodes and processes in parallel."""
        # Extract text from each episode, skip empties
        episode_texts: list[tuple[EpisodeRow, str]] = []
        for ep in episodes:
            text = extract_payload_text(ep.payload)
            if text:
                episode_texts.append((ep, text[:4000]))  # Cap per-episode text

        if not episode_texts:
            return []

        # Group into batches by total character count
        batches = self._create_batches(episode_texts)
        logger.info("compile_batched", episodes=len(episode_texts), batches=len(batches))

        # Run batches in parallel with concurrency limit
        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
        tasks = [self._process_batch(batch, semaphore) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        memories: list[MemoryRow] = []
        for result in batch_results:
            memories.extend(result)

        logger.info("compile_complete", total_memories=len(memories))
        return memories

    def _create_batches(
        self, episode_texts: list[tuple[EpisodeRow, str]]
    ) -> list[list[tuple[EpisodeRow, str]]]:
        """Group episodes into batches that fit within the char budget."""
        batches: list[list[tuple[EpisodeRow, str]]] = []
        current_batch: list[tuple[EpisodeRow, str]] = []
        current_chars = 0

        for ep, text in episode_texts:
            text_len = len(text)
            # If single episode exceeds budget, it goes in its own batch
            if text_len >= _MAX_BATCH_CHARS:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_chars = 0
                batches.append([(ep, text)])
                continue

            if current_chars + text_len > _MAX_BATCH_CHARS:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append((ep, text))
            current_chars += text_len

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _process_batch(
        self,
        batch: list[tuple[EpisodeRow, str]],
        semaphore: asyncio.Semaphore,
    ) -> list[MemoryRow]:
        """Process a batch of episodes in a single LLM call."""
        async with semaphore:
            # Format the prompt with all episodes in this batch
            episode_blocks = []
            for i, (ep, text) in enumerate(batch):
                episode_blocks.append(f"--- Episode {i} ---\n{text}")
            combined_text = "\n\n".join(episode_blocks)

            try:
                raw_memories = await self._call_llm_async(combined_text, len(batch))
            except Exception:
                logger.warning("llm_batch_failed", episode_count=len(batch), exc_info=True)
                return []

            # Map memories back to their source episodes
            results: list[MemoryRow] = []
            for mem in raw_memories:
                # Determine which episode this memory belongs to
                ep_idx = mem.get("episode_index", 0)
                if not isinstance(ep_idx, int) or ep_idx < 0 or ep_idx >= len(batch):
                    ep_idx = 0
                source_ep = batch[ep_idx][0]

                kind = mem.get("kind", "episode_summary")
                if kind not in ("profile_fact", "episode_summary", "procedure"):
                    kind = "episode_summary"

                content = mem.get("content", "")
                if not content:
                    continue

                results.append(
                    MemoryRow(
                        id=uuid.uuid4(),
                        subject_id=source_ep.subject_id,
                        kind=kind,
                        content=content,
                        summary=mem.get("summary", content[:200]),
                        confidence=min(max(float(mem.get("confidence", 0.7)), 0.0), 1.0),
                        valid_from=source_ep.created_at or datetime.now(timezone.utc),
                        source_episode_ids=[source_ep.id],
                        metadata_={"compiler": "llm", "model": self._model},
                        status="active",
                    )
                )

            logger.info("llm_batch_done", episodes=len(batch), memories_extracted=len(results))
            return results

    async def _call_llm_async(self, text: str, episode_count: int) -> list[dict[str, Any]]:
        """Async LLM call via direct httpx to OpenAI-compatible endpoint."""
        client = self._get_client()

        # Adjust max tokens based on batch size
        max_tokens = min(_MAX_TOKENS, 500 * episode_count)

        resp = await client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract memories from these {episode_count} episode(s):\n\n{text}"},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
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
