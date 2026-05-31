"""Regression: a preserved-id memory collision must be a clean 400, not a 500.

import_subject's preserve_ids conflict guard checked only episode ids and was
gated on episodes being present, so a memories-only document skipped the check
entirely. A re-import then collided on the memories primary key and surfaced as
a raw DB IntegrityError (HTTP 500), after episodes were partially added,
instead of the ValueError (HTTP 400) the function documents.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from server.services.backup import _FORMAT_VERSION, import_subject


def _memories_only_doc(subject_id: str, memory_id: str) -> dict:
    eps: list = []
    mems = [
        {
            "id": memory_id,
            "subject_id": subject_id,
            "tenant_id": None,
            "kind": "profile_fact",
            "content": "User prefers dark mode",
            "summary": "dark mode",
            "confidence": 0.9,
            "valid_from": datetime.now(timezone.utc).isoformat(),
            "valid_to": None,
            "source_episode_ids": [],
            "metadata": {},
            "status": "active",
            "embedding": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    content_str = json.dumps({"episodes": eps, "memories": mems}, sort_keys=True)
    checksum = hashlib.sha256(content_str.encode()).hexdigest()
    return {
        "format_version": _FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "tenant_id": None,
        "counts": {"episodes": 0, "memories": 1},
        "episodes": eps,
        "memories": mems,
        "checksum": checksum,
    }


@pytest.mark.anyio
async def test_import_rejects_memory_id_conflict_with_valueerror(
    client: AsyncClient, subject_id: str
):
    memory_id = str(uuid.uuid4())
    doc = _memories_only_doc(subject_id, memory_id)

    # First import persists the memory with its preserved id.
    await import_subject(doc, target_subject_id=subject_id, preserve_ids=True)
    r = await client.get("/v1/memories/search", params={"subject_id": subject_id})
    assert r.status_code == 200
    assert any(m["id"] == memory_id for m in r.json()["memories"])

    # Second import collides on the memory id: must be a clean ValueError
    # (-> HTTP 400), not a DB IntegrityError (-> HTTP 500).
    with pytest.raises(ValueError, match="conflict"):
        await import_subject(doc, target_subject_id=subject_id, preserve_ids=True)
