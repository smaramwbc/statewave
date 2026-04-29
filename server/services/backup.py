"""Subject backup/restore — portable export/import for operators.

Exports all episodes and memories for a subject as a self-contained JSON
document. The document includes metadata, checksums, and full provenance.

This is an operator/admin tool for:
- Backing up a subject before risky operations
- Migrating subjects between Statewave instances
- Moving subjects between tenants
- Creating portable archives

This is NOT:
- A replacement for pg_dump (use that for full database backups)
- A point-in-time recovery tool
- Related to Subject Snapshots (those are in-DB bootstrap copies)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select

from server.db.engine import async_session_factory
from server.db.tables import EpisodeRow, MemoryRow

logger = structlog.stdlib.get_logger()

# Export format version — bump if schema changes
_FORMAT_VERSION = "1.0"


async def export_subject(subject_id: str, *, tenant_id: str | None = None) -> dict[str, Any]:
    """Export all episodes and memories for a subject as a portable document.

    Returns a dict suitable for JSON serialization.
    """
    async with async_session_factory() as session:
        # Fetch episodes
        ep_stmt = select(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
        if tenant_id is not None:
            ep_stmt = ep_stmt.where(EpisodeRow.tenant_id == tenant_id)
        ep_stmt = ep_stmt.order_by(EpisodeRow.created_at.asc())
        ep_result = await session.execute(ep_stmt)
        episodes = ep_result.scalars().all()

        # Fetch memories
        mem_stmt = select(MemoryRow).where(MemoryRow.subject_id == subject_id)
        if tenant_id is not None:
            mem_stmt = mem_stmt.where(MemoryRow.tenant_id == tenant_id)
        mem_stmt = mem_stmt.order_by(MemoryRow.created_at.asc())
        mem_result = await session.execute(mem_stmt)
        memories = mem_result.scalars().all()

    # Serialize
    episodes_data = [
        {
            "id": str(ep.id),
            "subject_id": ep.subject_id,
            "tenant_id": ep.tenant_id,
            "source": ep.source,
            "type": ep.type,
            "payload": ep.payload,
            "metadata": ep.metadata_,
            "provenance": ep.provenance,
            "created_at": ep.created_at.isoformat(),
            "last_compiled_at": ep.last_compiled_at.isoformat() if ep.last_compiled_at else None,
        }
        for ep in episodes
    ]

    memories_data = [
        {
            "id": str(mem.id),
            "subject_id": mem.subject_id,
            "tenant_id": mem.tenant_id,
            "kind": mem.kind,
            "content": mem.content,
            "summary": mem.summary,
            "confidence": mem.confidence,
            "valid_from": mem.valid_from.isoformat(),
            "valid_to": mem.valid_to.isoformat() if mem.valid_to else None,
            "source_episode_ids": [str(eid) for eid in (mem.source_episode_ids or [])],
            "metadata": mem.metadata_,
            "status": mem.status,
            "embedding": mem.embedding,
            "created_at": mem.created_at.isoformat(),
            "updated_at": mem.updated_at.isoformat(),
        }
        for mem in memories
    ]

    # Build document
    doc = {
        "format_version": _FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "tenant_id": tenant_id,
        "counts": {
            "episodes": len(episodes_data),
            "memories": len(memories_data),
        },
        "episodes": episodes_data,
        "memories": memories_data,
    }

    # Add checksum over data content
    content_str = json.dumps({"episodes": episodes_data, "memories": memories_data}, sort_keys=True)
    doc["checksum"] = hashlib.sha256(content_str.encode()).hexdigest()

    return doc


async def import_subject(
    doc: dict[str, Any],
    *,
    target_subject_id: str | None = None,
    target_tenant_id: str | None = None,
    preserve_ids: bool = True,
) -> dict[str, Any]:
    """Import a previously exported subject document.

    Options:
    - target_subject_id: override the subject_id (default: use original)
    - target_tenant_id: override the tenant_id (default: use original from doc)
    - preserve_ids: if True, keep original UUIDs (default); if False, generate new ones

    Returns summary of what was imported.

    Safety checks:
    - Validates format version
    - Validates checksum
    - Checks for ID conflicts when preserve_ids=True
    """
    # Validate format
    version = doc.get("format_version")
    if version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported format version: {version} (expected {_FORMAT_VERSION})")

    # Validate checksum
    episodes_data = doc.get("episodes", [])
    memories_data = doc.get("memories", [])
    content_str = json.dumps({"episodes": episodes_data, "memories": memories_data}, sort_keys=True)
    expected_checksum = hashlib.sha256(content_str.encode()).hexdigest()
    if doc.get("checksum") != expected_checksum:
        raise ValueError("Checksum mismatch — export file may be corrupted or tampered with")

    subject_id = target_subject_id or doc.get("subject_id")
    tenant_id = target_tenant_id if target_tenant_id is not None else doc.get("tenant_id")

    if not subject_id:
        raise ValueError("No subject_id specified and none found in export document")

    # ID remapping (old_id -> new_id) when not preserving
    id_map: dict[str, str] = {}

    async with async_session_factory() as session:
        # Check for conflicts if preserving IDs
        if preserve_ids and episodes_data:
            existing_ids = [ep["id"] for ep in episodes_data]
            conflict_stmt = select(EpisodeRow.id).where(
                EpisodeRow.id.in_([uuid.UUID(eid) for eid in existing_ids])
            )
            result = await session.execute(conflict_stmt)
            conflicts = result.scalars().all()
            if conflicts:
                raise ValueError(
                    f"ID conflict: {len(conflicts)} episode(s) already exist. "
                    "Use preserve_ids=false to generate new IDs, or delete existing data first."
                )

        # Import episodes
        for ep_data in episodes_data:
            old_id = ep_data["id"]
            new_id = uuid.UUID(old_id) if preserve_ids else uuid.uuid4()
            id_map[old_id] = str(new_id)

            row = EpisodeRow(
                id=new_id,
                subject_id=subject_id,
                tenant_id=tenant_id,
                source=ep_data["source"],
                type=ep_data["type"],
                payload=ep_data["payload"],
                metadata_=ep_data.get("metadata", {}),
                provenance=ep_data.get("provenance", {}),
                created_at=datetime.fromisoformat(ep_data["created_at"]),
                last_compiled_at=(
                    datetime.fromisoformat(ep_data["last_compiled_at"])
                    if ep_data.get("last_compiled_at")
                    else None
                ),
            )
            session.add(row)

        # Import memories
        for mem_data in memories_data:
            old_id = mem_data["id"]
            new_id = uuid.UUID(old_id) if preserve_ids else uuid.uuid4()

            # Remap source_episode_ids
            source_ep_ids = [
                uuid.UUID(id_map.get(eid, eid)) for eid in mem_data.get("source_episode_ids", [])
            ]

            row = MemoryRow(
                id=new_id,
                subject_id=subject_id,
                tenant_id=tenant_id,
                kind=mem_data["kind"],
                content=mem_data["content"],
                summary=mem_data.get("summary", ""),
                confidence=mem_data.get("confidence", 1.0),
                valid_from=datetime.fromisoformat(mem_data["valid_from"]),
                valid_to=(
                    datetime.fromisoformat(mem_data["valid_to"])
                    if mem_data.get("valid_to")
                    else None
                ),
                source_episode_ids=source_ep_ids,
                metadata_=mem_data.get("metadata", {}),
                status=mem_data.get("status", "active"),
                embedding=mem_data.get("embedding"),
                created_at=datetime.fromisoformat(mem_data["created_at"]),
                updated_at=datetime.fromisoformat(mem_data["updated_at"]),
            )
            session.add(row)

        await session.commit()

    logger.info(
        "subject_imported",
        subject_id=subject_id,
        episodes=len(episodes_data),
        memories=len(memories_data),
        preserve_ids=preserve_ids,
    )

    return {
        "subject_id": subject_id,
        "tenant_id": tenant_id,
        "episodes_imported": len(episodes_data),
        "memories_imported": len(memories_data),
        "ids_preserved": preserve_ids,
    }
