"""Vendor-neutral memory portability service.

One module, six operations:

  * `list_starter_packs`         — read `server/starter_packs/index.json`
  * `import_starter_pack`        — ingest a bundled pack into a new subject
  * `reseed_support_subject`     — rebuild `statewave-support-docs` from the
                                   bundled `statewave-support-agent` pack
  * `clone_subject`              — copy episodes/memories from one subject
                                   to another with provenance metadata
  * `export_memory_payload`      — produce a versioned JSON payload that
                                   the admin client encrypts client-side
  * `import_memory_payload`      — ingest a payload (after the admin client
                                   has decrypted it) with strict validation

These primitives back every memory action in the admin UI. There is one
generic ingest path (`_ingest_records`) reused by starter-pack import,
clone, and `.swmem` import — per the spec's "do not invent separate one-off
pipelines."

No memory content is logged anywhere in this module. Log lines carry
subject ids, counts, and pack ids only.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import structlog
from sqlalchemy import delete, func, select

from server.core.config import settings
from server.db import engine as engine_module
from server.db.tables import EpisodeRow, MemoryRow

logger = structlog.stdlib.get_logger()

PAYLOAD_FORMAT = "statewave-memory-payload"
PAYLOAD_FORMAT_VERSION = 1
STARTER_PACK_FORMAT = "statewave-starter-pack"
STARTER_PACK_INDEX_FORMAT = "statewave-starter-pack-index"

CloneScope = Literal[
    "episodes",
    "memories",
    "episodes_and_memories",
    "episodes_memories_sources",
]
ConflictStrategy = Literal["create_copy", "merge", "cancel"]

_STARTER_PACK_KIND = Literal["support_docs", "demo_agent"]

_PACKS_ROOT = Path(__file__).resolve().parent.parent / "starter_packs"

# Subject-id regex matches what the rest of Statewave already accepts: a
# loose, URL-safe identifier. We pin a length cap and reject anything that
# could collide with internal prefixes used by the marketing widget's
# per-visitor subjects (`demo_web_*`).
_SUBJECT_ID_RE = re.compile(r"^[A-Za-z0-9_.\-:]{1,128}$")
_RESERVED_PREFIXES = ("demo_web_",)


class StarterPackError(Exception):
    """Domain error with an attached HTTP status hint for the API layer."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ─── Helpers ────────────────────────────────────────────────────────────────


def _validate_subject_id(
    value: str,
    *,
    field: str = "subject_id",
    allow_reserved: bool = False,
) -> str:
    """Validate a subject id.

    `allow_reserved` exists because the reserved-prefix guard is about
    preventing operators from CREATING new subjects in the marketing
    widget's namespace (`demo_web_*`). Reading from such a subject —
    cloning it, exporting it, or otherwise inspecting it — is a legitimate
    operator action and shouldn't be blocked. Source-side call sites
    therefore set `allow_reserved=True`; target-side call sites keep the
    default and the guard fires.
    """
    if not value or not _SUBJECT_ID_RE.match(value):
        raise StarterPackError(
            f"{field!r} must be 1-128 characters of letters, digits, "
            "underscore, dot, dash, or colon.",
            status_code=400,
        )
    if not allow_reserved and any(value.startswith(p) for p in _RESERVED_PREFIXES):
        raise StarterPackError(
            f"{field!r} uses a reserved prefix ({', '.join(_RESERVED_PREFIXES)}).",
            status_code=400,
        )
    return value


def _suggest_target_subject_id(base: str) -> str:
    """Derive a fresh subject id from a suggested base.

    If `base` carries a reserved prefix (e.g. `demo_web_`), the prefix is
    rewritten to `imported-` so the new id can never land back in the
    marketing widget's namespace. This matters most on `.swmem` import:
    archives generated from a per-visitor subject carry the original
    `demo_web_<uuid>__support-agent` id, and we want a clean
    `imported-<uuid>__support-agent-<hex>` target instead of a collision.
    """
    base = base or "imported-subject"
    for prefix in _RESERVED_PREFIXES:
        if base.startswith(prefix):
            base = "imported-" + base[len(prefix):]
            break
    suffix = uuid.uuid4().hex[:8]
    candidate = f"{base}-{suffix}"
    return _validate_subject_id(candidate)


async def _count_subject(subject_id: str) -> tuple[int, int]:
    """Return (episode_count, memory_count) for a subject. 0/0 if missing."""
    async with engine_module.get_session_factory()() as session:
        ep = await session.scalar(
            select(func.count()).select_from(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
        )
        mem = await session.scalar(
            select(func.count()).select_from(MemoryRow).where(MemoryRow.subject_id == subject_id)
        )
        return int(ep or 0), int(mem or 0)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            out.append(json.loads(raw))
    return out


def _read_starter_pack_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise StarterPackError(
            f"Starter pack at {directory.name!r} is missing manifest.json.",
            status_code=500,
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != STARTER_PACK_FORMAT:
        raise StarterPackError(
            f"Starter pack {directory.name!r} has unexpected format "
            f"{manifest.get('format')!r}; expected {STARTER_PACK_FORMAT!r}.",
            status_code=500,
        )
    return manifest


def _starter_pack_dir(pack_id: str) -> tuple[Path, dict[str, Any]]:
    """Resolve `pack_id` to a directory + manifest. Raises 404 on miss."""
    index = _read_index()
    entry = next((p for p in index if p["pack_id"] == pack_id), None)
    if not entry:
        raise StarterPackError(
            f"Starter pack {pack_id!r} not found.", status_code=404
        )
    directory = _PACKS_ROOT / entry["directory"]
    manifest = _read_starter_pack_manifest(directory)
    return directory, manifest


def _read_index() -> list[dict[str, Any]]:
    index_path = _PACKS_ROOT / "index.json"
    if not index_path.exists():
        return []
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if raw.get("format") != STARTER_PACK_INDEX_FORMAT:
        raise StarterPackError(
            "Starter pack index has unexpected format.", status_code=500
        )
    return list(raw.get("packs", []))


# ─── Public: list starter packs ──────────────────────────────────────────────


def list_starter_packs() -> list[dict[str, Any]]:
    """Return manifest metadata for every bundled starter pack.

    Pack content (episodes/memories) is NOT included — only metadata, so
    the admin UI can render selectable cards. Manifest read errors on a
    single pack become a `__error` field on that entry rather than failing
    the whole list, so a corrupt pack can't break the picker.
    """
    out: list[dict[str, Any]] = []
    for entry in _read_index():
        directory = _PACKS_ROOT / entry["directory"]
        try:
            manifest = _read_starter_pack_manifest(directory)
        except StarterPackError as e:
            out.append({"pack_id": entry["pack_id"], "kind": entry.get("kind"), "__error": str(e)})
            continue
        out.append(
            {
                "pack_id": manifest["pack_id"],
                "kind": entry.get("kind"),
                "display_name": manifest.get("display_name"),
                "description": manifest.get("description"),
                "version": manifest.get("version"),
                "created_at": manifest.get("created_at"),
                "subject_id_suggestion": manifest.get("subject_id_suggestion"),
                "episode_count": manifest.get("episode_count", 0),
                "memory_count": manifest.get("memory_count", 0),
                "source_count": manifest.get("source_count", 0),
                "tags": manifest.get("tags", []),
            }
        )
    return out


# ─── Internal: ingest pipeline shared by starter-pack/clone/import ──────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _ingest_records_async(
    *,
    target_subject_id: str,
    target_tenant_id: str | None,
    episodes_data: Iterable[dict[str, Any]],
    memories_data: Iterable[dict[str, Any]],
    extra_provenance: dict[str, Any],
    extra_metadata: dict[str, Any],
) -> dict[str, int]:
    eps = list(episodes_data)
    mems = list(memories_data)
    if len(eps) > settings.memory_import_max_episodes:
        raise StarterPackError(
            f"Too many episodes: {len(eps)} > limit {settings.memory_import_max_episodes}.",
            status_code=413,
        )
    if len(mems) > settings.memory_import_max_memories:
        raise StarterPackError(
            f"Too many memories: {len(mems)} > limit {settings.memory_import_max_memories}.",
            status_code=413,
        )

    id_map: dict[str, str] = {}
    async with engine_module.get_session_factory()() as session:
        for ep in eps:
            old_id = ep.get("id") or ep.get("original_episode_id")
            new_id = uuid.uuid4()
            if old_id:
                id_map[str(old_id)] = str(new_id)
            metadata = dict(ep.get("metadata") or {})
            metadata.update(extra_metadata)
            provenance = dict(ep.get("provenance") or {})
            provenance.update(extra_provenance)
            row = EpisodeRow(
                id=new_id,
                subject_id=target_subject_id,
                tenant_id=target_tenant_id,
                source=ep.get("source") or "memory-pack",
                type=ep.get("type") or "imported",
                payload=ep.get("payload") or {},
                metadata_=metadata,
                provenance=provenance,
                created_at=_parse_iso_or_now(ep.get("created_at")),
                last_compiled_at=_parse_iso_or_none(ep.get("last_compiled_at")),
            )
            session.add(row)

        for mem in mems:
            new_id = uuid.uuid4()
            source_eps: list[uuid.UUID] = []
            for raw in mem.get("source_episode_ids") or []:
                raw_str = str(raw)
                mapped = id_map.get(raw_str, raw_str)
                try:
                    source_eps.append(uuid.UUID(mapped))
                except (ValueError, TypeError):
                    continue
            metadata = dict(mem.get("metadata") or {})
            metadata.update(extra_metadata)
            row = MemoryRow(
                id=new_id,
                subject_id=target_subject_id,
                tenant_id=target_tenant_id,
                kind=mem.get("kind") or "fact",
                content=mem.get("content") or "",
                summary=mem.get("summary") or mem.get("content") or "",
                confidence=float(mem.get("confidence", 0.9)),
                valid_from=_parse_iso_or_now(mem.get("valid_from")),
                valid_to=_parse_iso_or_none(mem.get("valid_to")),
                source_episode_ids=source_eps,
                metadata_=dict(metadata),
                status=mem.get("status") or "active",
                embedding=mem.get("embedding"),
                created_at=_parse_iso_or_now(mem.get("created_at")),
                updated_at=_parse_iso_or_now(mem.get("updated_at")),
            )
            session.add(row)

        await session.commit()

    return {"episodes": len(eps), "memories": len(mems)}


def _parse_iso_or_now(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.now(timezone.utc)


def _parse_iso_or_none(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


# ─── Public: starter pack import ────────────────────────────────────────────


async def import_starter_pack(
    *,
    pack_id: str,
    target_subject_id: str | None,
    target_display_name: str | None,
    target_tenant_id: str | None,
    conflict_strategy: ConflictStrategy = "create_copy",
) -> dict[str, Any]:
    directory, manifest = _starter_pack_dir(pack_id)
    suggested = manifest.get("subject_id_suggestion") or pack_id

    if target_subject_id:
        target_subject_id = _validate_subject_id(target_subject_id, field="target_subject_id")

    final_subject_id = await _resolve_target_subject(
        explicit=target_subject_id,
        suggested=suggested,
        conflict_strategy=conflict_strategy,
    )

    episodes_data = _load_jsonl(directory / "episodes.jsonl")
    memories_data = _load_jsonl(directory / "memories.jsonl")

    extra_metadata = {
        "starter_pack_id": pack_id,
        "starter_pack_version": manifest.get("version"),
        "imported_at": _now_iso(),
    }
    if target_display_name:
        extra_metadata["display_name"] = target_display_name

    counts = await _ingest_records_async(
        target_subject_id=final_subject_id,
        target_tenant_id=target_tenant_id,
        episodes_data=episodes_data,
        memories_data=memories_data,
        extra_provenance={"starter_pack_id": pack_id},
        extra_metadata=extra_metadata,
    )

    logger.info(
        "starter_pack_imported",
        pack_id=pack_id,
        target_subject_id=final_subject_id,
        episodes=counts["episodes"],
        memories=counts["memories"],
        conflict_strategy=conflict_strategy,
    )

    return {
        "pack_id": pack_id,
        "target_subject_id": final_subject_id,
        "imported_episodes": counts["episodes"],
        "imported_memories": counts["memories"],
        "imported_sources": 0,
        "conflict_strategy": conflict_strategy,
        "imported_at": _now_iso(),
    }


async def _resolve_target_subject(
    *,
    explicit: str | None,
    suggested: str,
    conflict_strategy: ConflictStrategy,
) -> str:
    """Pick the final subject id given the conflict strategy."""
    if explicit:
        ep_count, mem_count = await _count_subject(explicit)
        already_populated = ep_count > 0 or mem_count > 0
        if not already_populated:
            return explicit
        if conflict_strategy == "merge":
            return explicit
        if conflict_strategy == "cancel":
            raise StarterPackError(
                f"Subject {explicit!r} already has data ({ep_count} episodes, "
                f"{mem_count} memories). Pick a different target id, choose "
                "'merge', or delete the existing subject first.",
                status_code=409,
            )
        # create_copy: pick a new id derived from the explicit one
        return _suggest_target_subject_id(explicit)

    # No explicit id: derive from the suggested base.
    #
    # If `suggested` lives in a reserved namespace (e.g. an archive
    # captured from a `demo_web_*` visitor subject), we MUST go through
    # `_suggest_target_subject_id` so the prefix gets rewritten — using
    # the suggested id verbatim would land the new subject back in the
    # marketing widget's namespace.
    if any(suggested.startswith(p) for p in _RESERVED_PREFIXES):
        return _suggest_target_subject_id(suggested)

    ep_count, mem_count = await _count_subject(suggested)
    if ep_count == 0 and mem_count == 0:
        return _validate_subject_id(suggested)
    return _suggest_target_subject_id(suggested)


# ─── Public: support reseed (vendor-neutral) ────────────────────────────────


async def reseed_support_subject(*, reason: str | None = None) -> dict[str, Any]:
    """Rebuild the shared `statewave-support-docs` subject from the bundled pack.

    Idempotent: every call deletes existing episodes/memories on the target
    subject before re-importing. Per-visitor `demo_web_*__statewave-support`
    subjects are not touched — this targets the configured shared subject id
    (`settings.support_subject_id`) only.
    """
    target = settings.support_subject_id
    pack_id = settings.support_starter_pack_id

    directory, manifest = _starter_pack_dir(pack_id)
    episodes_data = _load_jsonl(directory / "episodes.jsonl")
    memories_data = _load_jsonl(directory / "memories.jsonl")

    # Idempotent reseed: wipe the shared subject's existing rows. The strict
    # subject-id match means per-visitor `demo_web_<uuid>__statewave-support`
    # rows are not affected.
    async with engine_module.get_session_factory()() as session:
        await session.execute(delete(EpisodeRow).where(EpisodeRow.subject_id == target))
        await session.execute(delete(MemoryRow).where(MemoryRow.subject_id == target))
        await session.commit()

    counts = await _ingest_records_async(
        target_subject_id=target,
        target_tenant_id=None,
        episodes_data=episodes_data,
        memories_data=memories_data,
        extra_provenance={"starter_pack_id": pack_id, "support_reseed": True},
        extra_metadata={
            "starter_pack_id": pack_id,
            "starter_pack_version": manifest.get("version"),
            "imported_at": _now_iso(),
            "reseed_reason": reason,
        },
    )

    logger.info(
        "support_reseed_completed",
        target_subject_id=target,
        pack_id=pack_id,
        episodes=counts["episodes"],
        memories=counts["memories"],
    )

    return {
        "subject_id": target,
        "pack_id": pack_id,
        "pack_version": manifest.get("version"),
        "imported_episodes": counts["episodes"],
        "imported_memories": counts["memories"],
        "reseeded_at": _now_iso(),
        "reason": reason,
    }


# ─── Public: clone subject ──────────────────────────────────────────────────


async def clone_subject(
    *,
    source_subject_id: str,
    target_subject_id: str | None,
    target_display_name: str | None,
    target_tenant_id: str | None,
    clone_scope: CloneScope = "episodes_memories_sources",
    cloned_by: str | None = None,
) -> dict[str, Any]:
    """Clone a subject's records into a new subject.

    Reads the source through the existing `export_subject` primitive and
    routes the result through the same `_ingest_records_async` ingest
    pipeline used by starter-pack import — so clone, starter-pack import,
    and `.swmem` import share one write path.

    Source-side validation accepts reserved-prefix ids so operators can
    fork a visitor memory subject (`demo_web_<uuid>__support-agent`) for
    inspection. Target-side validation keeps the guard so the new id
    never lands back in the marketing widget's namespace.

    Provenance metadata is written onto every copied record:
      * `cloned_from_subject_id` — source subject id
      * `cloned_at` — ISO timestamp of the clone operation
      * `cloned_by` — operator email when supplied by the admin proxy
      * `original_episode_id` / `original_memory_id` — the source record
        id, so a future export of the cloned subject can still reference
        the original observation.

    Sources / citations are NOT yet first-class records in the storage
    schema — `source_count` always returns 0 even when the scope is
    `episodes_memories_sources`. This is documented as a known limitation;
    the rest of the scopes are honoured.
    """
    # Source side: allow reserved-prefix ids so operators can clone visitor
    # memory subjects (e.g. demo_web_<uuid>__support-agent) for inspection.
    # Target side keeps the guard so the clone can't *create* a new id in
    # that namespace.
    _validate_subject_id(source_subject_id, field="source_subject_id", allow_reserved=True)

    if target_subject_id:
        _validate_subject_id(target_subject_id, field="target_subject_id")

    src_eps, src_mems = await _count_subject(source_subject_id)
    if src_eps == 0 and src_mems == 0:
        raise StarterPackError(
            f"Source subject {source_subject_id!r} not found "
            "(no episodes or memories under that id).",
            status_code=404,
        )

    final_target = await _resolve_target_subject(
        explicit=target_subject_id,
        suggested=f"{source_subject_id}-clone",
        conflict_strategy="cancel",  # clone never overwrites without explicit retry
    )

    # Read source records via the existing backup primitive — same shape the
    # export endpoint produces, so the ingest path stays unified.
    from server.services.backup import export_subject

    source_doc = await export_subject(source_subject_id, tenant_id=None)

    include_episodes = clone_scope in (
        "episodes",
        "episodes_and_memories",
        "episodes_memories_sources",
    )
    include_memories = clone_scope in (
        "memories",
        "episodes_and_memories",
        "episodes_memories_sources",
    )

    eps = source_doc.get("episodes", []) if include_episodes else []
    mems = source_doc.get("memories", []) if include_memories else []

    cloned_at = _now_iso()
    extra_provenance: dict[str, Any] = {
        "cloned_from_subject_id": source_subject_id,
        "cloned_at": cloned_at,
    }
    if cloned_by:
        extra_provenance["cloned_by"] = cloned_by
    extra_metadata = dict(extra_provenance)
    if target_display_name:
        extra_metadata["display_name"] = target_display_name

    # Stamp each record with its original id BEFORE the ingest pipeline
    # rewrites the row id. The pipeline already preserves any keys we
    # pass on `metadata` / `provenance`; we just need to lift the source
    # id from the top-level shape it arrives in.
    eps_with_provenance = [
        {
            **ep,
            "metadata": {
                **(ep.get("metadata") or {}),
                "original_episode_id": ep.get("id"),
            },
            "provenance": {
                **(ep.get("provenance") or {}),
                "original_episode_id": ep.get("id"),
            },
        }
        for ep in eps
    ]
    mems_with_provenance = [
        {
            **mem,
            "metadata": {
                **(mem.get("metadata") or {}),
                "original_memory_id": mem.get("id"),
            },
        }
        for mem in mems
    ]

    counts = await _ingest_records_async(
        target_subject_id=final_target,
        target_tenant_id=target_tenant_id,
        episodes_data=eps_with_provenance,
        memories_data=mems_with_provenance,
        extra_provenance=extra_provenance,
        extra_metadata=extra_metadata,
    )

    logger.info(
        "subject_cloned",
        source_subject_id=source_subject_id,
        target_subject_id=final_target,
        clone_scope=clone_scope,
        episode_count=counts["episodes"],
        memory_count=counts["memories"],
        cloned_by=cloned_by,
    )

    return {
        "status": "cloned",
        "source_subject_id": source_subject_id,
        "target_subject_id": final_target,
        "target_display_name": target_display_name,
        "clone_scope": clone_scope,
        "episode_count": counts["episodes"],
        "memory_count": counts["memories"],
        # Sources/citations are not yet first-class cloneable records —
        # see the docstring above. Returning the literal 0 rather than
        # omitting the key keeps the response shape stable.
        "source_count": 0,
        "cloned_at": cloned_at,
    }


# ─── Public: export memory payload ──────────────────────────────────────────


async def export_memory_payload(
    *,
    subject_ids: list[str],
    tenant_id: str | None,
    export_scope: CloneScope = "episodes_memories_sources",
) -> dict[str, Any]:
    if not subject_ids:
        raise StarterPackError("subject_ids is required.", status_code=400)
    if len(subject_ids) > settings.memory_import_max_subjects:
        raise StarterPackError(
            f"Too many subjects in one export: {len(subject_ids)} > "
            f"limit {settings.memory_import_max_subjects}.",
            status_code=413,
        )
    for sid in subject_ids:
        # Same source-side carve-out as clone: exporting a visitor subject
        # for inspection / backup is legitimate.
        _validate_subject_id(sid, field="subject_ids", allow_reserved=True)

    include_episodes = export_scope in (
        "episodes",
        "episodes_and_memories",
        "episodes_memories_sources",
    )
    include_memories = export_scope in (
        "memories",
        "episodes_and_memories",
        "episodes_memories_sources",
    )

    from server.services.backup import export_subject

    subjects: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []

    for sid in subject_ids:
        doc = await export_subject(sid, tenant_id=tenant_id)
        subjects.append(
            {
                "original_subject_id": sid,
                "tenant_id": doc.get("tenant_id"),
                "display_name": None,
                "tags": [],
                "metadata": {},
                "exported_at": doc.get("exported_at"),
            }
        )
        if include_episodes:
            for ep in doc.get("episodes", []):
                episodes.append(
                    {
                        "original_episode_id": ep.get("id"),
                        "subject_id": ep.get("subject_id") or sid,
                        "source": ep.get("source"),
                        "type": ep.get("type"),
                        "payload": ep.get("payload"),
                        "metadata": ep.get("metadata") or {},
                        "provenance": ep.get("provenance") or {},
                        "created_at": ep.get("created_at"),
                        "last_compiled_at": ep.get("last_compiled_at"),
                    }
                )
        if include_memories:
            for mem in doc.get("memories", []):
                memories.append(
                    {
                        "original_memory_id": mem.get("id"),
                        "subject_id": mem.get("subject_id") or sid,
                        "kind": mem.get("kind"),
                        "content": mem.get("content"),
                        "summary": mem.get("summary"),
                        "confidence": mem.get("confidence"),
                        "valid_from": mem.get("valid_from"),
                        "valid_to": mem.get("valid_to"),
                        "source_episode_ids": mem.get("source_episode_ids") or [],
                        "status": mem.get("status"),
                        "metadata": mem.get("metadata") or {},
                        "created_at": mem.get("created_at"),
                        "updated_at": mem.get("updated_at"),
                    }
                )

    payload = {
        "format": PAYLOAD_FORMAT,
        "format_version": PAYLOAD_FORMAT_VERSION,
        "export_id": uuid.uuid4().hex,
        "exported_at": _now_iso(),
        "export_scope": export_scope,
        "subjects": subjects,
        "episodes": episodes,
        "memories": memories,
        "sources": [],
        "metadata": {
            "statewave_version": "0.1",
            "subject_count": len(subjects),
            "episode_count": len(episodes),
            "memory_count": len(memories),
            "source_count": 0,
        },
    }
    logger.info(
        "memory_export_built",
        subject_count=len(subjects),
        episode_count=len(episodes),
        memory_count=len(memories),
        export_scope=export_scope,
    )
    return payload


# ─── Public: import memory payload ──────────────────────────────────────────

_ALLOWED_PAYLOAD_KEYS = {
    "format",
    "format_version",
    "export_id",
    "exported_at",
    "export_scope",
    "subjects",
    "episodes",
    "memories",
    "sources",
    "metadata",
}


async def import_memory_payload(
    *,
    payload: dict[str, Any],
    target_tenant_id: str | None,
    conflict_strategy: ConflictStrategy = "create_copy",
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise StarterPackError("Payload must be a JSON object.", status_code=400)
    if payload.get("format") != PAYLOAD_FORMAT:
        raise StarterPackError(
            f"Unexpected payload format {payload.get('format')!r}; "
            f"expected {PAYLOAD_FORMAT!r}.",
            status_code=400,
        )
    if int(payload.get("format_version", 0)) != PAYLOAD_FORMAT_VERSION:
        raise StarterPackError(
            f"Unsupported format_version {payload.get('format_version')}; "
            f"expected {PAYLOAD_FORMAT_VERSION}.",
            status_code=400,
        )

    extra = set(payload.keys()) - _ALLOWED_PAYLOAD_KEYS
    if extra:
        raise StarterPackError(
            f"Payload contains unknown fields: {sorted(extra)}",
            status_code=400,
        )

    # Hard limit by serialized size — same number the request body limit guard
    # uses, kept here in case a future ingress strips it.
    serialized = json.dumps(payload)
    if len(serialized.encode("utf-8")) > settings.memory_import_max_bytes:
        raise StarterPackError(
            f"Payload exceeds size limit ({settings.memory_import_max_bytes} bytes).",
            status_code=413,
        )

    subjects = payload.get("subjects") or []
    episodes = payload.get("episodes") or []
    memories = payload.get("memories") or []
    if len(subjects) > settings.memory_import_max_subjects:
        raise StarterPackError(
            f"Too many subjects: {len(subjects)} > limit {settings.memory_import_max_subjects}.",
            status_code=413,
        )

    # Map each `original_subject_id` to a final target subject id. With
    # `create_copy` (the default) every imported subject gets a fresh id;
    # `merge` reuses the original; `cancel` aborts on any pre-existing
    # populated subject.
    id_map: dict[str, str] = {}
    for s in subjects:
        original = s.get("original_subject_id")
        if not original:
            continue
        # Source-side: the original_subject_id is metadata from the archive
        # so we accept reserved-prefix ids (e.g. `demo_web_*` archives
        # captured from the marketing widget). The resulting target id is
        # rewritten to `imported-*` by `_suggest_target_subject_id` so the
        # new subject doesn't land in the marketing widget's namespace.
        # `merge` and `cancel` strategies still pass the original through
        # to `_resolve_target_subject` as `explicit`, which validates it
        # WITHOUT the source-side carve-out — that's intentional, an admin
        # cannot merge into the visitor namespace.
        _validate_subject_id(
            original,
            field="subjects[].original_subject_id",
            allow_reserved=True,
        )
        final = await _resolve_target_subject(
            explicit=original if conflict_strategy != "create_copy" else None,
            suggested=original,
            conflict_strategy=conflict_strategy,
        )
        id_map[original] = final

    # Group episodes / memories by their source subject and ingest each group
    # under the mapped target id.
    eps_by_subject: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        original = ep.get("subject_id") or ""
        if not original:
            continue
        eps_by_subject.setdefault(original, []).append(ep)

    mems_by_subject: dict[str, list[dict[str, Any]]] = {}
    for mem in memories:
        original = mem.get("subject_id") or ""
        if not original:
            continue
        mems_by_subject.setdefault(original, []).append(mem)

    total_episodes = 0
    total_memories = 0
    target_subject_ids: list[str] = []

    for original_id in {*eps_by_subject.keys(), *mems_by_subject.keys()}:
        if original_id not in id_map:
            # Subject metadata wasn't included; create a fresh target id so
            # the records aren't dropped silently.
            id_map[original_id] = _suggest_target_subject_id(original_id)
        target_id = id_map[original_id]
        target_subject_ids.append(target_id)
        counts = await _ingest_records_async(
            target_subject_id=target_id,
            target_tenant_id=target_tenant_id,
            episodes_data=eps_by_subject.get(original_id, []),
            memories_data=mems_by_subject.get(original_id, []),
            extra_provenance={
                "imported_from_export_id": payload.get("export_id"),
                "original_subject_id": original_id,
                "imported_at": _now_iso(),
            },
            extra_metadata={
                "imported_from_export_id": payload.get("export_id"),
                "original_subject_id": original_id,
                "imported_at": _now_iso(),
            },
        )
        total_episodes += counts["episodes"]
        total_memories += counts["memories"]

    logger.info(
        "memory_payload_imported",
        export_id=payload.get("export_id"),
        target_subjects=target_subject_ids,
        episode_count=total_episodes,
        memory_count=total_memories,
        conflict_strategy=conflict_strategy,
    )

    return {
        "imported_at": _now_iso(),
        "export_id": payload.get("export_id"),
        "conflict_strategy": conflict_strategy,
        "subject_id_map": id_map,
        "imported_subjects": list(id_map.values()),
        "imported_episodes": total_episodes,
        "imported_memories": total_memories,
        "imported_sources": 0,
    }


