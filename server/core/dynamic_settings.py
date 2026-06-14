"""DB-backed override layer for env-driven settings.

Reads `system_settings` / `tenant_settings` on demand and resolves with the
precedence chain:

    tenant_override → global_db → env (pydantic Settings) → hardcoded_default

Env continues to be honoured for anything not overridden in the DB, so
existing deployments keep working without any config change. The first PATCH
against a setting causes the DB layer to start participating.

Process-wide cache: settings tables are read on every resolution by default;
a tiny TTL cache (``_DB_CACHE_TTL_SECONDS``) keeps hot paths (`/v1/context`,
webhook fan-out) from doing a SELECT-per-request. The cache is invalidated
whenever ``apply_global_override`` / ``delete_global_override`` /
``apply_tenant_override`` / ``delete_tenant_override`` runs in this process.
Cross-process invalidation (multi-replica) is out of scope for v0; running
operators see eventual consistency within the TTL window.

Audit + test-probe live here too — see ``record_audit`` and ``test_probe``.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    delete,
    desc,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.config import settings as env_settings
from server.core.settings_catalogue import (
    CATALOGUE,
    SettingSpec,
    get_spec,
    is_secret,
    redact,
)

# ─── Constants ────────────────────────────────────────────────────────────

# Hot paths hit this every request — too long and operator edits feel
# laggy; too short and we re-query the DB pointlessly. 5s is a deliberate
# floor — operator UX of "I changed the setting and the next request
# picked it up" still feels instant.
_DB_CACHE_TTL_SECONDS: float = 5.0


# ─── Boot-time snapshots ─────────────────────────────────────────────────
#
# `_env_baseline` is the pristine env value for every catalogued key,
# captured at module import BEFORE any DB override mutates env_settings.
# This is what the UI labels as `source: "env"` — operators expect "env"
# to mean "your deployment env file", not "the post-boot mutated value".
#
# `_applied_values` is what env_settings.X currently holds — i.e. what the
# live process has actually committed to. Mutated in lock-step with
# `apply_db_overrides_to_settings()` so the snapshot can detect "this DB
# override hasn't been picked up yet" by comparing effective vs applied.
#
# This split is what makes the "Restart required" banner correct: it
# fires iff DB effective ≠ process applied, instead of the naïve
# "source ≠ env" which can never clear (the DB row always exists after
# a patch, even after restart). Without this split, an operator would
# see "Restart required" forever after their first override.
_env_baseline: dict[str, Any] = {}
_applied_values: dict[str, Any] = {}


def _initialise_boot_snapshots() -> None:
    """Populate the baseline + applied snapshots from the live Settings
    object. Idempotent: only fills entries it hasn't seen yet, so a
    sibling test that mutates env_settings between calls can't poison
    the baseline."""
    for key in CATALOGUE:
        if key not in _env_baseline:
            _env_baseline[key] = getattr(env_settings, key, None)
        if key not in _applied_values:
            _applied_values[key] = getattr(env_settings, key, None)


# Capture at import. The pydantic Settings has already loaded env vars by
# the time this module is imported (the `from .config import settings`
# above triggered it), so the values here are the true env state.
_initialise_boot_snapshots()

# Hard cap on audit rows so a script that flips a setting every second
# can't unbounded-grow the table. ~100K rows ≈ 30MB and covers >2y of
# day-to-day operator activity. Past this, the oldest rows are rolled off
# inside ``record_audit`` (cheap O(1)-ish DELETE).
_AUDIT_HARD_CAP: int = 100_000

# Loud sentinel for the no-cached-value-yet branch.
_SENTINEL = object()


# ─── Tables (Core, not ORM) ───────────────────────────────────────────────
#
# We define the tables here in their own MetaData rather than adding them to
# ``server/db/tables.py``. Settings are infrastructure, not application
# objects — they have no relationships to the rest of the schema, are
# never exported, never appear in receipts or snapshots, and are only
# touched by this module + the admin endpoints. Keeping them off the main
# DeclarativeBase avoids polluting it with rows that should never be
# auto-discovered by snapshot / clone routines.

_metadata = MetaData()

system_settings = Table(
    "system_settings",
    _metadata,
    Column("key", String(128), primary_key=True),
    Column("value", JSONB, nullable=False),
    Column("category", String(64), nullable=False),
    Column("is_secret", Boolean, nullable=False, default=False),
    Column("set_by", String(256), nullable=True),
    Column("set_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

system_settings_audit = Table(
    "system_settings_audit",
    _metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("key", String(128), nullable=False),
    Column("old_value", JSONB, nullable=True),
    Column("new_value", JSONB, nullable=True),
    Column("action", String(32), nullable=False),
    Column("tenant_id", String(256), nullable=True),
    Column("changed_by", String(256), nullable=True),
    Column(
        "changed_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    Column("note", Text, nullable=True),
)

tenant_settings = Table(
    "tenant_settings",
    _metadata,
    Column("tenant_id", String(256), primary_key=True),
    Column("key", String(128), primary_key=True),
    Column("value", JSONB, nullable=False),
    Column("is_secret", Boolean, nullable=False, default=False),
    Column("set_by", String(256), nullable=True),
    Column("set_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)


# ─── Cache ────────────────────────────────────────────────────────────────


@dataclass
class _CachedScope:
    """In-process snapshot of one (tenant_id or None) settings scope."""

    values: dict[str, Any]
    expires_at: float


_global_cache: _CachedScope | None = None
_tenant_cache: dict[str, _CachedScope] = {}
_cache_lock = asyncio.Lock()


def _now() -> float:
    return time.monotonic()


async def _load_global_overrides(session: AsyncSession) -> dict[str, Any]:
    """Read all rows from system_settings into a {key: value} dict."""
    rows = (await session.execute(select(system_settings))).mappings().all()
    return {row["key"]: row["value"] for row in rows}


async def _load_tenant_overrides(session: AsyncSession, tenant_id: str) -> dict[str, Any]:
    """Read all rows for one tenant from tenant_settings."""
    rows = (
        await session.execute(
            select(tenant_settings).where(tenant_settings.c.tenant_id == tenant_id)
        )
    ).mappings().all()
    return {row["key"]: row["value"] for row in rows}


async def _global_overrides_cached(session: AsyncSession) -> dict[str, Any]:
    """Return cached global overrides, refreshing if past TTL."""
    global _global_cache
    async with _cache_lock:
        if _global_cache is None or _global_cache.expires_at < _now():
            values = await _load_global_overrides(session)
            _global_cache = _CachedScope(values=values, expires_at=_now() + _DB_CACHE_TTL_SECONDS)
        return _global_cache.values


async def _tenant_overrides_cached(session: AsyncSession, tenant_id: str) -> dict[str, Any]:
    async with _cache_lock:
        cached = _tenant_cache.get(tenant_id)
        if cached is None or cached.expires_at < _now():
            values = await _load_tenant_overrides(session, tenant_id)
            _tenant_cache[tenant_id] = _CachedScope(
                values=values, expires_at=_now() + _DB_CACHE_TTL_SECONDS
            )
        return _tenant_cache[tenant_id].values


def _invalidate_global() -> None:
    global _global_cache
    _global_cache = None


def _invalidate_tenant(tenant_id: str) -> None:
    _tenant_cache.pop(tenant_id, None)


def invalidate_cache(tenant_id: str | None = None) -> None:
    """Invalidate cached overrides. Tests and reseed scripts call this
    directly; mutation helpers below do it transparently."""
    if tenant_id is None:
        _invalidate_global()
    else:
        _invalidate_tenant(tenant_id)


# ─── Env-default reads ────────────────────────────────────────────────────


def _env_default(key: str) -> Any:
    """Return the original env-time value for `key`.

    Reads from `_env_baseline` (captured at module import, before any DB
    overrides applied) so the UI's "env" source label reflects the
    deployment env — not the post-boot mutation. Falls through to the
    live Settings attribute for keys not yet baselined (test fixtures
    that touch a key before `_initialise_boot_snapshots` ran)."""
    if key in _env_baseline:
        return _env_baseline[key]
    return getattr(env_settings, key, None)


def _applied_value(key: str) -> Any:
    """Return env_settings.X as the live process is reading it right now.

    Different from `_env_default` whenever a DB override was applied at
    boot — those mutated env_settings to the DB value, so the process
    "applied" what the DB said at the time. Used by the snapshot to
    compute `pending_restart`."""
    if key in _applied_values:
        return _applied_values[key]
    return getattr(env_settings, key, None)


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


# ─── Validation ───────────────────────────────────────────────────────────


class SettingValidationError(ValueError):
    """Raised when a PATCH value doesn't match the catalogue spec.

    Carries a short, human-readable message — the admin endpoint surfaces
    it as a 400 with `code = settings.invalid`."""


def _validate_value(spec: SettingSpec, value: Any) -> Any:
    """Type-check + light coerce a PATCH value against the spec.

    Heavy domain validation (e.g. a URL ping-test for `webhook_url`) is the
    job of ``test_probe`` — this function only rejects shape mismatches
    that can't possibly work."""
    if value is None:
        if spec.kind == "string_or_null":
            return None
        raise SettingValidationError(f"{spec.key}: null is not allowed for kind {spec.kind}")

    if spec.kind in ("string", "string_or_null"):
        if not isinstance(value, str):
            raise SettingValidationError(f"{spec.key}: expected string, got {type(value).__name__}")
        # Enum check happens BEFORE format check: if `allowed_values` is set,
        # nothing else applies — value must be exactly one of those.
        if spec.allowed_values is not None and value not in spec.allowed_values:
            hint = _suggest(value, spec.allowed_values)
            allowed = ", ".join(repr(v) for v in spec.allowed_values)
            msg = f"{spec.key}: {value!r} is not allowed (must be one of: {allowed})"
            if hint:
                msg += f". Did you mean {hint!r}?"
            raise SettingValidationError(msg)
        if spec.format == "url" and value:
            if not (value.startswith("http://") or value.startswith("https://")):
                raise SettingValidationError(
                    f"{spec.key}: URL must start with http:// or https://"
                )
        return value

    if spec.kind == "int":
        # Reject bool — Python bool is an int subclass; an accidental
        # `true` would silently land as 1 otherwise.
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingValidationError(f"{spec.key}: expected int, got {type(value).__name__}")
        _check_range(spec, value)
        return value

    if spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingValidationError(f"{spec.key}: expected number, got {type(value).__name__}")
        coerced = float(value)
        _check_range(spec, coerced)
        return coerced

    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise SettingValidationError(f"{spec.key}: expected bool, got {type(value).__name__}")
        return value

    if spec.kind == "json":
        if not _is_jsonable(value):
            raise SettingValidationError(f"{spec.key}: value is not JSON-serialisable")
        return value

    raise SettingValidationError(f"{spec.key}: unknown kind {spec.kind!r}")


def _check_range(spec: SettingSpec, value: float | int) -> None:
    if spec.min_value is not None and value < spec.min_value:
        raise SettingValidationError(
            f"{spec.key}: {value} is below minimum {spec.min_value}"
        )
    if spec.max_value is not None and value > spec.max_value:
        raise SettingValidationError(
            f"{spec.key}: {value} is above maximum {spec.max_value}"
        )


def _suggest(value: str, options: tuple[str, ...]) -> str | None:
    """Return the closest match from `options` if it's within an edit-
    distance of 2, else None. Used to turn 'lllm' → 'Did you mean llm?'
    instead of a bare 'not allowed' error.

    Pure Levenshtein, no fuzzy-string library dependency."""
    best: tuple[int, str | None] = (10**9, None)
    for opt in options:
        d = _levenshtein(value.lower(), opt.lower())
        if d < best[0]:
            best = (d, opt)
    if best[1] is not None and best[0] <= 2:
        return best[1]
    return None


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                cur[-1] + 1,           # insert
                prev[j] + 1,           # delete
                prev[j - 1] + (ca != cb),  # substitute
            ))
        prev = cur
    return prev[-1]


# ─── Public API: resolution ───────────────────────────────────────────────


async def get_setting(
    key: str,
    session: AsyncSession,
    *,
    tenant_id: str | None = None,
) -> Any:
    """Resolve the effective value of one setting.

    Precedence: tenant_override → global_db → env. Returns ``None`` only
    if env itself is None (e.g. unset `api_key`).
    """
    spec = get_spec(key)
    if spec is None:
        # Unknown keys never resolve via the DB layer — caller should
        # be reading via the Settings object directly.
        return _env_default(key)

    if tenant_id is not None and spec.tenant_overridable:
        tenant_values = await _tenant_overrides_cached(session, tenant_id)
        if key in tenant_values:
            return tenant_values[key]

    global_values = await _global_overrides_cached(session)
    if key in global_values:
        return global_values[key]

    return _env_default(key)


async def get_effective_snapshot(
    session: AsyncSession,
    *,
    tenant_id: str | None = None,
    redact_secrets: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return effective values for every catalogued setting.

    Shape:

        {
          "litellm_model": {
            "value": "gpt-4o-mini",
            "source": "env" | "global_db" | "tenant_db",
            "is_secret": False,
            "category": "llm",
            "kind": "string",
            "env_name": "STATEWAVE_LITELLM_MODEL",
            "description": "...",
            "hot_reloadable": True,
            "tenant_overridable": True,
            "editable": True,
          },
          ...
        }

    Secrets are redacted by default — pass ``redact_secrets=False`` only
    on internal call paths that need the raw value (never on a response
    that leaves the process).
    """
    global_values = await _global_overrides_cached(session)
    tenant_values: dict[str, Any] = {}
    if tenant_id is not None:
        tenant_values = await _tenant_overrides_cached(session, tenant_id)

    out: dict[str, dict[str, Any]] = {}
    for key, spec in CATALOGUE.items():
        if tenant_id is not None and spec.tenant_overridable and key in tenant_values:
            value = tenant_values[key]
            source = "tenant_db"
        elif key in global_values:
            value = global_values[key]
            source = "global_db"
        else:
            value = _env_default(key)
            source = "env"
        applied = _applied_value(key)
        # `pending_restart`: true iff the live process is running a
        # different value than the DB currently resolves to. Only
        # meaningful for non-hot-reloadable settings; hot-reloadable
        # ones are read per-request and never need a restart.
        #
        # The diff (effective vs applied) — not source ≠ env — is what
        # makes the banner CLEAR on its own after the operator restarts:
        # post-restart `_applied_values[key]` equals the DB value, so
        # the diff disappears.
        #
        # Tenant overrides are deliberately excluded: they're
        # per-request lookups picked up live by `get_setting()`, so
        # changing one never needs a process restart.
        pending_restart = (
            not spec.hot_reloadable
            and source != "tenant_db"
            and value != applied
        )
        # Redaction is the LAST step so we compare raw values for the
        # `pending_restart` diff but still ship `•••<tail>` to the wire.
        if redact_secrets and spec.is_secret:
            value = redact(value, key=key)
            applied = redact(applied, key=key)
        out[key] = {
            "value": value,
            "applied_value": applied,
            "pending_restart": pending_restart,
            "source": source,
            "is_secret": spec.is_secret,
            "category": spec.category,
            "kind": spec.kind,
            "env_name": spec.env_name,
            "description": spec.description,
            "hot_reloadable": spec.hot_reloadable,
            "tenant_overridable": spec.tenant_overridable,
            "editable": spec.editable,
            "allowed_values": list(spec.allowed_values) if spec.allowed_values else None,
            "min_value": spec.min_value,
            "max_value": spec.max_value,
            "format": spec.format,
        }
    return out


# ─── Public API: mutation ────────────────────────────────────────────────


async def apply_global_override(
    key: str,
    value: Any,
    session: AsyncSession,
    *,
    changed_by: str | None = None,
    note: str | None = None,
) -> Any:
    """UPSERT a global override and append an audit row.

    Returns the validated (and possibly coerced) value as persisted. Raises
    ``SettingValidationError`` for unknown keys / non-editable keys / bad
    shape. The caller is expected to commit; this function never commits
    on its own so it composes with the request-scoped session pattern.
    """
    spec = get_spec(key)
    if spec is None:
        raise SettingValidationError(f"unknown setting key: {key!r}")
    if not spec.editable:
        raise SettingValidationError(f"{key!r} is not editable (env / deploy-time only)")
    validated = _validate_value(spec, value)

    # Capture old value for audit — env vs prior DB row.
    old_global = await _load_global_overrides(session)
    old_value = old_global.get(key, _env_default(key))

    existing = (
        await session.execute(select(system_settings).where(system_settings.c.key == key))
    ).first()
    if existing is None:
        await session.execute(
            insert(system_settings).values(
                key=key,
                value=validated,
                category=spec.category,
                is_secret=spec.is_secret,
                set_by=changed_by,
            )
        )
    else:
        await session.execute(
            update(system_settings)
            .where(system_settings.c.key == key)
            .values(
                value=validated,
                category=spec.category,
                is_secret=spec.is_secret,
                set_by=changed_by,
                set_at=func.now(),
            )
        )

    await record_audit(
        session,
        key=key,
        old_value=old_value,
        new_value=validated,
        action="patch",
        changed_by=changed_by,
        tenant_id=None,
        note=note,
    )
    _invalidate_global()
    return validated


async def delete_global_override(
    key: str,
    session: AsyncSession,
    *,
    changed_by: str | None = None,
    note: str | None = None,
) -> None:
    """Drop a global override row (revert the setting to its env value).

    Idempotent — deleting a non-existent override is a no-op.
    """
    spec = get_spec(key)
    if spec is None:
        raise SettingValidationError(f"unknown setting key: {key!r}")

    old_global = await _load_global_overrides(session)
    if key not in old_global:
        return  # nothing to do — already at env default
    old_value = old_global[key]

    await session.execute(delete(system_settings).where(system_settings.c.key == key))
    await record_audit(
        session,
        key=key,
        old_value=old_value,
        new_value=_env_default(key),
        action="delete",
        changed_by=changed_by,
        tenant_id=None,
        note=note,
    )
    _invalidate_global()


async def apply_tenant_override(
    tenant_id: str,
    key: str,
    value: Any,
    session: AsyncSession,
    *,
    changed_by: str | None = None,
    note: str | None = None,
) -> Any:
    """UPSERT a per-tenant override + append audit. See :func:`apply_global_override`."""
    spec = get_spec(key)
    if spec is None:
        raise SettingValidationError(f"unknown setting key: {key!r}")
    if not spec.editable:
        raise SettingValidationError(f"{key!r} is not editable")
    if not spec.tenant_overridable:
        raise SettingValidationError(f"{key!r} is not tenant-overridable")
    validated = _validate_value(spec, value)

    old_tenant = await _load_tenant_overrides(session, tenant_id)
    old_value = old_tenant.get(key, _SENTINEL)
    audit_old = old_value if old_value is not _SENTINEL else None

    existing = (
        await session.execute(
            select(tenant_settings).where(
                tenant_settings.c.tenant_id == tenant_id,
                tenant_settings.c.key == key,
            )
        )
    ).first()
    if existing is None:
        await session.execute(
            insert(tenant_settings).values(
                tenant_id=tenant_id,
                key=key,
                value=validated,
                is_secret=spec.is_secret,
                set_by=changed_by,
            )
        )
    else:
        await session.execute(
            update(tenant_settings)
            .where(
                tenant_settings.c.tenant_id == tenant_id,
                tenant_settings.c.key == key,
            )
            .values(
                value=validated,
                is_secret=spec.is_secret,
                set_by=changed_by,
                set_at=func.now(),
            )
        )

    await record_audit(
        session,
        key=key,
        old_value=audit_old,
        new_value=validated,
        action="patch",
        changed_by=changed_by,
        tenant_id=tenant_id,
        note=note,
    )
    _invalidate_tenant(tenant_id)
    return validated


async def delete_tenant_override(
    tenant_id: str,
    key: str,
    session: AsyncSession,
    *,
    changed_by: str | None = None,
    note: str | None = None,
) -> None:
    """Drop a tenant override (revert to global / env fallback)."""
    spec = get_spec(key)
    if spec is None:
        raise SettingValidationError(f"unknown setting key: {key!r}")

    old_tenant = await _load_tenant_overrides(session, tenant_id)
    if key not in old_tenant:
        return
    old_value = old_tenant[key]

    await session.execute(
        delete(tenant_settings).where(
            tenant_settings.c.tenant_id == tenant_id,
            tenant_settings.c.key == key,
        )
    )
    await record_audit(
        session,
        key=key,
        old_value=old_value,
        new_value=None,
        action="delete",
        changed_by=changed_by,
        tenant_id=tenant_id,
        note=note,
    )
    _invalidate_tenant(tenant_id)


# ─── Audit ────────────────────────────────────────────────────────────────


async def record_audit(
    session: AsyncSession,
    *,
    key: str,
    old_value: Any,
    new_value: Any,
    action: str,
    changed_by: str | None,
    tenant_id: str | None,
    note: str | None = None,
) -> None:
    """Append one row to system_settings_audit.

    Secrets are redacted before persistence so a leaked audit dump can't
    reconstruct historical credentials. The caller commits the surrounding
    transaction.
    """
    redacted_old = redact(old_value, key=key) if is_secret(key) else old_value
    redacted_new = redact(new_value, key=key) if is_secret(key) else new_value

    await session.execute(
        insert(system_settings_audit).values(
            key=key,
            old_value=redacted_old,
            new_value=redacted_new,
            action=action,
            tenant_id=tenant_id,
            changed_by=changed_by,
            note=note,
        )
    )

    # Roll off the oldest rows past the hard cap — keeps audit growth
    # bounded without operator intervention. Approximate: we only DELETE
    # when the count exceeds the cap, and we trim down to 90% so we
    # don't run the cull on every single insert at the boundary.
    count_row = (await session.execute(select(func.count(system_settings_audit.c.id)))).scalar()
    count = int(count_row or 0)
    if count > _AUDIT_HARD_CAP:
        target = int(_AUDIT_HARD_CAP * 0.9)
        excess = count - target
        # Find the changed_at boundary of the oldest `excess` rows and
        # delete everything older than it. One round-trip.
        cutoff = (
            await session.execute(
                select(system_settings_audit.c.changed_at)
                .order_by(system_settings_audit.c.changed_at)
                .limit(1)
                .offset(excess)
            )
        ).scalar()
        if cutoff is not None:
            await session.execute(
                delete(system_settings_audit).where(
                    system_settings_audit.c.changed_at < cutoff
                )
            )


async def list_audit(
    session: AsyncSession,
    *,
    key: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent audit rows, newest first. Optionally filter to one key."""
    limit = max(1, min(limit, 1000))
    stmt = select(system_settings_audit).order_by(desc(system_settings_audit.c.changed_at)).limit(limit)
    if key is not None:
        stmt = stmt.where(system_settings_audit.c.key == key)
    rows = (await session.execute(stmt)).mappings().all()
    return [
        {
            "id": str(row["id"]),
            "key": row["key"],
            "old_value": row["old_value"],
            "new_value": row["new_value"],
            "action": row["action"],
            "tenant_id": row["tenant_id"],
            "changed_by": row["changed_by"],
            "changed_at": row["changed_at"].isoformat() if row["changed_at"] else None,
            "note": row["note"],
        }
        for row in rows
    ]


# ─── Test probe ───────────────────────────────────────────────────────────


@dataclass
class ProbeResult:
    """Outcome of a non-persisting validity check."""

    ok: bool
    detail: str
    extra: dict[str, Any] | None = None


async def test_probe(
    key: str,
    value: Any,
    *,
    session: AsyncSession | None = None,
) -> ProbeResult:
    """Run a side-effect-free validity check for a candidate value.

    Used by the admin UI's "Test" button before committing a change to a
    risky setting (LLM creds, webhook URL). Currently covers:

      * ``litellm_*`` — round-trips a 1-token completion against the
        proposed credentials
      * ``webhook_url`` — issues an OPTIONS / HEAD request and checks for
        2xx/3xx
      * everything else — runs the same shape validation as a PATCH
        without writing

    Returns a :class:`ProbeResult` instead of raising so the UI can show a
    friendly red toast without crashing the page on transient failures.
    """
    spec = get_spec(key)
    if spec is None:
        return ProbeResult(ok=False, detail=f"unknown setting key: {key!r}")

    try:
        _validate_value(spec, value)
    except SettingValidationError as exc:
        return ProbeResult(ok=False, detail=str(exc))

    # Provider-specific probes — kept minimal to avoid pulling in heavy
    # deps at import time. Each branch self-contains the import so a probe
    # for one provider doesn't pull in the others' libs.
    if key in {"litellm_model", "litellm_api_key", "litellm_api_base"}:
        return await _probe_litellm(key, value, session=session)

    if key == "webhook_url":
        return await _probe_webhook_url(value)

    return ProbeResult(ok=True, detail="shape ok (no provider probe defined)")


async def _probe_litellm(
    key: str, value: Any, *, session: AsyncSession | None
) -> ProbeResult:
    """Send a 1-token completion using the proposed override layered onto
    the rest of the env-driven LiteLLM config.

    Routed through ``server.services.llm.aping_with_overrides`` so this
    module never imports LiteLLM directly — the adapter-isolation rule
    (enforced by ``tests/test_llm_adapter_isolation.py``) keeps provider
    SDKs in one place.
    """
    from server.services.llm import aping_with_overrides

    # Resolve the rest of the LiteLLM config from current env, then patch in
    # the candidate value. This way the operator can test a new key without
    # also having to provide model + base.
    overrides: dict[str, Any] = {}
    if session is not None:
        snap = await get_effective_snapshot(session, redact_secrets=False)
        overrides = {k: v["value"] for k, v in snap.items()}
    else:
        overrides = {k: _env_default(k) for k in CATALOGUE}
    overrides[key] = value

    model = overrides.get("litellm_model") or "gpt-4o-mini"
    api_key = overrides.get("litellm_api_key")
    api_base = overrides.get("litellm_api_base")
    timeout = overrides.get("litellm_timeout_seconds") or 30.0

    if not api_key:
        return ProbeResult(ok=False, detail="no api key available to probe with")

    ok, detail = await aping_with_overrides(
        model=model, api_key=api_key, api_base=api_base, timeout=float(timeout)
    )
    return ProbeResult(ok=ok, detail=detail or ("ok" if ok else "failed"), extra={"model": model})


async def apply_db_overrides_to_settings() -> dict[str, Any]:
    """Read ``system_settings`` and mutate ``env_settings`` in place.

    Called once at FastAPI lifespan startup, BEFORE webhook / LLM
    singletons read their config. This is the lightweight alternative to
    plumbing dynamic-settings reads through every call site: the running
    process behaves *as if* the DB overrides were set in the env.

    Refreshes `_applied_values` after every key it mutates so the
    snapshot's `pending_restart` flag is accurate: a key written to the
    DB after this point will diff against this baseline and report
    `pending_restart=true`, then clear on next restart when this
    function re-runs.

    Returns the dict of {key: applied_value} so the caller can log it.
    Idempotent — calling twice is fine. Failures (table missing, DB
    unreachable) propagate so the caller can decide whether to keep
    booting; the lifespan handler logs and continues, so a corrupt
    settings table never blocks the server from coming up.
    """
    from server.db import engine as engine_module

    # Baselines must be in place before we mutate anything. Cheap +
    # idempotent — re-fills nothing it already saw at import time.
    _initialise_boot_snapshots()

    applied: dict[str, Any] = {}
    async with engine_module.get_session_factory()() as session:
        global_values = await _load_global_overrides(session)
    for key, value in global_values.items():
        spec = CATALOGUE.get(key)
        if spec is None or not hasattr(env_settings, key):
            continue
        try:
            object.__setattr__(env_settings, key, value)
            applied[key] = value
        except Exception:
            # pydantic-settings v2 may freeze attributes — fall back to
            # __dict__ direct write (still picked up by getattr).
            try:
                env_settings.__dict__[key] = value
                applied[key] = value
            except Exception:
                continue

    # Re-snapshot the applied values from env_settings AFTER all mutations.
    # Reading from env_settings (not from `applied`) so settings the
    # operator never touched still show the correct boot-time value.
    for key in CATALOGUE:
        _applied_values[key] = getattr(env_settings, key, None)

    return applied


async def _probe_webhook_url(value: Any) -> ProbeResult:
    """Issue a short HEAD against the URL — we just want to know it
    resolves + responds. 2xx/3xx → ok; 4xx → caller will still send
    POSTs (some webhook receivers reject HEAD), so we report 405/404 as
    "endpoint reachable" rather than fail; 5xx → fail."""
    if value in (None, ""):
        return ProbeResult(ok=True, detail="empty url → webhooks disabled")
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return ProbeResult(ok=False, detail="must be http(s)://")
    try:
        import httpx  # type: ignore[import-not-found]
    except Exception as exc:
        return ProbeResult(ok=False, detail=f"httpx unavailable: {exc}")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.head(value)
    except Exception as exc:
        return ProbeResult(ok=False, detail=f"{type(exc).__name__}: {exc}")
    if r.status_code >= 500:
        return ProbeResult(ok=False, detail=f"endpoint returned {r.status_code}", extra={"status": r.status_code})
    return ProbeResult(
        ok=True,
        detail=f"endpoint reachable (HEAD → {r.status_code})",
        extra={"status": r.status_code},
    )
