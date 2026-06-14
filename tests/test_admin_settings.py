"""DB-backed integration tests for the admin settings endpoints.

Mirror the pattern used by `test_admin_subjects.py`: skip when Postgres
isn't migrated, otherwise exercise the full HTTP → endpoint → DB → audit
pipeline against a real Postgres referenced by ``STATEWAVE_DATABASE_URL``.
"""

from __future__ import annotations

import os

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def _require_migrated_postgres():
    from server.core.config import settings

    url = (
        os.environ.get("STATEWAVE_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or settings.database_url
    )
    try:
        engine = create_async_engine(url, pool_pre_ping=True)
        async with engine.connect() as conn:
            # 0026 head check — if this column-less probe succeeds the
            # table exists with the expected shape.
            await conn.execute(text("SELECT key FROM system_settings LIMIT 0"))
            await conn.execute(text("SELECT id FROM system_settings_audit LIMIT 0"))
        await engine.dispose()
    except Exception as exc:
        pytest.skip(
            "Postgres not migrated to 0026_system_settings "
            f"({type(exc).__name__}: {exc}). "
            "Run `alembic upgrade head`."
        )


@pytest.fixture(autouse=True)
async def _clean_settings_tables(_require_migrated_postgres):
    """Each test starts with an empty settings + audit + tenant_settings.
    Tearing down after is also important — other tests in the suite might
    inadvertently depend on env-only behaviour.

    Depends on ``_require_migrated_postgres`` so we don't try to truncate
    against an unreachable DB (which would explode the teardown instead
    of skipping cleanly)."""
    from server.core.config import settings as env_settings

    url = (
        os.environ.get("STATEWAVE_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or env_settings.database_url
    )

    async def _truncate():
        engine = create_async_engine(url)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("DELETE FROM system_settings"))
                await conn.execute(text("DELETE FROM system_settings_audit"))
                await conn.execute(text("DELETE FROM tenant_settings"))
        finally:
            await engine.dispose()

    await _truncate()
    # Wipe in-process cache so the next read sees the cleaned tables.
    from server.core.dynamic_settings import invalidate_cache

    invalidate_cache()
    yield
    await _truncate()
    invalidate_cache()


# ─── GET /admin/settings ─────────────────────────────────────────────────


async def test_list_settings_returns_every_catalogued_key(client: AsyncClient):
    from server.core.settings_catalogue import CATALOGUE

    resp = await client.get("/admin/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["settings"].keys()) == set(CATALOGUE.keys())
    # No DB rows yet → every source must be `env`.
    for entry in data["settings"].values():
        assert entry["source"] == "env"


async def test_list_settings_redacts_secrets(client: AsyncClient, monkeypatch):
    """Secrets must NEVER leave the server in cleartext, even on first
    list. The redacted preview shows the last 3 chars so the operator can
    sanity-check which key is currently loaded without ever exposing the
    full value."""
    from server.core.config import settings as env_settings

    monkeypatch.setattr(env_settings, "litellm_api_key", "sk-secret-xyz")
    resp = await client.get("/admin/settings")
    assert resp.status_code == 200
    api_key = resp.json()["settings"]["litellm_api_key"]
    assert api_key["is_secret"] is True
    assert api_key["value"] == "•••xyz"


# ─── PATCH /admin/settings/{key} ─────────────────────────────────────────


async def test_patch_setting_persists_and_flips_source(client: AsyncClient):
    resp = await client.patch(
        "/admin/settings/litellm_model",
        json={"value": "gpt-4o", "changed_by": "test@local"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["value"] == "gpt-4o"

    # Source flips to global_db on subsequent reads
    follow = await client.get("/admin/settings/litellm_model")
    assert follow.status_code == 200
    assert follow.json()["value"] == "gpt-4o"
    assert follow.json()["source"] == "global_db"


async def test_patch_setting_rejects_wrong_type(client: AsyncClient):
    """An int field must reject a string payload — without this, an
    operator typing "60s" into a number input would silently break the
    rate limiter."""
    resp = await client.patch(
        "/admin/settings/rate_limit_rpm",
        json={"value": "sixty"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["code"] == "settings.invalid"


async def test_patch_rejects_non_editable_key(client: AsyncClient):
    resp = await client.patch(
        "/admin/settings/database_url",
        json={"value": "postgresql://x"},
    )
    assert resp.status_code == 400


async def test_patch_rejects_unknown_key(client: AsyncClient):
    resp = await client.patch(
        "/admin/settings/totally_made_up",
        json={"value": "x"},
    )
    assert resp.status_code == 400


# ─── DELETE /admin/settings/{key} ────────────────────────────────────────


async def test_delete_reverts_to_env(client: AsyncClient):
    # First patch it…
    await client.patch(
        "/admin/settings/litellm_model", json={"value": "gpt-4o"}
    )
    # …then revert.
    resp = await client.delete("/admin/settings/litellm_model")
    assert resp.status_code == 200
    follow = await client.get("/admin/settings/litellm_model")
    assert follow.json()["source"] == "env"


async def test_delete_unset_setting_is_idempotent(client: AsyncClient):
    resp = await client.delete("/admin/settings/litellm_model")
    assert resp.status_code == 200


# ─── Audit ───────────────────────────────────────────────────────────────


async def test_audit_log_records_patch_and_delete(client: AsyncClient):
    await client.patch(
        "/admin/settings/litellm_model",
        json={"value": "gpt-4o", "changed_by": "alice", "note": "trying new model"},
    )
    await client.delete(
        "/admin/settings/litellm_model",
        params={"changed_by": "alice", "note": "rolling back"},
    )

    resp = await client.get("/admin/settings/audit/log", params={"key": "litellm_model"})
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 2
    # Newest first
    assert entries[0]["action"] == "delete"
    assert entries[1]["action"] == "patch"
    assert entries[1]["changed_by"] == "alice"
    assert entries[1]["new_value"] == "gpt-4o"


async def test_audit_redacts_secret_values(client: AsyncClient):
    """An audit log of a secret edit must NOT preserve the raw secret —
    historical credentials must remain unrecoverable from the audit
    table, otherwise a breach widens past 'current key' to 'every key ever'."""
    await client.patch(
        "/admin/settings/litellm_api_key",
        json={"value": "sk-rotated-abc"},
    )
    resp = await client.get(
        "/admin/settings/audit/log", params={"key": "litellm_api_key"}
    )
    entries = resp.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["new_value"] == "•••abc"


# ─── Tenant overrides ────────────────────────────────────────────────────


async def test_tenant_override_takes_precedence_over_global(client: AsyncClient):
    # Global override
    await client.patch(
        "/admin/settings/litellm_model", json={"value": "global-model"}
    )
    # Tenant override on top
    await client.patch(
        "/admin/settings/tenants/tenant-abc/litellm_model",
        json={"value": "tenant-model"},
    )

    # No tenant param → global_db wins
    g = await client.get("/admin/settings/litellm_model")
    assert g.json()["value"] == "global-model"
    assert g.json()["source"] == "global_db"

    # With tenant param → tenant_db wins
    t = await client.get(
        "/admin/settings/litellm_model", params={"tenant_id": "tenant-abc"}
    )
    assert t.json()["value"] == "tenant-model"
    assert t.json()["source"] == "tenant_db"


async def test_tenant_override_refused_for_non_overridable_key(client: AsyncClient):
    """The catalogue narrows tenant overrides to LLM + webhook + rate-
    limit. Anything else MUST be refused — otherwise an operator could
    quietly per-tenant something with global consequences."""
    resp = await client.patch(
        "/admin/settings/tenants/tenant-abc/compile_batch_size",
        json={"value": 100},
    )
    assert resp.status_code == 400


# ─── Test probe ──────────────────────────────────────────────────────────


async def test_probe_endpoint_does_not_persist(client: AsyncClient):
    """A 'Test' click must not write — that's the whole point. Reading
    after a test should still show the env source."""
    resp = await client.post(
        "/admin/settings/test",
        json={"key": "litellm_model", "value": "gpt-99"},
    )
    assert resp.status_code == 200
    follow = await client.get("/admin/settings/litellm_model")
    assert follow.json()["source"] == "env"


async def test_probe_endpoint_returns_shape_error(client: AsyncClient):
    resp = await client.post(
        "/admin/settings/test",
        json={"key": "rate_limit_rpm", "value": "not-a-number"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "expected int" in body["detail"]


# ─── pending_restart flag ────────────────────────────────────────────────


async def test_pending_restart_set_after_patch_clears_after_simulated_restart(
    client: AsyncClient,
):
    """Regression for the banner that never cleared.

    Patching a non-hot-reloadable setting must mark `pending_restart=true`
    (process is still running the old value). Calling
    `apply_db_overrides_to_settings` simulates a server restart picking
    up the new value; afterwards `pending_restart` must be false even
    though the DB row still exists.
    """
    from server.core.dynamic_settings import apply_db_overrides_to_settings

    await client.patch("/admin/settings/compiler_type", json={"value": "llm"})
    snap = (await client.get("/admin/settings/compiler_type")).json()
    assert snap["value"] == "llm"
    assert snap["source"] == "global_db"
    assert snap["pending_restart"] is True
    assert snap["applied_value"] != "llm"

    # Simulate a process restart — env_settings is re-mutated, applied
    # snapshot is updated.
    await apply_db_overrides_to_settings()

    snap = (await client.get("/admin/settings/compiler_type")).json()
    assert snap["value"] == "llm"
    assert snap["source"] == "global_db"
    assert snap["pending_restart"] is False
    assert snap["applied_value"] == "llm"


async def test_pending_restart_set_after_delete_without_restart(client: AsyncClient):
    """DELETE returns the row to env-baseline, but the process is still
    running the previously-applied value until restart — that must show
    as pending_restart."""
    from server.core.dynamic_settings import apply_db_overrides_to_settings

    await client.patch("/admin/settings/compiler_type", json={"value": "llm"})
    await apply_db_overrides_to_settings()  # simulate restart so it's applied
    await client.delete("/admin/settings/compiler_type")

    snap = (await client.get("/admin/settings/compiler_type")).json()
    assert snap["source"] == "env"
    assert snap["pending_restart"] is True
    # applied_value still reflects the post-boot value the process is
    # actually using, not the env baseline.
    assert snap["applied_value"] == "llm"


async def test_pending_restart_false_for_tenant_override(client: AsyncClient):
    """Tenant overrides are read per-request via get_setting(); they
    never require a restart, no matter how often they change."""
    await client.patch(
        "/admin/settings/tenants/t-1/litellm_model",
        json={"value": "gpt-4o"},
    )
    snap = (
        await client.get("/admin/settings/litellm_model", params={"tenant_id": "t-1"})
    ).json()
    assert snap["source"] == "tenant_db"
    assert snap["pending_restart"] is False
