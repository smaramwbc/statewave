"""Tests for the `/admin/readiness-check` rule engine.

These tests pin the rule set — adding a new check (or weakening a
severity) requires touching this file, which keeps the production-
readiness contract reviewable as a single artifact rather than spread
across the admin module.

Tests are deliberately mostly assertions about "is rule X present for
config Y", not about exact strings — the messages are user-facing and
will get polished, but the IDs and severities are the contract.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def _ids(client: AsyncClient) -> set[str]:
    resp = await client.get("/admin/readiness-check")
    assert resp.status_code == 200
    return {i["id"] for i in resp.json()["issues"]}


async def _severity_of(client: AsyncClient, issue_id: str) -> str | None:
    resp = await client.get("/admin/readiness-check")
    for i in resp.json()["issues"]:
        if i["id"] == issue_id:
            return i["severity"]
    return None


# ─── critical: no auth ──────────────────────────────────────────────────


async def test_no_auth_is_critical(client: AsyncClient, monkeypatch):
    """The default Statewave config has no API key — the rule must fire
    critical so an operator can't miss it during the dev → prod hand-off."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "api_key", None)
    ids = await _ids(client)
    assert "no_backend_auth" in ids
    assert await _severity_of(client, "no_backend_auth") == "critical"


async def test_dev_placeholder_key_is_critical(client: AsyncClient, monkeypatch):
    """The known dev defaults shouldn't satisfy 'auth on' — they're
    public knowledge from the quickstart docs and trivial to guess."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "api_key", "dev-local-placeholder")
    ids = await _ids(client)
    # `no_backend_auth` is absent (auth IS set, just badly)
    assert "no_backend_auth" not in ids
    assert "dev_placeholder_api_key" in ids
    assert await _severity_of(client, "dev_placeholder_api_key") == "critical"


async def test_strong_key_passes(client: AsyncClient, monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "api_key", "k_" + "a" * 30)
    ids = await _ids(client)
    assert "no_backend_auth" not in ids
    assert "dev_placeholder_api_key" not in ids


# ─── high: CORS / debug / stub embeddings ───────────────────────────────


async def test_wildcard_cors_is_high(client: AsyncClient, monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "cors_origins", ["*"])
    ids = await _ids(client)
    assert "permissive_cors" in ids
    assert await _severity_of(client, "permissive_cors") == "high"


async def test_explicit_origins_pass(client: AsyncClient, monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "cors_origins", ["https://admin.example.com"])
    ids = await _ids(client)
    assert "permissive_cors" not in ids


async def test_debug_logging_is_high(client: AsyncClient, monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "debug", True)
    ids = await _ids(client)
    assert "debug_logging" in ids
    assert await _severity_of(client, "debug_logging") == "high"


async def test_stub_embeddings_is_high(client: AsyncClient, monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "embedding_provider", "stub")
    ids = await _ids(client)
    assert "stub_embeddings" in ids


async def test_litellm_embeddings_pass(client: AsyncClient, monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "embedding_provider", "litellm")
    ids = await _ids(client)
    assert "stub_embeddings" not in ids


# ─── medium: rate limit / strict schema ──────────────────────────────────


async def test_no_rate_limit_is_medium(client: AsyncClient, monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "rate_limit_rpm", 0)
    ids = await _ids(client)
    assert "no_rate_limit" in ids
    assert await _severity_of(client, "no_rate_limit") == "medium"


async def test_rate_limit_set_passes(client: AsyncClient, monkeypatch):
    from server.core.config import settings

    monkeypatch.setattr(settings, "rate_limit_rpm", 600)
    ids = await _ids(client)
    assert "no_rate_limit" not in ids


# ─── shape contract ──────────────────────────────────────────────────────


async def test_every_issue_has_required_fields(client: AsyncClient):
    """The frontend reads severity + id + title + summary unconditionally
    and crashes on a missing field. Pin the shape."""
    resp = await client.get("/admin/readiness-check")
    for i in resp.json()["issues"]:
        assert i["id"]
        assert i["severity"] in {"critical", "high", "medium", "low"}
        assert i["title"]
        assert i["summary"]
        # `fix` is optional — informational-only issues legitimately
        # omit it. When present, it must declare a kind.
        if i.get("fix") is not None:
            assert i["fix"]["kind"] in {"setting", "wizard", "admin_tab", "env"}


async def test_fix_staged_marks_issue_when_db_override_clears_it(
    client: AsyncClient, monkeypatch,
):
    """The bug we just fixed: operator saves `debug=false` in the UI,
    the DB has the new value, but `settings.debug` is still True
    until restart. Without `fix_staged`, the readiness card keeps
    showing the unchanged warning and the operator thinks the save
    was ignored.

    Force `settings.debug=True` so the rule fires live, PATCH the DB
    override to false, then assert the response carries
    `fix_staged=true` on the debug_logging issue. The post-restart
    case is covered by the existing pending-restart tests in
    test_admin_settings.py."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "debug", True)
    await client.patch("/admin/settings/debug", json={"value": False})

    resp = await client.get("/admin/readiness-check")
    issues = resp.json()["issues"]
    debug_issue = next((i for i in issues if i["id"] == "debug_logging"), None)
    assert debug_issue is not None, "debug_logging should still fire (live=True)"
    assert debug_issue.get("fix_staged") is True, (
        "fix_staged must be True — the DB has the resolving value queued"
    )

    # Cleanup so other tests don't see the override.
    await client.delete("/admin/settings/debug")


async def test_fix_staged_absent_when_no_override(client: AsyncClient, monkeypatch):
    """Mirror of the above: issues firing live with NO DB override
    must not be marked staged. Otherwise the UI shows "Pending
    restart" on issues that nothing's actually queued for."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "debug", True)
    resp = await client.get("/admin/readiness-check")
    issues = resp.json()["issues"]
    debug_issue = next((i for i in issues if i["id"] == "debug_logging"), None)
    assert debug_issue is not None
    assert "fix_staged" not in debug_issue or debug_issue["fix_staged"] is False


async def test_every_setting_fix_target_is_in_the_catalogue(
    client: AsyncClient, monkeypatch
):
    """Lock down a class of regression: a readiness rule pointing at
    `fix: {kind:'setting', key: 'foo'}` only works if 'foo' is in the
    catalogue. Otherwise the admin UI deep-links to a key it can't
    find, the editor never opens, and the operator sees a silent
    nothing-happens after clicking Fix.

    Force EVERY rule to fire by setting the live config to the worst
    possible state, then walk every issue's fix descriptor and assert
    the catalogue knows the key. If a future rule introduces a new
    fix target that we forgot to catalogue, this test catches it
    before it reaches an operator."""
    from server.core.config import settings
    from server.core.settings_catalogue import CATALOGUE

    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "cors_origins", ["*"])
    monkeypatch.setattr(settings, "debug", True)
    monkeypatch.setattr(settings, "embedding_provider", "stub")
    monkeypatch.setattr(settings, "rate_limit_rpm", 0)
    monkeypatch.setattr(settings, "strict_schema", False)
    monkeypatch.setattr(settings, "region", None)
    monkeypatch.setattr(settings, "webhook_url", None)

    resp = await client.get("/admin/readiness-check")
    issues = resp.json()["issues"]
    setting_fixes = [
        i for i in issues if i.get("fix") and i["fix"]["kind"] == "setting"
    ]
    # If this is empty we're not testing anything useful; sanity-check.
    assert setting_fixes, "expected at least one kind=setting fix to fire"
    for i in setting_fixes:
        key = i["fix"]["key"]
        assert key in CATALOGUE, (
            f"readiness rule {i['id']!r} points at setting key {key!r}, "
            f"but it's not in CATALOGUE. Add a SettingSpec for it, or "
            f"change the fix descriptor to kind='env'."
        )
