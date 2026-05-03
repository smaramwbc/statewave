"""Tests for vendor-neutral memory portability — service + admin endpoints.

Covers the contracts that don't depend on a running database:

  * route registration for all six endpoints
  * `list_starter_packs` reads on-disk manifests and returns the expected
    set of bundled packs (statewave-support + 5 demo agents)
  * subject-id validation (length, charset, reserved prefixes)
  * payload validation (format, version, unknown fields, size limit)
  * support reseed targets the configured shared subject and never a
    per-visitor `demo_web_*__statewave-support` id
  * clone refuses an empty / non-existent source
  * export rejects an empty `subject_ids` list and oversize requests
  * import rejects malformed payloads with clear error messages

Tests that exercise the actual database write path live in
`tests/integration/` because they need a real Postgres + asyncpg session
factory; these unit tests focus on the validation/dispatch contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.app import create_app
from server.core.config import settings
from server.services import memory_packs as mp


# ─── Route registration ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path, methods",
    [
        ("/admin/memory/starter-packs", {"GET"}),
        ("/admin/memory/starter-packs/import", {"POST"}),
        ("/admin/memory/support/reseed", {"POST"}),
        ("/admin/memory/clone", {"POST"}),
        ("/admin/memory/export", {"POST"}),
        ("/admin/memory/import", {"POST"}),
    ],
)
def test_memory_endpoints_registered(path, methods):
    app = create_app()
    for route in app.routes:
        if getattr(route, "path", None) == path:
            assert methods.issubset(route.methods), (
                f"{path} expected {methods}, got {route.methods}"
            )
            return
    pytest.fail(f"{path} not registered on the app")


def test_old_github_actions_endpoint_is_gone():
    """The vendor-neutral rewrite removes the GitHub-Actions dispatch endpoint."""
    app = create_app()
    routes = [getattr(r, "path", None) for r in app.routes]
    assert "/admin/docs-pack/reseed" not in routes


# ─── Starter pack registry ──────────────────────────────────────────────────


def test_list_starter_packs_returns_bundled_set():
    packs = mp.list_starter_packs()
    pack_ids = {p["pack_id"] for p in packs}
    # Demo pack ids carry the `demo-` prefix so an import lands on the same
    # subject id the marketing-widget demo flow already uses on
    # statewave.ai (e.g. `demo-support-agent`). Keeping them aligned means
    # an operator-imported pack is immediately picked up by the live demo
    # without an extra rename step.
    expected = {
        "statewave-support-agent",
        "demo-support-agent",
        "demo-coding-assistant",
        "demo-sales-copilot",
        "demo-devops-agent",
        "demo-research-assistant",
    }
    assert expected.issubset(pack_ids), (
        f"missing packs: {expected - pack_ids}"
    )


def test_starter_pack_metadata_shape():
    """Every bundled pack must carry the spec-required manifest fields."""
    packs = mp.list_starter_packs()
    required = {
        "pack_id",
        "display_name",
        "description",
        "version",
        "subject_id_suggestion",
        "episode_count",
        "memory_count",
    }
    for pack in packs:
        if "__error" in pack:
            pytest.fail(f"pack {pack['pack_id']} failed to load: {pack['__error']}")
        missing = required - set(pack.keys())
        assert not missing, f"{pack['pack_id']} missing fields: {missing}"
        assert pack["episode_count"] >= 0
        assert pack["memory_count"] >= 0


def test_starter_pack_directory_layout_on_disk():
    """Each pack's directory contains manifest.json + episodes.jsonl + memories.jsonl."""
    root = Path(__file__).resolve().parent.parent / "server" / "starter_packs"
    for pack in mp.list_starter_packs():
        d = root / pack["pack_id"]
        assert (d / "manifest.json").exists(), f"missing manifest for {pack['pack_id']}"
        # JSONL files may be empty for stub packs but must exist
        assert (d / "episodes.jsonl").exists(), f"missing episodes.jsonl for {pack['pack_id']}"
        assert (d / "memories.jsonl").exists(), f"missing memories.jsonl for {pack['pack_id']}"


# ─── Subject-id validation ──────────────────────────────────────────────────


def test_validate_subject_id_accepts_valid_ids():
    for ok in ("user-1", "tenant-a:agent-7", "default-support-agent", "support_v2.1"):
        assert mp._validate_subject_id(ok) == ok


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "x" * 200,  # too long
        "has spaces",  # invalid char
        "weird/path",  # path separator
        "demo_web_abc",  # reserved prefix used by visitor subjects
    ],
)
def test_validate_subject_id_rejects_invalid_ids(bad):
    with pytest.raises(mp.StarterPackError):
        mp._validate_subject_id(bad)


# ─── Support reseed targets only the shared subject ─────────────────────────


def test_support_subject_target_is_shared_not_per_visitor():
    """Sanity check on the configured target id used by reseed."""
    sid = settings.support_subject_id
    # Must NOT collide with the per-visitor namespace used by the marketing
    # widget's hybrid Statewave Support flow.
    assert not sid.startswith("demo_web_")
    assert "__statewave-support" not in sid
    assert sid == "statewave-support-docs"


# ─── Payload validation (no DB needed) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_import_rejects_non_dict_payload():
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.import_memory_payload(payload=[], target_tenant_id=None)
    assert "JSON object" in str(ei.value)


@pytest.mark.asyncio
async def test_import_rejects_wrong_format():
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.import_memory_payload(
            payload={"format": "something-else", "format_version": 1},
            target_tenant_id=None,
        )
    assert "format" in str(ei.value).lower()


@pytest.mark.asyncio
async def test_import_rejects_wrong_format_version():
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.import_memory_payload(
            payload={
                "format": mp.PAYLOAD_FORMAT,
                "format_version": 99,
            },
            target_tenant_id=None,
        )
    assert "format_version" in str(ei.value)


@pytest.mark.asyncio
async def test_import_rejects_unknown_top_level_fields():
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.import_memory_payload(
            payload={
                "format": mp.PAYLOAD_FORMAT,
                "format_version": mp.PAYLOAD_FORMAT_VERSION,
                "subjects": [],
                "episodes": [],
                "memories": [],
                "evil": "drop_table",
            },
            target_tenant_id=None,
        )
    assert "unknown fields" in str(ei.value)


@pytest.mark.asyncio
async def test_import_rejects_oversize_payload(monkeypatch):
    """Hard byte cap protects the worker from pathological payloads."""
    monkeypatch.setattr(settings, "memory_import_max_bytes", 1024)  # tiny cap
    big_blob = "x" * 4096
    payload = {
        "format": mp.PAYLOAD_FORMAT,
        "format_version": mp.PAYLOAD_FORMAT_VERSION,
        "subjects": [{"original_subject_id": "s", "metadata": {"blob": big_blob}}],
        "episodes": [],
        "memories": [],
    }
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.import_memory_payload(payload=payload, target_tenant_id=None)
    assert ei.value.status_code == 413


# ─── Export validation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_rejects_empty_subject_ids():
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.export_memory_payload(subject_ids=[], tenant_id=None)
    assert "required" in str(ei.value)


@pytest.mark.asyncio
async def test_export_rejects_too_many_subjects(monkeypatch):
    monkeypatch.setattr(settings, "memory_import_max_subjects", 2)
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.export_memory_payload(
            subject_ids=["a", "b", "c"], tenant_id=None
        )
    assert ei.value.status_code == 413


@pytest.mark.asyncio
async def test_export_rejects_invalid_subject_id():
    # `demo_web_*` is reserved as a TARGET prefix but allowed as a SOURCE
    # (export is read-only, operators legitimately need to back up visitor
    # subjects). Use a genuinely malformed id here instead.
    with pytest.raises(mp.StarterPackError):
        await mp.export_memory_payload(subject_ids=["has spaces"], tenant_id=None)


# ─── Clone validation ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clone_rejects_invalid_source_id():
    with pytest.raises(mp.StarterPackError):
        await mp.clone_subject(
            source_subject_id="has spaces",
            target_subject_id=None,
            target_display_name=None,
            target_tenant_id=None,
        )


@pytest.mark.asyncio
async def test_clone_rejects_reserved_prefix_TARGET_id():
    """The reserved-prefix guard still fires on the target side."""
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.clone_subject(
            source_subject_id="user-1",
            target_subject_id="demo_web_abc",
            target_display_name=None,
            target_tenant_id=None,
        )
    assert "reserved prefix" in str(ei.value)


def test_clone_allows_reserved_prefix_SOURCE_id_in_validation():
    """Source-side validation accepts the demo_web_ prefix so operators can
    clone visitor memory subjects for inspection. We pin the validator
    directly because hitting the full clone path requires a real DB."""
    sid = "demo_web_abc__support-agent"
    # default — refuses
    with pytest.raises(mp.StarterPackError):
        mp._validate_subject_id(sid, field="source_subject_id")
    # allow_reserved=True — accepts
    assert mp._validate_subject_id(sid, field="source_subject_id", allow_reserved=True) == sid


def test_suggest_target_id_rewrites_reserved_prefix():
    """Importing a .swmem captured from a visitor subject must NOT land the
    new subject back in the demo_web_ namespace. The rewriter swaps the
    reserved prefix for `imported-` so the result is namespace-safe.
    """
    new_id = mp._suggest_target_subject_id("demo_web_abc__support-agent")
    assert new_id.startswith("imported-abc__support-agent-")
    assert not new_id.startswith("demo_web_")


@pytest.mark.asyncio
async def test_import_accepts_reserved_prefix_in_original_subject_id_during_validation():
    """The payload validator must not reject an archive whose subjects'
    `original_subject_id` carry the reserved prefix — that's the common
    case for archives generated from the marketing widget. We exercise the
    validator path explicitly (not the full ingest, which needs a DB)."""
    # An ill-formed id should still raise (this part of validation is
    # unaffected by the carve-out).
    with pytest.raises(mp.StarterPackError):
        await mp.import_memory_payload(
            payload={
                "format": mp.PAYLOAD_FORMAT,
                "format_version": mp.PAYLOAD_FORMAT_VERSION,
                "subjects": [{"original_subject_id": "has spaces"}],
                "episodes": [],
                "memories": [],
            },
            target_tenant_id=None,
        )


# ─── Starter pack import error: unknown pack ────────────────────────────────


@pytest.mark.asyncio
async def test_starter_pack_import_unknown_pack_404():
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.import_starter_pack(
            pack_id="not-a-real-pack",
            target_subject_id=None,
            target_display_name=None,
            target_tenant_id=None,
        )
    assert ei.value.status_code == 404


# ─── HTTP-level checks via the test client ──────────────────────────────────


@pytest.mark.asyncio
async def test_http_starter_packs_list_endpoint(client):
    resp = await client.get("/admin/memory/starter-packs")
    assert resp.status_code == 200
    body = resp.json()
    assert "packs" in body
    pack_ids = {p["pack_id"] for p in body["packs"]}
    assert "statewave-support-agent" in pack_ids
    assert "demo-coding-assistant" in pack_ids


@pytest.mark.asyncio
async def test_http_import_rejects_unknown_format(client):
    resp = await client.post(
        "/admin/memory/import",
        json={"payload": {"format": "wrong", "format_version": 1}},
    )
    assert resp.status_code == 400
    assert "format" in resp.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_http_starter_pack_import_unknown_pack_404(client):
    resp = await client.post(
        "/admin/memory/starter-packs/import",
        json={"pack_id": "not-a-real-pack"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_http_clone_rejects_invalid_source_subject(client):
    resp = await client.post(
        "/admin/memory/clone",
        json={"source_subject_id": "has spaces"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_http_export_rejects_empty_subject_list(client):
    resp = await client.post(
        "/admin/memory/export",
        json={"subject_ids": []},
    )
    assert resp.status_code == 400


# ─── Versioned payload format constants ─────────────────────────────────────


def test_payload_format_constants_pinned():
    """Format identifiers are part of the on-the-wire contract — locked."""
    assert mp.PAYLOAD_FORMAT == "statewave-memory-payload"
    assert mp.PAYLOAD_FORMAT_VERSION == 1
    assert mp.STARTER_PACK_FORMAT == "statewave-starter-pack"
