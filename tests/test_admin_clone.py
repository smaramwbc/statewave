"""Tests for the subject-clone admin endpoint and service.

Pins the contracts the admin UI depends on without requiring a real
Postgres — the validation paths in `clone_subject` cover most of the
spec, and the route-registration tests confirm wiring. End-to-end
clones that actually write to the database live in
`tests/integration/` so they can scope a real test DB.

Coverage here:
  * route is registered as POST /admin/memory/clone
  * validation refuses unsafe target ids
  * source-side validation accepts the reserved demo_web_ prefix
    (operators must be able to fork visitor subjects for inspection)
  * payload validator picks the right CloneScope literal types
  * 404 fires for a source subject that does not exist (no episodes
    AND no memories under that id)
  * the response shape matches the spec (status, *_count keys, etc.)
  * the legacy `copied_*` keys are NOT emitted (regression guard for the
    rename to `episode_count` / `memory_count` / `source_count`)
"""

from __future__ import annotations

import pytest

from server.app import create_app
from server.services import memory_packs as mp


# ─── Route registration ──────────────────────────────────────────────────────


def test_clone_route_registered():
    app = create_app()
    routes = [getattr(r, "path", None) for r in app.routes]
    assert "/admin/memory/clone" in routes


def test_clone_route_is_post_only():
    app = create_app()
    for route in app.routes:
        if getattr(route, "path", None) == "/admin/memory/clone":
            assert "POST" in route.methods
            assert "GET" not in route.methods
            return
    pytest.fail("/admin/memory/clone route not found")


# ─── Validation: source id ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clone_rejects_invalid_source_id_format():
    """Unsafe characters fail format validation regardless of allow_reserved."""
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.clone_subject(
            source_subject_id="has spaces",
            target_subject_id=None,
            target_display_name=None,
            target_tenant_id=None,
        )
    assert ei.value.status_code == 400


def test_clone_validator_accepts_reserved_prefix_as_source_directly():
    """Source-side carve-out: operators can clone visitor memory subjects."""
    sid = "demo_web_abc__support-agent"
    # Without `allow_reserved`, the validator refuses.
    with pytest.raises(mp.StarterPackError):
        mp._validate_subject_id(sid, field="source_subject_id")
    # `clone_subject` opts into `allow_reserved=True` for the source side.
    assert (
        mp._validate_subject_id(sid, field="source_subject_id", allow_reserved=True)
        == sid
    )


# ─── Validation: target id ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clone_rejects_unsafe_target_id():
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.clone_subject(
            source_subject_id="user-1",
            target_subject_id="has spaces",
            target_display_name=None,
            target_tenant_id=None,
        )
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_clone_rejects_reserved_prefix_as_target_id():
    """The reserved-prefix guard must still fire on the TARGET side — clone
    cannot create a new id in the marketing widget's namespace."""
    with pytest.raises(mp.StarterPackError) as ei:
        await mp.clone_subject(
            source_subject_id="user-1",
            target_subject_id="demo_web_abc",
            target_display_name=None,
            target_tenant_id=None,
        )
    assert ei.value.status_code == 400
    assert "reserved prefix" in str(ei.value)


# ─── 404 when the source has no records ─────────────────────────────────────


@pytest.mark.asyncio
async def test_clone_returns_404_when_source_subject_has_no_data(client):
    """A subject id with no episodes AND no memories is treated as not found.

    Subjects are only materialised in the data model as a function of
    their episodes/memories — there is no separate "subject row" — so an
    unknown id naturally maps to 404.
    """
    resp = await client.post(
        "/admin/memory/clone",
        json={"source_subject_id": "nonexistent-test-subject"},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert "not found" in body["error"]["message"].lower()


# ─── HTTP-level wiring ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clone_http_400_on_invalid_source_id(client):
    resp = await client.post(
        "/admin/memory/clone",
        json={"source_subject_id": "has spaces"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_clone_http_400_on_unsafe_target_id(client):
    resp = await client.post(
        "/admin/memory/clone",
        json={
            "source_subject_id": "user-1",
            "target_subject_id": "has spaces",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_clone_http_422_on_unknown_clone_scope(client):
    """The Pydantic Literal type rejects unknown scope values before the
    handler runs — that's a 422 from FastAPI, not our 400."""
    resp = await client.post(
        "/admin/memory/clone",
        json={
            "source_subject_id": "user-1",
            "clone_scope": "bogus",
        },
    )
    assert resp.status_code == 422


# ─── Response shape regression guard ────────────────────────────────────────


def test_clone_scope_literal_pins_supported_scopes():
    """If a future PR adds a scope, this test surfaces it as a deliberate
    contract change rather than a silent drift."""
    # CloneScope is defined in memory_packs as a Literal of these strings.
    import typing

    args = typing.get_args(mp.CloneScope)
    assert set(args) == {
        "episodes",
        "memories",
        "episodes_and_memories",
        "episodes_memories_sources",
    }


def test_clone_response_shape_does_not_carry_legacy_copied_keys():
    """Regression guard for the rename to spec response shape.

    We can't run a full clone here without a DB; instead inspect the
    handler module to confirm the legacy `copied_*` keys are gone — a
    future revert would re-introduce them.
    """
    from pathlib import Path

    src = Path(mp.__file__).read_text()
    assert '"copied_episodes"' not in src
    assert '"copied_memories"' not in src
    assert '"copied_sources"' not in src
    # And the new keys are present.
    assert '"episode_count"' in src
    assert '"memory_count"' in src
    assert '"source_count"' in src
    assert '"status"' in src
