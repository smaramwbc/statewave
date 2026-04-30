"""Tests for the admin dashboard aggregation endpoint."""

from __future__ import annotations

import pytest


def test_dashboard_endpoint_registered():
    """Verify the dashboard endpoint is registered on the app."""
    from server.app import app

    routes = [r.path for r in app.routes]
    assert "/admin/dashboard" in routes


def test_dashboard_endpoint_is_get():
    """Verify the dashboard endpoint accepts GET."""
    from server.app import app

    for route in app.routes:
        if hasattr(route, "path") and route.path == "/admin/dashboard":
            assert "GET" in route.methods
            break
    else:
        pytest.fail("/admin/dashboard route not found")


# ─── Citing Memories (Reverse Provenance) ─────────────────────────────────────


def test_citing_memories_endpoint_registered():
    """Verify the citing-memories endpoint is registered on the app."""
    from server.app import app

    routes = [r.path for r in app.routes]
    assert "/admin/subjects/{subject_id}/episodes/{episode_id}/citing-memories" in routes


def test_citing_memories_endpoint_is_get():
    """Verify the citing-memories endpoint accepts GET."""
    from server.app import app

    for route in app.routes:
        if (
            hasattr(route, "path")
            and route.path == "/admin/subjects/{subject_id}/episodes/{episode_id}/citing-memories"
        ):
            assert "GET" in route.methods
            break
    else:
        pytest.fail("citing-memories route not found")


# ─── Session Timeline ─────────────────────────────────────────────────────────


def test_session_timeline_endpoint_registered():
    """Verify the session timeline endpoint is registered on the app."""
    from server.app import app

    routes = [r.path for r in app.routes]
    assert "/admin/subjects/{subject_id}/sessions/{session_id}/timeline" in routes


def test_session_timeline_endpoint_is_get():
    """Verify the session timeline endpoint accepts GET."""
    from server.app import app

    for route in app.routes:
        if (
            hasattr(route, "path")
            and route.path == "/admin/subjects/{subject_id}/sessions/{session_id}/timeline"
        ):
            assert "GET" in route.methods
            break
    else:
        pytest.fail("session timeline route not found")
