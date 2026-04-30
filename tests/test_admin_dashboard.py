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
