"""Data residency enforcement (v0.9, issue #161).

Model: per-region deployment + metadata-pinned tenants. Each server
process declares its region via ``STATEWAVE_REGION`` (Settings.region).
Each tenant optionally pins to a region via
``tenant_configs.config.region``. Requests for a pinned tenant are
only allowed to run inside the matching region; anywhere else the
request is refused with HTTP 403 / ``residency.mismatch``.

Why metadata-pinned tenants and not in-DB partitioning:

  * **Correctness boundary at the application layer**, not DNS or
    the load balancer. A misrouted request that bypassed the LB
    hostname would still be caught by this module before any DB
    access happens. Belt-and-suspenders by design.
  * **Separate Postgres / pgvector deployments per region** are the
    intended infra topology, but the residency story does not depend
    on the DB layer for correctness — the application is the source
    of truth for "is this request allowed here?"
  * **Auditable**: a tenant's pinned region is plain data in
    ``tenant_configs.config.region``, queryable, patch-able through
    the existing admin surface, and recorded on every emitted
    receipt's ``region`` field.

What this module does NOT do (intentional, v0.9 scope):

  * **No cross-region reads.** Total isolation. An admin in region A
    cannot read tenant B's data in region B. The pinned region IS
    the access boundary; unified federated audit is a future
    feature, not implicit cross-region access.
  * **No automatic region rebalancing.** Pinning a tenant is an
    operator action that requires their data to be in the target
    region first; this module does not move data.
  * **No in-DB region column on every row.** That would conflate
    residency with multi-tenancy. Residency is per-tenant, not
    per-row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResidencyMismatch:
    """Returned by ``check_residency`` when a request is refused.

    The reason is intentionally simple — the caller (API layer)
    surfaces it as HTTP 403 with ``error.code = residency.mismatch``
    and a message that names the tenant region. Server region is
    NOT echoed in the response: an attacker probing the residency
    boundary should not learn what region they hit. The deny is
    flat: "this tenant lives elsewhere", not "you reached us-east
    while the tenant is in eu-west".
    """

    tenant_region: str
    """The region the tenant is pinned to (safe to surface: the
    operator gave it to us)."""
    server_region: str
    """The region THIS process is running in. NEVER returned on the
    wire; logged for operators only."""


def check_residency(
    *,
    tenant_config: dict[str, Any] | None,
    server_region: str | None,
) -> ResidencyMismatch | None:
    """Pure residency check. Returns ``None`` when the request is
    allowed; returns a ``ResidencyMismatch`` otherwise.

    Three cases collapse to allow:

      1. ``server_region`` is ``None`` — single-region deployment,
         residency disabled entirely. (A regression that left the
         setting unset must not start refusing requests.)
      2. ``tenant_config.region`` is ``None`` or absent — tenant is
         not pinned; it can run anywhere. Legacy tenants and
         single-region tenants take this path.
      3. ``tenant_config.region == server_region`` — exact match,
         allowed.

    Anything else returns ``ResidencyMismatch``. Comparison is exact
    string equality; case mismatches are treated as different
    regions on purpose (we'd rather fail loudly than silently allow
    ``EU`` and ``eu`` to mean the same thing).
    """
    if not server_region:
        return None
    tenant_region = (tenant_config or {}).get("region")
    if tenant_region is None or tenant_region == "":
        return None
    if not isinstance(tenant_region, str):
        # Defensive: a malformed JSONB value should not silently allow
        # the request. Treat as "configured but invalid" and refuse.
        return ResidencyMismatch(
            tenant_region=str(tenant_region),
            server_region=server_region,
        )
    if tenant_region == server_region:
        return None
    return ResidencyMismatch(
        tenant_region=tenant_region,
        server_region=server_region,
    )


def validate_region_pin(
    *,
    proposed_region: str,
    server_region: str | None,
) -> str | None:
    """Operator-facing validation: returns a refusal message when an
    operator tries to pin a tenant to a region this server doesn't
    serve.

    Pinning a tenant to a region different from the local
    ``server_region`` would immediately lock the tenant out of this
    deployment — every subsequent request would 403. That's almost
    always an operator mistake, so we refuse the pin at the admin
    endpoint unless ``server_region is None`` (single-region or
    development mode, where the check is meaningless).

    Returns:
        - ``None`` when the pin is safe.
        - A human-readable refusal message otherwise. The admin
          endpoint surfaces this in a 422 ``residency.invalid_pin``.
    """
    if not server_region:
        # No locally-known region — operator is in dev / single-region
        # mode. Allow the pin so they can author tenant configs that
        # will be deployed to a multi-region environment later.
        return None
    if proposed_region == server_region:
        return None
    return (
        f"this server is running in region {server_region!r}; pinning a "
        f"tenant to {proposed_region!r} from here would lock the tenant "
        "out (every subsequent request would 403). Patch the pin from "
        "a server running in the target region, or set "
        "STATEWAVE_REGION to match if this server should serve it."
    )
