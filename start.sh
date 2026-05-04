#!/bin/sh
# Boot script for the Fly.io machine.
#
# We wait for Postgres to be reachable before running migrations. On Fly
# (especially with auto_stop_machines=suspend) the database can briefly
# refuse connections during a wake-up, which previously crashed alembic and
# put the machine in a restart loop. The retry loop turns that into a soft
# delay instead of a fatal boot.

set -e

echo "Waiting for database..."
python - <<'PY'
import asyncio
import os
import sys
import urllib.parse
import asyncpg

URL = os.environ.get("STATEWAVE_DATABASE_URL") or os.environ.get("DATABASE_URL")
if not URL:
    sys.exit("Neither STATEWAVE_DATABASE_URL nor DATABASE_URL is set.")

# asyncpg.connect doesn't understand the SQLAlchemy '+asyncpg' driver suffix.
URL = URL.replace("+asyncpg", "")

# SQLAlchemy URLs may carry SSL hints as query params (ssl=disable / ssl=require)
# that SQLAlchemy translates internally. Raw asyncpg interprets unknown params
# as "SET <param>=<value>" runtime knobs and chokes — drop ssl* params from the
# URL and surface them as a kwarg instead.
parsed = urllib.parse.urlsplit(URL)
qs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
ssl_kwarg = None
clean_qs = []
for k, v in qs:
    if k.lower() in ("ssl", "sslmode"):
        if v.lower() in ("disable", "false", "off", "0", ""):
            ssl_kwarg = False
        elif v.lower() in ("require", "verify-full", "verify-ca", "true", "on", "1"):
            ssl_kwarg = True
        # else: leave to asyncpg defaults
    else:
        clean_qs.append((k, v))
URL = urllib.parse.urlunsplit(
    (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(clean_qs), parsed.fragment)
)

ATTEMPTS = 30  # ~60s with 2s backoff
DELAY_S = 2.0


async def wait():
    last_exc = None
    for i in range(1, ATTEMPTS + 1):
        try:
            kwargs = {"timeout": 5}
            if ssl_kwarg is not None:
                kwargs["ssl"] = ssl_kwarg
            conn = await asyncpg.connect(URL, **kwargs)
            await conn.close()
            print(f"  database reachable (attempt {i})")
            return
        except Exception as e:  # noqa: BLE001 — any failure is a retry signal
            last_exc = e
            print(f"  attempt {i}/{ATTEMPTS}: {e!r}", file=sys.stderr)
            await asyncio.sleep(DELAY_S)
    sys.exit(f"Postgres unreachable after {ATTEMPTS * DELAY_S:.0f}s: {last_exc!r}")


asyncio.run(wait())
PY

echo "Running database migrations..."
alembic upgrade head

# ─── Auto-bootstrap: Statewave Support docs pack ────────────────────────
#
# Seeds the `statewave-support-docs` subject so the docs-grounded
# "Statewave Support" persona answers from real content out of the box.
# Triggers only when:
#   1. STATEWAVE_BOOTSTRAP_DOCS_PACK is unset or true (default true), AND
#   2. The docs corpus is reachable at STATEWAVE_DOCS_PATH (default /docs).
#
# Idempotent — the bootstrap script exits 2 when the subject already has
# episodes, which we treat as "already seeded, nothing to do". Production
# deploys (Fly) don't ship the corpus inside the image, so the path check
# silently skips this. Operators who want to disable explicitly can set
# STATEWAVE_BOOTSTRAP_DOCS_PACK=false.
#
# NOTE: This live-docs path is now superseded by the bundled-pack
# auto-update below. We keep the live-docs path for dev environments
# that mount `/docs` (so a docs change on the host is picked up without
# a build_support_pack regen + image rebuild). When neither
# STATEWAVE_BOOTSTRAP_DOCS_PACK is true nor `/docs` is mounted, the
# bundled-pack auto-update covers production.
DOCS_PATH="${STATEWAVE_DOCS_PATH:-/docs}"
BOOTSTRAP="${STATEWAVE_BOOTSTRAP_DOCS_PACK:-true}"
DOCS_MOUNTED=0
if [ "$BOOTSTRAP" = "true" ] && [ -d "$DOCS_PATH" ]; then
    DOCS_MOUNTED=1
    echo "Auto-bootstrap: docs pack will seed from /docs after API is ready (DOCS_PATH=$DOCS_PATH)"
    (
        # Disable errexit inside the subshell — the bootstrap script
        # exits 2 to signal "subject already populated", which is a
        # success outcome for us. Without `set +e`, the inherited
        # `set -e` from the parent script would kill this subshell
        # before the case statement below could log the outcome.
        set +e
        # Wait up to 60s for /healthz before attempting the seed. Falls
        # back silently if the API never comes up — the user-visible
        # error will be the failed startup itself, not a confusing
        # bootstrap-side log.
        for i in $(seq 1 60); do
            if curl -sS -m 2 -o /dev/null http://127.0.0.1:8100/healthz 2>/dev/null; then
                break
            fi
            sleep 1
        done
        STATEWAVE_DOCS_PATH="$DOCS_PATH" \
        STATEWAVE_URL="http://127.0.0.1:8100" \
            python -m scripts.bootstrap_docs_pack
        rc=$?
        case "$rc" in
            0) echo "Auto-bootstrap: Statewave Support docs pack seeded from /docs." ;;
            2) echo "Auto-bootstrap: docs pack already populated — skipped." ;;
            *) echo "Auto-bootstrap: bootstrap exited with code $rc (server is still serving)." >&2 ;;
        esac
    ) &
fi

# ─── Auto-update: support pack from bundled JSONL ────────────────────────
#
# Reseeds `statewave-support-docs` from the bundled
# `statewave-support-agent` starter pack baked into the image. The reseed
# endpoint is version-aware: if the live subject already carries the
# bundled pack's version it returns a no-op without touching any rows.
# Calling on every container restart is therefore idempotent — fresh
# installs get seeded, image upgrades pick up the new pack, no-op when
# already current. Selective purge means operator-added rows survive.
#
# Skipped when the live-docs path above ran (operator opted into refreshing
# from the mounted /docs directly) — that path produces equivalent rows so
# running both back-to-back would be redundant.
#
# Operators who want to disable can set
# STATEWAVE_AUTO_UPDATE_SUPPORT_PACK=false. The drawer's manual "Restore"
# button still works either way (it sets force=true).
AUTO_UPDATE="${STATEWAVE_AUTO_UPDATE_SUPPORT_PACK:-true}"
if [ "$AUTO_UPDATE" = "true" ] && [ "$DOCS_MOUNTED" = "0" ]; then
    echo "Auto-update: support pack will reseed (or no-op) after API is ready."
    (
        set +e
        for i in $(seq 1 60); do
            if curl -sS -m 2 -o /dev/null http://127.0.0.1:8100/healthz 2>/dev/null; then
                break
            fi
            sleep 1
        done
        AUTH_HEADER=""
        if [ -n "$STATEWAVE_API_KEY" ]; then
            AUTH_HEADER="X-API-Key: $STATEWAVE_API_KEY"
        fi
        BODY='{"reason":"auto-update on container start"}'
        RESP=$(curl -sS -m 900 -X POST \
            -H "Content-Type: application/json" \
            ${AUTH_HEADER:+-H "$AUTH_HEADER"} \
            -d "$BODY" \
            http://127.0.0.1:8100/admin/memory/support/reseed 2>&1)
        rc=$?
        if [ "$rc" = "0" ]; then
            outcome=$(echo "$RESP" | grep -oE '"outcome":"[a-z_]+"' | head -1 | cut -d'"' -f4)
            ver=$(echo "$RESP" | grep -oE '"installed_version":"[^"]*"' | head -1 | cut -d'"' -f4)
            case "$outcome" in
                already_current) echo "Auto-update: support pack already at v$ver — no-op." ;;
                seeded)          echo "Auto-update: support pack freshly seeded at v$ver." ;;
                auto_updated)    echo "Auto-update: support pack upgraded to v$ver." ;;
                "")              echo "Auto-update: reseed call returned (could not parse outcome): $RESP" >&2 ;;
                *)               echo "Auto-update: support pack reseeded ($outcome) at v$ver." ;;
            esac
        else
            echo "Auto-update: reseed call failed (curl rc=$rc). Server still serving." >&2
        fi
    ) &
elif [ "$AUTO_UPDATE" = "true" ]; then
    echo "Auto-update: skipped (live-docs bootstrap is handling this restart)."
else
    echo "Auto-update: STATEWAVE_AUTO_UPDATE_SUPPORT_PACK=false — support pack will not auto-reseed."
fi

echo "Starting Statewave server..."
exec uvicorn server.app:app --host 0.0.0.0 --port 8100
