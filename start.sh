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
import asyncpg

URL = os.environ.get("STATEWAVE_DATABASE_URL") or os.environ.get("DATABASE_URL")
if not URL:
    sys.exit("Neither STATEWAVE_DATABASE_URL nor DATABASE_URL is set.")

# asyncpg.connect doesn't understand the SQLAlchemy '+asyncpg' driver suffix.
URL = URL.replace("+asyncpg", "")

ATTEMPTS = 30  # ~60s with 2s backoff
DELAY_S = 2.0


async def wait():
    last_exc = None
    for i in range(1, ATTEMPTS + 1):
        try:
            conn = await asyncpg.connect(URL, timeout=5)
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

echo "Starting Statewave server..."
exec uvicorn server.app:app --host 0.0.0.0 --port 8100
