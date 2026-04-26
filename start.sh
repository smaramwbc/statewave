#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting Statewave server..."
exec uvicorn server.app:app --host 0.0.0.0 --port 8100
