#!/usr/bin/env bash
# Poll /readyz until status=ready or timeout. Prints elapsed seconds on success.
set -euo pipefail

BASE_URL="${1:-http://localhost:8100}"
BASE_URL="${BASE_URL%/}"
TIMEOUT="${2:-60}"
INTERVAL="${3:-2}"

start=$SECONDS
deadline=$((start + TIMEOUT))

while (( SECONDS < deadline )); do
  if resp=$(curl -sf "${BASE_URL}/readyz" 2>/dev/null); then
    status=$(printf '%s' "$resp" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))")
    if [[ "$status" == "ready" ]]; then
      elapsed=$((SECONDS - start))
      echo "readyz: status=ready in ${elapsed}s"
      printf '%s\n' "$elapsed" > "${STATEWAVE_COLD_TIMING_FILE:-/tmp/statewave_cold_ready_seconds}"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done

echo "readyz: did not reach status=ready within ${TIMEOUT}s (url=${BASE_URL})" >&2
exit 1
