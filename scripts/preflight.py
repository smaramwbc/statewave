#!/usr/bin/env python
"""Migration preflight check.

Run this BEFORE applying migrations to verify:
1. Database is reachable
2. Current schema revision is known
3. Pending migrations are identified
4. Go/no-go recommendation

Usage:
    python scripts/preflight.py
    DATABASE_URL=postgresql+asyncpg://... python scripts/preflight.py

Exit codes:
    0 — all clear, safe to migrate (or already up to date)
    1 — problem detected, review before proceeding
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> int:
    from server.services.migrations import check_migration_status

    print("=" * 60)
    print("  Statewave Migration Preflight Check")
    print("=" * 60)
    print()

    status = await check_migration_status()

    if status.error:
        print(f"❌ ERROR: {status.error}")
        print()
        print("Action: Fix the error above before proceeding.")
        return 1

    print(f"  Current revision : {status.current_revision or '(none — fresh DB)'}")
    print(f"  Expected head    : {status.expected_head}")
    print(f"  Pending          : {status.pending_count} migration(s)")
    print()

    if status.pending_revisions:
        print("  Pending migrations:")
        for rev in status.pending_revisions:
            print(f"    → {rev}")
        print()

    if status.is_compatible:
        print("✅ Schema is up to date. No migration needed.")
        return 0

    if status.current_revision is None:
        print("⚠️  Fresh database detected. All migrations will be applied.")
        print()
        print("  Recommendation: proceed with `alembic upgrade head`")
        return 0

    print(f"⚠️  {status.pending_count} migration(s) pending.")
    print()
    print("  Pre-migration checklist:")
    print("    1. Back up the database")
    print("    2. Ensure no active connections from old app version")
    print("    3. Run: alembic upgrade head")
    print("    4. Verify: python scripts/preflight.py")
    print("    5. Start the new app version")
    print()
    print("  Rollback if needed:")
    print(f"    alembic downgrade {status.current_revision}")
    print()
    print("✅ Safe to proceed with migration.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
