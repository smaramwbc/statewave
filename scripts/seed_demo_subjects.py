"""Re-seed the website demo subjects with rich many-to-one provenance.

The website hero visualization in statewave-web reads the production demo
subjects via /v1/timeline. Earlier seeds produced 1:1 memories (each memory
citing exactly one episode), which makes the visualization look like a
spider plot of radial lines instead of the layered "memory↔episode"
structure that Statewave actually produces in real usage.

This script wipes each demo subject and imports a hand-crafted document
where every memory cites 1–5 source episodes, using the same admin import
endpoint that the official backup/restore tooling uses.

Usage:
    STATEWAVE_API_KEY=... python scripts/seed_demo_subjects.py
    STATEWAVE_API_KEY=... python scripts/seed_demo_subjects.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("STATEWAVE_URL", "https://statewave-api.fly.dev")
API_KEY = os.environ.get("STATEWAVE_API_KEY", "")
FORMAT_VERSION = "1.0"


# ─── Narrative seed data ─────────────────────────────────────────────────────
#
# Per subject:
#   "episodes": list of (source, type, payload) triples — order is chronological
#   "memories": list of (kind, content, summary, confidence, source_episode_idxs)
#
# source_episode_idxs is a list of indices into the episodes list — that's how
# we encode the many-to-one provenance without UUIDs in the static data. The
# script resolves them to real UUIDs at build time.

SEEDS: dict[str, dict[str, Any]] = {
    "demo-support-agent": {
        "episodes": [
            ("zendesk", "ticket_opened", {
                "channel": "email",
                "content": "Sarah Chen from Globex Corporation reported login failure on mobile app",
                "priority": "high",
            }),
            ("zendesk", "customer_reply", {
                "content": "Attached screenshot showing error code AUTH-503",
                "attachments": 1,
            }),
            ("zendesk", "agent_action", {
                "action": "escalate",
                "content": "Escalated to L2 — authentication service issue confirmed",
            }),
            ("zendesk", "agent_action", {
                "action": "password_reset",
                "content": "Password reset link sent; Sarah confirmed access restored",
            }),
            ("zendesk", "ticket_closed", {
                "content": "Login issue resolved; root cause was session token expiry on mobile clients",
                "resolution": "fixed",
            }),
            ("crm", "profile_update", {
                "content": "Verified profile: Sarah Chen, Enterprise plan, primary email sarah@globex.com",
            }),
            ("crm", "preference_recorded", {
                "content": "Prefers email contact over phone; bilingual (DE/EN)",
            }),
            ("zendesk", "ticket_opened", {
                "channel": "email",
                "content": "Sarah requested $50 refund for a failed payment retry charge",
                "priority": "normal",
            }),
            ("zendesk", "agent_action", {
                "action": "refund",
                "content": "Refund of $50 issued to original payment method (Visa ending 4112)",
            }),
            ("zendesk", "ticket_closed", {
                "content": "Refund confirmed received by Sarah",
                "resolution": "fixed",
            }),
            ("analytics", "usage_spike", {
                "content": "Globex API usage doubled this quarter (Q1 → Q2 2026)",
            }),
            ("crm", "account_review", {
                "content": "Annual review: Globex on track for plan upgrade in Q2 2026",
            }),
        ],
        "memories": [
            ("profile_fact", "Sarah Chen at Globex Corporation, Enterprise plan, primary email sarah@globex.com", "customer identity", 0.96, [0, 5]),
            ("profile_fact", "Prefers email contact over phone; bilingual (DE/EN)", "contact preference", 0.92, [6]),
            ("episode_summary", "AUTH-503 mobile login issue resolved — root cause: session token expiry on mobile clients; fixed via password reset", "resolved support issue", 0.95, [0, 1, 2, 3, 4]),
            ("episode_summary", "$50 refund issued to Visa ending 4112 for a failed payment retry; confirmed received", "completed refund", 0.97, [7, 8, 9]),
            ("profile_fact", "API usage doubled Q1→Q2 2026; account on track for plan upgrade in Q2 2026", "growth signal", 0.88, [10, 11]),
        ],
    },

    "demo-coding-assistant": {
        "episodes": [
            ("github", "pr_opened", {
                "content": "Bob opened PR #142: fix useEffect cleanup race",
                "repo": "taskflow/web",
                "pr_number": 142,
            }),
            ("github", "pr_review", {
                "content": "Reviewer noted missing dependency array in PR #142",
                "pr_number": 142,
            }),
            ("github", "pr_merged", {
                "content": "PR #142 merged: fix useEffect cleanup race",
                "pr_number": 142,
            }),
            ("slack", "channel_msg", {
                "channel": "#engineering",
                "content": "Bob mentioned migrating from raw SQL to SQLAlchemy in Taskflow",
            }),
            ("slack", "dm", {
                "content": "Bob: project Taskflow uses FastAPI on the backend with Postgres",
            }),
            ("github", "issue_opened", {
                "content": "Postgres connection pool exhaustion under load in Taskflow",
                "repo": "taskflow/api",
            }),
            ("github", "issue_resolved", {
                "content": "Resolved: increased pool size to 50 base with 100 overflow",
            }),
            ("linear", "ticket_assigned", {
                "content": "Bob assigned to Q1 OKR: migrate auth from custom JWT to FastAPI middleware",
            }),
            ("github", "commit", {
                "content": "feat(auth): integrate fastapi-users for OAuth + magic link",
                "repo": "taskflow/api",
            }),
            ("linear", "ticket_done", {
                "content": "Auth migration complete; deprecated old custom JWT helpers",
            }),
        ],
        "memories": [
            ("profile_fact", "Bob is lead engineer on Taskflow — stack: FastAPI, SQLAlchemy, Postgres", "developer identity + stack", 0.94, [3, 4]),
            ("procedure", "When useEffect race conditions appear, audit the dependency array completeness (per fix in PR #142)", "debugging procedure", 0.9, [0, 1, 2]),
            ("procedure", "Taskflow Postgres pool sizing: 50 base / 100 overflow (resolves load contention)", "architecture decision", 0.92, [5, 6]),
            ("episode_summary", "Auth stack migrated to fastapi-users (OAuth + magic link); old custom JWT helpers deprecated", "completed migration", 0.95, [7, 8, 9]),
        ],
    },

    "demo-sales-copilot": {
        "episodes": [
            ("salesforce", "lead_created", {
                "content": "Acme Corp inbound lead from website; 200 employees",
            }),
            ("salesforce", "contact_added", {
                "content": "Primary contact: Maria Lopez, VP Engineering at Acme Corp",
            }),
            ("calendar", "meeting", {
                "content": "Discovery call held with Maria Lopez on 2026-04-09",
                "duration_min": 45,
            }),
            ("notes", "meeting_summary", {
                "content": "Maria evaluating 3 vendors; main concern: data residency in EU",
            }),
            ("salesforce", "opportunity_created", {
                "content": "Opportunity 'Acme Pilot' created — $80k ARR, Stage: Discovery",
            }),
            ("calendar", "meeting", {
                "content": "Demo call scheduled with Acme CTO on 2026-04-17",
            }),
            ("notes", "meeting_summary", {
                "content": "CTO demo: positive on architecture; asked about SOC 2 compliance",
            }),
            ("salesforce", "stage_change", {
                "content": "Opp 'Acme Pilot' moved to Negotiation",
            }),
            ("salesforce", "proposal_sent", {
                "content": "Sent proposal: $84k ARR with EU data residency add-on",
            }),
            ("notes", "meeting_summary", {
                "content": "Maria pushed back on price; offered 3-year deal at $76k/yr",
            }),
            ("salesforce", "stage_change", {
                "content": "Opp 'Acme Pilot' Closed Won at $228k TCV (3yr at $76k/yr)",
            }),
            ("analytics", "competitor_signal", {
                "content": "Won against Vendor X on data residency + faster integration timeline",
            }),
        ],
        "memories": [
            ("profile_fact", "Acme Corp, 200 employees; primary contact Maria Lopez (VP Engineering); CTO involved in technical eval", "account profile", 0.95, [0, 1, 6]),
            ("profile_fact", "Acme requires EU data residency and SOC 2 compliance", "deal requirements", 0.93, [3, 6]),
            ("episode_summary", "Opp 'Acme Pilot' Closed Won — $228k TCV over 3 years at $76k/yr", "won deal", 0.98, [4, 7, 8, 9, 10]),
            ("profile_fact", "Won against Vendor X on data residency + faster integration timeline", "competitive intel", 0.9, [11]),
            ("episode_summary", "Two key calls: discovery on 2026-04-09 with Maria, demo on 2026-04-17 with CTO", "engagement timeline", 0.91, [2, 5, 6]),
        ],
    },

    "demo-devops-agent": {
        "episodes": [
            ("pagerduty", "alert", {
                "content": "Alert: API p95 latency exceeded 800ms in production",
                "severity": "high",
            }),
            ("datadog", "metric", {
                "content": "Postgres connection pool at 95% utilization",
            }),
            ("slack", "incident", {
                "content": "Engineer opened incident channel #inc-432",
            }),
            ("github", "commit", {
                "content": "Hotfix: increased Postgres pool size from 20 to 50",
                "repo": "infra/api",
            }),
            ("pagerduty", "resolved", {
                "content": "Alert resolved; p95 latency back to <200ms",
            }),
            ("notes", "postmortem", {
                "content": "Postmortem: pool exhaustion triggered by sudden 3x traffic spike",
            }),
            ("terraform", "plan", {
                "content": "Terraform plan: scale Postgres instance from db-2x to db-4x",
            }),
            ("terraform", "apply", {
                "content": "Applied: Postgres scaled to db-4x; rolling restart completed",
            }),
            ("datadog", "dashboard", {
                "content": "New SLO dashboard published: API latency, error rate, queue depth",
            }),
            ("runbook", "update", {
                "content": "Updated runbook: pool exhaustion playbook now references the SLO dashboard",
            }),
        ],
        "memories": [
            ("episode_summary", "Incident #432: API p95 latency spike caused by Postgres pool exhaustion under a 3x traffic surge; resolved", "resolved incident", 0.96, [0, 1, 2, 4, 5]),
            ("procedure", "Postgres: pool size increased to 50; instance scaled db-2x → db-4x", "infra decision", 0.94, [3, 6, 7]),
            ("procedure", "On p95 latency alert, check Postgres pool utilization first; reference the SLO dashboard runbook", "incident procedure", 0.91, [8, 9]),
            ("profile_fact", "Production Postgres: db-4x instance, pool size 50, monitored via SLO dashboard", "current infra state", 0.95, [6, 7, 8]),
        ],
    },

    "demo-research-assistant": {
        "episodes": [
            ("arxiv", "paper_saved", {
                "content": "Paper saved: 'Memory-augmented LLMs for long-horizon tasks' (Nakamura et al., 2026)",
            }),
            ("notes", "annotation", {
                "content": "Highlight: 47% performance gain on multi-session benchmarks (Nakamura)",
            }),
            ("arxiv", "paper_saved", {
                "content": "Paper saved: 'RAG vs Memory: a 2026 retrospective' (Singh et al., 2026)",
            }),
            ("notes", "annotation", {
                "content": "Key claim: structured memory beats RAG on temporal reasoning (Singh)",
            }),
            ("notion", "doc_updated", {
                "content": "Lit review draft — section 3: structured memory architectures",
            }),
            ("notion", "doc_updated", {
                "content": "Added Statewave to comparison table (provenance + token-bounded retrieval)",
            }),
            ("slack", "dm", {
                "content": "Discussed with PI: focus thesis on memory-augmented support agents",
            }),
            ("arxiv", "paper_saved", {
                "content": "Paper saved: 'Customer support agent benchmarks' (Le, 2025)",
            }),
            ("notes", "annotation", {
                "content": "Cited Le 2025 as baseline for our eval methodology",
            }),
            ("notion", "doc_updated", {
                "content": "Methodology section: 3 eval suites adapted from Le 2025 + Statewave benchmark",
            }),
        ],
        "memories": [
            ("profile_fact", "Thesis focus: memory-augmented LLMs for customer-support agent workflows", "research direction", 0.95, [4, 6]),
            ("episode_summary", "Structured memory (e.g. Statewave) outperforms RAG on temporal reasoning (per Singh 2026); Nakamura reports 47% gain on multi-session benchmarks", "literature finding", 0.92, [0, 1, 2, 3]),
            ("artifact_ref", "Active citations: Nakamura 2026, Singh 2026, Le 2025", "bibliography", 0.94, [0, 2, 7]),
            ("procedure", "Eval methodology: 3 suites adapted from Le 2025 baseline plus Statewave's own benchmark", "methodology spec", 0.9, [7, 8, 9]),
        ],
    },
}


def build_document(subject_id: str, seed: dict[str, Any], *, base_time: datetime) -> dict[str, Any]:
    """Convert a narrative seed into the export document accepted by /admin/import."""
    eps_seed = seed["episodes"]
    mems_seed = seed["memories"]

    # Episode UUIDs + timestamps spaced 1h apart starting from base_time
    episode_ids: list[str] = [str(uuid.uuid4()) for _ in eps_seed]
    episodes: list[dict[str, Any]] = []
    for i, (source, ep_type, payload) in enumerate(eps_seed):
        created_at = base_time + timedelta(hours=i)
        episodes.append({
            "id": episode_ids[i],
            "subject_id": subject_id,
            "tenant_id": None,
            "source": source,
            "type": ep_type,
            "payload": payload,
            "metadata": {},
            "provenance": {},
            "created_at": created_at.isoformat(),
            "last_compiled_at": (created_at + timedelta(seconds=30)).isoformat(),
        })

    # Memory UUIDs; valid_from is the latest source episode's created_at
    memories: list[dict[str, Any]] = []
    for kind, content, summary, confidence, source_idxs in mems_seed:
        if not source_idxs:
            raise ValueError(f"Memory {content!r} has no source_episode_idxs")
        latest_ep_time = max(
            datetime.fromisoformat(episodes[idx]["created_at"]) for idx in source_idxs
        )
        memory_id = str(uuid.uuid4())
        memories.append({
            "id": memory_id,
            "subject_id": subject_id,
            "tenant_id": None,
            "kind": kind,
            "content": content,
            "summary": summary,
            "confidence": confidence,
            "valid_from": latest_ep_time.isoformat(),
            "valid_to": None,
            "source_episode_ids": [episode_ids[idx] for idx in source_idxs],
            "metadata": {},
            "status": "active",
            "embedding": None,
            "created_at": (latest_ep_time + timedelta(minutes=1)).isoformat(),
            "updated_at": (latest_ep_time + timedelta(minutes=1)).isoformat(),
        })

    # Build the document with checksum (matches server's hashing rule)
    content_str = json.dumps({"episodes": episodes, "memories": memories}, sort_keys=True)
    checksum = hashlib.sha256(content_str.encode()).hexdigest()

    return {
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "tenant_id": None,
        "counts": {"episodes": len(episodes), "memories": len(memories)},
        "episodes": episodes,
        "memories": memories,
        "checksum": checksum,
    }


def http_request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    api_key: str,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except Exception:
            payload = {"error": str(e)}
        return e.code, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Base API URL (default: {DEFAULT_URL})")
    parser.add_argument("--dry-run", action="store_true", help="Build documents but don't wipe or import")
    parser.add_argument("--only", help="Comma-separated subject IDs to seed (default: all)")
    parser.add_argument(
        "--wipe-all",
        action="store_true",
        help="Wipe ALL subjects (not just the demo ones) before seeding — full reset",
    )
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: STATEWAVE_API_KEY env var required", file=sys.stderr)
        return 2

    selected = set(args.only.split(",")) if args.only else set(SEEDS.keys())
    targets = [(sid, seed) for sid, seed in SEEDS.items() if sid in selected]
    if not targets:
        print("ERROR: no matching subjects", file=sys.stderr)
        return 2

    base_time = datetime.now(timezone.utc) - timedelta(days=30)

    # Optional: wipe ALL subjects (every non-demo subject too) before seeding.
    # This gives a fully clean DB with only the curated demo data.
    if args.wipe_all and not args.dry_run:
        print("\n=== full wipe (--wipe-all) ===")
        list_url = f"{args.url}/v1/subjects"
        status, body = http_request("GET", list_url, api_key=API_KEY)
        if status >= 400:
            print(f"  list subjects FAILED status={status} body={body}", file=sys.stderr)
            return 1
        all_subjects = body.get("subjects", []) if isinstance(body, dict) else []
        print(f"  found {len(all_subjects)} existing subject(s)")
        for s in all_subjects:
            sid = s.get("subject_id") if isinstance(s, dict) else s
            if not sid:
                continue
            del_url = f"{args.url}/v1/subjects/{sid}"
            d_status, d_body = http_request("DELETE", del_url, api_key=API_KEY)
            if d_status not in (200, 204, 404):
                print(f"  wipe FAILED for {sid}: status={d_status} body={d_body}", file=sys.stderr)
                return 1
            deleted = d_body.get("episodes_deleted", "?") if isinstance(d_body, dict) else "?"
            print(f"  wiped {sid} (episodes_deleted={deleted})")

    for subject_id, seed in targets:
        print(f"\n=== {subject_id} ===")
        doc = build_document(subject_id, seed, base_time=base_time)
        ep_count = len(doc["episodes"])
        mem_count = len(doc["memories"])
        total_refs = sum(len(m["source_episode_ids"]) for m in doc["memories"])
        ratio = total_refs / mem_count if mem_count else 0
        cited_eps = {
            eid for m in doc["memories"] for eid in m["source_episode_ids"]
        }
        orphans = ep_count - len(cited_eps)
        print(f"  built: {ep_count} episodes, {mem_count} memories, "
              f"{total_refs} refs (avg {ratio:.1f} eps/memory), {orphans} orphan episode(s)")

        if args.dry_run:
            print("  [dry-run] skipping wipe + import")
            continue

        # Wipe
        del_url = f"{args.url}/v1/subjects/{subject_id}"
        status, body = http_request("DELETE", del_url, api_key=API_KEY)
        if status not in (200, 204, 404):
            print(f"  WIPE FAILED status={status} body={body}", file=sys.stderr)
            return 1
        deleted = body.get("episodes_deleted", body.get("deleted", "?")) if isinstance(body, dict) else "?"
        print(f"  wiped: {status} (episodes_deleted={deleted})")

        # Import
        import_url = f"{args.url}/admin/import"
        status, body = http_request(
            "POST",
            import_url,
            body={"document": doc, "preserve_ids": True},
            api_key=API_KEY,
        )
        if status >= 400:
            print(f"  IMPORT FAILED status={status} body={body}", file=sys.stderr)
            return 1
        print(f"  imported: status={status} -> "
              f"{body.get('episodes_imported')} episodes, "
              f"{body.get('memories_imported')} memories")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
