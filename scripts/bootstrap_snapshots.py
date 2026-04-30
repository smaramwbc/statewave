"""Bootstrap subject snapshots for demo instant seeding.

Usage:
    python -m scripts.bootstrap_snapshots

Idempotent: skips snapshots that already exist with the same name+version.
To rebuild, bump TEMPLATE_VERSION or delete old snapshots first.

Requires:
    STATEWAVE_URL (default http://localhost:8100)
    STATEWAVE_API_KEY (optional)
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

STATEWAVE_URL = os.environ.get("STATEWAVE_URL", "http://localhost:8100")
STATEWAVE_API_KEY = os.environ.get("STATEWAVE_API_KEY", "")
TEMPLATE_VERSION = 2  # Bump when persona seed data changes — v2 adds session_id

PERSONAS: dict[str, list[dict]] = {
    "sarah-startup": [
        {"source": "support_chat", "type": "customer_profile", "session_id": "onboarding-001", "payload": {"text": "Customer Sarah Chen, CTO at NovaTech. Enterprise plan since January. Team of 12 developers.", "name": "Sarah Chen", "company": "NovaTech", "role": "CTO", "plan": "enterprise", "team_size": 12}},
        {"source": "support_chat", "type": "technical_context", "session_id": "onboarding-001", "payload": {"text": "Stack: Python 3.11, FastAPI, PostgreSQL, deployed on AWS us-east-1 using Terraform. Uses GitHub Actions for CI/CD.", "language": "python", "framework": "fastapi", "database": "postgresql", "cloud": "aws", "region": "us-east-1"}},
        {"source": "support_chat", "type": "preference", "session_id": "onboarding-001", "payload": {"text": "Sarah prefers email for follow-ups, not Slack. Likes brief technical answers without too much hand-holding.", "contact_preference": "email", "communication_style": "brief_technical"}},
        {"source": "support_chat", "type": "past_issue", "session_id": "ticket-tls-feb", "payload": {"text": "February 15: Sarah had a TLS connection error with PostgreSQL. Resolved by adding sslmode=require to connection string.", "issue": "TLS connection error", "solution": "sslmode=require", "date": "2026-02-15"}},
        {"source": "support_chat", "type": "past_issue", "session_id": "ticket-billing-mar", "payload": {"text": "March 3: Billing discrepancy — charged for 15 seats but only has 12. Refund issued, confirmed correct seat count.", "issue": "billing_overcharge", "resolution": "refund_issued", "correct_seats": 12, "date": "2026-03-03"}},
        {"source": "support_chat", "type": "feature_request", "session_id": "ticket-feature-mar", "payload": {"text": "Sarah requested read replica support for their PostgreSQL setup. Noted for Q2 roadmap.", "feature": "read_replicas", "status": "requested", "date": "2026-03-10"}},
    ],
    "marcus-agency": [
        {"source": "support_chat", "type": "customer_profile", "session_id": "onboarding-001", "payload": {"text": "Marcus Rivera, founder of PixelForge agency. Pro plan. Uses API to build white-label dashboards for his clients.", "name": "Marcus Rivera", "company": "PixelForge", "role": "Founder", "plan": "pro", "use_case": "white_label_dashboards"}},
        {"source": "support_chat", "type": "technical_context", "session_id": "onboarding-001", "payload": {"text": "Stack: Next.js, TypeScript, Vercel, Supabase. Uses our REST API heavily — averaging 50k calls/day across client dashboards.", "framework": "nextjs", "language": "typescript", "hosting": "vercel", "database": "supabase", "api_usage": "50k_calls_per_day"}},
        {"source": "support_chat", "type": "preference", "session_id": "onboarding-001", "payload": {"text": "Marcus strongly prefers Slack DMs for communication. Hates long emails. Wants bullet-point answers only.", "contact_preference": "slack_dm", "communication_style": "bullet_points_only", "dislikes": "long_emails"}},
        {"source": "support_chat", "type": "past_issue", "session_id": "ticket-ratelimit-mar", "payload": {"text": "March 20: Hit API rate limit (10k/hour) during client demo. Temporary limit increase granted to 25k/hour.", "issue": "api_rate_limit", "resolution": "temp_increase", "date": "2026-03-20"}},
        {"source": "support_chat", "type": "past_issue", "session_id": "ticket-webhook-apr", "payload": {"text": "April 2: Webhook deliveries failing to Vercel endpoint. Fixed by updating webhook URL from hobby to pro Vercel domain.", "issue": "webhook_delivery_failure", "solution": "update_vercel_domain", "date": "2026-04-02"}},
    ],
    "priya-enterprise": [
        {"source": "support_chat", "type": "customer_profile", "session_id": "onboarding-001", "payload": {"text": "Priya Sharma, Lead Architect at MedSecure Health. Enterprise plan, 85-person engineering org. Healthcare vertical — strict compliance requirements.", "name": "Priya Sharma", "company": "MedSecure Health", "role": "Lead Architect", "plan": "enterprise", "team_size": 85, "vertical": "healthcare"}},
        {"source": "support_chat", "type": "technical_context", "session_id": "onboarding-001", "payload": {"text": "Multi-region deployment: primary in us-east-1, DR in eu-west-1. Uses Kubernetes, Java/Spring Boot, and Oracle DB. Needs HIPAA and SOC2 compliance.", "regions": ["us-east-1", "eu-west-1"], "orchestration": "kubernetes", "language": "java", "framework": "spring_boot", "database": "oracle", "compliance": ["hipaa", "soc2"]}},
        {"source": "support_chat", "type": "preference", "session_id": "onboarding-001", "payload": {"text": "Priya requires all communications via their Jira Service Desk. No informal channels. Needs audit trails for every support interaction.", "contact_preference": "jira_service_desk", "requires_audit_trail": True, "communication_style": "formal_documented"}},
        {"source": "support_chat", "type": "past_issue", "session_id": "ticket-residency-jan", "payload": {"text": "January 28: Data residency concern — needed guarantee that EU user data stays in eu-west-1. Confirmed region pinning is active on their Enterprise plan.", "issue": "data_residency", "resolution": "region_pinning_confirmed", "date": "2026-01-28"}},
        {"source": "support_chat", "type": "upcoming_event", "session_id": "ticket-audit-prep", "payload": {"text": "Security audit scheduled for May 2026. Priya needs our SOC2 Type II report and data processing agreement (DPA) updated before then.", "event": "security_audit", "date": "2026-05", "needs": ["soc2_type_ii_report", "updated_dpa"]}},
        {"source": "support_chat", "type": "past_issue", "session_id": "ticket-failover-mar", "payload": {"text": "March 15: Failover test from us-east-1 to eu-west-1 took 45 seconds. Priya's SLA requires <30s. Engineering working on fix — ETA April 30.", "issue": "failover_too_slow", "current": "45s", "target": "30s", "fix_eta": "2026-04-30"}},
    ],
}


async def bootstrap_persona(client: httpx.AsyncClient, persona_id: str, episodes: list[dict]):
    """Bootstrap a single persona snapshot."""
    # Check if snapshot already exists
    resp = await client.get(f"{STATEWAVE_URL}/admin/snapshots")
    if resp.status_code == 200:
        existing = resp.json().get("snapshots", [])
        for s in existing:
            if s["name"] == persona_id and s["version"] == TEMPLATE_VERSION:
                print(f"  ✓ Snapshot '{persona_id}' v{TEMPLATE_VERSION} already exists, skipping")
                return

    # Use a temporary subject to build from
    temp_subject = f"_bootstrap_tmp/{persona_id}"

    # Clean up any previous temp data
    try:
        await client.delete(f"{STATEWAVE_URL}/v1/subjects/{temp_subject}")
    except Exception:
        pass

    # Ingest episodes
    print(f"  → Ingesting {len(episodes)} episodes...")
    for ep in episodes:
        resp = await client.post(
            f"{STATEWAVE_URL}/v1/episodes",
            json={"subject_id": temp_subject, **ep},
        )
        if resp.status_code not in (200, 201):
            print(f"    ERROR ingesting: {resp.status_code} {resp.text}")
            return

    # Compile memories
    print("  → Compiling memories...")
    resp = await client.post(
        f"{STATEWAVE_URL}/v1/memories/compile",
        json={"subject_id": temp_subject},
    )
    if resp.status_code not in (200, 201):
        print(f"    ERROR compiling: {resp.status_code} {resp.text}")
        return

    # Create snapshot from the compiled subject
    print("  → Creating snapshot...")
    resp = await client.post(
        f"{STATEWAVE_URL}/admin/snapshots",
        json={
            "name": persona_id,
            "source_subject_id": temp_subject,
            "version": TEMPLATE_VERSION,
            "metadata": {"type": "demo_persona", "template_version": TEMPLATE_VERSION},
        },
    )
    if resp.status_code not in (200, 201):
        print(f"    ERROR creating snapshot: {resp.status_code} {resp.text}")
        return

    data = resp.json()
    print(f"  ✓ Snapshot '{persona_id}' v{TEMPLATE_VERSION} created: {data.get('episode_count', '?')} episodes, {data.get('memory_count', '?')} memories")

    # Clean up temp subject
    await client.delete(f"{STATEWAVE_URL}/v1/subjects/{temp_subject}")


async def main():
    print(f"=== Statewave Snapshot Bootstrap (v{TEMPLATE_VERSION}) ===")
    print(f"Server: {STATEWAVE_URL}")
    print()

    headers = {"Content-Type": "application/json"}
    if STATEWAVE_API_KEY:
        headers["X-API-Key"] = STATEWAVE_API_KEY

    async with httpx.AsyncClient(headers=headers, timeout=120.0) as client:
        # Health check
        try:
            resp = await client.get(f"{STATEWAVE_URL}/healthz")
            resp.raise_for_status()
        except Exception as e:
            print(f"ERROR: Cannot reach server at {STATEWAVE_URL}: {e}")
            sys.exit(1)

        for persona_id, episodes in PERSONAS.items():
            print(f"[{persona_id}]")
            await bootstrap_persona(client, persona_id, episodes)
            print()

    print("Done! All snapshots bootstrapped.")


if __name__ == "__main__":
    asyncio.run(main())
