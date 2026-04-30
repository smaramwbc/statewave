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
TEMPLATE_VERSION = 4  # v4: adds real support conversations with user/agent turns + resolutions

# Each persona has:
# - episodes: the raw data (context + support conversation turns)
# - resolutions: session resolution records for SLA tracking
PERSONAS: dict[str, dict] = {
    "sarah-startup": {
        "episodes": [
            # Onboarding context (no SLA tracking needed)
            {"source": "crm", "type": "customer_profile", "session_id": "onboarding-001", "payload": {"text": "Customer Sarah Chen, CTO at NovaTech. Enterprise plan since January. Team of 12 developers.", "name": "Sarah Chen", "company": "NovaTech", "role": "CTO", "plan": "enterprise", "team_size": 12}},
            {"source": "crm", "type": "technical_context", "session_id": "onboarding-001", "payload": {"text": "Stack: Python 3.11, FastAPI, PostgreSQL, deployed on AWS us-east-1 using Terraform. Uses GitHub Actions for CI/CD.", "language": "python", "framework": "fastapi", "database": "postgresql", "cloud": "aws", "region": "us-east-1"}},
            {"source": "crm", "type": "preference", "session_id": "onboarding-001", "payload": {"text": "Sarah prefers email for follow-ups, not Slack. Likes brief technical answers without too much hand-holding.", "contact_preference": "email", "communication_style": "brief_technical"}},

            # TLS ticket - resolved quickly (good SLA)
            {"source": "user", "type": "support_message", "session_id": "ticket-tls-feb", "payload": {"text": "Getting SSL/TLS connection errors when connecting to PostgreSQL. Error: 'SSL SYSCALL error: EOF detected'. Started happening after your maintenance window.", "channel": "email"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-tls-feb", "payload": {"text": "Hi Sarah, thanks for reaching out. This is a known issue after the maintenance — you'll need to add sslmode=require to your connection string. Can you try that?", "agent": "Alex"}},
            {"source": "user", "type": "support_message", "session_id": "ticket-tls-feb", "payload": {"text": "That fixed it, thanks! Quick turnaround as always.", "channel": "email"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-tls-feb", "payload": {"text": "Great! I'll mark this as resolved. Let me know if anything else comes up.", "agent": "Alex"}},

            # Billing ticket - resolved (good SLA)  
            {"source": "user", "type": "support_message", "session_id": "ticket-billing-mar", "payload": {"text": "Our invoice shows 15 seats but we only have 12 developers. Can you check this?", "channel": "email"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-billing-mar", "payload": {"text": "Hi Sarah, I see the discrepancy. Looks like 3 seats weren't removed after your team changes in February. I've corrected this and issued a refund of $450. Should appear in 3-5 business days.", "agent": "Jordan"}},
            {"source": "user", "type": "support_message", "session_id": "ticket-billing-mar", "payload": {"text": "Perfect, thank you Jordan!", "channel": "email"}},

            # Feature request - open (still being tracked)
            {"source": "user", "type": "support_message", "session_id": "ticket-feature-mar", "payload": {"text": "We're scaling up and need read replica support for our PostgreSQL setup. Is this on the roadmap?", "channel": "email"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-feature-mar", "payload": {"text": "Hi Sarah, great question! Read replica support is planned for Q2. I've added your request to our tracking. Would you like me to notify you when it ships?", "agent": "Alex"}},
            {"source": "user", "type": "support_message", "session_id": "ticket-feature-mar", "payload": {"text": "Yes please, that would be great.", "channel": "email"}},
        ],
        "resolutions": [
            {"session_id": "ticket-tls-feb", "status": "resolved", "resolution_summary": "Added sslmode=require to connection string to fix TLS errors after maintenance window"},
            {"session_id": "ticket-billing-mar", "status": "resolved", "resolution_summary": "Corrected seat count from 15 to 12, issued $450 refund"},
            {"session_id": "ticket-feature-mar", "status": "open", "resolution_summary": "Feature request for read replicas - tracking for Q2"},
        ],
    },
    "marcus-agency": {
        "episodes": [
            # Onboarding context
            {"source": "crm", "type": "customer_profile", "session_id": "onboarding-001", "payload": {"text": "Marcus Rivera, founder of PixelForge agency. Pro plan. Uses API to build white-label dashboards for his clients.", "name": "Marcus Rivera", "company": "PixelForge", "role": "Founder", "plan": "pro", "use_case": "white_label_dashboards"}},
            {"source": "crm", "type": "technical_context", "session_id": "onboarding-001", "payload": {"text": "Stack: Next.js, TypeScript, Vercel, Supabase. Uses our REST API heavily — averaging 50k calls/day across client dashboards.", "framework": "nextjs", "language": "typescript", "hosting": "vercel", "database": "supabase", "api_usage": "50k_calls_per_day"}},
            {"source": "crm", "type": "preference", "session_id": "onboarding-001", "payload": {"text": "Marcus strongly prefers Slack DMs for communication. Hates long emails. Wants bullet-point answers only.", "contact_preference": "slack_dm", "communication_style": "bullet_points_only", "dislikes": "long_emails"}},

            # Rate limit ticket - resolved (slightly slow first response)
            {"source": "user", "type": "support_message", "session_id": "ticket-ratelimit-mar", "payload": {"text": "URGENT: Hit rate limit during live client demo!! Need immediate help", "channel": "slack"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-ratelimit-mar", "payload": {"text": "On it Marcus. Temporarily increasing your limit from 10k to 25k/hour. Should be active in ~2 mins.", "agent": "Sam"}},
            {"source": "user", "type": "support_message", "session_id": "ticket-ratelimit-mar", "payload": {"text": "working now, lifesaver 🙏", "channel": "slack"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-ratelimit-mar", "payload": {"text": "Great! The increase is temp (7 days). Ping me if you need to discuss permanent upgrade.", "agent": "Sam"}},

            # Webhook ticket - resolved 
            {"source": "user", "type": "support_message", "session_id": "ticket-webhook-apr", "payload": {"text": "Webhooks stopped working to our Vercel endpoint. Getting timeouts.", "channel": "slack"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-webhook-apr", "payload": {"text": "Checking... I see the issue. Your webhook URL uses Vercel hobby tier which has cold start delays. Update to your pro domain and it should work.", "agent": "Sam"}},
            {"source": "user", "type": "support_message", "session_id": "ticket-webhook-apr", "payload": {"text": "Updated. All good now, thanks!", "channel": "slack"}},
        ],
        "resolutions": [
            {"session_id": "ticket-ratelimit-mar", "status": "resolved", "resolution_summary": "Temporary rate limit increase from 10k to 25k/hour for 7 days"},
            {"session_id": "ticket-webhook-apr", "status": "resolved", "resolution_summary": "Webhook timeouts fixed by updating from Vercel hobby to pro domain"},
        ],
    },
    "priya-enterprise": {
        "episodes": [
            # Onboarding context
            {"source": "crm", "type": "customer_profile", "session_id": "onboarding-001", "payload": {"text": "Priya Sharma, Lead Architect at MedSecure Health. Enterprise plan, 85-person engineering org. Healthcare vertical — strict compliance requirements.", "name": "Priya Sharma", "company": "MedSecure Health", "role": "Lead Architect", "plan": "enterprise", "team_size": 85, "vertical": "healthcare"}},
            {"source": "crm", "type": "technical_context", "session_id": "onboarding-001", "payload": {"text": "Multi-region deployment: primary in us-east-1, DR in eu-west-1. Uses Kubernetes, Java/Spring Boot, and Oracle DB. Needs HIPAA and SOC2 compliance.", "regions": ["us-east-1", "eu-west-1"], "orchestration": "kubernetes", "language": "java", "framework": "spring_boot", "database": "oracle", "compliance": ["hipaa", "soc2"]}},
            {"source": "crm", "type": "preference", "session_id": "onboarding-001", "payload": {"text": "Priya requires all communications via their Jira Service Desk. No informal channels. Needs audit trails for every support interaction.", "contact_preference": "jira_service_desk", "requires_audit_trail": True, "communication_style": "formal_documented"}},

            # Data residency ticket - resolved
            {"source": "user", "type": "support_message", "session_id": "ticket-residency-jan", "payload": {"text": "For GDPR compliance, we need written confirmation that EU user data in our system stays in eu-west-1 and never touches US servers.", "channel": "jira"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-residency-jan", "payload": {"text": "Hi Priya, I can confirm that region pinning is active on your Enterprise plan. EU data is isolated to eu-west-1. I'm attaching our data residency documentation and can provide a signed letter if needed for your compliance records.", "agent": "Taylor"}},
            {"source": "user", "type": "support_message", "session_id": "ticket-residency-jan", "payload": {"text": "The documentation is sufficient. Please add the signed letter to our account file for our May audit.", "channel": "jira"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-residency-jan", "payload": {"text": "Done. Letter uploaded to your account documents. Let me know if you need anything else for the audit.", "agent": "Taylor"}},

            # Audit prep - open (upcoming)
            {"source": "user", "type": "support_message", "session_id": "ticket-audit-may", "payload": {"text": "Security audit scheduled for May 15. We need: 1) SOC2 Type II report 2) Updated DPA 3) Penetration test results from last 12 months", "channel": "jira"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-audit-may", "payload": {"text": "Hi Priya, I've started gathering these documents. SOC2 Type II and pen test results are ready. The updated DPA is with legal — ETA April 28. I'll send everything as a package once complete.", "agent": "Taylor"}},
            {"source": "user", "type": "support_message", "session_id": "ticket-audit-may", "payload": {"text": "Thank you. Please ensure the DPA reflects our EU data residency requirements specifically.", "channel": "jira"}},

            # Failover ticket - open (engineering working on fix)
            {"source": "user", "type": "support_message", "session_id": "ticket-failover-mar", "payload": {"text": "Our failover test from us-east-1 to eu-west-1 took 45 seconds. Our SLA with clients requires <30s. This is a blocker for our Q2 rollout.", "channel": "jira"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-failover-mar", "payload": {"text": "Hi Priya, I've escalated this to our infrastructure team. Initial analysis shows the delay is in DNS propagation. Engineering is working on a fix with ETA April 30.", "agent": "Taylor"}},
            {"source": "user", "type": "support_message", "session_id": "ticket-failover-mar", "payload": {"text": "Please keep me updated weekly. This is critical for our healthcare clients.", "channel": "jira"}},
            {"source": "agent", "type": "support_response", "session_id": "ticket-failover-mar", "payload": {"text": "Understood. I've set up weekly status updates every Monday. Next update: April 7.", "agent": "Taylor"}},
        ],
        "resolutions": [
            {"session_id": "ticket-residency-jan", "status": "resolved", "resolution_summary": "Confirmed EU data residency, provided documentation and signed letter for compliance audit"},
            {"session_id": "ticket-audit-may", "status": "open", "resolution_summary": "Gathering SOC2 Type II, DPA, and pen test docs for May 15 audit - DPA ETA April 28"},
            {"session_id": "ticket-failover-mar", "status": "open", "resolution_summary": "Failover taking 45s vs required 30s - engineering fix ETA April 30"},
        ],
    },
}


async def bootstrap_persona(client: httpx.AsyncClient, persona_id: str, data: dict):
    """Bootstrap a single persona snapshot."""
    episodes = data["episodes"]
    resolutions = data.get("resolutions", [])

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

    # Create resolutions for SLA tracking
    if resolutions:
        print(f"  → Creating {len(resolutions)} resolution records...")
        for res in resolutions:
            resp = await client.post(
                f"{STATEWAVE_URL}/v1/resolutions",
                json={"subject_id": temp_subject, **res},
            )
            if resp.status_code not in (200, 201):
                print(f"    ERROR creating resolution: {resp.status_code} {resp.text}")

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

    result = resp.json()
    print(f"  ✓ Snapshot '{persona_id}' v{TEMPLATE_VERSION} created: {result.get('episode_count', '?')} episodes, {result.get('memory_count', '?')} memories")

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

        for persona_id, persona_data in PERSONAS.items():
            print(f"[{persona_id}]")
            await bootstrap_persona(client, persona_id, persona_data)
            print()

    print("Done! All snapshots bootstrapped.")


if __name__ == "__main__":
    asyncio.run(main())
