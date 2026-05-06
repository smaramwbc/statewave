"""New-episode content for the 5 demo persona packs.

Authoring notes:
  * Every episode references something concrete: a named contact, a date,
    a ticket number, a stack detail, a SHA, a URL, a metric, or a decision
    record. No generic filler.
  * Each persona's new episodes deepen the established narrative — the
    same protagonist, same company / project / paper, same stack. They do
    not introduce new identities.
  * Episode types cover the mix the persona realistically experiences in
    the underlying tooling (zendesk for support, github for coding,
    HubSpot/email for sales, datadog/fly/pagerduty for devops, zotero
    /notion for research).
  * Ordering within each list is roughly chronological, latest last.
"""

from __future__ import annotations

from typing import Any

# Triple shape: (source, type, payload-dict). The payload's `content` field is
# the narrative text the LLM compiler reads; other keys carry structured
# metadata that the inspector renders.
Episode = tuple[str, str, dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────
# demo-support-agent — Maya Hassan @ Northwind Logistics
# Established: EU-Paris, Team plan, dispatch-automation, async-only stack
# (FastAPI + asyncpg + mypy --strict), webhook pattern
# hooks.northwind-logistics.fr/statewave/<event-type>, EU on-call routing rule.
# Wow moments: feature rollout (async + webhook + EU on-call), perf re-occurrence
# (#4937 root cause), renewal status, login-slow triage.
# ──────────────────────────────────────────────────────────────────────────

DEMO_SUPPORT_AGENT: list[Episode] = [
    ("zendesk", "ticket_opened", {
        "channel": "email",
        "ticket_id": "#4823",
        "priority": "normal",
        "content": "Maya asked why the dispatch-automation subject was rate-limited at "
                   "5 req/s when their Team plan documents 50 req/s. She suspects the "
                   "limit is per-IP rather than per-API-key.",
    }),
    ("zendesk", "agent_action", {
        "ticket_id": "#4823",
        "action": "clarify",
        "content": "Confirmed: rate limits are per-IP sliding-window, not per-key. "
                   "Northwind's edge proxy collapses to a single egress IP, so all "
                   "drivers' shipment writes share the 50 r/s budget. Recommended "
                   "either spreading egress across 4 IPs or batching via "
                   "/v1/episodes/batch (200 per call).",
    }),
    ("zendesk", "ticket_closed", {
        "ticket_id": "#4823",
        "resolution": "answered",
        "content": "Maya picked the batch path. Routing copilot ingestion now uses "
                   "/v1/episodes/batch at 4-wide parallelism, exactly matching the "
                   "post-#4937 backfill recipe.",
    }),
    ("zendesk", "ticket_opened", {
        "channel": "email",
        "ticket_id": "#4915",
        "priority": "normal",
        "content": "Northwind's procurement asked whether Statewave signs a "
                   "BAA-lite for non-PHI personal data under GDPR. They process "
                   "driver pickup-window data which is identifiable but not health.",
    }),
    ("zendesk", "agent_action", {
        "ticket_id": "#4915",
        "action": "legal_handoff",
        "content": "Routed to legal@statewave.ai with a copy of Northwind's DPA "
                   "request. Internal note: BAA proper is HIPAA-only; for GDPR "
                   "Article 28 processor coverage, the existing DPA + SCCs are "
                   "sufficient. Legal to send the standard processor addendum.",
    }),
    ("zendesk", "ticket_closed", {
        "ticket_id": "#4915",
        "resolution": "answered",
        "content": "Standard processor addendum signed by Maya's procurement on "
                   "2026-04-29. No BAA needed for Northwind's data scope.",
    }),
    ("crm", "feature_request", {
        "request_id": "FR-2026-117",
        "content": "Maya followed up on her 2026-04-20 request for per-event-type "
                   "webhook filtering. Use case: only memory.compiled and "
                   "subject.health_degraded should fan out to dispatch ops; the rest "
                   "are noise. Promised an ETA at next QBR.",
    }),
    ("crm", "feature_request", {
        "request_id": "FR-2026-128",
        "content": "Maya requested per-subject snapshot export so they can ship a "
                   "support-debug bundle for a single driver subject without "
                   "exporting the entire Northwind tenant. Sized at 'S' on the "
                   "feature board; tentative for v0.8.",
    }),
    ("zendesk", "ticket_opened", {
        "channel": "in-app-chat",
        "ticket_id": "#4954",
        "priority": "normal",
        "content": "Maya asked whether webhook delivery retry can be tuned per "
                   "event-type. Their dispatch-ops board can drop transient "
                   "memory.compiled events but must never drop "
                   "subject.health_degraded.",
    }),
    ("zendesk", "agent_action", {
        "ticket_id": "#4954",
        "action": "answer",
        "content": "Retry policy is currently global: 5 attempts, exponential "
                   "backoff capped at 5 minutes, dead-letter after that. Per-type "
                   "override is on the roadmap; recommended Maya add it to their "
                   "FR-2026-117 feature-request thread so the two land together.",
    }),
    ("zendesk", "ticket_closed", {
        "ticket_id": "#4954",
        "resolution": "answered",
        "content": "Maya merged the request into FR-2026-117 with a comment noting "
                   "the retry-policy dependency.",
    }),
    ("crm", "renewal_kickoff", {
        "content": "Renewal kickoff call held 2026-05-02 with Maya + Account "
                   "Manager Priya M. Maya confirmed the dock-ops expansion (15 new "
                   "seats projected) is still planned for Q3, contingent on the "
                   "snapshot-export feature shipping. Quote due to Maya by 2026-06-15.",
    }),
    ("crm", "expansion_signal", {
        "content": "Maya mentioned Northwind is evaluating Statewave as the memory "
                   "layer for their cross-border customs-clearance team in 2027 — a "
                   "second use case beyond dispatch-automation. Marked as "
                   "qualifying-stage opportunity in HubSpot, est. value $40k ARR.",
    }),
    ("github", "issue_opened", {
        "issue": "northwind-logistics/dispatch-routing#212",
        "content": "Northwind engineer Théo opened an issue against their own "
                   "internal repo asking whether replacing in-process LRU with the "
                   "Statewave context cache improves p95 on the routing copilot. "
                   "Maya tagged us for a benchmark methodology suggestion.",
    }),
    ("crm", "internal_note", {
        "author": "Priya M. (AM)",
        "content": "Maya promoted to Director of Dispatch Automation as of "
                   "2026-04-25. New direct: Théo Bernard (senior eng). Account "
                   "decision-maker for renewal/expansion is now Maya + her CTO "
                   "Henri Laurent. Henri prefers async written updates over calls.",
    }),
    ("zendesk", "ticket_opened", {
        "channel": "email",
        "ticket_id": "#5012",
        "priority": "high",
        "content": "Northwind reported elevated latency again on the routing-fleet "
                   "subject — p95 climbed from 280ms to 1.4s over the last 6 hours. "
                   "Maya asked for triage before paging.",
    }),
    ("zendesk", "agent_action", {
        "ticket_id": "#5012",
        "action": "investigation",
        "content": "Triage applied the post-#4937 lesson: checked per-subject memory "
                   "count first. Routing-fleet subject hit 58k memories (within "
                   "5k of the 60k threshold from #4937). Recommended splitting along "
                   "the same axis as #4937 (per-region routing-fleet-{region}) before "
                   "ingest volume crosses the threshold again.",
    }),
    ("zendesk", "ticket_closed", {
        "ticket_id": "#5012",
        "resolution": "fixed",
        "content": "Northwind shipped the per-region split on 2026-05-04. p95 back "
                   "to 290ms within 30 minutes of the migration. No paging needed "
                   "because the per-subject memory-count check caught it pre-cliff.",
    }),
    ("crm", "qbr_prep", {
        "content": "QBR scheduled 2026-05-12 with Maya + Henri. Agenda: (1) renewal "
                   "+ dock-ops expansion quote, (2) FR-2026-117 + FR-2026-128 ETAs, "
                   "(3) the customs-clearance qualifying opp, (4) recap of #4937, "
                   "#5012 — same root cause, faster catch.",
    }),
    ("zendesk", "ticket_opened", {
        "channel": "email",
        "ticket_id": "#5031",
        "priority": "low",
        "content": "Maya asked whether Statewave's snapshot format is stable across "
                   "minor versions for their compliance retention (3-year shelf "
                   "life requirement under EU transport regs).",
    }),
    ("zendesk", "agent_action", {
        "ticket_id": "#5031",
        "action": "answer",
        "content": "Snapshots use the statewave-memory-payload v1 envelope with "
                   "version-aware loaders — backward compatibility within format_version "
                   "1 is contractual. Cross-major-version restore (v1 → v2) will need "
                   "a one-time migration tool, ETA in roadmap as 'snapshot upgrader'.",
    }),
    ("crm", "internal_note", {
        "author": "Customer Success (Lina)",
        "content": "Northwind health score: 92/100 (up from 87 last month). Drivers: "
                   "two clean ticket resolutions in May, no escalations, expansion in "
                   "the pipeline. Risk markers: dependency on FR-2026-117 for the "
                   "Q3 dock-ops expansion timing.",
    }),
]


# ──────────────────────────────────────────────────────────────────────────
# demo-coding-assistant — Priya @ Stratus
# Established: TS strict, FastAPI/SQLModel/Pydantic v2, ADR-007 pricing service
# on Postgres+SQLModel, Dashboards V2 behind FLAG_DASHBOARD_V2, auth refactor
# (collapse JWT + session-pool acquisition), test policy: never mock the DB.
# Wow moments: PR #389 fix shape, ADR-007 stack, 150-line diff convention,
# `any` → no.
# ──────────────────────────────────────────────────────────────────────────

DEMO_CODING_ASSISTANT: list[Episode] = [
    ("github", "pr_merged", {
        "repo": "stratus/api",
        "pr_number": 412,
        "sha": "8c1f2e9",
        "content": "PR #412 merged: feat(auth): collapse jwt + session-pool acquisition "
                   "into get_current_user dependency. Closes the auth refactor in "
                   "flight since late April. Files removed: 4 (jwt_utils.py, "
                   "session_pool.py, current_user_old.py, auth_legacy_helpers.py). "
                   "Net diff: -340 / +118.",
    }),
    ("github", "pr_review", {
        "repo": "stratus/api",
        "pr_number": 412,
        "reviewer": "Priya",
        "content": "Priya approved with one nit: rename `_resolve_user` to "
                   "`_resolve_authenticated_user` for clarity. Verified test policy: "
                   "the new dependency is exercised against a real Postgres in "
                   "test_auth_dep.py.",
    }),
    ("github", "issue_opened", {
        "repo": "stratus/api",
        "issue": 415,
        "content": "Issue #415: post-#412 cleanup — JWT_SECRET env var still listed "
                   "in 3 places (.env.example, docker-compose.yml, fly.toml). After "
                   "the refactor only fly.toml needs it. Priya assigned to herself.",
    }),
    ("github", "pr_opened", {
        "repo": "stratus/api",
        "pr_number": 418,
        "sha": "1e9a3b4",
        "content": "PR #418: chore(env): drop unused JWT_SECRET references. "
                   "Resolves #415. 12-line diff.",
    }),
    ("github", "pr_merged", {
        "repo": "stratus/api",
        "pr_number": 418,
        "content": "Merged after one review round. Closes #415.",
    }),
    ("notion", "doc_published", {
        "doc": "ADR-008: Rate-limit middleware",
        "content": "ADR-008 finalised by Priya: chose slowapi over a custom Redis-token-bucket "
                   "for Stratus' API rate limiting. Reasons: slowapi already integrates "
                   "with FastAPI dependencies (matches the auth pattern post-#412), "
                   "Redis is already in stack for queueing, and the limit-per-API-key "
                   "scoping matches Stratus' tenant model.",
    }),
    ("github", "pr_opened", {
        "repo": "stratus/api",
        "pr_number": 423,
        "sha": "4c2d8f1",
        "content": "PR #423: feat(api): add slowapi-based rate limiting per ADR-008. "
                   "100 r/m default, 1000 r/m for paid tenants, 10 r/m for "
                   "unauthenticated requests. Conventional-commit prefix: feat.",
    }),
    ("github", "pr_review_requested", {
        "repo": "stratus/api",
        "pr_number": 423,
        "reviewer": "Priya",
        "content": "Priya flagged: PR is 187 lines — over the 150-line convention. "
                   "Suggested splitting into (1) middleware + tests, (2) per-tenant "
                   "config, (3) dashboard V2 surfacing. Author agreed to stack.",
    }),
    ("github", "pr_merged", {
        "repo": "stratus/api",
        "pr_number": 425,
        "content": "PR #425 (1/3 of the rate-limit stack): middleware + integration "
                   "tests. 92 lines, merged with one approval.",
    }),
    ("github", "pr_merged", {
        "repo": "stratus/api",
        "pr_number": 426,
        "content": "PR #426 (2/3): per-tenant config schema + admin endpoint. "
                   "78 lines. Test coverage hits the new config table via "
                   "real-Postgres fixtures (no DB mocking).",
    }),
    ("github", "pr_merged", {
        "repo": "stratus/web",
        "pr_number": 211,
        "content": "PR #426 (3/3): Dashboards V2 surfaces rate-limit usage card. "
                   "Behind FLAG_DASHBOARD_V2 — disabled in prod still, enabled in "
                   "staging since 2026-04-22.",
    }),
    ("launchdarkly", "flag_change", {
        "flag": "FLAG_DASHBOARD_V2",
        "content": "Priya flipped FLAG_DASHBOARD_V2 to 5% of paid tenants in prod "
                   "on 2026-05-03. Targeting rule: tenant_id in the early-access "
                   "list (5 tenants pre-agreed for the rollout). Plan: 5% → 25% "
                   "→ 100% over 14 days.",
    }),
    ("github", "issue_opened", {
        "repo": "stratus/web",
        "issue": 219,
        "content": "Issue #219: pricing-service endpoint /v1/pricing/quote returns "
                   "500 intermittently for ~1 in 200 requests. Stack trace points "
                   "to the SQLModel session — Priya suspects another session-pool "
                   "race like PR #389.",
    }),
    ("github", "pr_opened", {
        "repo": "stratus/api",
        "pr_number": 432,
        "sha": "5e0c8a2",
        "content": "PR #432: fix(pricing): wrap session-pool acquisition with the "
                   "same lock pattern from PR #389. Adds a regression test against "
                   "real Postgres that hammers the endpoint at 50 r/s for 30s.",
    }),
    ("github", "pr_merged", {
        "repo": "stratus/api",
        "pr_number": 432,
        "content": "Merged 2026-05-04. /v1/pricing/quote 500s dropped from 0.5% "
                   "to 0% over 24h after deploy. Closes #219.",
    }),
    ("slack", "channel_msg", {
        "channel": "#engineering",
        "content": "Priya posted: 'Reminder: tests/conftest.py now starts a real "
                   "Postgres-15 container per session via testcontainers. Don't "
                   "switch back to MagicMock — it cost us PR #389 and #432.' "
                   "Thread had 4 thumbs-up reactions.",
    }),
    ("github", "pr_opened", {
        "repo": "stratus/web",
        "pr_number": 218,
        "content": "PR #218: chore(types): move TenantConfig + RateLimitTier from "
                   "app packages into @stratus/types. New consumers: SDK and "
                   "frontend. Diff: 64 lines.",
    }),
    ("github", "pr_merged", {
        "repo": "stratus/web",
        "pr_number": 218,
        "content": "Merged. Frontend + SDK both pick up the shared types via "
                   "the next pnpm install.",
    }),
    ("calendar", "meeting", {
        "title": "Pair-programming with new junior (Olu)",
        "content": "Priya scheduled 90-minute sessions every Thursday with Olu "
                   "(joined 2026-05-01) to walk through Stratus' codebase. Week 1 "
                   "covered the auth dependency post-#412 and the test-with-real-DB "
                   "policy; Olu's first PR is queued behind ADR-008's stack landing.",
    }),
    ("github", "pr_opened", {
        "repo": "stratus/web",
        "pr_number": 224,
        "author": "Olu",
        "content": "Olu opened first PR: docs(adr): index ADR-001 through ADR-008 "
                   "in /docs/adr/README.md with a one-line summary each. 38 lines.",
    }),
    ("github", "pr_merged", {
        "repo": "stratus/web",
        "pr_number": 224,
        "content": "Priya approved with: 'Welcome aboard, Olu. Conventional-commits "
                   "format ✓, under 150 lines ✓, no console.log ✓ — clean first PR.'",
    }),
    ("launchdarkly", "flag_change", {
        "flag": "FLAG_DASHBOARD_V2",
        "content": "FLAG_DASHBOARD_V2 advanced from 5% to 25% on 2026-05-05 — no "
                   "regressions in the 5% cohort over 48h. Sentry error rate flat. "
                   "Next gate: 100% on 2026-05-12 if 25% stays clean for 7 days.",
    }),
]


# ──────────────────────────────────────────────────────────────────────────
# demo-sales-copilot — Tom (4-account pipeline)
# Established: Acme (closed expansion 5-seat 2026-04-12, Q3 renewal, Sarah Chen
# decision-maker), BetaTech (in negotiation, David Kim email-only, SOC 2 sent),
# Cirrus (churned end-Jan, Mem0 lost-on-determinism note), Delta Health (HIPAA
# prospect, Dr. Janelle Martinez), HubSpot-only, AE routing (Marco/Priya M.).
# Wow moments: meeting prefs (Sarah Chen), pipeline ARR, BetaTech state, Mem0
# positioning.
# ──────────────────────────────────────────────────────────────────────────

DEMO_SALES_COPILOT: list[Episode] = [
    ("hubspot", "deal_stage_change", {
        "deal": "BetaTech 2026-Q2",
        "stage_from": "negotiation",
        "stage_to": "security-review",
        "content": "BetaTech moved to security-review on 2026-04-29. David Kim "
                   "kicked off the questionnaire — 47 items, mostly SOC 2 cross-mapped. "
                   "Tom assigned to backfill the legacy questions; legal owns the BAA-lite "
                   "negotiation (BetaTech is healthcare-adjacent, asked for HIPAA-style "
                   "language even though they're not a covered entity).",
    }),
    ("email", "outbound", {
        "to": "david.kim@betatech.io",
        "subject": "BetaTech security questionnaire — items 31-47",
        "content": "Tom sent answers to the remaining 17 items. Highlight: question 38 "
                   "(data residency) confirmed Statewave runs in customer-chosen Fly "
                   "regions; BetaTech selected EU-Paris to match their existing "
                   "infrastructure.",
    }),
    ("email", "inbound", {
        "from": "david.kim@betatech.io",
        "subject": "Re: BetaTech security questionnaire — items 31-47",
        "content": "David acknowledged receipt. Asked one follow-up: confirm "
                   "encryption at rest in Postgres. Tom replied within 2h with the "
                   "fly volume encryption + LUKS-on-host details.",
    }),
    ("hubspot", "deal_note", {
        "deal": "BetaTech 2026-Q2",
        "content": "BetaTech procurement is comparing pricing against Pinecone "
                   "(per their 2026-05-01 procurement call). Pinecone quoted "
                   "$2,200/month for the equivalent volume. Our quote is $3,800/month "
                   "but includes the LLM compiler — Pinecone is vector-only.",
    }),
    ("hubspot", "deal_note", {
        "deal": "BetaTech 2026-Q2",
        "content": "Pricing concession: offered BetaTech 12-month annual at $42k "
                   "(15% discount vs the $48k monthly run-rate). David said he'd "
                   "take it back to legal for BAA-lite + the price together.",
    }),
    ("hubspot", "deal_stage_change", {
        "deal": "BetaTech 2026-Q2",
        "stage_from": "security-review",
        "stage_to": "contract",
        "content": "BetaTech moved to contract stage on 2026-05-04. Security "
                   "questionnaire complete. Counter-signed BAA-lite + redlined MSA "
                   "in flight. Expected close: 2026-05-10.",
    }),
    ("hubspot", "contact_update", {
        "contact": "Sarah Chen (Acme)",
        "content": "Sarah confirmed 2026-05-02: BluePeak engineering integration "
                   "is now in pilot. 10-seat opportunity firming for Q4 2026. Tom "
                   "added a calendar reminder for 2026-08-01 to start the expansion "
                   "conversation 60 days ahead of the integration go-live.",
    }),
    ("calendar", "meeting", {
        "title": "Acme expansion - BluePeak integration scope",
        "with": "Sarah Chen",
        "when": "2026-05-12T17:00:00Z (10am PT)",
        "content": "30-minute call held 2026-05-12 per Sarah's standing preference "
                   "(Tuesday 10am PT, never Mondays). Sarah confirmed BluePeak's "
                   "engineering org is 12 people; expansion sized at 10 seats; "
                   "decision in 60 days.",
    }),
    ("hubspot", "deal_create", {
        "deal": "Acme expansion 2026-Q4",
        "amount": 48000,
        "content": "Created Acme expansion 2026-Q4 deal: $48k incremental ARR "
                   "(10 seats × $4,800/seat-yr). Stage: qualifying. Owner: Marco. "
                   "Linked to parent Acme renewal Q3 2026.",
    }),
    ("hubspot", "deal_note", {
        "deal": "Delta Health 2026-Q3",
        "content": "Dr. Janelle Martinez (CMIO at Delta Health) requested a deeper "
                   "PoC scoping. Triage-assistant memory across patient encounters; "
                   "120-bed hospital; targeting one specialty (cardiology) for the "
                   "PoC first. Tom and AE Priya M. drafted PoC scope: 6 weeks, "
                   "fixed-fee $15k.",
    }),
    ("email", "outbound", {
        "to": "j.martinez@deltahealth.org",
        "subject": "Delta Health PoC scope - cardiology pilot",
        "content": "Tom sent the PoC scope document. 6 weeks: weeks 1-2 BAA + "
                   "data-flow review, weeks 3-4 ingest pipeline + memory schema, "
                   "weeks 5-6 evaluation against current triage workflow. Success "
                   "criteria: 80%+ memory-recall accuracy on a held-out patient set.",
    }),
    ("email", "inbound", {
        "from": "j.martinez@deltahealth.org",
        "subject": "Re: Delta Health PoC scope",
        "content": "Janelle approved PoC scope, asked to add IRB review to weeks 1-2. "
                   "Tom flagged the IRB requirement to legal — small adjustment, "
                   "no impact on timeline.",
    }),
    ("hubspot", "deal_stage_change", {
        "deal": "Delta Health 2026-Q3",
        "stage_from": "discovery",
        "stage_to": "PoC",
        "content": "Delta Health PoC kicks off 2026-06-01. PoC fee: $15k (booked). "
                   "Conversion target: full $90k/yr Enterprise plan if PoC succeeds.",
    }),
    ("hubspot", "deal_note", {
        "deal": "Cirrus Robotics (churned)",
        "content": "Re-engagement attempt: Liam Park (former CTO, now at a "
                   "different robotics startup, Helio Mobility) reached out 2026-05-03 "
                   "asking if Statewave's per-fleet metering had landed. Tom logged "
                   "the inquiry as a new opportunity (Helio Mobility 2026-Q4) and "
                   "noted the Cirrus-loss lesson: lead with deterministic ranking "
                   "+ provenance, NOT pricing.",
    }),
    ("hubspot", "deal_create", {
        "deal": "Helio Mobility 2026-Q4",
        "amount": 0,
        "stage": "qualifying",
        "content": "Created Helio Mobility 2026-Q4 deal. Decision-maker: Liam Park "
                   "(CTO). Use case: per-driver memory across a 200-driver fleet. "
                   "Owner: Marco (continuity from the Cirrus account). Estimated "
                   "ARR if qualifying clears: $32k.",
    }),
    ("calendar", "meeting", {
        "title": "Helio Mobility discovery call",
        "with": "Liam Park",
        "when": "2026-05-08T18:00:00Z",
        "content": "30-minute call. Liam asked directly about per-fleet vs per-subject "
                   "metering. Tom confirmed metering is per-subject (drivers map "
                   "1:1 to subjects); for fleet-wide questions a parent fleet "
                   "subject is the recommended pattern. Liam to internally "
                   "compare against Mem0 again before next call.",
    }),
    ("slack", "internal_dm", {
        "to": "Marco (AE)",
        "content": "Tom DM'd Marco: 'Reminder for Helio call — Liam burned us at "
                   "Cirrus on Mem0 pricing. Lead with provenance + deterministic "
                   "ranking, NOT volume discount. Have the determinism deck ready.'",
    }),
    ("hubspot", "pipeline_review", {
        "content": "May 2026 pipeline review with VP Sales: $94k Q2 ARR confirmed "
                   "($24k Acme expansion already closed + $48k BetaTech in "
                   "contract + $22k Cirrus lost). Forward pipeline (Q3): $63k "
                   "(Acme renewal $30k + Delta Health PoC→Enterprise $90k weighted "
                   "at 30% = $27k + Helio qualifying $32k weighted 20% = $6k).",
    }),
    ("hubspot", "deal_note", {
        "deal": "Acme renewal Q3 2026",
        "content": "Sarah Chen confirmed renewal terms 2026-05-06: keep Team plan, "
                   "annual billing, no price change. Auto-renewal clause stays. "
                   "Renewal contract sent same day via DocuSign; expected counter-sig "
                   "within 5 business days.",
    }),
    ("hubspot", "deal_stage_change", {
        "deal": "Acme renewal Q3 2026",
        "stage_from": "renewal-prep",
        "stage_to": "contract-out",
        "content": "Acme renewal contract out for counter-sig. Owner: Marco. "
                   "Q3 ARR confirmed at $30k (no churn risk).",
    }),
    ("email", "internal", {
        "to": "AE handoff distribution",
        "subject": "Q2 close pack",
        "content": "Tom emailed the AE team with the Q2 close pack: Acme expansion "
                   "(closed → CS handoff to Lina), BetaTech (in contract → CS "
                   "handoff prep), Delta Health PoC (kicks off June). Each section "
                   "addressed to the correct AE per the routing rule (Marco: Acme + "
                   "Cirrus + Helio; Priya M.: BetaTech + Delta Health).",
    }),
    ("hubspot", "deal_won", {
        "deal": "BetaTech 2026-Q2",
        "amount": 42000,
        "content": "BetaTech closed 2026-05-10 at $42k annual (15% discount applied). "
                   "Counter-signed by David Kim's procurement. Handoff to CS team "
                   "Lina Ortega 2026-05-11.",
    }),
]


# ──────────────────────────────────────────────────────────────────────────
# demo-devops-agent — Riya, SRE on nimbus-api
# Established: Fly iad+lhr, March outage (pool exhaustion → PgBouncer mitigation),
# 6-min replica failover drill, secrets rotation 90-day cadence, Datadog,
# alert thresholds (post-2026-04-08 noisy-alerts post-mortem), TLS rotation
# (post-2026-04-15 incident), 99.95% SLA, no-Friday-afternoon deploys.
# Wow moments: alert thresholds verbatim, rollback command, deploy window,
# replica promotion runbook.
# ──────────────────────────────────────────────────────────────────────────

DEMO_DEVOPS_AGENT: list[Episode] = [
    ("crm", "team_update", {
        "content": "SRE on-call lead Riya Patel (PT, primary US rotation) is the "
                   "operator for nimbus-api as of 2026-04-15. Riya took over from "
                   "the previous lead after the TLS-incident post-mortem; reports to "
                   "Director of Infrastructure Marcus Chen.",
    }),
    ("pagerduty", "incident", {
        "incident_id": "P-7142",
        "severity": "sev-3",
        "content": "Page 2026-04-19 03:42 UTC: error rate >2% for 5m on nimbus-api in iad. "
                   "Riya (primary) took the page within 2 minutes. Root cause: a "
                   "downstream LLM provider rate-limited 503ed; our backoff retry "
                   "amplified. Mitigation: tightened in-flight cap from 8 to 4 "
                   "concurrent compile requests (matches the v0.6 default).",
    }),
    ("notion", "postmortem", {
        "doc": "Post-mortem P-7142",
        "content": "5-page post-mortem published 2026-04-21. Action items: (1) cap "
                   "LLM in-flight requests at 4 by default in fly.toml (done), "
                   "(2) add a Datadog monitor for upstream-503-rate (done, deployed "
                   "2026-04-22), (3) document the LLM-503 amplification pattern in "
                   "the runbook (done, runbook v3.7).",
    }),
    ("datadog", "monitor_create", {
        "monitor": "nimbus-api: upstream LLM 503 rate >5% for 3m",
        "content": "New Datadog monitor created 2026-04-22: pages secondary on-call "
                   "if upstream LLM 503 rate exceeds 5% for 3 minutes. Threshold "
                   "tuned to fire on real provider degradation but not on "
                   "single-request transient errors.",
    }),
    ("github", "pr_merged", {
        "repo": "nimbus/api",
        "pr_number": 312,
        "sha": "0b9e2a1",
        "content": "PR #312 merged 2026-04-22: chore(fly): cap llm_in_flight to 4. "
                   "Riya approved. Deployment used the standard rolling strategy "
                   "from the deploy runbook (no Friday afternoon, not in 02:00-04:00 "
                   "UTC Wed maintenance).",
    }),
    ("calendar", "meeting", {
        "title": "Quarterly DR drill",
        "when": "2026-05-05T17:00:00Z",
        "content": "Quarterly DR drill: Postgres replica promotion + region "
                   "failover lhr → iad. Riya led. Total time 7 minutes (target "
                   "<10). Drill notes added to runbook v3.8.",
    }),
    ("notion", "runbook_update", {
        "doc": "Runbook v3.8 — Postgres replica promotion",
        "content": "Updated 2026-05-05 after the DR drill. New step: confirm "
                   "PgBouncer pool stats are quiescent before flipping the "
                   "replication target (added because the drill showed a 90s spike "
                   "of stale pooled connections post-promotion). Drill duration "
                   "updated to 7 minutes (target <10).",
    }),
    ("fly", "deploy", {
        "app": "nimbus-api",
        "image_label": "deployment-01HW9X7",
        "content": "Deploy 2026-04-23 14:12 UTC by Riya: fly deploy --remote-only. "
                   "Health check passed; p95 returned to 220ms within 90s. Rolling "
                   "strategy: 2/2 machines updated lease-by-lease. No downtime.",
    }),
    ("fly", "secret_rotation", {
        "content": "Riya rotated STATEWAVE_API_KEY on 2026-05-01 — the 90-day "
                   "cadence triggered. Stage → verify staging /readyz → prod deploy "
                   "in three steps. Old key revoked at 2026-05-01 18:00 UTC after "
                   "all consumers confirmed adoption (4 consumers: nimbus-web, "
                   "nimbus-admin, nimbus-billing, nimbus-eval).",
    }),
    ("fly", "secret_rotation", {
        "content": "DATABASE_URL credentials rotated 2026-05-01 alongside the API "
                   "key (same 90-day cadence, batched). Replica connection string "
                   "updated last to avoid a stale pool window.",
    }),
    ("pagerduty", "incident", {
        "incident_id": "P-7218",
        "severity": "sev-4",
        "content": "Page 2026-04-30 11:08 UTC: disk >85% on nimbus-pg primary for "
                   "10m. Riya checked: WAL growth from a long-running ETL caused the "
                   "spike. Mitigation: vacuum + WAL archive flush. Volume increased "
                   "from 200GB to 250GB the next day to add headroom.",
    }),
    ("notion", "postmortem", {
        "doc": "Post-mortem P-7218",
        "content": "1-page postmortem (sev-4 lighter format). Action items: (1) "
                   "add a 7-day disk-growth-rate monitor (done), (2) make the WAL "
                   "retention 14 days instead of 30 to reduce baseline (deferred — "
                   "needs compliance sign-off for the 30-day audit window).",
    }),
    ("calendar", "meeting", {
        "title": "Capacity planning: scale lhr 1→2 machines",
        "when": "2026-05-06T16:00:00Z",
        "content": "EU traffic up 40% MoM since BetaTech onboarding. Riya proposed "
                   "scaling lhr from 1 to 2 nimbus-api machines pre-emptively. "
                   "Approved by Marcus (Director of Infrastructure). Deploy "
                   "scheduled for 2026-05-08 (Thursday, within the deploy window).",
    }),
    ("fly", "scale", {
        "app": "nimbus-api",
        "region": "lhr",
        "content": "Scaled nimbus-api lhr from 1→2 machines on 2026-05-08 14:30 UTC. "
                   "Verified: p95 in lhr dropped from 410ms (under load) to 290ms "
                   "within an hour. iad unchanged at 220ms.",
    }),
    ("notion", "runbook_update", {
        "doc": "Runbook v3.9 — Cache stampede mitigation",
        "content": "New runbook section added 2026-05-09 after a near-miss on "
                   "2026-05-07 where a cold-start in lhr coincided with a "
                   "campaign-driven traffic spike. Mitigation: pre-warmed cache "
                   "via a startup task that issues 5 representative /v1/context "
                   "queries before health-check returns ready.",
    }),
    ("github", "pr_merged", {
        "repo": "nimbus/api",
        "pr_number": 327,
        "sha": "f3c1d8e",
        "content": "PR #327 merged 2026-05-09: feat(startup): pre-warm context "
                   "cache before /readyz returns 200. Reduces cold-start tail "
                   "latency from 1.2s to 280ms. Riya approved.",
    }),
    ("datadog", "dashboard_update", {
        "content": "Datadog dashboard 'nimbus-api SLO' updated 2026-05-04 to "
                   "reduce anomaly-detection sensitivity from 'medium' to 'low' on "
                   "the request-rate widget — was firing on legitimate weekend "
                   "traffic dips. p95/p99 widgets unchanged.",
    }),
    ("pagerduty", "rotation_handoff", {
        "content": "Weekly rotation handoff 2026-05-04 09:00 UTC: primary US Riya → "
                   "secondary EU Theo. No active incidents. One pending: monitor "
                   "the lhr scaling behavior post-2026-05-08 deploy. Theo to verify "
                   "p95 staying under 300ms in lhr through the week.",
    }),
    ("github", "pr_opened", {
        "repo": "nimbus/api",
        "pr_number": 334,
        "content": "PR #334: feat(deploy): add `fly deploy --strategy bluegreen` "
                   "option in the runbook for zero-touch deploys during business "
                   "hours. Riya: 'Optional — keep rolling as default; bluegreen "
                   "for high-risk migrations only.'",
    }),
    ("notion", "runbook_update", {
        "doc": "Runbook v3.10 — Bluegreen deploy",
        "content": "New section: when to use bluegreen vs rolling. Rolling = "
                   "default (auth changes, schema-additive deploys). Bluegreen = "
                   "high-risk (anything touching the LLM-compile pipeline or "
                   "PgBouncer pool config). Tied to PR #334 landing.",
    }),
    ("datadog", "alert_tune", {
        "content": "Tuned the disk-growth-rate alert added post-P-7218: changed "
                   "from 7-day rate to a 24-hour delta with hourly evaluation. "
                   "Catches WAL spikes faster while avoiding the false-positive "
                   "from baseline weekly growth.",
    }),
    ("calendar", "meeting", {
        "title": "Monthly SRE review with Marcus",
        "when": "2026-05-15T16:00:00Z",
        "content": "Monthly review: SLA tracking 99.99% (one 4-min TLS incident). "
                   "Two post-mortems published this period (P-7142, P-7218). Three "
                   "runbook updates (v3.7 → v3.10). Marcus approved the cache-stampede "
                   "preemptive work.",
    }),
]


# ──────────────────────────────────────────────────────────────────────────
# demo-research-assistant — Arushi, NeurIPS 2026 submission with Mei Wu
# Established: APA 7th, paper "Scoring Memories with Provenance" (intro + related
# work done, sec 3 in progress), Zotero+Obsidian+pandoc, mid-May section 3
# milestone, full draft to Mei by 2026-07-01, abstract due 2026-08-15.
# Wow moments: APA citation format for preprints, lit review status, deadline
# recall, compiler-density experiment recall.
# ──────────────────────────────────────────────────────────────────────────

DEMO_RESEARCH_ASSISTANT: list[Episode] = [
    ("zotero", "import", {
        "content": "Imported Liu et al. 2025 'Subject-Design Patterns for "
                   "Multi-Tenant Agent Memory' (DOI 10.48550/arXiv.2509.13104). "
                   "Tagged: subject-design, multi-tenant. Reading queue position 1.",
    }),
    ("obsidian", "note", {
        "doc": "reading/liu-2025-subject-design.md",
        "content": "1-line takeaway from Liu 2025: subject-per-user with optional "
                   "session metadata is empirically the best default for "
                   "multi-tenant agent memory; subject-per-session devolves to "
                   "fragmentation in long-running deployments.",
    }),
    ("zotero", "import", {
        "content": "Imported Voigt 2026 'Provenance-Bound Retrieval Revisited' "
                   "(arXiv 2603.04219). Tagged: retrieval, provenance. Reading "
                   "queue position 2.",
    }),
    ("obsidian", "note", {
        "doc": "reading/voigt-2026-provenance-revisited.md",
        "content": "1-line takeaway from Voigt 2026: their provenance-bound "
                   "retrieval is a re-derivation of last year's ICLR paper "
                   "('Provenance-Bound Retrieval', the Arushi+Mei one) — cite "
                   "as concurrent work in section 2.",
    }),
    ("notion", "draft_progress", {
        "doc": "neurips-2026-section-3",
        "content": "Section 3 (methodology) drafted to ~75% on 2026-05-04. "
                   "Subsections done: 3.1 ranking-formula derivation, 3.2 "
                   "deterministic-bundle theorem statement (proof in appendix). "
                   "Pending: 3.3 implementation reference + 3.4 evaluation "
                   "harness setup.",
    }),
    ("notion", "draft_progress", {
        "doc": "neurips-2026-section-3",
        "content": "Section 3 hit 100% on 2026-05-09 — under the mid-May milestone "
                   "by 6 days. 3.3 reference implementation links Statewave's "
                   "context.py; 3.4 harness is the eval framework Mei built for the "
                   "ICLR paper, extended.",
    }),
    ("zoom", "meeting", {
        "with": "Mei Wu",
        "when": "2026-05-10T17:00:00Z (10am PT)",
        "content": "60-minute review with Mei. Mei approved 3.1 + 3.2; flagged 3.3 "
                   "as too implementation-specific (recommended generalising the "
                   "language so reviewers don't read it as a Statewave ad). 3.4 "
                   "approved with one note: add 95% CI on the headline numbers.",
    }),
    ("notion", "draft_revision", {
        "doc": "neurips-2026-section-3",
        "content": "Applied Mei's feedback 2026-05-12: rewrote 3.3 in the abstract — "
                   "'an open-source typed-memory implementation' instead of "
                   "'Statewave'. Statewave appears in 3.4 only as the eval target. "
                   "Added bootstrap CI to all headline numbers in 3.4.",
    }),
    ("zotero", "import", {
        "content": "Imported Park & Singh 2026 (NeurIPS preprint, arXiv 2604.11827) "
                   "'Memory-Stream Scoring at Scale'. Tagged: memory-stream, "
                   "scoring, generative-agents. Reading queue position 3.",
    }),
    ("obsidian", "note", {
        "doc": "reading/park-singh-2026-scaling-memory-stream.md",
        "content": "1-line takeaway from Park & Singh 2026: memory-stream scoring "
                   "(Park 2024) extended to 1M-memory subjects via approximate "
                   "k-NN; their LLM-grading retrieval-time call costs 4x ours. "
                   "Cite in section 2 (related work) as an alternative scaling path "
                   "we explicitly chose against.",
    }),
    ("notion", "experiment_log", {
        "doc": "experiments/retrieval-budget-sensitivity",
        "content": "Experiment 'retrieval-budget-sensitivity' designed 2026-05-06: "
                   "vary max_tokens 200/800/2000/4000 across the 200-episode "
                   "synthetic dataset. Hypothesis: deterministic ranking should "
                   "produce monotonic non-decreasing accuracy with budget; non-"
                   "deterministic baselines should oscillate. Run scheduled for "
                   "2026-05-13.",
    }),
    ("notion", "experiment_log", {
        "doc": "experiments/retrieval-budget-sensitivity",
        "content": "Run completed 2026-05-13. Result: hypothesis confirmed. "
                   "Statewave deterministic ranking: 67% → 79% → 86% → 89% accuracy "
                   "across budgets 200/800/2000/4000. Naive RAG baseline: 51% → "
                   "73% → 71% → 79% (non-monotonic at the 2000 budget — confirms "
                   "the determinism failure mode). Going into figure 4 of the "
                   "paper.",
    }),
    ("calendar", "deadline_reminder", {
        "content": "Reminder set 2026-05-13 for 2026-08-15 NeurIPS abstract "
                   "deadline (94 days out) and 2026-08-22 full paper (101 days). "
                   "Internal milestone: full draft to Mei by 2026-07-01 (49 days).",
    }),
    ("email", "outbound", {
        "to": "mei.wu@berkeley.edu",
        "subject": "Section 3 + retrieval-budget figure",
        "content": "Sent Mei the revised section 3 plus figure 4 (retrieval-budget "
                   "sensitivity result). Asked her to review by 2026-05-20 so I can "
                   "merge feedback before starting section 4 (experimental design).",
    }),
    ("email", "inbound", {
        "from": "mei.wu@berkeley.edu",
        "subject": "Re: Section 3 + retrieval-budget figure",
        "content": "Mei replied 2026-05-15 with detailed comments. Section 3 now "
                   "looks publication-ready. Figure 4: she suggested adding error "
                   "bars on the naive-RAG baseline to make the determinism gap "
                   "more visually obvious. Arushi accepted; figure regen on "
                   "2026-05-16.",
    }),
    ("calendar", "decline", {
        "content": "Arushi declined a co-author invitation from a former MIT lab "
                   "member on 2026-05-08 — unrelated paper on tool-use agents, "
                   "would have pulled focus from NeurIPS. Polite decline citing "
                   "the 2026-08-22 deadline.",
    }),
    ("notion", "draft_progress", {
        "doc": "neurips-2026-section-4",
        "content": "Section 4 (experiments) outlined 2026-05-17. Three subsections: "
                   "4.1 synthetic-dataset benchmark (compiler density + retrieval "
                   "budget — both already run), 4.2 real-task evaluation "
                   "(multi-turn support-agent recall), 4.3 ablation (kind priority "
                   "off, recency off, semantic off).",
    }),
    ("conference", "iclr_poster_prep", {
        "content": "ICLR 2026 poster ('Provenance-Bound Retrieval', accepted "
                   "2026-03-26) due to printer 2026-06-15. Arushi started layout "
                   "in Inkscape 2026-05-14. Mei's lab supplies the foam-board.",
    }),
    ("email", "outbound", {
        "to": "iclr-poster@conf.iclr.cc",
        "subject": "ICLR 2026 poster slot — Arushi",
        "content": "Confirmed poster session 4 (Wednesday afternoon). Arushi to "
                   "fly into Vienna 2026-05-12 morning, present 2026-05-13. Mei "
                   "joining remotely via the conference Discord.",
    }),
    ("zotero", "import", {
        "content": "Imported Chen et al. 2026 'Eval Harnesses for Multi-Session "
                   "Agent Memory' (arXiv 2605.02338). Tagged: evaluation, "
                   "multi-session. Reading queue position 4.",
    }),
    ("obsidian", "note", {
        "doc": "reading/chen-2026-eval-harness.md",
        "content": "1-line takeaway from Chen 2026: their eval harness is the "
                   "first published one for multi-session agent recall — directly "
                   "relevant to our section 4.2. Will adopt their dataset format "
                   "for our real-task experiments + cite generously.",
    }),
    ("calendar", "focus_block", {
        "content": "Reading-block schedule held: every weekday 09:00-10:30 PT, "
                   "no meetings, no Slack. Dropped one slot on 2026-05-13 for the "
                   "experiment run; re-added 2026-05-14. Email and Slack still "
                   "closed during blocks per habit.",
    }),
    ("notion", "draft_progress", {
        "doc": "neurips-2026-section-4",
        "content": "Section 4.1 first draft complete 2026-05-19. Builds directly "
                   "on the compiler-density (2026-02-25) and retrieval-budget "
                   "(2026-05-13) experiments. Both figures + tables in place; "
                   "narrative wraps 2 days ahead of internal pace target.",
    }),
]


# ──────────────────────────────────────────────────────────────────────────
# Public registry — used by build_demo_packs.py
# ──────────────────────────────────────────────────────────────────────────

NEW_EPISODES: dict[str, list[Episode]] = {
    "demo-support-agent": DEMO_SUPPORT_AGENT,
    "demo-coding-assistant": DEMO_CODING_ASSISTANT,
    "demo-sales-copilot": DEMO_SALES_COPILOT,
    "demo-devops-agent": DEMO_DEVOPS_AGENT,
    "demo-research-assistant": DEMO_RESEARCH_ASSISTANT,
}


if __name__ == "__main__":
    # Quick sanity print: counts per persona.
    for persona, eps in NEW_EPISODES.items():
        print(f"{persona}: {len(eps)} new episodes")
