"""Docs-grounded support agent eval — measures retrieval/ranking quality
against the canonical statewave-support-docs subject.

For each canonical question we ask three things of Statewave's
`/v1/context` response:

  1. Doc match — did at least one expected doc appear in the resolved
     citation set (top-K source episodes by doc_path)?
  2. Term recall — what fraction of expected substantive terms appears
     anywhere in the retrieved context (facts + procedures)?
  3. Groundability — were at least 2 retrieved facts substantive
     enough (contain at least one expected term) for the LLM to
     ground a real answer?

Plus one aggregate signal:

  * Citation diversity — how many unique doc paths are cited across
    all queries combined? Low diversity is the bug shape we observed:
    the same 4 docs appearing across very different questions.

Designed to run before and after a server-side ranking change so we
can compare scores objectively. Mocks nothing — hits a live
Statewave instance.

Usage:
    STATEWAVE_URL=https://statewave-api.fly.dev \\
    STATEWAVE_API_KEY=... \\
    python eval_docs_support.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_URL = "https://statewave-api.fly.dev"
SUBJECT_ID = "statewave-support-docs"
TOP_K_CITATIONS = 4  # matches resolveDocSources cap in the widget


# ─── Canonical question set ──────────────────────────────────────────
#
# For each question we list:
#   * expected_doc_paths — at least ONE of these should be cited
#   * expected_terms     — substantive vocabulary the right answer relies
#                          on. Stopwords, verbs of asking, and the word
#                          "Statewave" itself are excluded — those would
#                          inflate term recall trivially.
#
# These are not exhaustive — they're the minimum signal we'd expect a
# competent docs-grounded retriever to surface. The eval is a floor,
# not a ceiling.

@dataclass
class Question:
    task: str
    expected_doc_paths: list[str]
    expected_terms: list[str]


QUESTIONS: list[Question] = [
    Question(
        task="What database does Statewave require?",
        expected_doc_paths=["architecture/overview.md", "product.md", "why-statewave.md"],
        expected_terms=["postgres", "pgvector"],
    ),
    Question(
        task="How do I deploy Statewave on Fly.io?",
        expected_doc_paths=["deployment/guide.md"],
        expected_terms=["fly", "deploy"],
    ),
    Question(
        task="What data leaves the box during compilation?",
        expected_doc_paths=[
            "architecture/privacy-and-data-flow.md",
            "architecture/compiler-modes.md",
        ],
        expected_terms=["compil", "llm", "local"],
    ),
    Question(
        task="Heuristic vs LLM compilation — when to pick which?",
        expected_doc_paths=["architecture/compiler-modes.md"],
        expected_terms=["heuristic", "llm"],
    ),
    Question(
        task="Do I need a GPU for Statewave?",
        expected_doc_paths=["deployment/hardware-and-scaling.md"],
        expected_terms=["gpu", "hardware"],
    ),
    Question(
        task="How do I back up and restore a subject?",
        expected_doc_paths=["dev/backup-restore.md"],
        expected_terms=["backup", "restore"],
    ),
    Question(
        task="How does context ranking work?",
        expected_doc_paths=["architecture/ranking.md"],
        expected_terms=["ranking", "score"],
    ),
    Question(
        task="Is Statewave self-hosted only?",
        expected_doc_paths=["product.md", "why-statewave.md", "README.md", "deployment/guide.md"],
        expected_terms=["self-hosted", "self-host"],
    ),
]


# ─── HTTP helpers ────────────────────────────────────────────────────


def _api_url() -> str:
    return os.environ.get("STATEWAVE_URL", DEFAULT_URL).rstrip("/")


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    key = os.environ.get("STATEWAVE_API_KEY", "")
    if key:
        h["X-API-Key"] = key
    return h


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{_api_url()}{path}",
        data=json.dumps(body).encode(),
        headers=_headers(),
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=90).read())


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{_api_url()}{path}", headers=_headers())
    return json.loads(urllib.request.urlopen(req, timeout=90).read())


# ─── Retrieval ────────────────────────────────────────────────────────


def fetch_episodes_by_id() -> dict[str, dict]:
    """One-time admin fetch of all docs episodes — used to resolve
    source_episode_ids to doc_path. Mirrors fetchAllEpisodesAdmin in
    the widget. limit=200 is the server's accepted max."""
    out: dict[str, dict] = {}
    for offset in (0, 200):
        page = _get(
            f"/admin/subjects/{SUBJECT_ID}/episodes?limit=200&offset={offset}"
        )
        eps = page.get("episodes", []) if isinstance(page, dict) else page
        if not eps:
            break
        for ep in eps:
            out[ep["id"]] = ep
        if len(eps) < 200:
            break
    return out


def resolve_citations(context: dict, ep_by_id: dict[str, dict]) -> list[str]:
    """Mirror the widget's resolveDocSources: walk memories.source_episode_ids
    in rank order, dedup by doc_path, cap at TOP_K_CITATIONS."""
    seen: set[str] = set()
    sources: list[str] = []
    memories = list(context.get("facts", [])) + list(context.get("procedures", []))
    for mem in memories:
        if len(sources) >= TOP_K_CITATIONS:
            break
        for sid in mem.get("source_episode_ids", []) or []:
            ep = ep_by_id.get(sid)
            if not ep:
                continue
            doc_path = (ep.get("provenance") or {}).get("doc_path")
            if doc_path and doc_path not in seen:
                seen.add(doc_path)
                sources.append(doc_path)
                if len(sources) >= TOP_K_CITATIONS:
                    break
    return sources


# ─── Scoring ──────────────────────────────────────────────────────────


@dataclass
class QuestionResult:
    task: str
    citations: list[str]
    doc_match: bool
    expected_doc_hits: list[str]
    term_recall: float  # 0.0–1.0
    matched_terms: list[str]
    groundable: bool
    fact_count: int


@dataclass
class EvalSummary:
    target: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def doc_match_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.doc_match) / len(self.results)

    @property
    def avg_term_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.term_recall for r in self.results) / len(self.results)

    @property
    def groundable_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.groundable) / len(self.results)

    @property
    def citation_diversity(self) -> int:
        all_docs: set[str] = set()
        for r in self.results:
            all_docs.update(r.citations)
        return len(all_docs)


def score_question(q: Question, context: dict, ep_by_id: dict[str, dict]) -> QuestionResult:
    citations = resolve_citations(context, ep_by_id)

    # Doc match: at least one expected doc cited
    expected_set = set(q.expected_doc_paths)
    expected_hits = [d for d in citations if d in expected_set]
    doc_match = len(expected_hits) > 0

    # Term recall: fraction of expected terms in retrieved context
    facts = context.get("facts", []) + context.get("procedures", [])
    blob = " ".join(m.get("content", "").lower() for m in facts)
    matched_terms = [t for t in q.expected_terms if t.lower() in blob]
    term_recall = len(matched_terms) / max(1, len(q.expected_terms))

    # Groundable: at least 2 facts each containing some expected term
    grounded_fact_count = sum(
        1
        for m in facts
        if any(t.lower() in m.get("content", "").lower() for t in q.expected_terms)
    )
    groundable = grounded_fact_count >= 2

    return QuestionResult(
        task=q.task,
        citations=citations,
        doc_match=doc_match,
        expected_doc_hits=expected_hits,
        term_recall=term_recall,
        matched_terms=matched_terms,
        groundable=groundable,
        fact_count=len(facts),
    )


# ─── Runner ───────────────────────────────────────────────────────────


def run() -> EvalSummary:
    target = _api_url()
    print(f"Eval target: {target}")
    print(f"Subject:     {SUBJECT_ID}")
    print(f"Questions:   {len(QUESTIONS)}")
    print()

    print("Loading episode→doc_path map (one-time admin fetch)...")
    ep_by_id = fetch_episodes_by_id()
    print(f"  Loaded {len(ep_by_id)} episodes")
    print()

    summary = EvalSummary(target=target)
    for q in QUESTIONS:
        ctx = _post(
            "/v1/context",
            {"subject_id": SUBJECT_ID, "task": q.task, "max_tokens": 600},
        )
        result = score_question(q, ctx, ep_by_id)
        summary.results.append(result)

        ok = "✓" if result.doc_match else "✗"
        gnd = "✓" if result.groundable else "✗"
        print(f"  {ok} doc_match  {gnd} groundable  recall={result.term_recall:.2f}  {q.task[:64]}")
        if result.expected_doc_hits:
            print(f"        cited expected: {result.expected_doc_hits}")
        if not result.doc_match:
            print(f"        cited: {result.citations}  expected: {q.expected_doc_paths}")
        if result.matched_terms:
            print(f"        terms hit: {result.matched_terms}")
        else:
            print(f"        terms missed: {q.expected_terms}")

    print()
    print("─" * 70)
    print(f"  doc_match_rate     {summary.doc_match_rate:.0%}  ({sum(1 for r in summary.results if r.doc_match)}/{len(summary.results)})")
    print(f"  avg_term_recall    {summary.avg_term_recall:.2f}")
    print(f"  groundable_rate    {summary.groundable_rate:.0%}  ({sum(1 for r in summary.results if r.groundable)}/{len(summary.results)})")
    print(f"  citation_diversity {summary.citation_diversity} unique docs across all queries")
    print("─" * 70)

    return summary


def emit_json(summary: EvalSummary, path: str) -> None:
    """Dump a JSON snapshot for before/after diffing."""
    data: dict[str, Any] = {
        "target": summary.target,
        "doc_match_rate": summary.doc_match_rate,
        "avg_term_recall": summary.avg_term_recall,
        "groundable_rate": summary.groundable_rate,
        "citation_diversity": summary.citation_diversity,
        "questions": [
            {
                "task": r.task,
                "doc_match": r.doc_match,
                "groundable": r.groundable,
                "term_recall": r.term_recall,
                "citations": r.citations,
                "matched_terms": r.matched_terms,
                "fact_count": r.fact_count,
            }
            for r in summary.results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSnapshot written to {path}")


if __name__ == "__main__":
    try:
        summary = run()
    except urllib.error.HTTPError as e:
        print(f"\nERROR: {e.code} {e.reason}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1].startswith("--out="):
        emit_json(summary, sys.argv[1].split("=", 1)[1])
