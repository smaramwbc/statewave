"""Recording the decision that retires a memory.

Both compile-time producers of supersession — the deterministic resolver
(`server.services.conflicts`) and the LLM reconcile pass
(`server.services.reconcile`) — describe what they decided through
:class:`SupersessionDecision` and stage it here. Recording only one of them
would be worse than recording neither: reconcile is on by default and retires
more memories than the resolver does, so an admin view reading a
conflicts-only table would answer "no successor" where it used to answer with
a guess.

What a decision may carry
-------------------------
Ids, a rule name, a claim KEY, and the numbers behind a scored rule. NOT the
memory content, NOT the claim values, NOT the reconcile model's rationale
string — the rationale in particular is generated FROM the memories and would
smuggle their text in. A reader joins `memories` for anything renderable, so
deleting a subject's memories still takes its text with it (#419 decision;
#423 is open on subject text outliving deletion and must not be widened).

These rows are not receipts. A receipt attests to bytes handed to a caller; a
supersession record is a decision about stored state that nobody was served.
Nothing here is chained, signed, or hashed into a receipt body.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from server.db.tables import SupersessionRecordRow

# Rule names are the producers' OWN strategy strings — the ones
# `conflicts.py` already puts in its `memory_superseded` log line — so a
# recorded rule and a logged strategy are the same vocabulary.
RULE_CLAIM_CONTRADICTION = "claim_contradiction"
RULE_CLAIM_DUPLICATE = "claim_duplicate"
RULE_LEXICAL = "lexical"
# Reconcile logs an action rather than a strategy; its two retiring actions
# are recorded under the same convention.
RULE_RECONCILE_UPDATE = "reconcile_update"
RULE_RECONCILE_DELETE = "reconcile_delete"


@dataclass(frozen=True)
class SupersessionDecision:
    """One "this memory retires that one, because" decision.

    ``superseding_memory_id`` is None when the decision named a successor
    that was never persisted — reconcile can accept a candidate that
    supersedes a committed memory and then drop that candidate in a later
    chunk. The loser is superseded either way, so the honest record is the
    decision with no successor rather than no record at all (which would send
    the admin view back to guessing).
    """

    superseded_memory_id: uuid.UUID
    superseding_memory_id: uuid.UUID | None
    rule: str
    claim_key: str | None = None
    score: float | None = None
    threshold: float | None = None


def record_supersessions(
    session,
    subject_id: str,
    decisions: list[SupersessionDecision],
    *,
    tenant_id: str | None = None,
    compile_job_id: str | None = None,
) -> None:
    """Stage one row per decision on the caller's session.

    Staged, not flushed or committed: the rows ride the compile batch's own
    transaction, so a batch that rolls back records nothing and a batch that
    commits records exactly the supersessions it performed.
    """
    if not decisions:
        return
    session.add_all(
        [
            SupersessionRecordRow(
                subject_id=subject_id,
                tenant_id=tenant_id,
                superseded_memory_id=d.superseded_memory_id,
                superseding_memory_id=d.superseding_memory_id,
                rule=d.rule,
                claim_key=d.claim_key,
                score=d.score,
                threshold=d.threshold,
                compile_job_id=compile_job_id,
                details={},
            )
            for d in decisions
        ]
    )
