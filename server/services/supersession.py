"""Recording the decision the resolver made about a memory.

Both compile-time producers of supersession — the deterministic resolver
(`server.services.conflicts`) and the LLM reconcile pass
(`server.services.reconcile`) — describe what they decided through
:class:`SupersessionDecision` and stage it here. Recording only one of them
would be worse than recording neither: reconcile is on by default and retires
more memories than the resolver does, so an admin view reading a
conflicts-only table would answer "no successor" where it used to answer with
a guess.

Not every decision retires something. Since #414 the lexical pass also records
the pairs it deliberately did NOT supersede, under
:data:`RULE_WIDENING_SKIPPED`. A skip row uses the same two id columns as a
supersession — `superseded_memory_id` is the memory that would have been
retired, `superseding_memory_id` the one that would have retired it — so a
reader that does not filter on :data:`SUPERSEDING_RULES` will report an ACTIVE
memory as superseded. The rule column is what tells them apart.

What a decision may carry
-------------------------
Ids, a rule name, a claim KEY, and the numbers behind a scored rule. NOT the
memory content, NOT the claim values, NOT the reconcile model's rationale
string — the rationale in particular is generated FROM the memories and would
smuggle their text in. A reader joins `memories` for anything renderable, so
deleting a subject's memories still takes its text with it (#419 decision;
#423 is open on subject text outliving deletion and must not be widened).
`details` is under the same rule: shape and counts, never the tokens
themselves, which are memory text by another name.

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
# The lexical pass declined to supersede: the newer statement's content words
# and numbers are a strict SUBSET of the older one's, so it drops a condition
# without adding anything and the two coexist (#414). Recorded because "how
# often does the guard fire, and on which memories" has to be answerable by
# query rather than by grepping logs.
RULE_WIDENING_SKIPPED = "widening_skipped"

# Rules that mean a memory WAS retired. A non-supersession decision is recorded
# in the same table and on the same two id columns, so every reader that asks
# "what retired this / what did this retire" must filter on this set or it will
# report an active memory as superseded. Membership is the contract; new rules
# opt in here explicitly.
SUPERSEDING_RULES = frozenset(
    {
        RULE_CLAIM_CONTRADICTION,
        RULE_CLAIM_DUPLICATE,
        RULE_LEXICAL,
        RULE_RECONCILE_UPDATE,
        RULE_RECONCILE_DELETE,
    }
)


@dataclass(frozen=True)
class SupersessionDecision:
    """One "this memory retires that one, because" decision — or, for a rule
    outside :data:`SUPERSEDING_RULES`, one "this memory did not retire that
    one, because".

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
    # Rule-specific shape, counts and flags only — never tokens, values or
    # content (see the module docstring). None stores an empty object, so a
    # reader never has to distinguish "no details" from "null".
    details: dict | None = None

    def supersedes(self) -> bool:
        """Whether this decision retired ``superseded_memory_id``."""
        return self.rule in SUPERSEDING_RULES


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
                details=d.details or {},
            )
            for d in decisions
        ]
    )
