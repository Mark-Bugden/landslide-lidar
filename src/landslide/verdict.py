"""Test statuses and the replication verdict (D25; BRIEF.md "Replication criteria" and Appendix A §1).

Every rule here is recorded in `docs/decisions/P1-13.md`. In short:

- A test is **PASS** or **FAIL** only when it is complete: the point estimate decides (estimate >= threshold passes;
  equality passes). A stage-2 result that never reaches the recall level is a FAIL.
- A test with any missing, failed, interrupted or otherwise unusable run is **INCOMPLETE**, which is never a pass and
  never a fail.
- Stage 2 has a fourth state, **NOT_RUN**: no stage-2 unit or CNN detection exists at all (the runner skips stage 2
  unless (a) and (b) both pass).
- The verdict, with the precedence of Appendix A §1:
  1. any completed test below its threshold (FAIL) -> NOT REPLICATED, even when other runs are missing, and so also
     when stage 2 was not run because (a) or (b) failed;
  2. otherwise, all three tests PASS -> REPLICATED;
  3. otherwise (something INCOMPLETE or NOT_RUN, nothing FAIL) -> INCOMPLETE.
- **Borderline** (reported, never changes the verdict): the CI straddles the threshold, i.e. it holds both passing
  and failing values: ci_low < threshold <= ci_high.

The verdict words come from the config (`replication.verdict_words`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PASS", "FAIL", "INCOMPLETE", "NOT_RUN", "TEST_STATUSES", "STAGE1_STATUSES", "TESTS", "DEFAULT_WORDS",
    "Verdict", "verdict_words", "passes", "is_borderline", "classify_test", "decide",
]

PASS = "PASS"
FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"
NOT_RUN = "NOT_RUN"
TEST_STATUSES = (PASS, FAIL, INCOMPLETE, NOT_RUN)
#: (a) and (b) are never "not run": the runner always trains them, so a missing unit makes them INCOMPLETE.
STAGE1_STATUSES = (PASS, FAIL, INCOMPLETE)
TESTS = ("a", "b", "stage2")

#: replicated, not replicated, incomplete (the order of `replication.verdict_words`).
DEFAULT_WORDS = ("REPLICATED", "NOT REPLICATED", "INCOMPLETE")

_LABELS = {"a": "(a)", "b": "(b)", "stage2": "stage 2"}


def verdict_words(cfg: Mapping[str, Any]) -> tuple[str, str, str]:
    """The three verdict words from `cfg["replication"]["verdict_words"]`, in the order replicated, not
    replicated, incomplete. Anything other than three distinct words raises."""
    words = tuple(str(w) for w in cfg["replication"]["verdict_words"])
    if len(words) != 3 or len(set(words)) != 3:
        raise ValueError(f"replication.verdict_words must hold three distinct words, got {words}")
    return words  # type: ignore[return-value]


def passes(estimate: float, threshold: float) -> bool:
    """The pre-registered rule: the point estimate passes if it is >= the threshold."""
    e = float(estimate)
    if not math.isfinite(e):
        raise ValueError(f"estimate {estimate!r} is not finite")
    return e >= float(threshold)


def is_borderline(ci_low: float | None, ci_high: float | None, threshold: float) -> bool | None:
    """True if the CI holds both passing and failing values (ci_low < threshold <= ci_high); None without a CI."""
    if ci_low is None or ci_high is None:
        return None
    lo, hi = float(ci_low), float(ci_high)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo < float(threshold) <= hi


def classify_test(*, complete: bool, estimate: float | None, threshold: float, reached: bool = True) -> str:
    """PASS / FAIL / INCOMPLETE of one test.

    - not `complete` -> INCOMPLETE, whatever the estimate (a partial result never passes or fails);
    - complete but `reached` False (stage 2: the recall level is never reached) -> FAIL;
    - otherwise PASS if `estimate` >= `threshold`, else FAIL. A complete, reached test needs a finite estimate.
    """
    if not complete:
        return INCOMPLETE
    if not reached:
        return FAIL
    if estimate is None:
        raise ValueError("a complete test with the level reached needs an estimate")
    return PASS if passes(estimate, threshold) else FAIL


@dataclass(frozen=True)
class Verdict:
    word: str
    statuses: dict[str, str]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.word, **self.statuses, "reasons": list(self.reasons)}


def decide(a: str, b: str, stage2: str, words: Sequence[str] = DEFAULT_WORDS) -> Verdict:
    """The replication verdict from the three test statuses (D25, Appendix A §1 precedence).

    `a` and `b` are PASS, FAIL or INCOMPLETE; `stage2` is PASS, FAIL, INCOMPLETE or NOT_RUN.
    """
    replicated, not_replicated, incomplete = (str(w) for w in words)
    statuses = {"a": a, "b": b, "stage2": stage2}
    for t in ("a", "b"):
        if statuses[t] not in STAGE1_STATUSES:
            raise ValueError(f"test {t}: status {statuses[t]!r} not in {STAGE1_STATUSES}")
    if stage2 not in TEST_STATUSES:
        raise ValueError(f"stage 2: status {stage2!r} not in {TEST_STATUSES}")

    failed = [t for t in TESTS if statuses[t] == FAIL]
    open_ = [t for t in TESTS if statuses[t] in (INCOMPLETE, NOT_RUN)]
    reasons: list[str] = []
    for t in failed:
        reasons.append(f"{_LABELS[t]} is complete and below its threshold (FAIL)")
    for t in open_:
        what = "was not run" if statuses[t] == NOT_RUN else "has missing or unusable runs (INCOMPLETE)"
        reasons.append(f"{_LABELS[t]} {what}")

    if failed:
        word = not_replicated
        if open_:
            reasons.append("a completed test below its threshold decides, even with other tests open "
                           "(Appendix A §1)")
        if stage2 == NOT_RUN and (a == FAIL or b == FAIL):
            reasons.append("stage 2 runs only if (a) and (b) both pass")
    elif not open_:
        word = replicated
        reasons.append("(a), (b) and stage 2 all pass on their point estimates")
    else:
        word = incomplete
        reasons.append("INCOMPLETE is never a pass")
    return Verdict(word=word, statuses=statuses, reasons=reasons)
