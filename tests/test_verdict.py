"""The replication verdict (D25; BRIEF.md "Replication criteria"; Appendix A §1): the full truth table and the rules
for one test's status and the borderline flag."""

import itertools
import math

import pytest

from landslide import verdict as V
from landslide.config import load_config

R, NR, INC = V.DEFAULT_WORDS
P, F, I, N = V.PASS, V.FAIL, V.INCOMPLETE, V.NOT_RUN


def expected_word(a: str, b: str, s2: str) -> str:
    """The brief's rules, written out independently of `decide`."""
    if a == F or b == F:
        # "If (a) or (b) fails, the verdict is FAIL", also when stage 2 was therefore not run, and "if a completed
        # test is below its threshold, the verdict is FAIL even when other runs are missing".
        return NR
    if a == P and b == P:
        # "If both pass and stage 2 fails, the verdict is FAIL. PASS requires all three." Stage 2 missing or not
        # run although (a) and (b) passed: a missing run, so INCOMPLETE.
        return {P: R, F: NR, I: INC, N: INC}[s2]
    # (a) or (b) INCOMPLETE and neither failed: a completed stage 2 below threshold still decides; otherwise
    # "a test with any missing run is INCOMPLETE, which is never a pass".
    return NR if s2 == F else INC


ALL = list(itertools.product(V.STAGE1_STATUSES, V.STAGE1_STATUSES, V.TEST_STATUSES))


def test_truth_table_has_every_combination():
    assert len(ALL) == 3 * 3 * 4 == 36
    assert [c for c in ALL if V.decide(*c).word == R] == [(P, P, P)]


@pytest.mark.parametrize(("a", "b", "s2"), ALL)
def test_truth_table(a, b, s2):
    v = V.decide(a, b, s2)
    assert v.word == expected_word(a, b, s2)
    assert v.statuses == {"a": a, "b": b, "stage2": s2}
    assert v.reasons


@pytest.mark.parametrize(("a", "b", "s2", "word"), [
    (P, P, P, R),     # all three pass
    (P, P, F, NR),    # both pass, stage 2 fails (incl. "50 % recall never reached")
    (F, P, N, NR),    # stage 2 not run because (a) failed
    (P, F, N, NR),    # stage 2 not run because (b) failed
    (F, F, N, NR),
    (I, F, N, NR),    # a completed test below threshold decides even with runs missing
    (F, I, N, NR),
    (I, P, N, INC),   # (a) missing a run: never a pass
    (P, I, N, INC),
    (P, P, N, INC),   # (a) and (b) pass but stage 2 never ran (e.g. the wall-clock cap)
    (P, P, I, INC),   # a stage-2 unit missing
    (I, I, N, INC),
])
def test_brief_sentences(a, b, s2, word):
    assert V.decide(a, b, s2).word == word


def test_not_run_reason_mentions_the_rule():
    reasons = " ".join(V.decide(F, P, N).reasons)
    assert "stage 2 runs only if (a) and (b) both pass" in reasons


def test_decide_rejects_unknown_statuses():
    with pytest.raises(ValueError):
        V.decide(N, P, P)  # (a) and (b) are never "not run"
    with pytest.raises(ValueError):
        V.decide(P, P, "BORDERLINE")


def test_verdict_words_come_from_config():
    assert V.verdict_words(load_config()) == V.DEFAULT_WORDS
    words = ("YES", "NO", "MAYBE")
    assert V.decide(P, P, P, words).word == "YES"
    assert V.decide(F, P, N, words).word == "NO"
    assert V.decide(I, P, N, words).word == "MAYBE"
    with pytest.raises(ValueError):
        V.verdict_words({"replication": {"verdict_words": ["A", "A", "B"]}})


def test_classify_test():
    # incomplete overrides any estimate
    assert V.classify_test(complete=False, estimate=0.99, threshold=0.75) == I
    assert V.classify_test(complete=False, estimate=None, threshold=0.75) == I
    # the point estimate decides; equality passes
    assert V.classify_test(complete=True, estimate=0.75, threshold=0.75) == P
    assert V.classify_test(complete=True, estimate=math.nextafter(0.75, 0), threshold=0.75) == F
    # stage 2 never reaching the recall level is a FAIL, never a pass
    assert V.classify_test(complete=True, estimate=None, threshold=0.30, reached=False) == F
    assert V.classify_test(complete=True, estimate=0.9, threshold=0.30, reached=False) == F
    with pytest.raises(ValueError):
        V.classify_test(complete=True, estimate=None, threshold=0.3)
    with pytest.raises(ValueError):
        V.classify_test(complete=True, estimate=float("nan"), threshold=0.3)


@pytest.mark.parametrize(("lo", "hi", "thr", "expected"), [
    (0.70, 0.80, 0.75, True),
    (0.76, 0.80, 0.75, False),   # all values pass
    (0.70, 0.74, 0.75, False),   # all values fail
    (0.70, 0.75, 0.75, True),    # the upper bound passes, the rest fails
    (0.75, 0.80, 0.75, False),   # the lower bound is a pass: every value passes
    (None, 0.8, 0.75, None),
    (float("nan"), 0.8, 0.75, None),
])
def test_borderline(lo, hi, thr, expected):
    assert V.is_borderline(lo, hi, thr) is expected
