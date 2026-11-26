import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from landslide import metrics as m
from landslide.config import load_config

# ------------------------------------------------------------------------------------------------ ROC AUC


@pytest.mark.parametrize(
    ("y", "s", "expected"),
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], 1.0),
        ([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1], 0.0),
        ([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], 0.5),
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.75),
        # ties: positives {3, 2}, negatives {2, 1, 2}: 3 beats all (3), 2 ties twice and beats 1 (2) -> 5/6
        ([1, 1, 0, 0, 0], [3, 2, 2, 1, 2], 5 / 6),
        # a tie between the only positive and the only negative
        ([1, 0], [7.0, 7.0], 0.5),
    ],
)
def test_roc_auc_known_values(y, s, expected):
    assert m.roc_auc(y, s) == pytest.approx(expected, abs=1e-15)
    assert m.roc_auc(np.asarray(y, dtype=bool), s) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("levels", [None, 3, 12])
@pytest.mark.parametrize("seed", range(5))
def test_roc_auc_matches_sklearn_and_midranks(levels, seed):
    rng = np.random.default_rng(seed)
    n = 500
    y = rng.random(n) < 0.4
    s = rng.normal(size=n) + 0.7 * y
    if levels is not None:  # heavy ties
        s = np.floor(s * levels / 3)
    auc = m.roc_auc(y, s)
    assert auc == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    r = rankdata(s, method="average")
    n1, n0 = y.sum(), (~y).sum()
    assert auc == pytest.approx((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), abs=1e-12)


def test_roc_auc_integer_weights_equal_duplication():
    rng = np.random.default_rng(7)
    y = rng.random(200) < 0.5
    s = np.round(rng.normal(size=200), 1)
    w = rng.integers(0, 4, size=200)
    idx = np.repeat(np.arange(200), w)
    assert m.roc_auc(y, s, weights=w) == m.roc_auc(y[idx], s[idx])
    assert m.roc_auc(y, s, weights=w) == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-12)


def test_roc_auc_rejects_bad_input():
    with pytest.raises(ValueError, match="both classes"):
        m.roc_auc([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="both classes"):
        m.roc_auc([1, 0], [0.1, 0.2], weights=[1, 0])
    with pytest.raises(ValueError, match="NaN"):
        m.roc_auc([1, 0], [0.1, np.nan])
    with pytest.raises(ValueError, match="0/1"):
        m.roc_auc([2, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="shape"):
        m.roc_auc([1, 0, 1], [0.1, 0.2])


# ------------------------------------------------------------------------------------------------ bootstrap


def _synthetic(n_sheets=40, per_sheet=20, seed=0):
    rng = np.random.default_rng(seed)
    sheet = np.repeat([f"{i:02d}-00-00" for i in range(n_sheets)], per_sheet)
    y = np.tile(np.arange(per_sheet) % 2 == 0, n_sheets)
    effect = rng.normal(0.8, 0.5, size=n_sheets).repeat(per_sheet)
    cnn = rng.normal(size=y.size) + effect * y
    base = np.round(rng.normal(size=y.size) + 0.3 * y, 1)  # coarse, many ties
    return y, sheet, cnn, base


def test_bootstrap_spec_reads_config():
    cfg = load_config()
    spec = m.bootstrap_spec(cfg)
    assert (spec.n_replicates, spec.ci, spec.seed, spec.method, spec.unit) == (2000, 0.95, 123, "percentile", "sheet")
    bad = {"eval": {"bootstrap": dict(cfg["eval"]["bootstrap"], method="bca")}}
    with pytest.raises(ValueError, match="percentile"):
        m.bootstrap_spec(bad)


def test_draws_follow_the_documented_recipe():
    cfg = load_config()
    y, sheet, *_ = _synthetic()
    d = m.draw_clusters_from_cfg(sheet, cfg)
    k = np.unique(sheet).size
    expected = np.random.default_rng(123).integers(0, k, size=(2000, k))
    np.testing.assert_array_equal(d.draw_idx, expected)
    assert d.n_replicates == 2000 and d.n_clusters == k and d.n_items == y.size
    np.testing.assert_array_equal(d.counts.sum(axis=1), np.full(2000, k))
    for b in (0, 1, 1999):
        np.testing.assert_array_equal(d.counts[b], np.bincount(d.draw_idx[b], minlength=k))
        assert np.bincount(d.replicate_items(b), minlength=y.size).tolist() == d.weights(b).tolist()


def test_bootstrap_is_deterministic():
    y, sheet, cnn, _ = _synthetic()
    r1 = m.bootstrap_auc(y, cnn, m.draw_clusters(sheet, 500, 123), ci=0.95)
    r2 = m.bootstrap_auc(y, cnn, m.draw_clusters(sheet, 500, 123), ci=0.95)
    assert r1 == r2
    np.testing.assert_array_equal(r1.replicates, r2.replicates)
    r3 = m.bootstrap_auc(y, cnn, m.draw_clusters(sheet, 500, 124), ci=0.95)
    assert r3.draws_key != r1.draws_key and (r3.ci_low, r3.ci_high) != (r1.ci_low, r1.ci_high)
    assert r1.estimate == m.roc_auc(y, cnn)
    assert r1.ci_low < r1.estimate < r1.ci_high
    assert (r1.n_replicates, r1.n_used, r1.n_degenerate) == (500, 500, 0)


def test_bootstrap_does_not_depend_on_row_order():
    y, sheet, cnn, _ = _synthetic()
    perm = np.random.default_rng(3).permutation(y.size)
    a = m.replicate_aucs(y, cnn, m.draw_clusters(sheet, 300, 123))
    b = m.replicate_aucs(y[perm], cnn[perm], m.draw_clusters(sheet[perm], 300, 123))
    np.testing.assert_array_equal(a, b)


def test_replicates_equal_explicit_resamples():
    y, sheet, cnn, base = _synthetic()
    d = m.draw_clusters(sheet, 200, 123)
    for s in (cnn, base):
        reps = m.replicate_aucs(y, s, d)
        for b in range(0, 200, 13):
            idx = d.replicate_items(b)
            assert reps[b] == m.roc_auc(y[idx], s[idx])  # exact: same integer arithmetic
            assert reps[b] == pytest.approx(roc_auc_score(y[idx], s[idx]), abs=1e-12)


def test_chunking_does_not_change_replicates(monkeypatch):
    y, sheet, cnn, base = _synthetic()
    d = m.draw_clusters(sheet, 101, 123)
    full = m.replicate_aucs(y, base, d)
    monkeypatch.setattr(m, "_CHUNK_CELLS", 3 * y.size)  # 3 replicates per chunk, last chunk partial
    np.testing.assert_array_equal(m.replicate_aucs(y, base, d), full)


def test_paired_differences_use_identical_draws():
    y, sheet, cnn, base = _synthetic()
    rnd = np.random.default_rng(9).random(y.size)
    d = m.draw_clusters(sheet, 400, 123)
    res = m.bootstrap_aucs(y, {"cnn": cnn, "slope": base, "random": rnd}, d, ci=0.95)
    diffs = m.paired_differences(res, reference="cnn")
    assert set(diffs) == {"slope", "random"}
    for name, s in (("slope", base), ("random", rnd)):
        r = diffs[name]
        assert r.estimate == res["cnn"].estimate - res[name].estimate
        np.testing.assert_array_equal(r.replicates, res["cnn"].replicates - res[name].replicates)
        assert (r.ci_low, r.ci_high) == m.percentile_ci(r.replicates, 0.95)
        for b in (0, 57, 399):  # each replicate difference is two AUCs on one explicit resample
            idx = d.replicate_items(b)
            assert r.replicates[b] == m.roc_auc(y[idx], cnn[idx]) - m.roc_auc(y[idx], s[idx])
    assert diffs["random"].ci_low > 0  # the CNN-like scorer beats random on every sensible resample


def test_paired_differences_refuse_different_draws():
    y, sheet, cnn, base = _synthetic()
    a = m.bootstrap_auc(y, cnn, m.draw_clusters(sheet, 100, 123), ci=0.95)
    b = m.bootstrap_auc(y, base, m.draw_clusters(sheet, 100, 124), ci=0.95)
    with pytest.raises(ValueError, match="different draws"):
        m.paired_differences({"cnn": a, "base": b}, reference="cnn")
    with pytest.raises(KeyError):
        m.paired_differences({"cnn": a}, reference="missing")


def test_degenerate_replicates_are_dropped_and_counted():
    # sheet A holds only positives, B and C only negatives: a replicate without A, or with A only, is one-class
    sheet = np.array(["A"] * 3 + ["B"] * 3 + ["C"] * 2)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    s1 = np.array([0.9, 0.5, 0.2, 0.6, 0.1, 0.3, 0.3, 0.8])
    s2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    d = m.draw_clusters(sheet, 2000, 123)
    a = int(np.flatnonzero(d.labels == "A")[0])
    expected = (d.counts[:, a] == 0) | (d.counts[:, a] == 3)
    assert 0 < expected.sum() < 2000
    res = m.bootstrap_aucs(y, {"s1": s1, "s2": s2}, d, ci=0.95)
    for r in res.values():
        np.testing.assert_array_equal(np.isnan(r.replicates), expected)
        assert r.n_degenerate == expected.sum()
        assert r.n_used == 2000 - expected.sum()
        assert (r.ci_low, r.ci_high) == m.percentile_ci(r.replicates[~expected], 0.95)
        assert np.isfinite(r.ci_low) and np.isfinite(r.ci_high)
    diff = m.paired_differences(res, reference="s1")["s2"]
    assert diff.n_degenerate == expected.sum() and diff.n_used == 2000 - expected.sum()


def test_degenerate_statistic_replicates_and_all_degenerate():
    d = m.draw_clusters(["A", "B", "C"], 50, 0)
    r = m.bootstrap_statistic(lambda w, c: 1.0 if (c == 1).all() else np.nan, d, ci=0.95)
    expected_used = int((d.counts == 1).all(axis=1).sum())  # only replicates that drew A, B and C once each
    assert 0 < expected_used < 50
    assert (r.n_used, r.n_degenerate) == (expected_used, 50 - expected_used)

    calls = []

    def finite_only_on_full_sample(w, c):
        calls.append(1)
        return 1.0 if len(calls) == 1 else np.nan

    r = m.bootstrap_statistic(finite_only_on_full_sample, d, ci=0.95)
    assert (r.estimate, r.n_used, r.n_degenerate) == (1.0, 0, 50)
    assert np.isnan(r.ci_low) and np.isnan(r.ci_high)
    with pytest.raises(ValueError, match="full sample"):
        m.bootstrap_statistic(lambda w, c: np.nan, d, ci=0.95)


def test_percentile_ci_uses_numpy_linear_quantiles():
    v = np.random.default_rng(1).normal(size=2000)
    lo, hi = m.percentile_ci(np.concatenate([v, [np.nan] * 5]), 0.95)
    assert (lo, hi) == tuple(np.quantile(v, [0.025, 0.975]))
    with pytest.raises(ValueError):
        m.percentile_ci(v, 1.5)


def test_bootstrap_statistic_matches_vectorised_auc():
    y, sheet, cnn, _ = _synthetic(n_sheets=15)
    d = m.draw_clusters(sheet, 150, 123)

    def auc(w, _counts):
        try:
            return m.roc_auc(y, cnn, weights=w)
        except ValueError:
            return np.nan

    r = m.bootstrap_statistic(auc, d, ci=0.95)
    ref = m.bootstrap_auc(y, cnn, d, ci=0.95)
    np.testing.assert_array_equal(r.replicates, ref.replicates)
    assert (r.estimate, r.ci_low, r.ci_high, r.draws_key) == (ref.estimate, ref.ci_low, ref.ci_high, ref.draws_key)


# ------------------------------------------------------------------------------------------------ seed ensemble


def test_seed_ensemble_aligns_on_id():
    ids = np.array([5, 1, 9, 3])
    p0 = pd.DataFrame({"id": ids, "prob": [0.1, 0.2, 0.3, 0.4]})
    p1 = pd.DataFrame({"id": ids[::-1], "prob": [0.8, 0.6, 0.4, 0.2]})  # reversed order
    p2 = pd.DataFrame({"id": ids[[2, 0, 3, 1]], "prob": [0.0, 0.3, 1.0, 0.5]})
    out = m.seed_ensemble({0: p0, 1: p1, 2: p2})
    assert out["id"].tolist() == ids.tolist()
    # seed values per id: 5 -> (0.1, 0.2, 0.3), 1 -> (0.2, 0.4, 0.5), 9 -> (0.3, 0.6, 0.0), 3 -> (0.4, 0.8, 1.0)
    np.testing.assert_allclose(out["prob"], [0.2, 1.1 / 3, 0.3, 2.2 / 3], atol=1e-15)
    pd.testing.assert_frame_equal(out, m.seed_ensemble([p0, p1, p2]))


def test_seed_ensemble_rejects_mismatch():
    p0 = pd.DataFrame({"id": [1, 2, 3], "prob": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="differs"):
        m.seed_ensemble([p0, pd.DataFrame({"id": [1, 2, 4], "prob": [0.1, 0.2, 0.3]})])
    with pytest.raises(ValueError, match="differs"):
        m.seed_ensemble([p0, pd.DataFrame({"id": [1, 2], "prob": [0.1, 0.2]})])
    with pytest.raises(ValueError, match="duplicate"):
        m.seed_ensemble([p0, pd.DataFrame({"id": [1, 2, 2, 3], "prob": [0.1, 0.2, 0.2, 0.3]})])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        m.seed_ensemble([p0, pd.DataFrame({"id": [1, 2, 3], "prob": [0.1, np.nan, 0.3]})])
    with pytest.raises(ValueError, match="no seed"):
        m.seed_ensemble([])


# ------------------------------------------------------------------------------------------------ pair accuracy


def test_pair_accuracy_hand_example():
    #            pair 0      pair 1      pair 2 (tie)  pair 3 (cross-sheet)  bg   pair 4 (incomplete)
    y = np.array([1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.2, 0.7, 0.4, 0.5, 0.5, 0.1, 0.6, 0.99, 0.8])
    pid = np.array([0, 0, 1, 1, 2, 2, 3, 3, -1, 4])
    sheet = np.array(["a", "a", "b", "b", "c", "c", "d", "e", "a", "f"])
    r = m.pair_accuracy(y, s, pid, sheet=sheet)
    assert (r.n_pairs, r.n_correct, r.n_wrong, r.n_ties, r.n_cross_sheet, r.n_incomplete) == (3, 1, 1, 1, 1, 1)
    assert r.accuracy == pytest.approx((1 + 0 + 0.5) / 3)
    np.testing.assert_array_equal(r.pair_ids, [0, 1, 2])
    np.testing.assert_array_equal(r.credit, [1.0, 0.0, 0.5])
    r_all = m.pair_accuracy(y, s, pid)  # without sheet: the cross-sheet pair counts too (and is wrong)
    assert (r_all.n_pairs, r_all.n_cross_sheet, r_all.accuracy) == (4, 0, pytest.approx(1.5 / 4))


def test_pair_accuracy_is_order_free_and_checks_pairs():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.3, 0.1, 0.2, 0.4])
    pid = np.array([7, 7, 2, 2])
    perm = [3, 0, 2, 1]
    assert m.pair_accuracy(y, s, pid).accuracy == m.pair_accuracy(y[perm], s[perm], pid[perm]).accuracy == 0.5
    with pytest.raises(ValueError, match="positives"):
        m.pair_accuracy([1, 1, 0], [0.1, 0.2, 0.3], [0, 0, 0])
    with pytest.raises(ValueError, match="negatives"):
        m.pair_accuracy([1, 0, 0], [0.1, 0.2, 0.3], [0, 0, 0])
    with pytest.raises(ValueError, match="integers"):
        m.pair_accuracy([1, 0], [0.1, 0.2], [0.5, 0.5])
    assert np.isnan(m.pair_accuracy([0, 0], [0.1, 0.2], [-1, -1]).accuracy)


# ------------------------------------------------------------------------------------------------ stage-2 PR sweep

# Hand-computed example, 4 targets, recall level 0.5 (TP >= 2):
#
#   score  outcomes            cum TP  cum FP  precision  recall
#   0.9    tp, fp (tie)        1       1       1/2        1/4
#   0.8    ignored             1       1       1/2        1/4    <- occupies a threshold, changes nothing
#   0.7    duplicate, fp       1       2       1/3        1/4
#   0.6    tp, fp (tie)        2       3       2/5        1/2    <- highest threshold with recall >= 0.5
#   0.5    tp, ignored         3       3       1/2        3/4
#   0.4    fp                  3       4       3/7        3/4
#   0.3    tp                  4       4       1/2        1
#   0.2    duplicate           4       4       1/2        1
#
# P@R50 = 2/5 at threshold 0.6 (if the tie at 0.6 were broken with the TP first it would wrongly be 2/4);
# interpolated = max(2/5, 1/2, 3/7, 1/2, 1/2) = 1/2; max recall = 1.
PR_SCORES = [0.9, 0.9, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2]
PR_OUTCOMES = ["tp", "fp", "ignored", "duplicate", "fp", "tp", "fp", "tp", "ignored", "fp", "tp", "duplicate"]


def test_pr_sweep_hand_example_with_ties():
    r = m.pr_sweep(PR_SCORES, PR_OUTCOMES, n_targets=4, recall_level=0.5)
    assert r.status == m.REACHED and r.reached
    np.testing.assert_array_equal(r.thresholds, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    np.testing.assert_array_equal(r.tp, [1, 1, 1, 2, 3, 3, 4, 4])
    np.testing.assert_array_equal(r.fp, [1, 1, 2, 3, 3, 4, 4, 4])
    np.testing.assert_allclose(r.precision, [1 / 2, 1 / 2, 1 / 3, 2 / 5, 1 / 2, 3 / 7, 1 / 2, 1 / 2])
    np.testing.assert_allclose(r.recall, [0.25, 0.25, 0.25, 0.5, 0.75, 0.75, 1.0, 1.0])
    assert r.precision_at_recall == pytest.approx(0.4)
    assert r.threshold_at_recall == 0.6
    assert r.interpolated_precision == pytest.approx(0.5)
    assert r.max_recall == 1.0
    assert (r.n_tp, r.n_fp, r.n_ignored, r.n_duplicate) == (4, 4, 2, 2)
    assert r.passes(0.3) and not r.passes(0.45)
    d = r.to_dict()
    assert d["status"] == "reached" and d["n_thresholds"] == 8


def test_pr_sweep_ignores_input_order():
    perm = np.random.default_rng(0).permutation(len(PR_SCORES))
    a = m.pr_sweep(PR_SCORES, PR_OUTCOMES, 4, 0.5)
    b = m.pr_sweep(np.asarray(PR_SCORES)[perm], np.asarray(PR_OUTCOMES)[perm], 4, 0.5)
    assert a == b


def test_pr_sweep_ignored_and_duplicates_never_change_precision():
    base = m.pr_sweep([0.9, 0.6, 0.3], ["tp", "fp", "tp"], 4, 0.5)
    extra = m.pr_sweep([0.9, 0.8, 0.6, 0.5, 0.3, 0.1], ["tp", "ignored", "fp", "duplicate", "tp", "ignored"], 4, 0.5)
    assert (base.precision_at_recall, base.threshold_at_recall, base.interpolated_precision, base.max_recall) == (
        extra.precision_at_recall, extra.threshold_at_recall, extra.interpolated_precision, extra.max_recall)
    assert extra.thresholds.size == 6 and base.thresholds.size == 3


def test_pr_sweep_recall_never_reached_is_a_fail():
    r = m.pr_sweep(PR_SCORES, PR_OUTCOMES, n_targets=10, recall_level=0.5)  # 4 TP of 10 targets
    assert r.status == m.NOT_REACHED and not r.reached
    assert r.precision_at_recall is None and r.threshold_at_recall is None and r.interpolated_precision is None
    assert r.max_recall == pytest.approx(0.4)
    assert not r.passes(0.0)
    assert r.to_dict()["precision_at_recall"] is None
    empty = m.pr_sweep([], [], n_targets=3, recall_level=0.5)
    assert empty.status == m.NOT_REACHED and empty.max_recall == 0.0 and not empty.passes(0.0)
    only_ignored = m.pr_sweep([0.5, 0.4], ["ignored", "fp"], n_targets=3, recall_level=0.5)
    assert not only_ignored.reached and np.isnan(only_ignored.precision[0])


def test_pr_sweep_recall_exactly_at_level_counts():
    r = m.pr_sweep([0.9, 0.8, 0.7], ["tp", "fp", "tp"], n_targets=4, recall_level=0.5)
    assert r.reached and r.precision_at_recall == pytest.approx(2 / 3) and r.threshold_at_recall == 0.7
    r = m.pr_sweep([0.9, 0.8, 0.7], ["tp", "fp", "tp"], n_targets=5, recall_level=0.5)
    assert not r.reached and r.max_recall == pytest.approx(0.4)


def test_pr_sweep_weights_equal_duplication():
    w = np.array([2, 1, 3, 0, 1, 2, 1, 1, 2, 1, 1, 2])
    idx = np.repeat(np.arange(len(PR_SCORES)), w)
    a = m.pr_sweep(PR_SCORES, PR_OUTCOMES, n_targets=9, recall_level=0.5, weights=w)
    b = m.pr_sweep(np.asarray(PR_SCORES)[idx], np.asarray(PR_OUTCOMES)[idx], n_targets=9, recall_level=0.5)
    assert a.to_dict() == b.to_dict()


def test_pr_sweep_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown outcomes"):
        m.pr_sweep([0.1], ["hit"], 1, 0.5)
    with pytest.raises(ValueError, match="n_targets"):
        m.pr_sweep([0.1], ["tp"], 0, 0.5)
    with pytest.raises(ValueError, match="true positives"):
        m.pr_sweep([0.1, 0.2], ["tp", "tp"], 1, 0.5)
    with pytest.raises(ValueError, match="length"):
        m.pr_sweep([0.1, 0.2], ["tp"], 1, 0.5)
    with pytest.raises(ValueError, match="NaN"):
        m.pr_sweep([np.nan], ["tp"], 1, 0.5)


def test_pr_sweep_reads_recall_level_from_config():
    cfg = load_config()
    th = cfg["eval"]["thresholds"]
    r = m.pr_sweep(PR_SCORES, PR_OUTCOMES, 4, th["stage2_recall"])
    assert r.recall_level == 0.5 and r.passes(th["stage2_precision_min"])
