"""landslide.baselines and scripts/run_baselines.py on synthetic meta tables only (nothing under data/ is read).

The synthetic table has every PLAN.md §4.3 `meta.csv` column: 14 blocks (10 flysch over 5 folds, 4 non-flysch), 3
sheets per block, matched positive/negative pairs on each sheet (negatives from every source) and background rows.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from landslide import baselines as B
from landslide import metrics as M
from landslide.config import config_hash, load_config
from landslide.paths import Paths
from landslide.train import select_unit_rows

CFG = load_config()
NAMES = ["slope_only", "susceptibility", "slope_susc_lr", "terrain_stats_lr", "terrain_stats_gbm"]
NEG_SOURCES = ("random", "random", "quarry", "spoil", "rock", "cut")

# (block, fold, group_b, kraj): 10 flysch blocks (F6..F9 are "F_other": in the (a) folds and the (b) test group) and
# 4 non-flysch blocks (N0, N1 ZLMS train; N2, N3 test).
BLOCKS = [(f"F{i}", i % 5, "train" if i < 6 else "test",
           ("Zlínský", "Moravskoslezský", "Jihomoravský", "Olomoucký")[i // 3 if i < 6 else 2 + (i - 6) // 2])
          for i in range(10)] + \
         [("N0", -1, "train", "Zlínský"), ("N1", -1, "train", "Moravskoslezský"),
          ("N2", -1, "test", "Jihomoravský"), ("N3", -1, "test", "Plzeňský")]
INNER_VAL = {"a_f0": ["F1"], "a_f1": ["F2"], "a_f2": ["F3"], "a_f3": ["F4"], "a_f4": ["F0"], "b": ["F2", "N1"],
             "s2": ["F7", "N3"]}
SHEETS_PER_BLOCK = 3


def synth_meta(pairs_per_sheet: int = 8, n_bg: int = 2, seed: int = 0, signal: float = 0.6) -> pd.DataFrame:
    """A meta table with the §4.3 columns; positives carry a weak signal in a few statistics."""
    rng = np.random.default_rng(seed)
    rows = []
    pair = 0
    for b, fold, g, kraj in BLOCKS:
        stratum = ("F_" if fold >= 0 else "N_") + ("ZLMS" if g == "train" else "other")
        for s in range(SHEETS_PER_BLOCK):
            sheet = f"{b}-{s:02d}"
            common = {"sheet": sheet, "kraj": kraj, "flysch": fold >= 0, "fold": fold, "group_b": g, "block": b,
                      "stratum": stratum}
            for j in range(pairs_per_sheet):
                slope = rng.uniform(5, 30)
                for lab in (1, 0):
                    src = "positive" if lab else NEG_SOURCES[pair % len(NEG_SOURCES)]
                    rows.append({**common, "label": lab, "source": src, "slope_true": slope, "pair_id": pair,
                                 "match_level": 0 if lab else 1, "polygon_objectid": 100 + pair if lab else -1})
                pair += 1
            for _ in range(n_bg):
                rows.append({**common, "label": 0, "source": "background", "slope_true": rng.uniform(0, 30),
                             "pair_id": -1, "match_level": 0, "polygon_objectid": -1})
    meta = pd.DataFrame(rows)
    n = len(meta)
    lab = meta["label"].to_numpy()
    meta.insert(0, "id", np.arange(n))
    meta["x"] = rng.uniform(-600000, -450000, n).round(1)
    meta["y"] = rng.uniform(-1200000, -1100000, n).round(1)
    meta["row"] = rng.integers(100, 2000, n)
    meta["col"] = rng.integers(100, 2000, n)
    meta["slope_mean"] = meta.pop("slope_true") + rng.normal(0, 0.3, n)  # matched: no signal
    meta["slope_bin"] = np.floor(meta["slope_mean"] / 2).astype(int)
    meta["slope_std"] = rng.gamma(4, 1.5, n) + signal * lab
    meta["slope_p90"] = meta["slope_mean"] + 1.5 * meta["slope_std"]
    for c in ("lrm_s_std", "lrm_l_std", "curv_prof_std", "curv_plan_std", "svf_std", "detr_std"):
        meta[c] = rng.gamma(3, 0.5, n) + (signal / 2 if c in ("curv_prof_std", "detr_std") else 0) * lab
    meta["svf_mean"] = rng.uniform(0.8, 1.0, n)
    meta["detr_range_2_98"] = 4 * meta["detr_std"] + rng.normal(0, 0.5, n)
    meta["tri"] = rng.gamma(2, 0.3, n)
    meta["susc_high_frac"] = np.clip(rng.beta(1, 4, n) + 0.1 * lab, 0, 1)
    meta["susc_mean"] = np.clip(rng.choice([0.0, 0.5, 1.0, 1.5, 2.0], n) + 0.25 * lab, 0, 2)
    order = ["id", "label", "x", "y", "row", "col", "sheet", "kraj", "flysch", "source", "polygon_objectid", "fold",
             "group_b", "block", "stratum", "slope_bin", "match_level", "pair_id", *B.STAT_COLUMNS]
    return meta[order]


def synth_splits() -> dict:
    return {
        "stage2": {"flysch": [], "nonflysch": []},
        "folds_a": {str(k): {"blocks": [b for b, f, _, _ in BLOCKS if f == k], "sheets": [], "n_eligible": 0}
                    for k in range(5)},
        "group_b": {"train_blocks": [b for b, _, g, _ in BLOCKS if g == "train"],
                    "test_blocks": [b for b, _, g, _ in BLOCKS if g == "test"]},
        "inner_val": INNER_VAL,
        "config_hash": "synthetic",
    }


def small_cfg(n: int = 200) -> dict:
    c = copy.deepcopy(CFG)
    c["eval"]["bootstrap"]["n"] = n
    return c


@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    return synth_meta()


@pytest.fixture(scope="module")
def splits() -> dict:
    return synth_splits()


@pytest.fixture(scope="module")
def res(meta, splits) -> B.BaselineResults:
    return B.run_baselines(meta, splits, CFG)


def _load_script():
    spec = importlib.util.spec_from_file_location("run_baselines", Paths().root / "scripts" / "run_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------------
# Rows: fitting never sees the test rows; the test rows are the CNN's
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("run", ["a_f0", "a_f1", "a_f2", "a_f3", "a_f4", "b"])
def test_fit_rows_are_train_plus_inner_val(meta, splits, run):
    u = B.predict_unit(run, meta, splits, CFG)
    r = select_unit_rows(run, meta, splits)
    np.testing.assert_array_equal(u.fit_rows, np.union1d(r.train, r.val))
    np.testing.assert_array_equal(u.test_rows, r.test)
    assert np.intersect1d(u.fit_rows, u.test_rows).size == 0
    assert len(r.val) > 0  # inner-val rows really are part of the fit
    assert set(meta["source"].iloc[u.fit_rows]) != {"background"} and "background" not in set(
        meta["source"].iloc[np.concatenate([u.fit_rows, u.test_rows])])
    assert [f.baseline for f in u.fits] == ["slope_susc_lr", "terrain_stats_lr", "terrain_stats_gbm"]
    for f in u.fits:
        assert (f.n_fit, f.n_train, f.n_inner_val, f.n_test) == (len(u.fit_rows), len(r.train), len(r.val),
                                                                  len(r.test))
        assert f.n_fit_pos == int(meta["label"].iloc[u.fit_rows].sum())


def test_estimators_receive_only_fit_rows(meta, splits, monkeypatch):
    """Spy on what every estimator is fitted on: rows are identified by their exact feature vectors."""
    seen = []
    orig = B._fit_predict

    def spy(spec, x_fit, y_fit, x_test, seed):
        seen.append((spec.name, x_fit.copy(), y_fit.copy(), x_test.copy()))
        return orig(spec, x_fit, y_fit, x_test, seed)

    monkeypatch.setattr(B, "_fit_predict", spy)
    for run in ["a_f0", "a_f1", "a_f2", "a_f3", "a_f4", "b"]:
        seen.clear()
        u = B.predict_unit(run, meta, splits, CFG)
        assert len(seen) == 3
        for name, x_fit, y_fit, x_test in seen:
            feats = next(s.features for s in B.baseline_specs(CFG) if s.name == name)
            allx = meta[list(feats)].to_numpy(dtype=np.float64)
            index = {tuple(v): i for i, v in enumerate(allx)}
            assert len(index) == len(meta)  # feature vectors identify rows
            fit_rows = np.sort([index[tuple(v)] for v in x_fit])
            test_rows = [index[tuple(v)] for v in x_test]
            np.testing.assert_array_equal(fit_rows, u.fit_rows)
            np.testing.assert_array_equal(test_rows, u.test_rows)
            assert not set(fit_rows) & set(test_rows)
            np.testing.assert_array_equal(np.sort(y_fit), np.sort(meta["label"].to_numpy()[u.fit_rows]))


@pytest.mark.parametrize(("test", "run"), [("a", "a_f0"), ("a", "a_f3"), ("b", "b")])
def test_test_labels_do_not_influence_their_own_predictions(meta, splits, test, run):
    """Flip the labels of one run's test rows: that run's predictions stay bit-identical. For (a) the other folds,
    which train on those rows, change, so the check has power."""
    base = B.predict_test(test, meta, splits, CFG)
    rows = select_unit_rows(run, meta, splits).test
    m2 = meta.copy()
    m2.loc[rows, "label"] = 1 - m2.loc[rows, "label"]
    alt = B.predict_test(test, m2, splits, CFG)
    np.testing.assert_array_equal(alt.rows, base.rows)
    mine = np.isin(base.rows, rows)
    assert mine.sum() == len(rows)
    for name in NAMES:
        np.testing.assert_array_equal(alt.scores[name][mine], base.scores[name][mine])
    if test == "a":
        for name in ("slope_susc_lr", "terrain_stats_lr", "terrain_stats_gbm"):
            assert not np.array_equal(alt.scores[name][~mine], base.scores[name][~mine])


def test_preds_rows_equal_select_unit_rows(meta, splits, res):
    ids = meta["id"].to_numpy()
    exp = []
    for k in range(5):
        t = select_unit_rows(f"a_f{k}", meta, splits).test
        exp.append(pd.DataFrame({"id": ids[t], "fold": k}))
    exp_a = pd.concat(exp).sort_values("id").reset_index(drop=True)
    pa = res.preds["a"]
    np.testing.assert_array_equal(pa["id"].to_numpy(), exp_a["id"].to_numpy())
    np.testing.assert_array_equal(pa["fold"].to_numpy(), exp_a["fold"].to_numpy())
    np.testing.assert_array_equal(res.preds["b"]["id"].to_numpy(), ids[select_unit_rows("b", meta, splits).test])
    # The pooled (a) test set is every flysch stage-1 patch; (b) is every "test"-group patch; no background.
    assert set(pa["id"]) == set(meta.loc[(meta["fold"] >= 0) & (meta["source"] != "background"), "id"])
    assert set(res.preds["b"]["id"]) == set(
        meta.loc[(meta["group_b"] == "test") & (meta["source"] != "background"), "id"])
    assert pa["id"].is_unique and res.preds["b"]["id"].is_unique


def test_unfitted_scores_are_the_columns(meta, res):
    for t in ("a", "b"):
        p = res.preds[t].set_index("id")
        m = meta.set_index("id").loc[p.index]
        np.testing.assert_array_equal(p["slope_only"].to_numpy(), m["slope_mean"].to_numpy())
        np.testing.assert_array_equal(p["susceptibility"].to_numpy(), m["susc_mean"].to_numpy())
        for n in ("slope_susc_lr", "terrain_stats_lr", "terrain_stats_gbm"):
            assert ((p[n] > 0) & (p[n] < 1)).all()


# ---------------------------------------------------------------------------------------------------------------
# Determinism and schema
# ---------------------------------------------------------------------------------------------------------------

def test_determinism(meta, splits):
    cfg = small_cfg()
    r1 = B.run_baselines(meta, splits, cfg)
    r2 = B.run_baselines(meta.copy(), copy.deepcopy(splits), cfg)
    for t in ("a", "b"):
        pd.testing.assert_frame_equal(r1.preds[t], r2.preds[t], check_exact=True)
    pd.testing.assert_frame_equal(r1.table, r2.table, check_exact=True)
    assert json.dumps(r1.summary, sort_keys=True) == json.dumps(r2.summary, sort_keys=True)
    # Row order of meta does not matter for the predictions of a patch.
    shuffled = meta.sample(frac=1.0, random_state=3).reset_index(drop=True)
    r3 = B.run_baselines(shuffled, splits, cfg)
    for t in ("a", "b"):
        a = r1.preds[t].set_index("id").sort_index()
        b = r3.preds[t].set_index("id").sort_index()
        pd.testing.assert_frame_equal(a[["slope_only", "susceptibility"]], b[["slope_only", "susceptibility"]])
        np.testing.assert_allclose(a["terrain_stats_lr"], b["terrain_stats_lr"], rtol=0, atol=1e-9)


def test_script_outputs_and_schema(meta, splits, tmp_path):
    mp, sp = tmp_path / "meta.csv", tmp_path / "splits.json"
    meta.to_csv(mp, index=False)
    sp.write_text(json.dumps(splits), encoding="utf-8")
    out = tmp_path / "out"
    rb = _load_script()
    assert rb.main(["--meta", str(mp), "--splits", str(sp), "--out", str(out), "--check-determinism"]) == 0

    pa = pd.read_csv(out / "baselines" / "preds_a.csv")
    pb = pd.read_csv(out / "baselines" / "preds_b.csv")
    assert list(pa.columns) == ["id", "fold", *NAMES]
    assert list(pb.columns) == ["id", *NAMES]
    assert pa["fold"].between(0, 4).all() and pa.notna().all().all() and pb.notna().all().all()

    info = json.loads((out / "baselines" / "build_info.json").read_text(encoding="utf-8"))
    assert info["config_hash"] == config_hash(load_config())
    assert info["files"]["preds_a.csv"]["rows"] == len(pa) and info["files"]["preds_b.csv"]["rows"] == len(pb)
    assert info["columns"]["preds_a.csv"] == list(pa.columns)
    assert info["determinism"] == {"recomputed": True, "identical": True}
    assert info["inputs"]["meta_rows"] == len(meta)

    tab = pd.read_csv(out / "results" / "baselines.csv", keep_default_na=False, na_values=[""])
    assert list(tab.columns) == B.table_columns(CFG)
    assert list(tab.columns[:11]) == ["test", "baseline", "kind", "breakdown", "group", "n", "n_pos", "n_neg",
                                      "n_sheets", "n_blocks", "auc"]
    assert {"sheet_ci_low", "sheet_ci_high", "block_ci_low", "block_ci_high"} <= set(tab.columns)
    assert not tab.duplicated(["test", "baseline", "breakdown", "group"]).any()
    kinds = tab.groupby("baseline")["kind"].unique().map(list).to_dict()
    assert kinds == {"slope_only": ["baseline"], "susceptibility": ["baseline"], "slope_susc_lr": ["classical_ml"],
                     "terrain_stats_lr": ["classical_ml"], "terrain_stats_gbm": ["classical_ml"]}
    # Every group carries all five scorers.
    assert (tab.groupby(["test", "breakdown", "group"])["baseline"].apply(sorted).map(tuple) ==
            tuple(sorted(NAMES))).all()

    js = json.loads((out / "results" / "baselines.json").read_text(encoding="utf-8"))
    assert js["config_hash"] == info["config_hash"] and js["task"] == "P1-11"
    assert set(js["tests"]) == {"a", "b"}
    for t in ("a", "b"):
        assert set(js["tests"][t]["results"]) == set(NAMES)
        r = js["tests"][t]["results"]["terrain_stats_lr"]
        assert r["kind"] == "classical_ml" and len(r["sheet_ci"]) == 2 and len(r["block_ci"]) == 2
        assert js["tests"][t]["best"]["baseline"] in NAMES
        assert js["matching_check"][t]["auc"] == js["tests"][t]["results"]["slope_only"]["auc"]
    assert js["bootstrap"] == {"n": 2000, "seed": 123, "ci": 0.95, "method": "percentile",
                               "units": ["sheet", "block"], "primary_unit": "sheet"}
    assert js["random_state"] == CFG["replication"]["seed"]
    assert js["outputs"]["preds_a"].endswith("preds_a.csv")
    # The JSON agrees with the table.
    row = tab[(tab["test"] == "a") & (tab["baseline"] == "terrain_stats_lr") & (tab["breakdown"] == "overall")]
    assert js["tests"]["a"]["results"]["terrain_stats_lr"]["auc"] == pytest.approx(float(row["auc"].iloc[0]),
                                                                                   abs=1e-15)
    # Nothing is written outside --out.
    assert sorted(p.name for p in (out / "baselines").iterdir()) == ["build_info.json", "preds_a.csv",
                                                                     "preds_b.csv"]
    assert sorted(p.name for p in (out / "results").iterdir()) == ["baselines.csv", "baselines.json"]


def test_script_refuses_missing_inputs(tmp_path):
    rb = _load_script()
    with pytest.raises(FileNotFoundError, match="P1-10"):
        rb.main(["--meta", str(tmp_path / "nope.csv"), "--splits", str(tmp_path / "nope.json"),
                 "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out" / "baselines").exists()


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------

def _tiny_meta() -> pd.DataFrame:
    # y = [1, 1, 0, 0, 0], score [3, 2, 2, 1, 2]: 3 beats all three negatives, 2 beats one and ties two -> 5/6.
    return pd.DataFrame({
        "id": [10, 11, 12, 13, 14], "label": [1, 1, 0, 0, 0],
        "source": ["positive", "positive", "random", "quarry", "random"],
        "sheet": ["s1", "s2", "s1", "s2", "s3"], "block": ["A", "A", "A", "B", "B"],
        "kraj": ["K1", "K1", "K1", "K2", "K2"], "flysch": [True, True, True, False, False],
        "slope_mean": [3.0, 2.0, 2.0, 1.0, 2.0], "susc_mean": [0.0, 0.0, 0.0, 0.0, 0.0]})


def test_scorer_with_known_auc():
    meta = _tiny_meta()
    specs = B.baseline_specs(CFG)[:2]
    tp = B.PooledPredictions(test="b", rows=np.arange(5), run=np.array(["b"] * 5, dtype=object),
                             scores={"slope_only": meta["slope_mean"].to_numpy(),
                                     "susceptibility": meta["susc_mean"].to_numpy()}, units=[])
    tab = B.evaluate_test("b", meta, tp, small_cfg(), specs)
    get = lambda bd, g, n: tab[(tab["breakdown"] == bd) & (tab["group"] == g) & (tab["baseline"] == n)].iloc[0]
    assert get("overall", "all", "slope_only")["auc"] == pytest.approx(5 / 6, abs=1e-15)
    assert get("overall", "all", "susceptibility")["auc"] == 0.5  # constant score: all ties
    r = get("overall", "all", "slope_only")
    assert (r["n"], r["n_pos"], r["n_neg"], r["n_sheets"], r["n_blocks"]) == (5, 2, 3, 3, 2)
    # Positives vs random negatives only: pos {3, 2} vs neg {2, 2} -> (1 + 1 + 0.5 + 0.5) / 4 = 0.75
    assert get("source", "random", "slope_only")["auc"] == pytest.approx(0.75, abs=1e-15)
    # Positives vs quarry: {3, 2} vs {1} -> 1
    assert get("source", "quarry", "slope_only")["auc"] == 1.0
    # No spoil negatives in this test set: the group has one class.
    sp = get("source", "spoil", "slope_only")
    assert math.isnan(sp["auc"]) and sp["n_neg"] == 0 and "one class" in sp["note"]
    # kraj K2 holds negatives only.
    k2 = get("kraj", "K2", "slope_only")
    assert math.isnan(k2["auc"]) and k2["n_pos"] == 0
    assert get("kraj", "K1", "slope_only")["auc"] == pytest.approx(0.75, abs=1e-15)  # {3, 2} vs {2}
    assert set(tab.loc[tab["breakdown"] == "flysch", "group"]) == {"flysch", "nonflysch"}


def test_auc_and_cis_use_the_shared_draws(meta, res):
    """Each AUC equals sklearn's, and each CI equals metrics.bootstrap_auc on draws over the same rows."""
    cfg = CFG
    lab = meta.set_index("id")
    for t in ("a", "b"):
        p = res.preds[t]
        m = lab.loc[p["id"]]
        y = m["label"].to_numpy()
        tab = res.table[(res.table["test"] == t) & (res.table["breakdown"] == "overall")].set_index("baseline")
        ds = M.draw_clusters_from_cfg(m["sheet"].to_numpy(), cfg)
        db = M.draw_clusters_from_cfg(m["block"].to_numpy(), cfg)
        for n in NAMES:
            assert tab.loc[n, "auc"] == pytest.approx(roc_auc_score(y, p[n]), abs=1e-12)
            rs = M.bootstrap_auc(y, p[n].to_numpy(), ds, ci=0.95)
            rb = M.bootstrap_auc(y, p[n].to_numpy(), db, ci=0.95)
            assert (tab.loc[n, "sheet_ci_low"], tab.loc[n, "sheet_ci_high"]) == (rs.ci_low, rs.ci_high)
            assert (tab.loc[n, "block_ci_low"], tab.loc[n, "block_ci_high"]) == (rb.ci_low, rb.ci_high)
            assert tab.loc[n, "sheet_n_used"] + tab.loc[n, "sheet_n_degenerate"] == 2000
            assert tab.loc[n, "n_sheets"] == m["sheet"].nunique() and tab.loc[n, "n_blocks"] == m["block"].nunique()


def test_breakdown_groups(meta, splits, res):
    tab = res.table
    groups = {t: tab[(tab["test"] == t) & (tab["baseline"] == "slope_only")] for t in ("a", "b")}
    ga = groups["a"]
    assert list(zip(ga["breakdown"], ga["group"])) == (
        [("overall", "all")] + [("fold", str(k)) for k in range(5)] +
        [("source", s) for s in ("random", "quarry", "spoil", "rock", "cut")])
    gb = groups["b"]
    assert list(zip(gb["breakdown"], gb["group"])) == (
        [("overall", "all"), ("kraj", "Jihomoravský"), ("kraj", "Olomoucký"), ("kraj", "Plzeňský"),
         ("flysch", "flysch"), ("flysch", "nonflysch")] +
        [("source", s) for s in ("random", "quarry", "spoil", "rock", "cut")])
    for t, g in groups.items():
        p = meta.set_index("id").loc[res.preds[t]["id"]]
        n_pos = int(p["label"].sum())
        for _, r in g[g["breakdown"] == "source"].iterrows():
            assert r["n_pos"] == n_pos and r["n_neg"] == int((p["source"] == r["group"]).sum())
        folds_or_kraje = g[g["breakdown"].isin(["fold", "kraj"])]
        assert folds_or_kraje["n"].sum() == len(p)
    # Per-fold AUCs equal the AUC of each fold's own test rows.
    pa = res.preds["a"]
    lab = meta.set_index("id").loc[pa["id"], "label"].to_numpy()
    fold_rows = tab[(tab["test"] == "a") & (tab["breakdown"] == "fold") & (tab["baseline"] == "terrain_stats_gbm")]
    for _, r in fold_rows.iterrows():
        f = pa["fold"].to_numpy() == int(r["group"])
        assert r["auc"] == pytest.approx(roc_auc_score(lab[f], pa.loc[f, "terrain_stats_gbm"]), abs=1e-12)


def test_signal_is_found_and_matching_check(meta, splits, res):
    """The synthetic positives differ in slope_std / curvature / detr only: the fitted scorers find it, slope-only
    (matched) stays near 0.5."""
    s = res.summary
    for t in ("a", "b"):
        r = s["tests"][t]["results"]
        assert r["terrain_stats_lr"]["auc"] > 0.6 and r["slope_susc_lr"]["auc"] > 0.55
        assert abs(r["slope_only"]["auc"] - 0.5) < 0.08
        assert s["matching_check"][t]["auc"] == r["slope_only"]["auc"]
        lo, hi = s["matching_check"][t]["sheet_ci"]
        assert s["matching_check"][t]["sheet_ci_contains_0.5"] == (lo <= 0.5 <= hi)
        best = s["tests"][t]["best"]
        assert best["auc"] == max(v["auc"] for v in r.values())
        assert s["tests"][t]["best_baseline"]["baseline"] in ("slope_only", "susceptibility")
        assert s["tests"][t]["best_classical_ml"]["baseline"] in ("slope_susc_lr", "terrain_stats_lr",
                                                                  "terrain_stats_gbm")
    assert set(s["matching_check"]["a"]["per_fold_auc"]) == {"0", "1", "2", "3", "4"}


def test_perfect_feature_gives_auc_one(splits):
    m = synth_meta(seed=3)
    m["tri"] = m["label"] * 10.0 + np.random.default_rng(1).uniform(0, 1, len(m))
    r = B.run_baselines(m, splits, small_cfg())
    for t in ("a", "b"):
        assert r.summary["tests"][t]["results"]["terrain_stats_lr"]["auc"] == 1.0
        # Boosting keeps fitting noise in the other 11 statistics after the first split, so not exactly 1.
        assert r.summary["tests"][t]["results"]["terrain_stats_gbm"]["auc"] > 0.99


def test_spike_reference_is_reference_only(res):
    tab = res.table
    filled = tab[tab["spike_auc"].notna()]
    assert list(zip(filled["test"], filled["baseline"], filled["breakdown"])) == [
        ("a", "terrain_stats_lr", "overall"), ("b", "terrain_stats_gbm", "overall")]
    assert filled["spike_auc"].tolist() == [0.695, 0.624]
    assert filled["spike_ci_low"].tolist() == [0.678, 0.603] and filled["spike_ci_high"].tolist() == [0.713, 0.647]
    ref = res.summary["spike_reference"]
    assert ref["a"]["auc"] == 0.695 and ref["b"]["baseline"] == "terrain_stats_gbm"
    a = res.summary["tests"]["a"]["spike_reference"]
    assert a["ours"] == res.summary["tests"]["a"]["results"]["terrain_stats_lr"]["auc"]
    assert a["ours_inside_spike_ci"] == (0.678 <= a["ours"] <= 0.713)


# ---------------------------------------------------------------------------------------------------------------
# Settings, convergence, input checks
# ---------------------------------------------------------------------------------------------------------------

def test_settings_are_sklearn_defaults_with_fixed_seed(res):
    seed = CFG["replication"]["seed"]
    sc = res.summary["scorers"]
    lr = sc["terrain_stats_lr"]["settings"]
    assert lr["scaler"]["class"] == "StandardScaler" and lr["lr"]["class"] == "LogisticRegression"
    p = lr["lr"]["params"]
    assert (p["C"], p["solver"], p["max_iter"], p["tol"], p["l1_ratio"], p["fit_intercept"], p["class_weight"],
            p["random_state"]) == (1.0, "lbfgs", 100, 1e-4, 0.0, True, None, seed)
    assert lr["scaler"]["params"] == {"copy": True, "with_mean": True, "with_std": True}
    g = sc["terrain_stats_gbm"]["settings"]["HistGradientBoostingClassifier"]["params"]
    assert (g["random_state"], g["early_stopping"], g["learning_rate"], g["max_iter"], g["max_leaf_nodes"],
            g["min_samples_leaf"], g["l2_regularization"], g["max_bins"]) == (seed, "auto", 0.1, 100, 31, 20, 0.0,
                                                                                255)
    assert sc["slope_only"]["kind"] == "baseline" and sc["terrain_stats_gbm"]["kind"] == "classical_ml"
    assert sc["terrain_stats_lr"]["features"] == list(B.TERRAIN_FEATURES) and len(B.TERRAIN_FEATURES) == 12
    assert sc["slope_susc_lr"]["features"] == ["slope_mean", "slope_std", "susc_mean", "susc_high_frac"]
    # Every LR fit converged, and the GBM used no early stopping (training sets far below 10,000 rows).
    assert res.summary["all_lr_fits_converged"]
    fits = res.summary["fits"]
    assert len(fits) == 3 * 6
    assert all(f["converged"] and f["n_iter"] < 100 for f in fits if f["model"] == "lr")
    assert all(f["early_stopping"] is False and f["n_iter"] == 100 for f in fits if f["model"] == "gbm")


def test_non_convergence_is_recorded_not_hidden(meta, splits, monkeypatch):
    orig = B.make_estimator

    def capped(spec, seed):
        est = orig(spec, seed)
        if spec.model == "lr":
            est.set_params(lr__max_iter=1)
        return est

    monkeypatch.setattr(B, "make_estimator", capped)
    u = B.predict_unit("a_f0", meta, splits, CFG)
    lr = [f for f in u.fits if f.model == "lr"]
    assert all(f.converged is False and f.n_iter == 1 for f in lr)
    assert all(any("ConvergenceWarning" in w for w in f.warnings) for f in lr)
    r = B.run_baselines(meta, splits, small_cfg(50))
    assert r.summary["all_lr_fits_converged"] is False


def test_lr_matches_a_plain_sklearn_fit(meta, splits):
    u = B.predict_unit("b", meta, splits, CFG)
    feats = list(B.SLOPE_SUSC_FEATURES)
    x = meta[feats].to_numpy(dtype=np.float64)
    mu, sd = x[u.fit_rows].mean(axis=0), x[u.fit_rows].std(axis=0)
    lr = LogisticRegression(random_state=CFG["replication"]["seed"]).fit((x[u.fit_rows] - mu) / sd,
                                                                         meta["label"].to_numpy()[u.fit_rows])
    np.testing.assert_allclose(u.scores["slope_susc_lr"], lr.predict_proba((x[u.test_rows] - mu) / sd)[:, 1],
                               rtol=1e-12, atol=1e-12)


def test_non_finite_statistic_raises(meta, splits):
    m = meta.copy()
    m.loc[m.index[5], "svf_std"] = np.nan
    with pytest.raises(ValueError, match="svf_std"):
        B.run_baselines(m, splits, small_cfg(50))
    with pytest.raises(KeyError, match="tri"):
        B.run_baselines(meta.drop(columns=["tri"]), splits, small_cfg(50))


def test_flags_read_from_csv_forms():
    for vals in ([True, False], ["True", "False"], [1, 0], ["1", "0"]):
        np.testing.assert_array_equal(B._flag(pd.Series(vals)), [True, False])
    with pytest.raises(ValueError):
        B._flag(pd.Series(["yes", "no"]))


def test_scorer_specs_from_config():
    # Appendix A §10, as the config's replication.baselines.scorers records it
    assert B.baseline_specs(CFG) == (
        B.BaselineSpec("slope_only", "column", ("slope_mean",)),
        B.BaselineSpec("susceptibility", "column", ("susc_mean",)),
        B.BaselineSpec("slope_susc_lr", "lr", B.SLOPE_SUSC_FEATURES),
        B.BaselineSpec("terrain_stats_lr", "lr", B.TERRAIN_FEATURES),
        B.BaselineSpec("terrain_stats_gbm", "gbm", B.TERRAIN_FEATURES))
    assert [s.name for s in B.baseline_specs(CFG)] == NAMES
    assert [s.kind for s in B.baseline_specs(CFG)] == ["baseline", "baseline", "classical_ml", "classical_ml",
                                                       "classical_ml"]
    c = copy.deepcopy(CFG)
    c["replication"]["baselines"] = {"scorers": [{"name": "slope_only", "model": "column", "features": ["slope_mean"]},
                                                 {"name": "x_lr", "model": "lr", "features": ["tri", "svf_mean"]}]}
    sp = B.baseline_specs(c)
    assert [(s.name, s.model, s.features) for s in sp] == [("slope_only", "column", ("slope_mean",)),
                                                          ("x_lr", "lr", ("tri", "svf_mean"))]
    for bad in ([{"name": "a", "model": "svm", "features": ["tri"]}],
                [{"name": "a", "model": "column", "features": ["tri", "svf_mean"]}],
                [{"name": "a", "model": "lr", "features": ["elevation"]}],
                [{"name": "a", "model": "lr", "features": ["tri"]},
                 {"name": "a", "model": "gbm", "features": ["tri"]}]):
        c["replication"]["baselines"] = {"scorers": bad}
        with pytest.raises(ValueError):
            B.baseline_specs(c)
    assert B.bootstrap_units(CFG) == ("sheet", "block")
    assert B.negative_sources(CFG) == ("random", "quarry", "spoil", "rock", "cut")
    assert B.random_state(CFG) == CFG["replication"]["seed"]


# ---------------------------------------------------------------------------------------------------------------
# Real outputs (skipped until the orchestrator has run the baselines)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.needs_data
@pytest.mark.skipif(not (Paths().baselines_dir / "preds_a.csv").exists() or not Paths().meta_csv.exists()
                    or not Paths().splits_json.exists(), reason="data/baselines is not built")
def test_real_preds_match_the_cnn_test_rows():
    p = Paths()
    meta = pd.read_csv(p.meta_csv)
    splits = json.loads(p.splits_json.read_text(encoding="utf-8"))
    ids = meta["id"].to_numpy()
    pa = pd.read_csv(p.baselines_dir / "preds_a.csv")
    pb = pd.read_csv(p.baselines_dir / "preds_b.csv")
    assert list(pa.columns) == ["id", "fold", *NAMES] and list(pb.columns) == ["id", *NAMES]
    for k in range(int(CFG["splits"]["n_folds_a"])):
        t = select_unit_rows(f"a_f{k}", meta, splits).test
        np.testing.assert_array_equal(np.sort(pa.loc[pa["fold"] == k, "id"].to_numpy()), np.sort(ids[t]))
    np.testing.assert_array_equal(np.sort(pb["id"].to_numpy()), np.sort(ids[select_unit_rows("b", meta, splits).test]))
    info = json.loads((p.baselines_dir / "build_info.json").read_text(encoding="utf-8"))
    assert info["config_hash"] == config_hash(load_config())
