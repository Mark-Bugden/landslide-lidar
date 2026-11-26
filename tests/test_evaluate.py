"""landslide.evaluate and scripts/evaluate.py on synthetic run directories (`eval_synth.py`); no real data is read.

Covers: integrity (missing, failed, interrupted, hash mismatch, invalid units, a config that differs from the frozen
one), the seed ensemble and pooled out-of-fold predictions against an independent computation, CIs and paired
differences on shared draws, breakdowns and pair accuracy, stage 2 (not reached = FAIL, not run, the sheet
bootstrap's not-reached rule, TPs counted on their target's sheet), the verdict wiring, the spike comparison, smoke
isolation and byte-identical re-runs.
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from eval_synth import build_world, make_cfg, stage2_sheets, synth_detections, write_json
from landslide import evaluate as E
from landslide import metrics as M
from landslide import verdict as V
from landslide.config import config_hash, load_config
from landslide.figures import learning_curves_png
from landslide.logs import setup_logging
from landslide.paths import REPO_ROOT, Paths

TOL = 1e-6  # metrics.json rounds to 6 decimals


@pytest.fixture(autouse=True)
def _no_data_env(monkeypatch):
    monkeypatch.delenv("LANDSLIDE_DATA_DIR", raising=False)


def evaluate(w, **kw):
    return E.run_evaluation(w.cfg, w.paths, **kw)


def ensemble(w, runs):
    """Independent pooled seed ensemble: {id: mean prob} from the probabilities the world wrote."""
    out = {}
    for r in runs:
        seeds = w.probs[r]
        ids = seeds[next(iter(seeds))][0]
        mean = np.mean([seeds[s][1] for s in sorted(seeds)], axis=0)
        out.update(zip(ids.tolist(), mean.tolist()))
    return pd.Series(out).sort_index()


def labels(w, ids):
    return w.meta.set_index("id").loc[ids, "label"].to_numpy()


# ------------------------------------------------------------------------------------------------ complete run


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    w = build_world(tmp_path_factory.mktemp("complete"))
    return w, evaluate(w)


def test_complete_world_is_replicated(complete):
    w, m = complete
    assert m["verdict"] == "REPLICATED"
    assert {t: m["tests"][t]["status"] for t in ("a", "b", "stage2")} == {"a": "PASS", "b": "PASS", "stage2": "PASS"}
    assert m["config_hash"] == config_hash(w.cfg)
    assert m["integrity"]["config_hash_source"] == "current_config"  # no configs/landslide.sha256 yet
    assert m["integrity"]["ok"] and m["integrity"]["n_done"] == 21
    assert m["git_commit"] == "c0ffee"
    assert m["smoke"] is False


def test_pooled_oof_ensemble_matches_independent_computation(complete):
    w, m = complete
    ens = ensemble(w, [f"a_f{k}" for k in range(5)])
    expected_ids = w.meta.loc[(w.meta.source != "background") & (w.meta.fold >= 0), "id"].sort_values()
    assert ens.index.tolist() == expected_ids.tolist()
    y = labels(w, ens.index)
    a = m["tests"]["a"]
    assert a["estimate"] == pytest.approx(roc_auc_score(y, ens.to_numpy()), abs=TOL)
    assert a["n"]["patches"] == len(ens) and a["n"]["positives"] == int(y.sum())
    # per fold: the seed ensemble of that fold; per seed: that seed pooled over the folds
    for k in range(5):
        e = ensemble(w, [f"a_f{k}"])
        assert a["per_fold"][str(k)]["auc"] == pytest.approx(roc_auc_score(labels(w, e.index), e.to_numpy()), abs=TOL)
    for s in (0, 1, 2):
        ids = np.concatenate([w.probs[f"a_f{k}"][s][0] for k in range(5)])
        p = np.concatenate([w.probs[f"a_f{k}"][s][1] for k in range(5)])
        assert a["per_seed"][str(s)]["auc"] == pytest.approx(roc_auc_score(labels(w, ids), p), abs=TOL)
    assert set(a["per_unit"]) == {f"a_f{k}_s{s}" for k in range(5) for s in (0, 1, 2)}
    eb = ensemble(w, ["b"])
    assert m["tests"]["b"]["estimate"] == pytest.approx(roc_auc_score(labels(w, eb.index), eb.to_numpy()), abs=TOL)
    assert m["tests"]["b"]["per_fold"] is None


def test_cis_and_paired_differences_share_draws(complete):
    w, m = complete
    spec = M.bootstrap_spec(w.cfg)
    ens = ensemble(w, [f"a_f{k}" for k in range(5)])
    items = w.meta.set_index("id").loc[ens.index]
    y = items["label"].to_numpy()
    base = pd.read_csv(w.paths.baselines_dir / "preds_a.csv").set_index("id").loc[ens.index]
    for unit in ("sheet", "block"):
        draws = M.draw_clusters(items[unit].to_numpy(), spec.n_replicates, spec.seed)
        r_cnn = M.replicate_aucs(y, ens.to_numpy(), draws)
        lo, hi = M.percentile_ci(r_cnn, spec.ci)
        ci = m["tests"]["a"][f"{unit}_ci"]
        assert (ci["low"], ci["high"]) == (pytest.approx(lo, abs=TOL), pytest.approx(hi, abs=TOL))
        assert ci["n_replicates"] == spec.n_replicates
        for name in ("slope_only", "terrain_stats_lr", "terrain_stats_gbm"):
            r_b = M.replicate_aucs(y, base[name].to_numpy(), draws)
            d = m["paired_diffs"]["a"][unit][name]
            dlo, dhi = M.percentile_ci(r_cnn - r_b, spec.ci)
            assert d["ci"]["low"] == pytest.approx(dlo, abs=TOL) and d["ci"]["high"] == pytest.approx(dhi, abs=TOL)
            assert d["estimate"] == pytest.approx(M.roc_auc(y, ens.to_numpy()) - M.roc_auc(y, base[name]), abs=TOL)
            # the baseline's own CI is on the same draws
            blo, bhi = M.percentile_ci(r_b, spec.ci)
            bci = m["baselines"]["a"]["scorers"][name][f"{unit}_ci"]
            assert bci["low"] == pytest.approx(blo, abs=TOL) and bci["high"] == pytest.approx(bhi, abs=TOL)


def test_pair_accuracy_scope_is_same_sheet(complete):
    w, m = complete
    pa = m["secondary"]["pair_accuracy"]["a"]
    assert pa["primary"] == "same_sheet"
    meta = w.meta
    neg = meta[(meta.label == 0) & (meta.pair_id >= 0) & (meta.fold >= 0)]
    n_by_level = neg["match_level"].value_counts().to_dict()
    same = pa["same_sheet"]
    # same sheet = match levels 1 and 3; the level-2 pairs (other sheet of the block) are dropped and counted
    assert same["n_pairs"] == n_by_level[1] + n_by_level[3]
    assert same["by_match_level"] == {"1": n_by_level[1], "3": n_by_level[3]}
    assert same["n_cross_sheet"] == n_by_level[2]
    assert pa["level_1"]["n_pairs"] == n_by_level[1]
    assert pa["all_pairs"]["n_pairs"] == len(neg)
    # value: share of same-sheet pairs whose positive scores higher
    ens = ensemble(w, [f"a_f{k}" for k in range(5)])
    pos = meta[(meta.label == 1) & (meta.fold >= 0)].set_index("pair_id")
    ng = neg.set_index("pair_id")
    keep = ng.index[ng.sheet == pos.loc[ng.index, "sheet"]]
    credit = (ens.loc[pos.loc[keep, "id"]].to_numpy() > ens.loc[ng.loc[keep, "id"]].to_numpy()).mean()
    assert same["accuracy"] == pytest.approx(credit, abs=TOL)
    assert same["sheet_ci"]["low"] <= same["accuracy"] <= same["sheet_ci"]["high"]


def test_breakdowns(complete):
    w, m = complete
    bd = m["tests"]["b"]["breakdowns"]
    test_b = w.meta[(w.meta.group_b == "test") & (w.meta.source != "background")]
    assert set(bd["by_kraj"]) == set(test_b.kraj)
    assert set(bd["by_flysch"]) == {"flysch", "non-flysch"}
    jm = bd["by_kraj_flysch"]["Jihomoravský | non-flysch"]
    assert m["secondary"]["jihomoravsky_nonflysch"]["auc"] == jm["auc"]
    sub = test_b[(test_b.kraj == "Jihomoravský") & ~test_b.flysch]
    e = ensemble(w, ["b"]).loc[sub.id]
    assert jm["auc"] == pytest.approx(roc_auc_score(sub.label, e.to_numpy()), abs=TOL)
    assert set(bd["by_source"]) == {"random", "quarry", "rock", "cut"}
    q = bd["by_source"]["quarry"]
    assert q["n"]["negatives"] == int((test_b.source == "quarry").sum())
    assert q["n"]["positives"] == int((test_b.label == 1).sum())
    assert set(m["tests"]["a"]["breakdowns"]) == {"by_source"}
    slope = m["secondary"]["slope_only_check"]["a"]
    a_items = w.meta[(w.meta.fold >= 0) & (w.meta.source != "background")]
    assert slope["auc"] == pytest.approx(roc_auc_score(a_items.label, a_items.slope_mean), abs=TOL)


def test_stage2_metrics(complete):
    w, m = complete
    s2 = m["tests"]["stage2"]
    dets = pd.read_parquet(w.paths.stage2 / "detections.parquet")
    cnn = dets[dets.scorer == "cnn"]
    n_t = int((w.targets.eligible & w.targets.sheet.isin(stage2_sheets(w.cfg, False))).sum())
    sw = M.pr_sweep(cnn.score, cnn.outcome, n_t, 0.5)
    assert s2["estimate"] == pytest.approx(sw.precision_at_recall, abs=TOL) and s2["reached"] is True
    assert s2["max_recall"] == pytest.approx(sw.max_recall, abs=TOL)
    assert s2["interpolated_precision"] == pytest.approx(sw.interpolated_precision, abs=TOL)
    assert s2["n"]["targets"] == n_t == 20 * 6
    ring = M.pr_sweep(cnn.score, cnn.outcome.replace("ignored", "fp"), n_t, 0.5)
    assert m["secondary"]["stage2_ring_as_fp"]["estimate"] == pytest.approx(ring.precision_at_recall, abs=TOL)
    strata = s2["breakdowns"]["by_stratum"]
    assert strata["flysch"]["n"]["targets"] == strata["nonflysch"]["n"]["targets"] == 60
    assert s2["sheet_ci"]["post_hoc"] is True and s2["block_ci"] is None
    assert set(m["baselines"]["stage2"]["scorers"]) == {"random", "slope_susc_lr"}
    assert set(m["paired_diffs"]["stage2"]["sheet"]) == {"random", "slope_susc_lr"}


def test_schema(complete):
    _, m = complete
    for key in ("schema_version", "config_hash", "git_commit", "verdict", "verdict_detail", "integrity", "tests",
                "baselines", "paired_diffs", "secondary", "spike_claims", "discrepancies", "bootstrap", "figures"):
        assert key in m
    for t in ("a", "b", "stage2"):
        for key in ("status", "estimate", "sheet_ci", "block_ci", "threshold", "pass", "borderline", "n",
                    "per_fold", "per_seed", "breakdowns", "complete", "missing_units", "problems"):
            assert key in m["tests"][t], (t, key)
    assert set(m["spike_claims"]) == set(E.SPIKE_CLAIMS)


def test_metrics_table_lists_claims_and_flags(complete):
    w, m = complete
    md = (w.paths.results / "metrics_table.md").read_text(encoding="utf-8")
    assert "**Verdict: REPLICATED**" in md
    assert "0.811" in md and "[0.796, 0.826]" in md
    # the synthetic CNN is far better than the spike: a discrepancy, which does not change the verdict
    assert "a_cnn" in m["discrepancies"]["flagged"] and m["verdict"] == "REPLICATED"
    assert "**yes**" in md


def test_learning_curve_figure(complete):
    w, m = complete
    png = (w.paths.figures / "learning_curves.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert m["figures"]["learning_curves"]["runs"] == ["a_f0", "a_f1", "a_f2", "a_f3", "a_f4", "b", "s2"]
    with pytest.raises(ValueError):
        learning_curves_png({}, {})


def test_stage2_allowed(complete):
    _, m = complete
    assert E.stage2_allowed(m)


# ------------------------------------------------------------------------------------------------ integrity


def test_missing_unit_is_incomplete(tmp_path):
    w = build_world(tmp_path, unit_status={"a_f2_s1": "missing"})
    m = evaluate(w)
    a = m["tests"]["a"]
    assert a["status"] == "INCOMPLETE" and a["estimate"] is None and a["pass"] is None
    assert a["missing_units"] == {"a_f2_s1": "missing"}
    assert m["integrity"]["missing"] == ["a_f2_s1"] and not m["integrity"]["ok"]
    assert m["tests"]["b"]["status"] == "PASS"
    assert m["verdict"] == "INCOMPLETE"
    # what is available is still reported, but never as the test's result
    assert a["per_fold"]["2"]["complete"] is False and a["per_fold"]["2"]["auc"] is None
    assert a["per_fold"]["0"]["auc"] is not None
    assert a["per_seed"]["1"]["auc"] is None and a["per_seed"]["0"]["auc"] is not None
    assert "a_f2_s1" not in a["per_unit"]
    # baselines are still evaluated on the full test set
    assert m["baselines"]["a"]["status"] == "ok" and m["paired_diffs"]["a"]["sheet"] == {}


def test_failed_and_interrupted_units(tmp_path):
    w = build_world(tmp_path, unit_status={"a_f0_s0": "failed", "b_s2": "interrupted"})
    m = evaluate(w)
    integ = m["integrity"]
    assert integ["failed"] == ["a_f0_s0"] and integ["interrupted"] == ["b_s2"]
    assert integ["units"]["a_f0_s0"]["error"] == "RuntimeError: boom"
    assert integ["units"]["b_s2"]["epochs_completed"] == 2
    assert m["tests"]["a"]["status"] == m["tests"]["b"]["status"] == "INCOMPLETE"
    assert m["verdict"] == "INCOMPLETE"


def test_completed_failure_beats_missing_runs(tmp_path):
    s2_missing = {f"s2_s{s}": "missing" for s in (0, 1, 2)}
    w = build_world(tmp_path, strength_b=-1.0, stage2=None, comparators=False,
                    unit_status={"a_f1_s0": "missing", **s2_missing})
    m = evaluate(w)
    assert m["tests"]["a"]["status"] == "INCOMPLETE"
    assert m["tests"]["b"]["status"] == "FAIL" and m["tests"]["b"]["pass"] is False
    assert m["tests"]["stage2"]["status"] == "NOT_RUN"
    assert m["verdict"] == "NOT REPLICATED"
    assert not E.stage2_allowed(m)


def test_hash_mismatch_unit(tmp_path):
    w = build_world(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "landslide.sha256").write_text(f"{w.chash}  configs/landslide.json\n", encoding="ascii")
    p = w.paths.run_dir("b", 1) / "done.json"
    done = json.loads(p.read_text(encoding="utf-8"))
    done["config_hash"] = "deadbeef"
    write_json(p, done)
    m = evaluate(w)
    assert m["integrity"]["config_hash_source"] == "frozen"
    assert m["integrity"]["hash_mismatch"] == ["b_s1"]
    assert m["tests"]["b"]["status"] == "INCOMPLETE" and m["tests"]["b"]["missing_units"] == {"b_s1": "hash_mismatch"}
    assert m["verdict"] == "INCOMPLETE"


def test_config_differs_from_frozen_hash(tmp_path):
    w = build_world(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "landslide.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(E.ConfigMismatchError):
        evaluate(w)


@pytest.mark.parametrize("breakage", ["n_test", "both_markers", "no_preds", "smoke_flag", "bad_prob"])
def test_invalid_units(tmp_path, breakage):
    w = build_world(tmp_path, stage2=None, comparators=False)
    d = w.paths.run_dir("a_f3", 2)
    done = json.loads((d / "done.json").read_text(encoding="utf-8"))
    if breakage == "n_test":
        done["n_test"] += 1
        write_json(d / "done.json", done)
    elif breakage == "both_markers":
        write_json(d / "failed.json", {"error": "x"})
    elif breakage == "no_preds":
        (d / "preds_test.csv").unlink()
    elif breakage == "smoke_flag":
        done["smoke"] = True
        write_json(d / "done.json", done)
    else:
        df = pd.read_csv(d / "preds_test.csv")
        df.loc[0, "prob"] = 1.5
        df.to_csv(d / "preds_test.csv", index=False)
    m = evaluate(w)
    assert m["integrity"]["invalid"] == ["a_f3_s2"]
    assert m["integrity"]["units"]["a_f3_s2"]["problems"]
    assert m["tests"]["a"]["status"] == "INCOMPLETE"


def test_predictions_must_cover_the_test_set(tmp_path):
    w = build_world(tmp_path, stage2=None, comparators=False)
    for s in (0, 1, 2):  # all seeds drop the same patch, so the ensemble itself is consistent
        d = w.paths.run_dir("a_f4", s)
        df = pd.read_csv(d / "preds_test.csv").iloc[1:]
        df.to_csv(d / "preds_test.csv", index=False)
        done = json.loads((d / "done.json").read_text(encoding="utf-8"))
        done["n_test"] = len(df)
        write_json(d / "done.json", done)
    m = evaluate(w)
    assert m["integrity"]["ok"]
    assert m["tests"]["a"]["status"] == "INCOMPLETE"
    assert "cover" in m["tests"]["a"]["problems"][0]


def test_seed_disagreement_is_incomplete(tmp_path):
    w = build_world(tmp_path, stage2=None, comparators=False)
    d = w.paths.run_dir("b", 0)
    df = pd.read_csv(d / "preds_test.csv").iloc[2:]
    df.to_csv(d / "preds_test.csv", index=False)
    done = json.loads((d / "done.json").read_text(encoding="utf-8"))
    done["n_test"] = len(df)
    write_json(d / "done.json", done)
    m = evaluate(w)
    assert m["tests"]["b"]["status"] == "INCOMPLETE"
    assert "disagree" in m["tests"]["b"]["problems"][0]


def test_ensemble_does_not_depend_on_row_order(tmp_path):
    w = build_world(tmp_path)
    evaluate(w)
    before = (w.paths.results / "metrics.json").read_bytes()
    p = w.paths.run_dir("a_f1", 2) / "preds_test.csv"
    pd.read_csv(p).sample(frac=1.0, random_state=3).to_csv(p, index=False)
    evaluate(w)
    assert (w.paths.results / "metrics.json").read_bytes() == before


# ------------------------------------------------------------------------------------------------ baselines


def test_missing_baselines_do_not_change_the_verdict(tmp_path):
    w = build_world(tmp_path, baselines=False)
    m = evaluate(w)
    assert m["baselines"]["a"]["status"] == m["baselines"]["b"]["status"] == "missing"
    assert m["paired_diffs"]["a"] == {"sheet": {}, "block": {}}
    assert m["verdict"] == "REPLICATED"
    assert m["discrepancies"]["comparisons"][3]["note"] == "not available in this evaluation"


def test_baseline_id_mismatch(tmp_path):
    w = build_world(tmp_path)
    p = w.paths.baselines_dir / "preds_b.csv"
    pd.read_csv(p).iloc[1:].to_csv(p, index=False)
    m = evaluate(w)
    assert m["baselines"]["b"]["status"] == "id_mismatch" and m["baselines"]["b"]["n_missing_ids"] == 1
    assert m["paired_diffs"]["b"] == {"sheet": {}, "block": {}}
    assert m["baselines"]["a"]["status"] == "ok"
    assert m["verdict"] == "REPLICATED"


# ------------------------------------------------------------------------------------------------ stage 2


def test_stage2_not_reached_is_a_fail(tmp_path):
    w = build_world(tmp_path, stage2="not_reached")
    m = evaluate(w)
    s2 = m["tests"]["stage2"]
    assert s2["complete"] and s2["reached"] is False and s2["estimate"] is None
    assert s2["status"] == "FAIL" and s2["pass"] is False
    assert m["verdict"] == "NOT REPLICATED"
    assert "not reached" in (w.paths.results / "metrics_table.md").read_text(encoding="utf-8")


def test_stage2_low_precision_is_a_fail(tmp_path):
    w = build_world(tmp_path, stage2="fail")
    m = evaluate(w)
    assert m["tests"]["stage2"]["reached"] is True and m["tests"]["stage2"]["estimate"] < 0.30
    assert m["tests"]["stage2"]["status"] == "FAIL" and m["verdict"] == "NOT REPLICATED"


def test_stage2_not_run(tmp_path):
    w = build_world(tmp_path, stage2=None, unit_status={f"s2_s{s}": "missing" for s in (0, 1, 2)})
    m = evaluate(w)
    assert m["tests"]["stage2"]["status"] == "NOT_RUN"
    assert m["verdict"] == "INCOMPLETE"  # (a) and (b) passed, so stage 2 should have run
    # the phase-1 comparators are still reported
    assert m["baselines"]["stage2"]["scorers"]["random"]["status"] == "ok"
    assert m["paired_diffs"]["stage2"]["sheet"] == {}


def test_stage2_units_without_detections_is_incomplete(tmp_path):
    w = build_world(tmp_path, stage2=None)
    m = evaluate(w)
    assert m["tests"]["stage2"]["status"] == "INCOMPLETE"
    assert any("no CNN detections" in p for p in m["tests"]["stage2"]["problems"])


def test_stage2_missing_unit_is_incomplete(tmp_path):
    w = build_world(tmp_path, unit_status={"s2_s1": "failed"})
    m = evaluate(w)
    assert m["tests"]["stage2"]["status"] == "INCOMPLETE" and m["tests"]["stage2"]["estimate"] is None
    assert m["verdict"] == "INCOMPLETE"


def test_stage2_missing_sheet_is_incomplete(tmp_path):
    w = build_world(tmp_path)
    p = w.paths.stage2 / "detections.parquet"
    d = pd.read_parquet(p)
    drop = stage2_sheets(w.cfg, False)[0]
    d[~((d.scorer == "cnn") & (d.sheet == drop))].to_parquet(p, index=False)
    m = evaluate(w)
    assert m["tests"]["stage2"]["status"] == "INCOMPLETE"
    assert any(drop in p_ for p_ in m["tests"]["stage2"]["problems"])


def test_stage2_bootstrap_counts_not_reached_as_zero(tmp_path):
    """Point estimate reaches 50 % recall; some sheet resamples do not. They count as precision 0 (kept)."""
    cfg = make_cfg(n_boot=300)
    w = build_world(tmp_path, cfg, stage2=None, comparators=False)
    sheets = stage2_sheets(cfg, False)
    fr = [1.0 if i % 2 == 0 else 1 / 6 for i in range(len(sheets))]  # 7 of 12 targets per sheet pair -> 0.58
    det = synth_detections("cnn", sheets, w.targets, tp_frac=fr, n_fp=3, quality=3.0, seed=7)
    det.to_parquet(w.paths.stage2 / "detections.parquet", index=False)
    m = evaluate(w)
    s2 = m["tests"]["stage2"]
    assert s2["reached"] is True and s2["status"] == "PASS"

    # independent replicates: explicit resamples (each drawn sheet's detections repeated), no weights
    spec = M.bootstrap_spec(cfg)
    draws = M.draw_clusters(np.array(sorted(sheets)), spec.n_replicates, spec.seed)
    nt = w.targets[w.targets.eligible & w.targets.sheet.isin(sheets)].groupby("sheet").size()
    by_sheet = {s: g for s, g in det.groupby("sheet")}
    reps, reached = [], []
    for b in range(draws.n_replicates):
        drawn = [draws.labels[k] for k in draws.draw_idx[b]]
        rs = pd.concat([by_sheet[s] for s in drawn], ignore_index=True)
        sw = M.pr_sweep(rs.score, rs.outcome, int(sum(nt[s] for s in drawn)), 0.5)
        reached.append(sw.reached)
        reps.append(sw.precision_at_recall if sw.reached else 0.0)
    reps, reached = np.array(reps), np.array(reached)
    assert 0 < (~reached).sum() < len(reps)
    lo, hi = M.percentile_ci(reps, spec.ci)
    ci = s2["sheet_ci"]
    assert ci["n_not_reached"] == int((~reached).sum())
    assert ci["n_used"] == spec.n_replicates and ci["n_degenerate"] == 0
    assert ci["low"] == pytest.approx(lo, abs=TOL) and ci["high"] == pytest.approx(hi, abs=TOL)
    # dropping those replicates instead would move the lower bound up (the bias the rule avoids)
    lo_drop, _ = M.percentile_ci(reps[reached], spec.ci)
    assert lo_drop > ci["low"]
    assert s2["borderline"] is V.is_borderline(lo, hi, 0.30)


def test_tp_counts_on_its_target_sheet(tmp_path):
    """A TP whose target is assigned to another reserved sheet is attributed to that sheet in the bootstrap, so a
    resample can never hold more TPs than targets."""
    cfg = make_cfg(n_boot=200)
    w = build_world(tmp_path, cfg, stage2=None, comparators=False)
    sheets = stage2_sheets(cfg, False)
    det = synth_detections("cnn", sheets, w.targets, tp_frac=0.8, n_fp=3, quality=3.0, seed=4)
    a, b = sheets[0], sheets[1]
    tp_b = det.index[(det.sheet == b) & (det.outcome == "tp")][0]
    det.loc[tp_b, "sheet"] = a  # the detection lies on sheet a, its target on sheet b
    det.to_parquet(w.paths.stage2 / "detections.parquet", index=False)
    frame = E.S2Frame.build(sheets, E.stage2_sheet_strata(cfg), w.targets)
    _, _, idx = E._scorer_arrays(det, frame)
    assert frame.sheets[idx[det.index.get_loc(tp_b)]] == b
    m = evaluate(w)
    assert m["tests"]["stage2"]["status"] == "PASS"


def test_stage2_bad_detections_are_incomplete_not_a_crash(tmp_path):
    w = build_world(tmp_path)
    p = w.paths.stage2 / "detections.parquet"
    d = pd.read_parquet(p)
    i = d.index[(d.scorer == "cnn") & (d.outcome == "tp")][0]
    d.loc[i, "target_objectid"] = 123  # not a target
    d.to_parquet(p, index=False)
    m = evaluate(w)
    assert m["tests"]["stage2"]["status"] == "INCOMPLETE"
    assert any("unusable" in p_ for p_ in m["tests"]["stage2"]["problems"])


def test_stage2_targets_can_be_passed_in(tmp_path):
    w = build_world(tmp_path)
    m1 = evaluate(w)
    (w.paths.polygons_parquet).unlink()
    m2 = evaluate(w, targets=w.targets[w.targets.eligible][["objectid", "sheet"]])
    assert m1["tests"]["stage2"] == m2["tests"]["stage2"]


# ------------------------------------------------------------------------------------------------ outputs


def test_byte_identical_reruns(tmp_path):
    w = build_world(tmp_path)
    files = [w.paths.results / "metrics.json", w.paths.results / "metrics_table.md",
             w.paths.figures / "learning_curves.png"]
    evaluate(w)
    first = [f.read_bytes() for f in files]
    for f in files:
        f.unlink()
    evaluate(w)
    assert [f.read_bytes() for f in files] == first
    text = first[0].decode("utf-8")
    assert "\r\n" not in text and text.endswith("}\n")
    assert json.loads(text) == json.loads(E.dumps_metrics(json.loads(text)))


def test_json_canonical_form():
    out = json.loads(E.dumps_metrics({"b": 0.12345678, "a": [float("nan"), -0.0000001, np.float32(0.5),
                                                              np.int64(3), True]}))
    assert out == {"a": [None, 0.0, 0.5, 3, True], "b": 0.123457}
    text = E.dumps_metrics({"b": 1, "a": 2}).decode()
    assert text.index('"a"') < text.index('"b"')


def test_spike_claims_are_the_briefs():
    brief = (REPO_ROOT / "docs" / "BRIEF.md").read_text(encoding="utf-8")
    for key, c in E.SPIKE_CLAIMS.items():
        assert c["brief_text"] in brief, key
        assert f"{c['value']:.3f}" in c["brief_text"], key
        if c["ci"]:
            assert f"[{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]" in c["brief_text"], key


def test_discrepancy_flag_boundaries():
    def flag(x):
        comps = E.compare_with_spike({"tests": {"a": {"estimate": x}}})["comparisons"]
        return next(c for c in comps if c["id"] == "a_cnn")["outside_spike_ci"]
    assert flag(0.811) is False
    assert flag(0.826) is False and flag(0.796) is False  # the bounds are inside
    assert flag(0.8261) is True and flag(0.79) is True
    assert flag(None) is None
    comps = E.compare_with_spike({"secondary": {"pair_accuracy": {"a": {"same_sheet": {"accuracy": 0.5}}}}})
    assert next(c for c in comps["comparisons"] if c["id"] == "a_pair_accuracy")["outside_spike_ci"] is None


# ------------------------------------------------------------------------------------------------ smoke mode


def _snapshot(root):
    if not root.exists():
        return {}
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_smoke_reads_and_writes_only_smoke_locations(tmp_path, monkeypatch):
    """scripts/evaluate.py --smoke with the real config: outputs only under data/smoke/, nothing in the tracked
    results/ or figures/ (or logs/)."""
    data = tmp_path / "data"
    monkeypatch.setenv("LANDSLIDE_DATA_DIR", str(data))
    cfg = load_config()
    w = build_world(REPO_ROOT, cfg, smoke=True)
    assert w.paths.runs == data / "smoke" / "runs"
    # a full-mode run directory must not be read in smoke mode
    full = Paths(smoke=False)
    write_json(full.run_dir("a_f0", 0) / "done.json", {"config_hash": "nope"})
    tracked = {d: _snapshot(REPO_ROOT / d) for d in ("results", "figures", "logs")}
    before = _snapshot(data)
    spec = importlib.util.spec_from_file_location("evaluate_script", REPO_ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        assert mod.main(["--smoke"]) == 0
    finally:
        setup_logging(None, console=False)  # close the log file handler
    for d, snap in tracked.items():
        assert _snapshot(REPO_ROOT / d) == snap, d
    new = set(_snapshot(data)) - set(before)
    assert new and all((data / "smoke") in p.parents for p in new)
    m = json.loads((data / "smoke" / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert m["smoke"] is True
    assert m["integrity"]["n_expected"] == len(cfg["smoke"]["runs"]) * len(cfg["smoke"]["seeds"])
    assert m["integrity"]["ok"]
    assert list(m["tests"]["a"]["per_fold"]) == [str(k) for k in cfg["smoke"]["folds"]]
    assert m["tests"]["stage2"]["sheets"] == stage2_sheets(cfg, True)
    assert m["tests"]["stage2"]["status"] in ("PASS", "FAIL")
    assert (data / "smoke" / "figures" / "learning_curves.png").exists()
    assert (data / "smoke" / "results" / "metrics_table.md").read_text(encoding="utf-8").count("SMOKE TEST") == 1
    assert os.environ["LANDSLIDE_DATA_DIR"] == str(data)
