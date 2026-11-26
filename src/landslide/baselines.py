"""Stage-1 non-ML baselines and classical-ML comparators (BRIEF.md Appendix A §8 and §10; PLAN.md P1-11).

Every rule here is recorded in `docs/decisions/P1-11.md`. In short:

- **Scorers** (higher = more landslide-like). Non-ML baselines, unfitted: `slope_only` = `slope_mean`,
  `susceptibility` = `susc_mean` (the `meta.csv` columns). Classical-ML comparators, fitted: `slope_susc_lr` and
  `terrain_stats_lr` (StandardScaler + LogisticRegression, sklearn defaults) and `terrain_stats_gbm`
  (HistGradientBoostingClassifier, sklearn defaults), with `random_state` = the config's `replication.seed`. The
  fitted score is `predict_proba` of class 1. No setting is tuned.
- **Rows.** `landslide.train.select_unit_rows(run, meta, splits)` defines each run's train, inner-val and test rows,
  exactly as for the CNN. A fitted scorer trains on that run's train + inner-val rows and predicts exactly its test
  rows. Test (a): one fit per fold `a_f0 … a_f<n-1>`, out-of-fold predictions pooled. Test (b): one fit (`b`).
- **Metrics.** ROC AUC (`landslide.metrics.roc_auc`, ties averaged) with cluster-bootstrap percentile CIs on the
  test rows, once with sheets and once with blocks as the unit (n, seed, CI level from `eval.bootstrap`). The draws
  are made once per subset and shared by all five scorers, so they are the draws P1-13 makes for the same rows.
  Breakdowns: per fold (a); by kraj and by flysch flag (b); every test positive against the test negatives of one
  source (random, then each look-alike type), for both tests. The slope-only AUC is the matching check (about 0.5).
- **Outputs** (written by `scripts/run_baselines.py`): `preds_a.csv` (`id`, `fold`, one column per scorer) and
  `preds_b.csv` (`id`, one column per scorer), one row per test patch in `meta.csv` row order; the breakdown table
  (one row per test x scorer x breakdown group, with a `kind` column) and a summary dict.

Nothing here reads files: `meta`, `splits` and `cfg` are passed in (tests use synthetic ones).
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from landslide import metrics as M
from landslide.train import BACKGROUND, select_unit_rows

__all__ = [
    "KIND_BASELINE",
    "KIND_CLASSICAL_ML",
    "SLOPE_SUSC_FEATURES",
    "TERRAIN_FEATURES",
    "DEFAULT_SPECS",
    "STAT_COLUMNS",
    "SPIKE_BEST_NON_CNN",
    "BaselineSpec",
    "FitRecord",
    "UnitPredictions",
    "PooledPredictions",
    "BaselineResults",
    "baseline_specs",
    "random_state",
    "bootstrap_units",
    "negative_sources",
    "make_estimator",
    "estimator_settings",
    "feature_matrix",
    "predict_unit",
    "predict_test",
    "table_columns",
    "breakdown_groups",
    "evaluate_test",
    "run_baselines",
]

KIND_BASELINE = "baseline"
KIND_CLASSICAL_ML = "classical_ml"
_MODELS = ("column", "lr", "gbm")
_KIND_OF_MODEL = {"column": KIND_BASELINE, "lr": KIND_CLASSICAL_ML, "gbm": KIND_CLASSICAL_ML}

#: Appendix A §10: slope mean and std, susceptibility mean and high-class fraction.
SLOPE_SUSC_FEATURES = ("slope_mean", "slope_std", "susc_mean", "susc_high_frac")
#: Appendix A §10: the 12 terrain statistics (the `meta.csv` column names of PLAN.md §4.3).
TERRAIN_FEATURES = ("slope_mean", "slope_std", "slope_p90", "lrm_s_std", "lrm_l_std", "curv_prof_std",
                    "curv_plan_std", "svf_mean", "svf_std", "detr_std", "detr_range_2_98", "tri")
#: Every statistic column of `meta.csv` (PLAN.md §4.3).
STAT_COLUMNS = TERRAIN_FEATURES + ("susc_mean", "susc_high_frac")

#: The spike's best non-CNN results (BRIEF.md "The spike's claims"). Reference only: never used to change anything.
SPIKE_BEST_NON_CNN = {
    "a": {"baseline": "terrain_stats_lr", "auc": 0.695, "ci_low": 0.678, "ci_high": 0.713},
    "b": {"baseline": "terrain_stats_gbm", "auc": 0.624, "ci_low": 0.603, "ci_high": 0.647},
}

#: Secondary bootstrap unit when the config has no `eval.bootstrap.secondary_unit` (P1-05 note; Appendix A §8).
_SECONDARY_UNIT_DEFAULT = "block"


@dataclass(frozen=True)
class BaselineSpec:
    """One scorer: `model` is "column" (the score is `features[0]` as it is), "lr" or "gbm"."""

    name: str
    model: str
    features: tuple[str, ...]

    @property
    def kind(self) -> str:
        return _KIND_OF_MODEL[self.model]

    @property
    def fitted(self) -> bool:
        return self.model != "column"


DEFAULT_SPECS = (
    BaselineSpec("slope_only", "column", ("slope_mean",)),
    BaselineSpec("susceptibility", "column", ("susc_mean",)),
    BaselineSpec("slope_susc_lr", "lr", SLOPE_SUSC_FEATURES),
    BaselineSpec("terrain_stats_lr", "lr", TERRAIN_FEATURES),
    BaselineSpec("terrain_stats_gbm", "gbm", TERRAIN_FEATURES),
)


def baseline_specs(cfg: Mapping[str, Any]) -> tuple[BaselineSpec, ...]:
    """The scorers: `DEFAULT_SPECS` (Appendix A §10), or `replication.baselines.scorers` if the config has it.

    The config form is a list of {"name", "model", "features"}. Names must be unique, models one of "column", "lr",
    "gbm" (a "column" scorer has exactly one feature) and features `meta.csv` statistic columns.
    """
    block = cfg.get("replication", {}).get("baselines", {}).get("scorers")
    if block is None:
        return DEFAULT_SPECS
    specs = tuple(BaselineSpec(str(s["name"]), str(s["model"]), tuple(str(f) for f in s["features"])) for s in block)
    names = [s.name for s in specs]
    if len(set(names)) != len(names) or not specs:
        raise ValueError(f"replication.baselines.scorers: names must be unique and non-empty, got {names}")
    for s in specs:
        if s.model not in _MODELS:
            raise ValueError(f"scorer {s.name!r}: model {s.model!r} is not one of {_MODELS}")
        if not s.features or (s.model == "column" and len(s.features) != 1):
            raise ValueError(f"scorer {s.name!r}: bad feature list {s.features}")
        bad = [f for f in s.features if f not in STAT_COLUMNS]
        if bad:
            raise ValueError(f"scorer {s.name!r}: {bad} are not meta.csv statistic columns")
    return specs


def random_state(cfg: Mapping[str, Any]) -> int:
    """The fixed `random_state` of every fitted scorer: the config's `replication.seed`."""
    return int(cfg["replication"]["seed"])


def bootstrap_units(cfg: Mapping[str, Any]) -> tuple[str, str]:
    """(primary, secondary) cluster units: `eval.bootstrap.unit` ("sheet") and the secondary unit ("block")."""
    b = cfg["eval"]["bootstrap"]
    units = (str(b["unit"]), str(b.get("secondary_unit", _SECONDARY_UNIT_DEFAULT)))
    if units[0] == units[1]:
        raise ValueError(f"the two bootstrap units must differ, got {units}")
    return units


def negative_sources(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Matched-negative sources for the per-source breakdown: "random", then `sampling.lookalike_types`."""
    return ("random", *(str(t) for t in cfg["sampling"]["lookalike_types"]))


# ---------------------------------------------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------------------------------------------

def make_estimator(spec: BaselineSpec, seed: int) -> Any:
    """A fresh, unfitted estimator for a fitted scorer. Every setting not named here is sklearn's default."""
    if spec.model == "lr":
        return Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(random_state=seed))])
    if spec.model == "gbm":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"scorer {spec.name!r} ({spec.model}) is not fitted")


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def estimator_settings(spec: BaselineSpec, seed: int) -> dict[str, Any]:
    """Every parameter of the estimator (sklearn `get_params`), per pipeline step, for the record."""
    if not spec.fitted:
        return {"score": f"meta.{spec.features[0]} as it is (no fitting)"}
    est = make_estimator(spec, seed)
    steps = est.steps if isinstance(est, Pipeline) else [(type(est).__name__, est)]
    return {name: {"class": type(step).__name__, "params": {k: _jsonable(v) for k, v in
                                                            sorted(step.get_params(deep=False).items())}}
            for name, step in steps}


# ---------------------------------------------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------------------------------------------

def feature_matrix(meta: pd.DataFrame, rows: np.ndarray, features: Sequence[str]) -> np.ndarray:
    """`meta[features]` at `rows` as float64 (n, k). Missing columns or non-finite values raise."""
    missing = [f for f in features if f not in meta.columns]
    if missing:
        raise KeyError(f"meta lacks statistic columns {missing}")
    x = meta[list(features)].iloc[rows].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        bad = [f for j, f in enumerate(features) if not np.isfinite(x[:, j]).all()]
        raise ValueError(f"non-finite values in {bad} among {len(rows)} rows")
    return x


@dataclass
class FitRecord:
    """What happened when one scorer was fitted for one run."""

    run: str
    baseline: str
    model: str
    n_fit: int
    n_fit_pos: int
    n_train: int
    n_inner_val: int
    n_test: int
    converged: bool | None
    n_iter: int | None
    early_stopping: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"run": self.run, "baseline": self.baseline, "model": self.model, "n_fit": self.n_fit,
                "n_fit_pos": self.n_fit_pos, "n_fit_neg": self.n_fit - self.n_fit_pos, "n_train": self.n_train,
                "n_inner_val": self.n_inner_val, "n_test": self.n_test, "converged": self.converged,
                "n_iter": self.n_iter, "early_stopping": self.early_stopping, "warnings": list(self.warnings)}


@dataclass
class UnitPredictions:
    """One run's fit rows (train + inner-val), test rows and test scores per scorer (aligned with `test_rows`)."""

    run: str
    fit_rows: np.ndarray
    train_rows: np.ndarray
    val_rows: np.ndarray
    test_rows: np.ndarray
    scores: dict[str, np.ndarray]
    fits: list[FitRecord]


def _fit_predict(spec: BaselineSpec, x_fit: np.ndarray, y_fit: np.ndarray, x_test: np.ndarray,
                 seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    est = make_estimator(spec, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(x_fit, y_fit)
        proba = est.predict_proba(x_test)
    final = est.steps[-1][1] if isinstance(est, Pipeline) else est
    classes = list(final.classes_)
    if classes != [0, 1]:
        raise ValueError(f"{spec.name}: expected classes [0, 1], got {classes}")
    msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    info: dict[str, Any] = {"warnings": msgs}
    if spec.model == "lr":
        info["n_iter"] = int(np.max(final.n_iter_))
        info["converged"] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    else:
        info["n_iter"] = int(final.n_iter_)
        info["converged"] = None
        info["early_stopping"] = bool(final.do_early_stopping_)
    return proba[:, 1].astype(np.float64), info


def predict_unit(run: str, meta: pd.DataFrame, splits: Mapping[str, Any], cfg: Mapping[str, Any],
                 specs: Sequence[BaselineSpec] | None = None) -> UnitPredictions:
    """Score the test rows of `run` with every scorer; fitted scorers train on the run's train + inner-val rows."""
    specs = tuple(specs) if specs is not None else baseline_specs(cfg)
    rows = select_unit_rows(run, meta, dict(splits))
    fit_rows = np.union1d(rows.train, rows.val)
    if np.intersect1d(fit_rows, rows.test).size:
        raise AssertionError(f"{run}: fit rows and test rows overlap")
    if rows.test.size == 0:
        raise ValueError(f"{run}: no test rows")
    y = meta["label"].to_numpy().astype(np.int64)
    seed = random_state(cfg)
    scores: dict[str, np.ndarray] = {}
    fits: list[FitRecord] = []
    for spec in specs:
        if not spec.fitted:
            scores[spec.name] = feature_matrix(meta, rows.test, spec.features)[:, 0]
            continue
        p, info = _fit_predict(spec, feature_matrix(meta, fit_rows, spec.features), y[fit_rows],
                               feature_matrix(meta, rows.test, spec.features), seed)
        if not np.isfinite(p).all():
            raise ValueError(f"{run} {spec.name}: non-finite predictions")
        scores[spec.name] = p
        fits.append(FitRecord(run=run, baseline=spec.name, model=spec.model, n_fit=int(fit_rows.size),
                              n_fit_pos=int(y[fit_rows].sum()), n_train=int(rows.train.size),
                              n_inner_val=int(rows.val.size), n_test=int(rows.test.size),
                              converged=info["converged"], n_iter=info["n_iter"],
                              early_stopping=info.get("early_stopping"), warnings=info["warnings"]))
    return UnitPredictions(run=run, fit_rows=fit_rows, train_rows=rows.train, val_rows=rows.val,
                           test_rows=rows.test, scores=scores, fits=fits)


def _runs_of_test(test: str, cfg: Mapping[str, Any]) -> list[str]:
    if test == "a":
        return [f"a_f{k}" for k in range(int(cfg["splits"]["n_folds_a"]))]
    if test == "b":
        return ["b"]
    raise ValueError(f"unknown test {test!r} (expected 'a' or 'b')")


@dataclass
class PooledPredictions:
    """A test's pooled predictions: positional `rows` into meta (ascending), the run of each row and the scores."""

    test: str
    rows: np.ndarray
    run: np.ndarray
    scores: dict[str, np.ndarray]
    units: list[UnitPredictions]

    @property
    def fold(self) -> np.ndarray:
        """The (a) fold of each row (the k of run a_f<k>); -1 for other runs."""
        return np.array([int(m.group(1)) if (m := re.fullmatch(r"a_f(\d+)", r)) else -1 for r in self.run],
                        dtype=np.int64)

    def to_frame(self, meta: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
        """The `preds_<test>.csv` table: `id`, (`fold` for test a), then one column per scorer."""
        cols: dict[str, Any] = {"id": meta["id"].to_numpy()[self.rows]}
        if self.test == "a":
            cols["fold"] = self.fold
        for n in names:
            cols[n] = self.scores[n]
        return pd.DataFrame(cols)


def predict_test(test: str, meta: pd.DataFrame, splits: Mapping[str, Any], cfg: Mapping[str, Any],
                 specs: Sequence[BaselineSpec] | None = None) -> PooledPredictions:
    """Test (a): every fold's run, out-of-fold predictions pooled; test (b): the `b` run. Rows in meta order."""
    specs = tuple(specs) if specs is not None else baseline_specs(cfg)
    units = [predict_unit(run, meta, splits, cfg, specs) for run in _runs_of_test(test, cfg)]
    rows = np.concatenate([u.test_rows for u in units])
    if np.unique(rows).size != rows.size:
        raise AssertionError(f"test {test}: a row is in the test set of two runs")
    order = np.argsort(rows, kind="stable")
    run = np.concatenate([np.full(u.test_rows.size, u.run, dtype=object) for u in units])[order]
    scores = {s.name: np.concatenate([u.scores[s.name] for u in units])[order] for s in specs}
    return PooledPredictions(test=test, rows=rows[order], run=run, scores=scores, units=units)


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------

#: Columns of `results/baselines.csv`: these, then `<unit>_<c>` for each bootstrap unit (sheet, then block) and each
#: c in `_PER_UNIT`, then `_TAIL`.
_HEAD = ("test", "baseline", "kind", "breakdown", "group", "n", "n_pos", "n_neg", "n_sheets", "n_blocks", "auc")
_PER_UNIT = ("ci_low", "ci_high", "n_used", "n_degenerate")
_TAIL = ("spike_auc", "spike_ci_low", "spike_ci_high", "note")


def table_columns(cfg: Mapping[str, Any]) -> list[str]:
    """The columns of `results/baselines.csv`, in order."""
    return [*_HEAD, *(f"{u}_{c}" for u in bootstrap_units(cfg) for c in _PER_UNIT), *_TAIL]


def _flag(values: pd.Series) -> np.ndarray:
    """A boolean column that may have been read from CSV as bool, 0/1 or "True"/"False"."""
    if values.dtype == bool:
        return values.to_numpy()
    s = values.astype(str).str.strip().str.lower()
    ok = s.isin(["true", "false", "1", "0", "1.0", "0.0"])
    if not ok.all():
        raise ValueError(f"cannot read flags {sorted(set(values[~ok].astype(str)))[:5]} as booleans")
    return s.isin(["true", "1", "1.0"]).to_numpy()


def breakdown_groups(test: str, meta: pd.DataFrame, tp: PooledPredictions,
                     cfg: Mapping[str, Any]) -> list[tuple[str, str, np.ndarray]]:
    """(breakdown, group, boolean mask over `tp.rows`) for every subset reported for `test`.

    Both tests: ("overall", "all"), and ("source", s) = every test positive plus the test negatives of source s, for
    s in `negative_sources(cfg)`. Test a: ("fold", "k") per fold. Test b: ("kraj", name) per kraj of the test rows
    (sorted) and ("flysch", "flysch" | "nonflysch").
    """
    n = tp.rows.size
    label = meta["label"].to_numpy()[tp.rows].astype(bool)
    source = meta["source"].astype(str).to_numpy()[tp.rows]
    out: list[tuple[str, str, np.ndarray]] = [("overall", "all", np.ones(n, dtype=bool))]
    if test == "a":
        fold = tp.fold
        out += [("fold", str(k), fold == k) for k in range(int(cfg["splits"]["n_folds_a"]))]
    elif test == "b":
        kraj = meta["kraj"].astype(str).to_numpy()[tp.rows]
        out += [("kraj", str(k), kraj == k) for k in sorted(set(kraj.tolist()))]
        fl = _flag(meta["flysch"].iloc[tp.rows])
        out += [("flysch", "flysch", fl), ("flysch", "nonflysch", ~fl)]
    out += [("source", s, label | (source == s)) for s in negative_sources(cfg)]
    return out


def evaluate_test(test: str, meta: pd.DataFrame, tp: PooledPredictions, cfg: Mapping[str, Any],
                  specs: Sequence[BaselineSpec] | None = None) -> pd.DataFrame:
    """One table row per scorer x breakdown group of `test`, with AUC and both bootstrap CIs.

    The cluster draws of a group are made once from the group's own rows (`metrics.draw_clusters_from_cfg` on the
    sheet, then the block labels) and shared by every scorer. A group without both classes gets NaN AUC and a note.
    """
    specs = tuple(specs) if specs is not None else baseline_specs(cfg)
    spec_ci = M.bootstrap_spec(cfg)
    units = bootstrap_units(cfg)
    label = meta["label"].to_numpy().astype(np.int64)[tp.rows]
    unit_labels = {u: meta[u].astype(str).to_numpy()[tp.rows] for u in units}
    sheets = meta["sheet"].astype(str).to_numpy()[tp.rows]
    blocks = meta["block"].astype(str).to_numpy()[tp.rows]
    spike = SPIKE_BEST_NON_CNN.get(test)
    records: list[dict[str, Any]] = []
    for breakdown, group, mask in breakdown_groups(test, meta, tp, cfg):
        y = label[mask]
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        base = {"test": test, "breakdown": breakdown, "group": group, "n": int(mask.sum()), "n_pos": n_pos,
                "n_neg": n_neg, "n_sheets": int(np.unique(sheets[mask]).size),
                "n_blocks": int(np.unique(blocks[mask]).size)}
        draws = None
        note = ""
        if n_pos == 0 or n_neg == 0:
            note = "no rows" if mask.sum() == 0 else "one class only: AUC undefined"
        else:
            draws = {u: M.draw_clusters_from_cfg(unit_labels[u][mask], cfg) for u in units}
        for spec in specs:
            rec: dict[str, Any] = {**base, "baseline": spec.name, "kind": spec.kind, "auc": math.nan}
            for u in units:
                rec.update({f"{u}_ci_low": math.nan, f"{u}_ci_high": math.nan, f"{u}_n_used": 0,
                            f"{u}_n_degenerate": 0})
            if draws is not None:
                s = tp.scores[spec.name][mask]
                rec["auc"] = M.roc_auc(y, s)
                for u in units:
                    r = M.bootstrap_auc(y, s, draws[u], ci=spec_ci.ci)
                    rec.update({f"{u}_ci_low": r.ci_low, f"{u}_ci_high": r.ci_high, f"{u}_n_used": r.n_used,
                                f"{u}_n_degenerate": r.n_degenerate})
            is_ref = spike is not None and breakdown == "overall" and spec.name == spike["baseline"]
            rec.update({"spike_auc": spike["auc"] if is_ref else math.nan,
                        "spike_ci_low": spike["ci_low"] if is_ref else math.nan,
                        "spike_ci_high": spike["ci_high"] if is_ref else math.nan, "note": note})
            records.append(rec)
    return pd.DataFrame(records, columns=table_columns(cfg))


# ---------------------------------------------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class BaselineResults:
    """Predictions per test, the breakdown table, the summary (for `results/baselines.json`) and the fit records."""

    preds: dict[str, pd.DataFrame]
    table: pd.DataFrame
    summary: dict[str, Any]
    fits: list[FitRecord]
    tests: dict[str, PooledPredictions]


def _none_if_nan(v: Any) -> Any:
    if isinstance(v, (float, np.floating)) and math.isnan(float(v)):
        return None
    return v.item() if isinstance(v, np.generic) else v


def _result_entry(row: pd.Series, units: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": row["kind"], "auc": _none_if_nan(row["auc"])}
    for u in units:
        out[f"{u}_ci"] = [_none_if_nan(row[f"{u}_ci_low"]), _none_if_nan(row[f"{u}_ci_high"])]
        out[f"{u}_n_used"] = int(row[f"{u}_n_used"])
        out[f"{u}_n_degenerate"] = int(row[f"{u}_n_degenerate"])
    return out


def _summary(cfg: Mapping[str, Any], specs: Sequence[BaselineSpec], tests: Mapping[str, PooledPredictions],
             table: pd.DataFrame, meta: pd.DataFrame, fits: Sequence[FitRecord]) -> dict[str, Any]:
    units = bootstrap_units(cfg)
    bs = M.bootstrap_spec(cfg)
    seed = random_state(cfg)
    out: dict[str, Any] = {
        "bootstrap": {"n": bs.n_replicates, "seed": bs.seed, "ci": bs.ci, "method": bs.method,
                      "units": list(units), "primary_unit": units[0]},
        "random_state": seed,
        "scorers": {s.name: {"kind": s.kind, "model": s.model, "features": list(s.features),
                             "settings": estimator_settings(s, seed)} for s in specs},
        "tests": {},
        "matching_check": {"scorer": "slope_only", "expected": "about 0.5 (Appendix A §8)"},
        "fits": [f.to_dict() for f in fits],
        "all_lr_fits_converged": all(f.converged for f in fits if f.model == "lr"),
        "spike_reference": {"note": "the spike's best non-CNN result per test (BRIEF.md); reference only, never "
                                    "used to change anything",
                            **{t: dict(v) for t, v in SPIKE_BEST_NON_CNN.items()}},
    }
    label = meta["label"].to_numpy().astype(np.int64)
    source = meta["source"].astype(str).to_numpy()
    for t, tp in tests.items():
        tt = table[table["test"] == t]
        overall = tt[tt["breakdown"] == "overall"].set_index("baseline")
        y = label[tp.rows]
        results = {s.name: _result_entry(overall.loc[s.name], units) for s in specs}
        valid = {k: v for k, v in results.items() if v["auc"] is not None}
        best = max(valid, key=lambda k: valid[k]["auc"]) if valid else None
        entry: dict[str, Any] = {
            "definition": cfg["eval"].get(f"primary_{t}"),
            "runs": [u.run for u in tp.units],
            "n": int(tp.rows.size), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "n_sheets": int(meta["sheet"].astype(str).iloc[tp.rows].nunique()),
            "n_blocks": int(meta["block"].astype(str).iloc[tp.rows].nunique()),
            "n_by_source": {str(k): int(v) for k, v in
                            sorted(pd.Series(source[tp.rows]).value_counts().items())},
            "results": results,
            "best": None if best is None else {"baseline": best, **valid[best]},
        }
        for kind in (KIND_BASELINE, KIND_CLASSICAL_ML):
            vk = {k: v for k, v in valid.items() if v["kind"] == kind}
            bk = max(vk, key=lambda k: vk[k]["auc"]) if vk else None
            entry[f"best_{kind}"] = None if bk is None else {"baseline": bk, "auc": vk[bk]["auc"],
                                                             f"{units[0]}_ci": vk[bk][f"{units[0]}_ci"]}
        if t == "a":
            folds = tt[tt["breakdown"] == "fold"]
            entry["per_fold_auc"] = {g: {r["baseline"]: _none_if_nan(r["auc"]) for _, r in d.iterrows()}
                                     for g, d in folds.groupby("group", sort=True)}
        ref = SPIKE_BEST_NON_CNN.get(t)
        if ref is not None and ref["baseline"] in results and results[ref["baseline"]]["auc"] is not None:
            ours = results[ref["baseline"]]["auc"]
            entry["spike_reference"] = {**ref, "ours": ours,
                                        "ours_inside_spike_ci": bool(ref["ci_low"] <= ours <= ref["ci_high"])}
        out["tests"][t] = entry
        if "slope_only" in results:
            so = results["slope_only"]
            lo, hi = so[f"{units[0]}_ci"]
            mc: dict[str, Any] = {**so, f"{units[0]}_ci_contains_0.5": None if lo is None else bool(lo <= 0.5 <= hi)}
            if t == "a":
                mc["per_fold_auc"] = {g: v["slope_only"] for g, v in entry["per_fold_auc"].items()}
            out["matching_check"][t] = mc
    return out


def run_baselines(meta: pd.DataFrame, splits: Mapping[str, Any], cfg: Mapping[str, Any],
                  tests: Sequence[str] = ("a", "b")) -> BaselineResults:
    """Predict tests (a) and (b) with every scorer, evaluate them, and assemble the outputs (no file I/O)."""
    meta = meta.reset_index(drop=True)
    if not meta["id"].is_unique:
        raise ValueError("meta.id is not unique")
    if (meta["source"].astype(str) == BACKGROUND).any() and not (
            meta.loc[meta["source"].astype(str) == BACKGROUND, "label"] == 0).all():
        raise ValueError("a background row has label 1")
    specs = baseline_specs(cfg)
    names = [s.name for s in specs]
    tps = {t: predict_test(t, meta, splits, cfg, specs) for t in tests}
    table = pd.concat([evaluate_test(t, meta, tps[t], cfg, specs) for t in tests], ignore_index=True)
    fits = [f for t in tests for u in tps[t].units for f in u.fits]
    summary = _summary(cfg, specs, tps, table, meta, fits)
    preds = {t: tps[t].to_frame(meta, names) for t in tests}
    return BaselineResults(preds=preds, table=table, summary=summary, fits=fits, tests=tps)
