"""Evaluation and replication verdict from saved predictions (PLAN.md P1-13; BRIEF.md Appendix A §1, §7-§9; D25).

`run_evaluation(cfg, paths)` reads only saved outputs and writes `results/metrics.json`, `results/metrics_table.md`
and `figures/learning_curves.png` (under `data/smoke/` with `Paths(smoke=True)`). Every rule is recorded in
`docs/decisions/P1-13.md`; the statistics all come from `landslide.metrics` (P1-05).

Inputs:
- run directories `paths.run_dir(run, seed)` (PLAN.md §4.3; P1-06): `done.json` / `failed.json` /
  `interrupted.json`, `history.csv`, `preds_val.csv`, `preds_test.csv`;
- `data/dataset/meta.csv` (labels, sheets, blocks, folds, groups, sources, pairs, match levels, `slope_mean`);
- `data/baselines/preds_a.csv`, `preds_b.csv` (P1-11), optional;
- `<stage2>/detections.parquet` (P1-12), optional, with the stage-2 targets read from
  `data/derived/polygons.parquet` (eligible polygons assigned to the evaluated stage-2 sheets) or passed in.

What it does, in order:
1. **Integrity.** Every expected unit (run x seed) is classified done / failed / interrupted / missing /
   hash_mismatch / invalid. A done unit must carry the expected config hash (`configs/landslide.sha256` once frozen,
   the current config's hash before) and complete, consistent outputs. The evaluation refuses to run when the config
   on disk differs from the frozen hash.
2. **Tests (a) and (b).** Seed ensemble per run, pooled out-of-fold predictions for (a); ROC AUC with sheet
   (primary) and block (secondary) cluster-bootstrap CIs; per fold, per seed and per unit; breakdowns; the
   baselines on the same draws and the paired differences CNN - baseline. A test with any unusable unit, or whose
   predictions do not cover exactly its test set, is INCOMPLETE.
3. **Stage 2.** P@R50, interpolated precision, max recall, per stratum, the ring counted as FP, and the post hoc
   sheet bootstrap of P@R50, in which a replicate that never reaches the recall level counts as precision 0.
4. **Verdict** (`landslide.verdict.decide`), the comparison with the spike's claims and the discrepancy flags.

Output files are byte-identical on re-runs: sorted keys, floats rounded to `FLOAT_DECIMALS`, no timestamps.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from landslide import metrics as M
from landslide import verdict as V
from landslide.config import config_hash, frozen_hash
from landslide.figures import learning_curves
from landslide.logs import get_logger
from landslide.paths import Paths

__all__ = [
    "SCHEMA_VERSION", "FLOAT_DECIMALS", "CNN", "SPIKE_CLAIMS", "ConfigMismatchError", "Unit", "expected_runs",
    "expected_seeds", "read_unit", "load_meta", "meta_test_mask", "run_evaluation", "stage2_allowed",
    "dumps_metrics", "metrics_table_md",
]

log = get_logger("evaluate")

SCHEMA_VERSION = 1
#: Floats in metrics.json are rounded to this many decimals (decisions use the unrounded values).
FLOAT_DECIMALS = 6
CNN = "cnn"
BACKGROUND = "background"
JIHOMORAVSKY = "Jihomoravský"

# Unit statuses
DONE, FAILED, INTERRUPTED, MISSING, HASH_MISMATCH, INVALID = (
    "done", "failed", "interrupted", "missing", "hash_mismatch", "invalid")
UNIT_STATUSES = (DONE, FAILED, INTERRUPTED, MISSING, HASH_MISMATCH, INVALID)
_MARKERS = ("done.json", "failed.json", "interrupted.json")
_UNIT_FILES = ("history.csv", "preds_val.csv", "preds_test.csv")

META_COLUMNS = ("id", "label", "sheet", "kraj", "flysch", "source", "fold", "group_b", "block", "pair_id",
                "match_level", "slope_mean")
S2_COLUMNS = ("scorer", "sheet", "score", "outcome", "target_objectid")

#: Appendix A §10 terminology. Stage-2 scorer names are P1-12's; unknown names are listed as "comparator".
SCORER_KINDS = {
    "slope_only": "non-ML baseline",
    "susceptibility": "non-ML baseline",
    "slope_susc_lr": "classical-ML comparator",
    "terrain_stats_lr": "classical-ML comparator",
    "terrain_stats_gbm": "classical-ML comparator (secondary)",
    "slope": "non-ML baseline",
    "random": "chance level",
}

S2_NOT_REACHED_RULE = ("a bootstrap replicate whose pooled detections never reach the recall level counts as "
                       "precision 0 (a fail at that replicate); it is kept, not dropped")

#: The spike's claims, copied from BRIEF.md "The spike's claims (to verify)". `brief_text` must appear verbatim in
#: docs/BRIEF.md (tested); `ours` is the dotted path of our number in metrics.json.
SPIKE_CLAIMS: dict[str, dict[str, Any]] = {
    "a_cnn": {"claim": "(a) held-out flysch sheets, ROC AUC (5 folds, pooled): CNN", "value": 0.811,
              "ci": [0.796, 0.826], "ours": "tests.a.estimate", "brief_text": "0.811 [0.796, 0.826]"},
    "b_cnn": {"claim": "(b) train Zlínský + Moravskoslezský, test other kraje, ROC AUC: CNN", "value": 0.749,
              "ci": [0.727, 0.770], "ours": "tests.b.estimate", "brief_text": "0.749 [0.727, 0.770]"},
    "stage2_cnn": {"claim": "Stage 2, 20 reserved sheets, precision at 50 % object recall: CNN", "value": 0.559,
                   "ci": [0.407, 0.812], "ci_note": "post hoc sheet-level bootstrap, not pre-registered",
                   "ours": "tests.stage2.estimate", "brief_text": "0.559 [0.407, 0.812]"},
    "a_best_noncnn": {"claim": "(a) best non-CNN: terrain-stats LR", "value": 0.695, "ci": [0.678, 0.713],
                      "ours": "baselines.a.scorers.terrain_stats_lr.auc",
                      "brief_text": "0.695 [0.678, 0.713] (terrain-stats LR)"},
    "b_best_noncnn": {"claim": "(b) best non-CNN: terrain-stats GBM", "value": 0.624, "ci": [0.603, 0.647],
                      "ours": "baselines.b.scorers.terrain_stats_gbm.auc",
                      "brief_text": "0.624 [0.603, 0.647] (terrain-stats GBM)"},
    "stage2_best_noncnn": {"claim": "Stage 2 best non-CNN: slope + susceptibility LR", "value": 0.136, "ci": None,
                           "ours": "baselines.stage2.scorers.slope_susc_lr.estimate",
                           "brief_text": "0.136 (slope + susceptibility LR)"},
    "stage2_random": {"claim": "Stage 2 random scores", "value": 0.085, "ci": None,
                      "ours": "baselines.stage2.scorers.random.estimate", "brief_text": "random 0.085"},
    "a_pair_accuracy": {"claim": "(a) same-sheet pair accuracy (post hoc)", "value": 0.824, "ci": None,
                        "ours": "secondary.pair_accuracy.a.same_sheet.accuracy",
                        "brief_text": "same-sheet pair accuracy 0.824 (a) and 0.762 (b)"},
    "b_pair_accuracy": {"claim": "(b) same-sheet pair accuracy (post hoc)", "value": 0.762, "ci": None,
                        "ours": "secondary.pair_accuracy.b.same_sheet.accuracy",
                        "brief_text": "same-sheet pair accuracy 0.824 (a) and 0.762 (b)"},
    "stage2_ring_as_fp": {"claim": "Stage 2 P@R50 with the 150 m ring counted as false positives (post hoc)",
                          "value": 0.418, "ci": None, "ours": "secondary.stage2_ring_as_fp.estimate",
                          "brief_text": "stage 2 is 0.418 if detections in the 150 m ring count as false positives"},
    "b_jihomoravsky_nonflysch": {"claim": "(b) non-flysch Jihomoravský ROC AUC (post hoc)", "value": 0.622,
                                 "ci": None, "ours": "secondary.jihomoravsky_nonflysch.auc",
                                 "brief_text": "non-flysch Jihomoravský AUC 0.622"},
}

DISCREPANCY_RULE = ("a discrepancy is our point estimate outside the spike's 95 % CI (bounds count as inside); "
                    "claims without a CI cannot be flagged; discrepancies never change the verdict")


class ConfigMismatchError(RuntimeError):
    """The config on disk differs from the frozen config (`configs/landslide.sha256`)."""


# ---------------------------------------------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------------------------------------------

def _clean(obj: Any) -> Any:
    """JSON-ready copy: floats rounded to FLOAT_DECIMALS (non-finite -> None, -0.0 -> 0.0), numpy -> python."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj) if isinstance(obj, (set, frozenset)) else obj
        return [_clean(v) for v in items]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return round(x, FLOAT_DECIMALS) + 0.0 if math.isfinite(x) else None
    if isinstance(obj, Path):
        return obj.as_posix()
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def dumps_metrics(obj: Mapping[str, Any]) -> bytes:
    """Canonical bytes of metrics.json: sorted keys, 2-space indent, UTF-8, LF, trailing newline."""
    text = json.dumps(_clean(obj), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------------------------
# Expected units and integrity
# ---------------------------------------------------------------------------------------------------------------

def expected_runs(cfg: Mapping[str, Any], smoke: bool) -> dict[str, list[str]]:
    """Runs per test: {"a": [a_f0..], "b": ["b"], "stage2": ["s2"]}. Smoke mode uses the config's `smoke` block."""
    if smoke:
        folds = [int(k) for k in cfg["smoke"]["folds"]]
        runs = [str(r) for r in cfg["smoke"]["runs"]]
    else:
        folds = list(range(int(cfg["splits"]["n_folds_a"])))
        runs = [f"a_f{k}" for k in folds] + ["b", "s2"]
    a_runs = [f"a_f{k}" for k in folds]
    unknown = [r for r in runs if r not in a_runs and r not in ("b", "s2")]
    if unknown:
        raise ValueError(f"unexpected runs {unknown} (folds {folds})")
    return {"a": [r for r in a_runs if r in runs], "b": [r for r in runs if r == "b"],
            "stage2": [r for r in runs if r == "s2"]}


def expected_seeds(cfg: Mapping[str, Any], smoke: bool) -> list[int]:
    return [int(s) for s in (cfg["smoke"]["seeds"] if smoke else cfg["train"]["seeds"])]


@dataclass
class Unit:
    """One training unit (run x seed) as found on disk."""

    run: str
    seed: int
    status: str = MISSING
    problems: list[str] = field(default_factory=list)
    info: dict[str, Any] | None = None  # the content of done.json / failed.json / interrupted.json
    preds_test: pd.DataFrame | None = None
    history: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return f"{self.run}_s{self.seed}"

    @property
    def ok(self) -> bool:
        return self.status == DONE

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "problems": list(self.problems)}
        info = self.info or {}
        if self.status in (DONE, HASH_MISMATCH):
            for k in ("config_hash", "git_commit", "git_dirty", "best_epoch", "best_val_auc", "epochs_run",
                      "stop_reason", "n_train", "n_val", "n_test", "seconds", "peak_gpu_mem_gb"):
                out[k] = info.get(k)
        elif self.status == FAILED:
            out["error"] = info.get("error")
        elif self.status == INTERRUPTED:
            out["epochs_completed"] = info.get("epochs_completed")
        return out


def _read_json(p: Path) -> dict[str, Any]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{p.name} does not hold a JSON object")
    return obj


def _read_preds(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if not {"id", "prob"} <= set(df.columns):
        raise ValueError(f"{p.name} lacks the columns id, prob")
    ids = df["id"].to_numpy()
    if len(df) and not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{p.name}: id must be integer")
    if not df["id"].is_unique:
        raise ValueError(f"{p.name}: duplicate ids")
    prob = df["prob"].to_numpy(dtype=np.float64)
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError(f"{p.name}: prob must be finite and in [0, 1]")
    return pd.DataFrame({"id": ids.astype(np.int64), "prob": prob})


def read_unit(paths: Paths, run: str, seed: int, expected_hash: str) -> Unit:
    """Classify one unit's directory (see the module docstring) and load its test predictions and history."""
    u = Unit(run, int(seed))
    d = paths.run_dir(run, seed)
    if (d / "history.csv").exists():
        try:
            u.history = pd.read_csv(d / "history.csv")
        except Exception as e:  # noqa: BLE001 - recorded, never fatal
            u.problems.append(f"history.csv unreadable: {type(e).__name__}: {e}")
    present = [m for m in _MARKERS if (d / m).exists()]
    if not present:
        if u.history is not None:
            u.problems.append("history.csv exists but there is no done, failed or interrupted file")
        return u
    if len(present) > 1:
        u.status = INVALID
        u.problems.append(f"several status files: {', '.join(present)}")
        return u
    marker = present[0]
    try:
        u.info = _read_json(d / marker)
    except (OSError, ValueError) as e:
        u.status = INVALID
        u.problems.append(f"{marker} unreadable: {type(e).__name__}: {e}")
        return u
    if marker == "failed.json":
        u.status = FAILED
        return u
    if marker == "interrupted.json":
        u.status = INTERRUPTED
        return u

    info, problems = u.info, []
    if str(info.get("unit", u.name)) != u.name:
        problems.append(f"done.json names unit {info.get('unit')!r}")
    if "run" in info and str(info["run"]) != run:
        problems.append(f"done.json names run {info['run']!r}")
    if "seed" in info and int(info["seed"]) != int(seed):
        problems.append(f"done.json names seed {info['seed']!r}")
    if "smoke" in info and bool(info["smoke"]) != paths.smoke:
        problems.append(f"done.json has smoke={info['smoke']!r} but this is a {'smoke' if paths.smoke else 'full'} "
                        f"evaluation")
    for f in _UNIT_FILES:
        if not (d / f).exists():
            problems.append(f"{f} is missing")
    if (d / "preds_test.csv").exists():
        try:
            u.preds_test = _read_preds(d / "preds_test.csv")
            if "n_test" in info and int(info["n_test"]) != len(u.preds_test):
                problems.append(f"preds_test.csv has {len(u.preds_test)} rows, done.json n_test = {info['n_test']}")
        except (OSError, ValueError) as e:
            problems.append(f"preds_test.csv unusable: {e}")
    if u.history is not None and not {"epoch", "train_loss", "train_auc", "val_loss", "val_auc"} <= set(
            u.history.columns):
        problems.append("history.csv lacks the Appendix A §7 columns")
    if info.get("config_hash") != expected_hash:
        problems.append(f"config hash {info.get('config_hash')!r} differs from the expected {expected_hash!r}")
        u.status = HASH_MISMATCH if len(problems) == 1 else INVALID
    else:
        u.status = INVALID if problems else DONE
    u.problems.extend(problems)
    return u


def _integrity(units: Mapping[str, Unit], expected_hash: str, hash_source: str, current_hash: str) -> dict[str, Any]:
    by_status = {s: sorted(n for n, u in units.items() if u.status == s) for s in UNIT_STATUSES}
    done = [u for u in units.values() if u.ok]
    commits = sorted({str(u.info["git_commit"]) for u in done if u.info and u.info.get("git_commit")})
    dirty = sorted(u.name for u in done if u.info and u.info.get("git_dirty") is True)
    return {
        "config_hash_expected": expected_hash,
        "config_hash_source": hash_source,
        "config_hash_current": current_hash,
        "n_expected": len(units),
        "n_done": len(done),
        "missing": by_status[MISSING],
        "failed": by_status[FAILED],
        "interrupted": by_status[INTERRUPTED],
        "hash_mismatch": by_status[HASH_MISMATCH],
        "invalid": by_status[INVALID],
        "git_commits": commits,
        "git_dirty_units": dirty,
        "ok": len(done) == len(units),
        "units": {n: u.summary() for n, u in sorted(units.items())},
    }


# ---------------------------------------------------------------------------------------------------------------
# Meta and test sets
# ---------------------------------------------------------------------------------------------------------------

def load_meta(path: Path | str) -> pd.DataFrame:
    """Read `meta.csv` with the columns the evaluation needs, sorted by id."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; the evaluation needs the dataset's meta.csv")
    df = pd.read_csv(p, dtype={"sheet": str, "kraj": str, "group_b": str, "block": str, "source": str,
                               "stratum": str})
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"meta.csv lacks columns {missing}")
    if not df["id"].is_unique:
        raise ValueError("meta.csv: duplicate ids")
    if not df["label"].isin((0, 1)).all():
        raise ValueError("meta.csv: label must be 0 or 1")
    fl = df["flysch"]
    if fl.dtype != bool:
        fl = fl.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if fl.isna().any():
            raise ValueError("meta.csv: flysch must be a boolean")
        df["flysch"] = fl.astype(bool)
    df["kraj"] = df["kraj"].fillna("unknown").astype(str)
    df["sheet"] = df["sheet"].astype(str)
    df["block"] = df["block"].astype(str)
    df["source"] = df["source"].astype(str)
    for c in ("id", "label", "fold", "pair_id", "match_level"):
        df[c] = df[c].astype(np.int64)
    return df.sort_values("id", kind="stable").reset_index(drop=True)


def _fold_of(run: str) -> int | None:
    m = re.fullmatch(r"a_f(\d+)", run)
    return int(m.group(1)) if m else None


def meta_test_mask(meta: pd.DataFrame, test: str, run: str | None = None) -> np.ndarray:
    """Rows of `meta` in the test set of `test` ("a" or "b"), or of one (a) fold's run. Background never counts."""
    nb = meta["source"].to_numpy() != BACKGROUND
    if test == "a":
        fold = meta["fold"].to_numpy()
        k = None if run is None else _fold_of(run)
        return nb & (fold >= 0) if k is None else nb & (fold == k)
    if test == "b":
        return nb & (meta["group_b"].astype(str).to_numpy() == "test")
    raise ValueError(f"unknown test {test!r}")


# ---------------------------------------------------------------------------------------------------------------
# Small statistical helpers (all statistics come from landslide.metrics)
# ---------------------------------------------------------------------------------------------------------------

def _ci_dict(r: M.BootstrapResult, unit: str) -> dict[str, Any]:
    return {"unit": unit, "low": r.ci_low, "high": r.ci_high, "level": r.ci, "n_replicates": r.n_replicates,
            "n_used": r.n_used, "n_degenerate": r.n_degenerate}


def _counts(items: pd.DataFrame) -> dict[str, int]:
    y = items["label"].to_numpy()
    return {"patches": int(len(items)), "positives": int((y == 1).sum()), "negatives": int((y == 0).sum()),
            "sheets": int(items["sheet"].nunique()), "blocks": int(items["block"].nunique())}


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    try:
        return M.roc_auc(y, s)
    except ValueError:
        return None


def _draws(labels: Sequence[Any] | np.ndarray, spec: M.BootstrapSpec) -> M.ClusterDraws:
    return M.draw_clusters(np.asarray(labels), spec.n_replicates, spec.seed)


def _auc_record(items: pd.DataFrame, score: np.ndarray, spec: M.BootstrapSpec) -> dict[str, Any]:
    """AUC with its sheet CI on a subset (own draws, same seed). One class only -> auc None."""
    n = _counts(items)
    if n["positives"] == 0 or n["negatives"] == 0:
        return {"auc": None, "sheet_ci": None, "n": n}
    r = M.bootstrap_auc(items["label"].to_numpy(), score, _draws(items["sheet"].to_numpy(), spec), ci=spec.ci)
    return {"auc": r.estimate, "sheet_ci": _ci_dict(r, "sheet"), "n": n}


# ---------------------------------------------------------------------------------------------------------------
# CNN predictions
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class CNNPreds:
    complete: bool
    problems: list[str]
    missing_units: dict[str, str]
    preds: pd.DataFrame | None = None  # id, prob (seed ensemble), run; sorted by id


def collect_cnn(test: str, runs: Sequence[str], seeds: Sequence[int], units: Mapping[str, Unit],
                meta: pd.DataFrame, smoke: bool) -> CNNPreds:
    """The pooled seed-ensemble test predictions of a test, or why the test is incomplete."""
    names = [f"{r}_s{s}" for r in runs for s in seeds]
    missing = {n: units[n].status for n in names if not units[n].ok}
    if not names:
        return CNNPreds(False, ["no run is expected for this test"], {})
    if missing:
        return CNNPreds(False, [f"{len(missing)} of {len(names)} units are not usable"], missing)
    problems: list[str] = []
    frames = []
    all_ids = set(meta["id"].tolist())
    for r in runs:
        try:
            ens = M.seed_ensemble({s: units[f"{r}_s{s}"].preds_test for s in seeds})
        except ValueError as e:
            problems.append(f"{r}: seed predictions disagree: {e}")
            continue
        ens["run"] = r
        frames.append(ens)
        ids = ens["id"].to_numpy()
        unknown = int((~np.isin(ids, list(all_ids))).sum())
        allowed = meta["id"].to_numpy()[meta_test_mask(meta, test, r if test == "a" else None)]
        outside = int((~np.isin(ids, allowed)).sum())
        if unknown:
            problems.append(f"{r}: {unknown} predicted ids are not in meta.csv")
        elif outside:
            problems.append(f"{r}: {outside} predicted ids are not in this run's test set")
    if problems:
        return CNNPreds(False, problems, {})
    pooled = pd.concat(frames, ignore_index=True)
    if not pooled["id"].is_unique:
        return CNNPreds(False, ["a patch is predicted by more than one run"], {})
    expected = set(meta["id"].to_numpy()[meta_test_mask(meta, test)].tolist())
    got = set(pooled["id"].tolist())
    if not got:
        problems.append("no test predictions")
    elif not smoke and got != expected:
        problems.append(f"the predictions cover {len(got & expected)} of the {len(expected)} test patches "
                        f"in meta.csv")
    if problems:
        return CNNPreds(False, problems, {})
    return CNNPreds(True, [], {}, pooled.sort_values("id", kind="stable").reset_index(drop=True))


def _per_fold_seed(test: str, runs: Sequence[str], seeds: Sequence[int], units: Mapping[str, Unit],
                   meta: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Per fold (seed ensemble), per seed (pooled over the test's runs) and per unit AUCs, where available."""
    label = pd.Series(meta["label"].to_numpy(), index=meta["id"].to_numpy())

    def auc_of(df: pd.DataFrame) -> tuple[float | None, int]:
        y = label.reindex(df["id"].to_numpy()).to_numpy()
        if np.isnan(y.astype(float)).any():
            return None, len(df)
        return _safe_auc(y.astype(np.int64), df["prob"].to_numpy()), len(df)

    per_unit = {}
    for r in runs:
        for s in seeds:
            u = units[f"{r}_s{s}"]
            if u.ok:
                a, n = auc_of(u.preds_test)
                per_unit[u.name] = {"auc": a, "n_patches": n}
    per_fold: dict[str, Any] | None = None
    if test == "a":
        per_fold = {}
        for r in runs:
            names = [f"{r}_s{s}" for s in seeds]
            rec: dict[str, Any] = {"run": r, "complete": all(units[n].ok for n in names), "auc": None}
            if rec["complete"] and names:
                try:
                    a, n = auc_of(M.seed_ensemble({s: units[f"{r}_s{s}"].preds_test for s in seeds}))
                    rec.update(auc=a, n_patches=n)
                except ValueError as e:
                    rec["problem"] = str(e)
            per_fold[str(_fold_of(r))] = rec
    per_seed = {}
    for s in seeds:
        names = [f"{r}_s{s}" for r in runs]
        rec = {"complete": bool(names) and all(units[n].ok for n in names), "auc": None}
        if rec["complete"]:
            pooled = pd.concat([units[n].preds_test for n in names], ignore_index=True)
            if pooled["id"].is_unique:
                a, n = auc_of(pooled)
                rec.update(auc=a, n_patches=n)
        per_seed[str(s)] = rec
    return per_fold, per_seed, per_unit


# ---------------------------------------------------------------------------------------------------------------
# Stage-1 baselines
# ---------------------------------------------------------------------------------------------------------------

def load_baselines(path: Path, items: pd.DataFrame, test: str, smoke: bool) -> tuple[dict[str, np.ndarray],
                                                                                       dict[str, Any]]:
    """Baseline scores aligned with `items` (sorted by id), and the file's status record."""
    rec: dict[str, Any] = {"file": path.name}
    if not path.exists():
        rec["status"] = "missing"
        return {}, rec
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        rec.update(status="unreadable", problem=f"{type(e).__name__}: {e}")
        return {}, rec
    scorers = [c for c in df.columns if c not in ("id", "fold")]
    if "id" not in df.columns or not df["id"].is_unique or not scorers:
        rec.update(status="invalid", problem="needs a unique id column and at least one scorer column")
        return {}, rec
    if CNN in scorers:
        rec.update(status="invalid", problem=f"a baseline may not be called {CNN!r}")
        return {}, rec
    ids = items["id"].to_numpy()
    n_missing = int((~np.isin(ids, df["id"].to_numpy())).sum())
    n_extra = int((~np.isin(df["id"].to_numpy(), ids)).sum())
    rec.update(n_rows=int(len(df)), n_missing_ids=n_missing, n_extra_ids=n_extra, scorers=sorted(scorers))
    if n_missing or (n_extra and not smoke):
        rec.update(status="id_mismatch", problem="the baseline ids are not the CNN test set")
        return {}, rec
    sub = df.set_index("id").reindex(ids)
    vals = sub[scorers].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all():
        rec.update(status="invalid", problem="non-finite baseline scores")
        return {}, rec
    if test == "a" and "fold" in df.columns:
        bad = int((sub["fold"].to_numpy().astype(np.int64) != items["fold"].to_numpy()).sum())
        rec["n_fold_mismatch"] = bad
        if bad:
            rec.update(status="fold_mismatch", problem=f"{bad} rows disagree with meta.csv on the fold")
            return {}, rec
    rec["status"] = "ok"
    return {c: sub[c].to_numpy(dtype=np.float64) for c in scorers}, rec


# ---------------------------------------------------------------------------------------------------------------
# Tests (a) and (b)
# ---------------------------------------------------------------------------------------------------------------

def _pair_accuracy_records(items: pd.DataFrame, score: np.ndarray, sheet_draws: M.ClusterDraws,
                           spec: M.BootstrapSpec) -> dict[str, Any]:
    """Pair accuracy: same sheet (primary, with a sheet CI on the test's draws), match level 1, all pairs."""
    y = items["label"].to_numpy().astype(bool)
    pid = items["pair_id"].to_numpy()
    sheet = items["sheet"].to_numpy()
    lvl = items["match_level"].to_numpy()
    neg = ~y & (pid >= 0)
    level_of = dict(zip(pid[neg].tolist(), lvl[neg].tolist()))

    same = M.pair_accuracy(y, score, pid, sheet=sheet)
    rec_same: dict[str, Any] = {**same.to_dict(), "sheet_ci": None, "by_match_level": {
        str(k): v for k, v in sorted(Counter(level_of[p] for p in same.pair_ids.tolist()).items())}}
    if same.n_pairs:
        pos = y & (pid >= 0)
        sheet_of = dict(zip(pid[pos].tolist(), sheet[pos].tolist()))
        index = {lab: i for i, lab in enumerate(sheet_draws.labels.tolist())}
        k = np.array([index[sheet_of[p]] for p in same.pair_ids.tolist()], dtype=np.int64)
        credit_sum = np.bincount(k, weights=same.credit, minlength=sheet_draws.n_clusters)
        n_pairs = np.bincount(k, minlength=sheet_draws.n_clusters).astype(np.float64)

        def stat(_w: np.ndarray, c: np.ndarray) -> float:
            den = float(c @ n_pairs)
            return float(c @ credit_sum) / den if den > 0 else math.nan

        rec_same["sheet_ci"] = _ci_dict(M.bootstrap_statistic(stat, sheet_draws, ci=spec.ci), "sheet")

    l1 = np.isin(pid, [p for p, lv in level_of.items() if lv == 1])
    level1 = M.pair_accuracy(y[l1], score[l1], pid[l1], sheet=sheet[l1])
    allp = M.pair_accuracy(y, score, pid)
    return {"primary": "same_sheet", "same_sheet": rec_same, "level_1": level1.to_dict(),
            "all_pairs": allp.to_dict()}


def evaluate_auc_test(test: str, cfg: Mapping[str, Any], spec: M.BootstrapSpec, meta: pd.DataFrame,
                      units: Mapping[str, Unit], runs: Sequence[str], seeds: Sequence[int], paths: Paths
                      ) -> dict[str, Any]:
    """Test (a) or (b): the test record, the baselines, the paired differences and the secondary checks."""
    thr = float(cfg["eval"]["thresholds"]["a_auc_min" if test == "a" else "b_auc_min"])
    cnn = collect_cnn(test, runs, seeds, units, meta, paths.smoke)
    if cnn.complete:
        items = meta[meta["id"].isin(cnn.preds["id"])].reset_index(drop=True)
        cnn_score = cnn.preds.set_index("id")["prob"].reindex(items["id"].to_numpy()).to_numpy()
    else:
        items = meta[meta_test_mask(meta, test)].reset_index(drop=True)
        cnn_score = None
    y = items["label"].to_numpy()
    n = _counts(items)
    if cnn.complete and not (n["positives"] and n["negatives"]):
        cnn.complete, cnn_score = False, None
        cnn.problems.append("the test set does not hold both classes, so it has no AUC")
    log.info("test (%s): %d patches on %d sheets; CNN %s", test, n["patches"], n["sheets"],
             "complete" if cnn.complete else "INCOMPLETE")

    base_scores, base_file = load_baselines(paths.baselines_dir / f"preds_{test}.csv", items, test, paths.smoke)
    scorers: dict[str, np.ndarray] = ({CNN: cnn_score} if cnn.complete else {}) | base_scores
    have_both = n["positives"] > 0 and n["negatives"] > 0
    res_sheet: dict[str, M.BootstrapResult] = {}
    res_block: dict[str, M.BootstrapResult] = {}
    sheet_draws = block_draws = None
    if have_both:
        sheet_draws = _draws(items["sheet"].to_numpy(), spec)
        block_draws = _draws(items["block"].to_numpy(), spec)
        if scorers:
            res_sheet = M.bootstrap_aucs(y, scorers, sheet_draws, ci=spec.ci)
            res_block = M.bootstrap_aucs(y, scorers, block_draws, ci=spec.ci)

    per_fold, per_seed, per_unit = _per_fold_seed(test, runs, seeds, units, meta)
    rec: dict[str, Any] = {
        "test": test, "definition": cfg["eval"]["primary_a" if test == "a" else "primary_b"], "metric": "roc_auc",
        "threshold": thr, "complete": cnn.complete, "problems": cnn.problems,
        "units": [f"{r}_s{s}" for r in runs for s in seeds], "missing_units": cnn.missing_units,
        "estimate": None, "pass": None, "borderline": None, "borderline_block": None,
        "sheet_ci": None, "block_ci": None, "n": n, "per_fold": per_fold, "per_seed": per_seed,
        "per_unit": per_unit, "breakdowns": None,
    }
    secondary: dict[str, Any] = {"slope_only_check": None, "pair_accuracy": None}
    if cnn.complete and CNN in res_sheet:
        rs, rb = res_sheet[CNN], res_block[CNN]
        rec.update(estimate=rs.estimate, sheet_ci=_ci_dict(rs, "sheet"), block_ci=_ci_dict(rb, "block"))
        rec["status"] = V.classify_test(complete=True, estimate=rs.estimate, threshold=thr)
        rec["pass"] = rec["status"] == V.PASS
        rec["borderline"] = V.is_borderline(rs.ci_low, rs.ci_high, thr)
        rec["borderline_block"] = V.is_borderline(rb.ci_low, rb.ci_high, thr)
        rec["breakdowns"] = _breakdowns(test, items, cnn_score, spec)
        secondary["pair_accuracy"] = _pair_accuracy_records(items, cnn_score, sheet_draws, spec)
        if test == "b":
            sub = (items["kraj"].to_numpy() == JIHOMORAVSKY) & ~items["flysch"].to_numpy()
            secondary["jihomoravsky_nonflysch"] = _auc_record(items[sub], cnn_score[sub], spec)
    else:
        rec["status"] = V.classify_test(complete=False, estimate=None, threshold=thr)
        if test == "b":
            secondary["jihomoravsky_nonflysch"] = None
    if have_both:
        slope = M.bootstrap_auc(y, items["slope_mean"].to_numpy(dtype=np.float64), sheet_draws, ci=spec.ci)
        secondary["slope_only_check"] = {"score": "meta.csv slope_mean (patch-mean slope)", "auc": slope.estimate,
                                         "sheet_ci": _ci_dict(slope, "sheet"), "expected": "about 0.5", "n": n}

    baselines: dict[str, Any] = {**base_file, "scorers": {}}
    fold = items["fold"].to_numpy()
    for name in sorted(base_scores):
        b: dict[str, Any] = {"kind": SCORER_KINDS.get(name, "comparator"), "auc": None, "sheet_ci": None,
                             "block_ci": None}
        if name in res_sheet:
            b.update(auc=res_sheet[name].estimate, sheet_ci=_ci_dict(res_sheet[name], "sheet"),
                     block_ci=_ci_dict(res_block[name], "block"))
        if test == "a":
            b["per_fold"] = {str(k): _safe_auc(y[fold == k], base_scores[name][fold == k])
                             for k in sorted(set(fold.tolist()))}
        baselines["scorers"][name] = b

    paired: dict[str, Any] = {"sheet": {}, "block": {}}
    if CNN in res_sheet and len(res_sheet) > 1:
        for unit, res in (("sheet", res_sheet), ("block", res_block)):
            for name, d in M.paired_differences(res, CNN).items():
                paired[unit][name] = {"estimate": d.estimate, "ci": _ci_dict(d, unit)}
    return {"test": rec, "baselines": baselines, "paired": paired, "secondary": secondary}


def _breakdowns(test: str, items: pd.DataFrame, score: np.ndarray, spec: M.BootstrapSpec) -> dict[str, Any]:
    """CNN AUC by negative source (both tests), and for (b) by kraj, by flysch flag and by kraj x flysch."""
    y = items["label"].to_numpy()
    src = items["source"].to_numpy()
    out: dict[str, Any] = {"by_source": {}}
    for s in sorted(set(src[y == 0].tolist())):
        m = (y == 1) | (src == s)
        out["by_source"][s] = _auc_record(items[m], score[m], spec)
    if test == "b":
        kraj = items["kraj"].to_numpy()
        fl = np.where(items["flysch"].to_numpy(), "flysch", "non-flysch")
        out["by_kraj"] = {k: _auc_record(items[kraj == k], score[kraj == k], spec) for k in sorted(set(kraj))}
        out["by_flysch"] = {f: _auc_record(items[fl == f], score[fl == f], spec) for f in sorted(set(fl))}
        combo = np.array([f"{k} | {f}" for k, f in zip(kraj, fl)], dtype=object)
        out["by_kraj_flysch"] = {c: _auc_record(items[combo == c], score[combo == c], spec)
                                 for c in sorted(set(combo.tolist()))}
    return out


# ---------------------------------------------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------------------------------------------

def stage2_sheet_strata(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Reserved stage-2 sheet -> "flysch" / "nonflysch" (config `replication.splits.stage2_sheets`)."""
    s2 = cfg["replication"]["splits"]["stage2_sheets"]
    return {str(s): st for st in ("flysch", "nonflysch") for s in s2[st]}


def load_detections(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, None
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        return None, f"{path.name} unreadable: {type(e).__name__}: {e}"
    missing = [c for c in S2_COLUMNS if c not in df.columns]
    if missing:
        return None, f"{path.name} lacks columns {missing}"
    df = df.copy()
    df["scorer"] = df["scorer"].astype(str)
    df["sheet"] = df["sheet"].astype(str)
    df["outcome"] = df["outcome"].astype(str)
    return df, None


def load_targets(path: Path, sheets: Sequence[str]) -> tuple[pd.DataFrame | None, str | None]:
    """Stage-2 targets: eligible polygons assigned to one of `sheets` (columns objectid, sheet)."""
    if not path.exists():
        return None, f"{path.name} is missing (stage-2 targets)"
    try:
        df = pd.read_parquet(path, columns=["objectid", "eligible", "sheet"])
    except Exception as e:  # noqa: BLE001
        return None, f"{path.name} unreadable: {type(e).__name__}: {e}"
    df = df[df["eligible"].astype(bool) & df["sheet"].isin(list(sheets))]
    return pd.DataFrame({"objectid": df["objectid"].astype(np.int64).to_numpy(),
                         "sheet": df["sheet"].astype(str).to_numpy()}), None


@dataclass
class S2Frame:
    sheets: np.ndarray            # sorted sheet ids = the bootstrap clusters
    stratum: np.ndarray           # "flysch" / "nonflysch" per sheet
    n_targets: np.ndarray         # targets per sheet (int64)
    target_sheet: dict[int, str]  # target objectid -> its assigned sheet

    @classmethod
    def build(cls, sheets: Sequence[str], strata: Mapping[str, str], targets: pd.DataFrame) -> S2Frame:
        sh = np.array(sorted(sheets), dtype=object)
        t = targets[targets["sheet"].isin(list(sh))]
        if not t["objectid"].is_unique:
            raise ValueError("duplicate target objectids")
        counts = t.groupby("sheet").size()
        return cls(sh, np.array([strata[s] for s in sh], dtype=object),
                   np.array([int(counts.get(s, 0)) for s in sh], dtype=np.int64),
                   dict(zip(t["objectid"].astype(np.int64).tolist(), t["sheet"].tolist())))


def _scorer_arrays(dets: pd.DataFrame, frame: S2Frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, outcomes and each detection's bootstrap sheet index (a TP counts on its target's sheet)."""
    present = set(dets["sheet"].tolist())
    outside = sorted(present - set(frame.sheets.tolist()))
    if outside:
        raise ValueError(f"detections on sheets outside the evaluated stage-2 sheets: {outside[:5]}")
    absent = sorted(set(frame.sheets.tolist()) - present)
    if absent:
        raise ValueError(f"no detections on {len(absent)} evaluated sheets: {absent[:5]}")
    scores = dets["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite detection scores")
    outcomes = dets["outcome"].to_numpy().astype(str)
    bad = sorted(set(outcomes.tolist()) - set(M.STAGE2_OUTCOMES))
    if bad:
        raise ValueError(f"unknown outcomes {bad}")
    tp = outcomes == "tp"
    oid = pd.to_numeric(dets["target_objectid"], errors="coerce").astype("float64").to_numpy()[tp]
    if np.isnan(oid).any():
        raise ValueError("a TP detection has no target_objectid")
    oid = oid.astype(np.int64)
    if len(set(oid.tolist())) != oid.size:
        raise ValueError("a target is matched by more than one TP")
    unknown = [o for o in oid.tolist() if o not in frame.target_sheet]
    if unknown:
        raise ValueError(f"TP targets that are not targets of the evaluated sheets: {unknown[:5]}")
    unit_sheet = dets["sheet"].to_numpy().astype(object)
    unit_sheet[tp] = [frame.target_sheet[o] for o in oid.tolist()]
    idx = np.searchsorted(frame.sheets, unit_sheet)
    return scores, outcomes, idx.astype(np.int64)


def _sweep(scores: np.ndarray, outcomes: np.ndarray, idx: np.ndarray, frame: S2Frame, level: float,
           sheet_mask: np.ndarray | None = None) -> M.PRSweep | None:
    keep = np.ones(scores.size, dtype=bool) if sheet_mask is None else sheet_mask[idx]
    nt = int(frame.n_targets.sum() if sheet_mask is None else frame.n_targets[sheet_mask].sum())
    if nt <= 0:
        return None
    return M.pr_sweep(scores[keep], outcomes[keep], nt, level)


def _sweep_record(sw: M.PRSweep | None, n_sheets: int) -> dict[str, Any]:
    if sw is None:
        return {"estimate": None, "reached": None, "note": "no targets"}
    return {"estimate": sw.precision_at_recall, "reached": sw.reached,
            "interpolated_precision": sw.interpolated_precision, "max_recall": sw.max_recall,
            "score_threshold": sw.threshold_at_recall, "recall_level": sw.recall_level,
            "n": {"sheets": n_sheets, "targets": sw.n_targets, "detections": int(sw.n_tp + sw.n_fp + sw.n_ignored
                                                                                 + sw.n_duplicate),
                  "tp": sw.n_tp, "fp": sw.n_fp, "ignored": sw.n_ignored, "duplicate": sw.n_duplicate}}


def _s2_bootstrap(scores: np.ndarray, outcomes: np.ndarray, idx: np.ndarray, frame: S2Frame, level: float,
                  draws: M.ClusterDraws, ci: float) -> M.BootstrapResult:
    """Sheet bootstrap of P@R (post hoc). A replicate that never reaches the level counts as precision 0."""
    nt = frame.n_targets

    def stat(_w: np.ndarray, c: np.ndarray) -> float:
        n_t = int(c @ nt)
        if n_t <= 0:
            return math.nan  # no target drawn: degenerate (D21 sense), dropped and counted
        sw = M.pr_sweep(scores, outcomes, n_t, level, weights=c[idx])
        return float(sw.precision_at_recall) if sw.reached else 0.0

    return M.bootstrap_statistic(stat, draws, ci=ci)


def _s2_scorer(scores: np.ndarray, outcomes: np.ndarray, idx: np.ndarray, frame: S2Frame, level: float,
               draws: M.ClusterDraws, spec: M.BootstrapSpec) -> tuple[dict[str, Any], M.BootstrapResult]:
    sw = _sweep(scores, outcomes, idx, frame, level)
    rec = _sweep_record(sw, len(frame.sheets))
    boot = _s2_bootstrap(scores, outcomes, idx, frame, level, draws, spec.ci)
    ci = _ci_dict(boot, "sheet")
    # With n_targets > 0, reaching recall >= level needs TP > 0, so precision 0 means "not reached".
    ci.update(post_hoc=True, not_reached_rule=S2_NOT_REACHED_RULE,
              n_not_reached=int(np.sum(boot.replicates == 0.0)))
    rec["sheet_ci"] = ci
    rec["by_stratum"] = {st: _sweep_record(_sweep(scores, outcomes, idx, frame, level, frame.stratum == st),
                                           int((frame.stratum == st).sum()))
                         for st in sorted(set(frame.stratum.tolist()))}
    ring = np.where(outcomes == "ignored", "fp", outcomes)
    rec["ring_as_fp"] = _sweep_record(_sweep(scores, ring, idx, frame, level), len(frame.sheets))
    return rec, boot


def evaluate_stage2(cfg: Mapping[str, Any], spec: M.BootstrapSpec, paths: Paths, units: Mapping[str, Unit],
                    runs: Sequence[str], seeds: Sequence[int], targets: pd.DataFrame | None = None
                    ) -> dict[str, Any]:
    thr = float(cfg["eval"]["thresholds"]["stage2_precision_min"])
    level = float(cfg["eval"]["thresholds"]["stage2_recall"])
    names = [f"{r}_s{s}" for r in runs for s in seeds]
    missing = {n: units[n].status for n in names if not units[n].ok}
    strata = stage2_sheet_strata(cfg)
    problems: list[str] = []

    dets, det_problem = load_detections(paths.stage2 / "detections.parquet")
    if det_problem:
        problems.append(det_problem)
    has_cnn = dets is not None and bool((dets["scorer"] == CNN).any())
    if paths.smoke:
        src = dets[dets["scorer"] == CNN] if has_cnn else dets
        sheets = sorted(set(src["sheet"].tolist())) if src is not None else []
    else:
        sheets = sorted(strata)
    outside = [s for s in sheets if s not in strata]
    frame = None
    if outside:
        problems.append(f"sheets that are not reserved stage-2 sheets: {outside[:5]}")
    elif sheets and dets is not None:
        if targets is None:
            targets, t_problem = load_targets(paths.polygons_parquet, sheets)
            if t_problem:
                problems.append(t_problem)
        if targets is not None:
            try:
                frame = S2Frame.build(sheets, strata, targets)
            except ValueError as e:
                problems.append(f"targets: {e}")
    draws = _draws(frame.sheets, spec) if frame is not None else None
    if draws is not None and draws.labels.tolist() != frame.sheets.tolist():
        raise AssertionError("stage-2 draws are not indexed by the sorted sheets")

    rec: dict[str, Any] = {
        "test": "stage2", "definition": cfg["stage2"]["precision_rule"], "metric": "precision_at_recall",
        "recall_level": level, "threshold": thr, "complete": False, "problems": problems, "units": names,
        "missing_units": missing, "sheets": sheets, "estimate": None, "reached": None, "pass": None,
        "borderline": None, "interpolated_precision": None, "max_recall": None, "score_threshold": None,
        "sheet_ci": None, "block_ci": None, "n": None, "per_fold": None, "per_seed": None, "breakdowns": None,
    }
    secondary: dict[str, Any] = {"stage2_ring_as_fp": None}
    cnn_boot = None
    any_trace = any(units[n].status != MISSING for n in names) or has_cnn
    if not any_trace:
        rec["status"] = V.NOT_RUN
        problems.append("stage 2 was not run (no stage-2 unit or CNN detection exists)")
    elif missing:
        rec["status"] = V.INCOMPLETE
        problems.append(f"{len(missing)} of {len(names)} stage-2 units are not usable")
    elif not has_cnn:
        rec["status"] = V.INCOMPLETE
        problems.append("no CNN detections (scorer 'cnn') in detections.parquet")
    elif frame is None:
        rec["status"] = V.INCOMPLETE
        problems.append("the stage-2 targets could not be built")
    elif int(frame.n_targets.sum()) == 0:
        rec["status"] = V.INCOMPLETE
        problems.append("no targets on the evaluated stage-2 sheets")
    elif paths.smoke and len(sheets) != int(cfg["smoke"]["stage2_sheets"]):
        rec["status"] = V.INCOMPLETE
        problems.append(f"smoke stage 2 covers {len(sheets)} sheets, expected {cfg['smoke']['stage2_sheets']}")
    else:
        try:
            arrays = _scorer_arrays(dets[dets["scorer"] == CNN], frame)
            srec, cnn_boot = _s2_scorer(*arrays, frame, level, draws, spec)
        except ValueError as e:
            rec["status"] = V.INCOMPLETE
            problems.append(f"CNN detections unusable: {e}")
        else:
            reached = bool(srec["reached"])
            rec.update(complete=True, estimate=srec["estimate"], reached=reached,
                       interpolated_precision=srec["interpolated_precision"], max_recall=srec["max_recall"],
                       score_threshold=srec["score_threshold"], sheet_ci=srec["sheet_ci"], n=srec["n"],
                       breakdowns={"by_stratum": srec["by_stratum"]})
            rec["status"] = V.classify_test(complete=True, estimate=srec["estimate"], threshold=thr,
                                            reached=reached)
            rec["pass"] = rec["status"] == V.PASS
            rec["borderline"] = V.is_borderline(srec["sheet_ci"]["low"], srec["sheet_ci"]["high"], thr)
            secondary["stage2_ring_as_fp"] = srec["ring_as_fp"]

    baselines: dict[str, Any] = {"file": "detections.parquet",
                                 "status": "ok" if dets is not None else ("missing" if det_problem is None
                                                                         else "unreadable"),
                                 "scorers": {}}
    paired: dict[str, Any] = {"sheet": {}}
    if dets is not None and frame is not None:
        for name in sorted(set(dets["scorer"].tolist()) - {CNN}):
            try:
                arrays = _scorer_arrays(dets[dets["scorer"] == name], frame)
                srec, boot = _s2_scorer(*arrays, frame, level, draws, spec)
            except ValueError as e:
                baselines["scorers"][name] = {"kind": SCORER_KINDS.get(name, "comparator"), "status": "error",
                                              "problem": str(e)}
                continue
            baselines["scorers"][name] = {"kind": SCORER_KINDS.get(name, "comparator"), "status": "ok", **srec}
            if cnn_boot is not None:
                d = M.paired_differences({CNN: cnn_boot, name: boot}, CNN)[name]
                paired["sheet"][name] = {"estimate": d.estimate, "ci": _ci_dict(d, "sheet"), "post_hoc": True}
    return {"test": rec, "baselines": baselines, "paired": paired, "secondary": secondary}


# ---------------------------------------------------------------------------------------------------------------
# Spike claims and discrepancies
# ---------------------------------------------------------------------------------------------------------------

def _get(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, Mapping) or k not in cur:
            return None
        cur = cur[k]
    return cur


def compare_with_spike(m: Mapping[str, Any]) -> dict[str, Any]:
    comps = []
    for key, c in SPIKE_CLAIMS.items():
        ours = _get(m, c["ours"])
        ours = float(ours) if isinstance(ours, (int, float, np.floating)) and not isinstance(ours, bool) else None
        ci = c.get("ci")
        if ours is None or not math.isfinite(ours):
            flag, note = None, "not available in this evaluation"
        elif ci is None:
            flag, note = None, "the spike gives no CI"
        else:
            flag, note = bool(ours < ci[0] or ours > ci[1]), None
        comps.append({"id": key, "claim": c["claim"], "spike": c["value"], "spike_ci": ci, "ours": ours,
                      "ours_path": c["ours"], "outside_spike_ci": flag, "note": note})
    return {"rule": DISCREPANCY_RULE, "flagged": [c["id"] for c in comps if c["outside_spike_ci"]],
            "comparisons": comps}


# ---------------------------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------------------------

def _expected_hash(cfg: Mapping[str, Any], sha_path: Path) -> tuple[str, str, str]:
    current = config_hash(dict(cfg))
    frozen = frozen_hash(sha_path)
    if frozen is not None and frozen != current:
        raise ConfigMismatchError(f"the config on disk (hash {current}) differs from the frozen config "
                                  f"({sha_path}: {frozen}); the evaluation must run with the frozen config")
    return (frozen, "frozen", current) if frozen is not None else (current, "current_config", current)


def run_evaluation(cfg: Mapping[str, Any], paths: Paths, *, targets: pd.DataFrame | None = None,
                   sha_path: Path | str | None = None, write: bool = True) -> dict[str, Any]:
    """Evaluate everything from saved outputs; write metrics.json, metrics_table.md and the learning curves.

    `targets` (objectid, sheet) overrides the stage-2 targets read from `paths.polygons_parquet`. `sha_path`
    defaults to `<paths.root>/configs/landslide.sha256`. Returns the metrics as written (rounded, JSON-ready).
    """
    sha = Path(sha_path) if sha_path is not None else paths.root / "configs" / "landslide.sha256"
    expected_hash, hash_source, current_hash = _expected_hash(cfg, sha)
    runs = expected_runs(cfg, paths.smoke)
    seeds = expected_seeds(cfg, paths.smoke)
    units = {f"{r}_s{s}": read_unit(paths, r, s, expected_hash)
             for t in ("a", "b", "stage2") for r in runs[t] for s in seeds}
    integrity = _integrity(units, expected_hash, hash_source, current_hash)
    log.info("units: %d of %d done; missing %s; failed %s; interrupted %s; hash mismatch %s; invalid %s",
             integrity["n_done"], integrity["n_expected"], integrity["missing"], integrity["failed"],
             integrity["interrupted"], integrity["hash_mismatch"], integrity["invalid"])

    meta = load_meta(paths.meta_csv)
    spec = M.bootstrap_spec(cfg)
    ev_a = evaluate_auc_test("a", cfg, spec, meta, units, runs["a"], seeds, paths)
    ev_b = evaluate_auc_test("b", cfg, spec, meta, units, runs["b"], seeds, paths)
    ev_s2 = evaluate_stage2(cfg, spec, paths, units, runs["stage2"], seeds, targets=targets)
    verdict = V.decide(ev_a["test"]["status"], ev_b["test"]["status"], ev_s2["test"]["status"],
                       V.verdict_words(cfg))
    log.info("verdict: %s (a %s, b %s, stage 2 %s)", verdict.word, *verdict.statuses.values())

    m: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "smoke": paths.smoke,
        "config_hash": expected_hash,
        "git_commit": integrity["git_commits"][0] if len(integrity["git_commits"]) == 1 else None,
        "verdict": verdict.word,
        "verdict_detail": {**verdict.statuses, "reasons": verdict.reasons,
                           "rule": "any completed test below its threshold -> NOT REPLICATED; else all three PASS "
                                   "-> REPLICATED; else INCOMPLETE (Appendix A §1, D25)"},
        "integrity": integrity,
        "tests": {"a": ev_a["test"], "b": ev_b["test"], "stage2": ev_s2["test"]},
        "baselines": {"a": ev_a["baselines"], "b": ev_b["baselines"], "stage2": ev_s2["baselines"]},
        "paired_diffs": {"definition": "CNN minus scorer, replicate by replicate on the same cluster resamples",
                         "a": ev_a["paired"], "b": ev_b["paired"], "stage2": ev_s2["paired"]},
        "secondary": {
            "slope_only_check": {"a": ev_a["secondary"]["slope_only_check"],
                                 "b": ev_b["secondary"]["slope_only_check"]},
            "pair_accuracy": {"a": ev_a["secondary"]["pair_accuracy"], "b": ev_b["secondary"]["pair_accuracy"]},
            "jihomoravsky_nonflysch": ev_b["secondary"].get("jihomoravsky_nonflysch"),
            "stage2_ring_as_fp": ev_s2["secondary"]["stage2_ring_as_fp"],
        },
        "bootstrap": {"n": spec.n_replicates, "ci": spec.ci, "seed": spec.seed, "method": spec.method,
                      "primary_unit": spec.unit, "secondary_unit": "block"},
        "spike_claims": {k: {kk: vv for kk, vv in c.items() if kk != "brief_text"} for k, c in SPIKE_CLAIMS.items()},
    }
    m["discrepancies"] = compare_with_spike(m)

    histories = {r: {u.seed: u.history for u in units.values() if u.run == r and u.history is not None
                     and len(u.history)}
                 for t in ("a", "b", "stage2") for r in runs[t]}
    histories = {r: h for r, h in histories.items() if h}
    best = {r: {u.seed: int(u.info["best_epoch"]) for u in units.values()
                if u.run == r and u.ok and u.info.get("best_epoch") is not None} for r in histories}
    m["figures"] = {"learning_curves": {"file": "learning_curves.png", "runs": list(histories)} if histories
                    else None}

    out = json.loads(dumps_metrics(m).decode("utf-8"))
    if write:
        _atomic_write(paths.results / "metrics.json", dumps_metrics(m))
        _atomic_write(paths.results / "metrics_table.md", metrics_table_md(out).encode("utf-8"))
        if histories:
            learning_curves(histories, best, paths.figures / "learning_curves.png")
        else:
            (paths.figures / "learning_curves.png").unlink(missing_ok=True)  # never leave a stale figure
        log.info("wrote %s, %s%s", paths.results / "metrics.json", paths.results / "metrics_table.md",
                 f", {paths.figures / 'learning_curves.png'}" if histories else "")
    return out


def stage2_allowed(metrics: Mapping[str, Any]) -> bool:
    """For the runner: stage 2 runs only if (a) and (b) both pass (Appendix A §1)."""
    return metrics["tests"]["a"]["status"] == V.PASS and metrics["tests"]["b"]["status"] == V.PASS


# ---------------------------------------------------------------------------------------------------------------
# metrics_table.md
# ---------------------------------------------------------------------------------------------------------------

def _f(x: Any) -> str:
    return "n/a" if x is None else f"{float(x):.3f}"


def _ci_txt(ci: Any) -> str:
    if not ci or ci.get("low") is None or ci.get("high") is None:
        return "n/a"
    return f"[{ci['low']:.3f}, {ci['high']:.3f}]"


def _spike_ci_txt(c: Mapping[str, Any]) -> str:
    return "" if not c["spike_ci"] else f"[{c['spike_ci'][0]:.3f}, {c['spike_ci'][1]:.3f}]"


def _yn(x: Any) -> str:
    return "n/a" if x is None else ("yes" if x else "no")


def _flag(comp: Mapping[str, Any] | None) -> str:
    if comp is None:
        return "n/a"
    f = comp["outside_spike_ci"]
    return "**yes**" if f else ("no" if f is False else f"n/a ({comp['note']})")


def metrics_table_md(m: Mapping[str, Any]) -> str:
    """Our numbers next to the spike's claims, from the (rounded) metrics dict. Deterministic text."""
    comps = {c["id"]: c for c in m["discrepancies"]["comparisons"]}
    t = m["tests"]
    vd = m["verdict_detail"]
    L: list[str] = []
    L.append("# Replication metrics")
    L.append("")
    L.append("Generated by `scripts/evaluate.py` from saved predictions; do not edit. The full record is "
             "`metrics.json`.")
    if m["smoke"]:
        L.append("")
        L.append("**SMOKE TEST.** These numbers come from the smoke subset and are never used to change anything.")
    L.append("")
    L.append(f"**Verdict: {m['verdict']}** ((a) {vd['a']}, (b) {vd['b']}, stage 2 {vd['stage2']})")
    L.append("")
    for r in vd["reasons"]:
        L.append(f"- {r}")
    L.append("")
    integ = m["integrity"]
    L.append(f"Config hash `{m['config_hash']}` ({integ['config_hash_source']}); training commit "
             f"`{m['git_commit'] or 'n/a'}`. Units done: {integ['n_done']} of {integ['n_expected']}.")
    for key in ("missing", "failed", "interrupted", "hash_mismatch", "invalid"):
        if integ[key]:
            L.append(f"- {key.replace('_', ' ')}: {', '.join(integ[key])}")
    L.append("")
    L.append("## Pre-registered tests")
    L.append("")
    L.append("The point estimate decides. Borderline = the sheet CI straddles the threshold (reported, never "
             "decisive). Discrepancy = our estimate outside the spike's 95 % CI (never changes the verdict).")
    L.append("")
    L.append("| Test | Threshold | Ours | Sheet 95 % CI | Block 95 % CI | Status | Borderline | Spike | "
             "Spike 95 % CI | Discrepancy |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = (("(a) held-out flysch, ROC AUC (pooled out-of-fold)", "a", "a_cnn"),
            ("(b) transfer to other kraje, ROC AUC", "b", "b_cnn"),
            ("Stage 2, precision at 50 % object recall", "stage2", "stage2_cnn"))
    for label, key, cid in rows:
        r, c = t[key], comps[cid]
        est = _f(r["estimate"]) if not (key == "stage2" and r.get("reached") is False) else "not reached"
        ci = _ci_txt(r["sheet_ci"]) + ("†" if key == "stage2" and r["sheet_ci"] else "")
        L.append(f"| {label} | ≥ {r['threshold']:.2f} | {est} | {ci} | {_ci_txt(r['block_ci'])} | {r['status']} | "
                 f"{_yn(r['borderline'])} | {c['spike']:.3f} | {_spike_ci_txt(c)}{'†' if key == 'stage2' else ''} | "
                 f"{_flag(c)} |")
    L.append("")
    L.append("† Post hoc sheet-level bootstrap, not pre-registered. A replicate that never reaches 50 % recall "
             "counts as precision 0.")
    L.append("")
    L.append("## Baselines and comparators")
    L.append("")
    L.append("| Test | Scorer | Kind | Ours | Sheet 95 % CI | CNN − scorer [sheet 95 % CI] | Spike | Discrepancy |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    spike_by_scorer = {("a", "terrain_stats_lr"): "a_best_noncnn", ("b", "terrain_stats_gbm"): "b_best_noncnn",
                       ("stage2", "slope_susc_lr"): "stage2_best_noncnn", ("stage2", "random"): "stage2_random"}
    n_rows = 0
    for key in ("a", "b", "stage2"):
        bl = m["baselines"][key]
        for name, b in sorted(bl.get("scorers", {}).items()):
            val = b.get("auc") if key != "stage2" else (b.get("estimate") if b.get("reached") is not False
                                                        else None)
            val_txt = "not reached" if key == "stage2" and b.get("reached") is False else _f(val)
            d = m["paired_diffs"][key]["sheet"].get(name)
            d_txt = f"{_f(d['estimate'])} {_ci_txt(d['ci'])}" if d else "n/a"
            cid = spike_by_scorer.get((key, name))
            c = comps.get(cid) if cid else None
            sp = "" if c is None else f"{c['spike']:.3f} {_spike_ci_txt(c)}".strip()
            L.append(f"| {key} | {name} | {b.get('kind', '')} | {val_txt} | {_ci_txt(b.get('sheet_ci'))} | {d_txt} | "
                     f"{sp} | {_flag(c) if c else ''} |")
            n_rows += 1
        if bl.get("status") not in (None, "ok"):
            L.append(f"| {key} | ({bl.get('file')}: {bl['status']}) | | | | | | |")
            n_rows += 1
    if not n_rows:
        L.append("| | (none available) | | | | | | |")
    L.append("")
    L.append("## Post hoc checks")
    L.append("")
    L.append("| Check | Ours | 95 % CI | Spike | Discrepancy |")
    L.append("| --- | --- | --- | --- | --- |")
    sec = m["secondary"]
    for label, cid, ci in (
            ("(a) same-sheet pair accuracy", "a_pair_accuracy",
             (_get(sec, "pair_accuracy.a.same_sheet.sheet_ci"))),
            ("(b) same-sheet pair accuracy", "b_pair_accuracy",
             (_get(sec, "pair_accuracy.b.same_sheet.sheet_ci"))),
            ("Stage 2 P@R50, ring counted as FP", "stage2_ring_as_fp", None),
            ("(b) non-flysch Jihomoravský ROC AUC", "b_jihomoravsky_nonflysch",
             _get(sec, "jihomoravsky_nonflysch.sheet_ci"))):
        c = comps[cid]
        L.append(f"| {label} | {_f(c['ours'])} | {_ci_txt(ci)} | {c['spike']:.3f} | {_flag(c)} |")
    for key in ("a", "b"):
        s = _get(sec, f"slope_only_check.{key}")
        if s:
            L.append(f"| ({key}) slope-only AUC (matching check, about 0.5 expected) | {_f(s['auc'])} | "
                     f"{_ci_txt(s['sheet_ci'])} | | |")
    L.append("")
    L.append("## Spread")
    L.append("")
    L.append("| Test | Per fold (seed ensemble) | Per seed |")
    L.append("| --- | --- | --- |")
    for key in ("a", "b"):
        r = t[key]
        pf = "n/a" if r["per_fold"] is None else ", ".join(f"{k}: {_f(v['auc'])}" for k, v in
                                                           sorted(r["per_fold"].items()))
        ps = ", ".join(f"s{k}: {_f(v['auc'])}" for k, v in sorted(r["per_seed"].items())) or "n/a"
        L.append(f"| ({key}) | {pf} | {ps} |")
    L.append("")
    return "\n".join(L)
