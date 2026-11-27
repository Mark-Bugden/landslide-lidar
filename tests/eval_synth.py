"""Synthetic inputs for the evaluation tests (P1-13): meta.csv, run directories, baseline predictions, stage-2
detections and targets, all written under a temporary root. Nothing here reads real data.

Layout of the synthetic dataset (2 sheets per block, `N_PAIRS` slope-matched pairs per sheet, 2 background patches
per sheet):
- flysch blocks FB0..FB9, fold = i % 5; FB0..FB6 are Zlínský / Moravskoslezský (group_b "train"), FB7..FB9 are
  Jihomoravský / Olomoucký (group_b "test");
- non-flysch blocks NB0..NB5, fold -1; NB0 Zlínský (train); NB1, NB2 Jihomoravský, NB3 Ústecký, NB4 Liberecký,
  NB5 Olomoucký (test).
Per sheet the negatives are matched at level 1 (same sheet) except one at level 3 (same sheet, adjacent bin) and
one at level 2 (the other sheet of the block). Negative sources rotate over random, quarry, rock, cut.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from landslide.config import config_hash, load_config
from landslide.paths import Paths

N_PAIRS = 6
N_BG = 2
SOURCES = ("random", "quarry", "random", "rock", "random", "cut")

FLYSCH_BLOCKS = [(f"FB{i}", i % 5, "train" if i < 7 else "test",
                  ("Zlínský", "Moravskoslezský")[i % 2] if i < 7 else ("Jihomoravský", "Olomoucký")[i % 2])
                 for i in range(10)]
NONFLYSCH_BLOCKS = [("NB0", -1, "train", "Zlínský"), ("NB1", -1, "test", "Jihomoravský"),
                    ("NB2", -1, "test", "Jihomoravský"), ("NB3", -1, "test", "Ústecký"),
                    ("NB4", -1, "test", "Liberecký"), ("NB5", -1, "test", "Olomoucký")]


def make_cfg(n_boot: int = 200) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg["eval"]["bootstrap"]["n"] = n_boot
    return cfg


def synth_meta(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    pair = 0
    blocks = [(b, f, g, k, True) for b, f, g, k in FLYSCH_BLOCKS] + \
             [(b, f, g, k, False) for b, f, g, k in NONFLYSCH_BLOCKS]
    for b, fold, group, kraj, flysch in blocks:
        stratum = ("F_" if flysch else "N_") + ("ZLMS" if group == "train" else "other")
        sheets = [f"{b}-s0", f"{b}-s1"]
        for si, sheet in enumerate(sheets):
            other = sheets[1 - si]
            for j in range(N_PAIRS):
                slope = float(rng.uniform(5, 30))
                level = 3 if j == N_PAIRS - 2 else (2 if j == N_PAIRS - 1 else 1)
                common = dict(kraj=kraj, flysch=flysch, fold=fold, group_b=group, block=b, stratum=stratum,
                              pair_id=pair, x=0.0, y=0.0)
                rows.append(dict(common, label=1, sheet=sheet, source="positive", match_level=0,
                                 slope_mean=slope, polygon_objectid=pair))
                rows.append(dict(common, label=0, sheet=other if level == 2 else sheet,
                                 source=SOURCES[j % len(SOURCES)],
                                 match_level=level, slope_mean=slope + float(rng.normal(0, 0.5)),
                                 polygon_objectid=-1))
                pair += 1
            for _ in range(N_BG):
                rows.append(dict(kraj=kraj, flysch=flysch, fold=fold, group_b=group, block=b, stratum=stratum,
                                 pair_id=-1, x=0.0, y=0.0, label=0, sheet=sheet, source="background",
                                 match_level=0, slope_mean=float(rng.uniform(0, 10)), polygon_objectid=-1))
    meta = pd.DataFrame(rows)
    meta.insert(0, "id", np.arange(len(meta)))
    meta["slope_bin"] = (meta["slope_mean"] // 2).astype(int)
    return meta


def unit_test_ids(meta: pd.DataFrame, run: str) -> np.ndarray:
    nb = meta["source"] != "background"
    if run.startswith("a_f"):
        return meta.loc[nb & (meta["fold"] == int(run[3:])), "id"].to_numpy()
    if run == "b":
        return meta.loc[nb & (meta["group_b"] == "test"), "id"].to_numpy()
    return np.empty(0, dtype=np.int64)


def cnn_probs(meta: pd.DataFrame, ids: np.ndarray, seed: int, strength: float) -> np.ndarray:
    """Seed-dependent probabilities with a shared per-patch component (so seeds correlate)."""
    y = meta.set_index("id").loc[ids, "label"].to_numpy()
    shared = np.random.default_rng(10_000).normal(0, 1, size=len(meta))[ids]
    own = np.random.default_rng(seed * 7919 + 17).normal(0, 0.6, size=len(meta))[ids]
    z = strength * (2 * y - 1) + shared + own
    return 1.0 / (1.0 + np.exp(-z))


def write_json(p: Path, obj: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_unit(paths: Paths, run: str, seed: int, ids: np.ndarray, probs: np.ndarray, chash: str, *,
               status: str = "done", epochs: int = 6, best_epoch: int = 3, commit: str = "c0ffee") -> Path:
    d = paths.run_dir(run, seed)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 100 * len(run))
    ep = np.arange(1, epochs + 1)
    pd.DataFrame({"epoch": ep, "lr": 3e-4, "train_loss": 0.7 - 0.05 * ep + rng.normal(0, 0.01, epochs),
                  "train_auc": 0.6 + 0.04 * ep, "val_loss": 0.68 - 0.03 * ep + rng.normal(0, 0.01, epochs),
                  "val_auc": 0.6 + 0.03 * ep - 0.01 * np.maximum(ep - best_epoch, 0),
                  "seconds": 1.0}).to_csv(d / "history.csv", index=False)
    if status == "failed":
        write_json(d / "failed.json", {"unit": f"{run}_s{seed}", "error": "RuntimeError: boom", "config_hash": chash})
        return d
    if status == "interrupted":
        write_json(d / "interrupted.json", {"unit": f"{run}_s{seed}", "epochs_completed": 2, "config_hash": chash})
        return d
    pd.DataFrame({"id": ids, "prob": probs}).to_csv(d / "preds_test.csv", index=False)
    pd.DataFrame({"id": [0, 1], "prob": [0.4, 0.6]}).to_csv(d / "preds_val.csv", index=False)
    write_json(d / "done.json", {
        "unit": f"{run}_s{seed}", "run": run, "seed": seed, "config_hash": chash, "git_commit": commit,
        "git_dirty": False, "best_epoch": best_epoch, "best_val_auc": 0.7, "n_train": 100, "n_val": 20,
        "n_test": int(len(ids)), "seconds": 12.5, "peak_gpu_mem_gb": 1.2, "epochs_run": epochs,
        "stop_reason": "early_stopping", "smoke": paths.smoke, "finished": "2026-11-27T00:00:00+00:00"})
    return d


def write_baselines(paths: Paths, meta: pd.DataFrame, seed: int = 5) -> None:
    rng = np.random.default_rng(seed)
    d = paths.baselines_dir
    d.mkdir(parents=True, exist_ok=True)
    for test, runs in (("a", [f"a_f{k}" for k in range(5)]), ("b", ["b"])):
        ids = np.sort(np.concatenate([unit_test_ids(meta, r) for r in runs]))
        m = meta.set_index("id").loc[ids]
        y = m["label"].to_numpy()
        df = pd.DataFrame({"id": ids, "slope_only": m["slope_mean"].to_numpy(),
                           "susceptibility": rng.integers(0, 3, size=len(ids)).astype(float),
                           "slope_susc_lr": rng.normal(0.2 * y, 1.0),
                           "terrain_stats_lr": rng.normal(0.6 * y, 1.0),
                           "terrain_stats_gbm": rng.normal(0.5 * y, 1.0)})
        if test == "a":
            df["fold"] = m["fold"].to_numpy()
        df.to_csv(d / f"preds_{test}.csv", index=False)


def stage2_sheets(cfg: dict[str, Any], smoke: bool) -> list[str]:
    s = cfg["replication"]["splits"]["stage2_sheets"]
    sheets = list(s["flysch"]) + list(s["nonflysch"])
    return sorted(sheets[:1] if smoke else sheets)


def synth_targets(sheets: list[str], per_sheet: int = 6) -> pd.DataFrame:
    rows = [{"objectid": 1000 * (i + 1) + j, "eligible": True, "sheet": s}
            for i, s in enumerate(sheets) for j in range(per_sheet)]
    rows += [{"objectid": 9_000_000 + i, "eligible": False, "sheet": None} for i in range(3)]
    rows += [{"objectid": 9_100_000, "eligible": True, "sheet": "99-99-99"}]
    return pd.DataFrame(rows)


def synth_detections(scorer: str, sheets: list[str], targets: pd.DataFrame, *, tp_frac: float | list[float],
                     n_fp: int = 10, quality: float = 1.5, seed: int = 0) -> pd.DataFrame:
    """Detections with known outcomes. `tp_frac` (per sheet or one value) = the share of a sheet's targets that get
    a TP detection; each TP also gets one lower-scored duplicate; every sheet gets `n_fp` FPs and 3 ignored."""
    rng = np.random.default_rng(seed)
    fr = tp_frac if isinstance(tp_frac, list) else [tp_frac] * len(sheets)
    rows = []
    for s, f in zip(sheets, fr):
        t = targets.loc[targets["eligible"] & (targets["sheet"] == s), "objectid"].to_numpy()
        n_tp = int(round(f * len(t)))
        for o in t[:n_tp]:
            sc = float(rng.normal(quality, 1))
            rows.append((scorer, s, sc, "tp", int(o)))
            rows.append((scorer, s, sc - abs(float(rng.normal(0.5, 0.2))), "duplicate", int(o)))
        rows += [(scorer, s, float(rng.normal(0, 1)), "fp", -1) for _ in range(n_fp)]
        rows += [(scorer, s, float(rng.normal(0.5, 1)), "ignored", -1) for _ in range(3)]
    df = pd.DataFrame(rows, columns=["scorer", "sheet", "score", "outcome", "target_objectid"])
    df["x"] = 0.0
    df["y"] = 0.0
    return df


@dataclass
class World:
    root: Path
    paths: Paths
    cfg: dict[str, Any]
    meta: pd.DataFrame
    chash: str
    probs: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    targets: pd.DataFrame | None = None


def build_world(root: Path, cfg: dict[str, Any] | None = None, *, smoke: bool = False, strength_a: float = 2.0,
                strength_b: float = 1.5, unit_status: dict[str, str] | None = None, stage2: str | None = "pass",
                baselines: bool = True, comparators: bool = True) -> World:
    """Write a complete synthetic evaluation input under `root`.

    `unit_status` maps unit names to "failed" / "interrupted" / "missing" (default done). `stage2` is "pass",
    "fail" (reached, low precision), "not_reached", or None (no stage-2 output at all; s2 units are still written
    unless marked missing).
    """
    cfg = cfg if cfg is not None else make_cfg()
    paths = Paths(smoke=smoke, root=root)
    chash = config_hash(cfg)
    meta = synth_meta()
    paths.dataset.mkdir(parents=True, exist_ok=True)
    meta.to_csv(paths.meta_csv, index=False)
    w = World(root, paths, cfg, meta, chash)
    unit_status = unit_status or {}
    if smoke:
        runs, seeds = list(cfg["smoke"]["runs"]), [int(s) for s in cfg["smoke"]["seeds"]]
    else:
        runs = [f"a_f{k}" for k in range(int(cfg["splits"]["n_folds_a"]))] + ["b", "s2"]
        seeds = [int(s) for s in cfg["train"]["seeds"]]
    for run in runs:
        ids = unit_test_ids(meta, run)
        if smoke and len(ids):
            ids = ids[: max(4, len(ids) // 2) // 2 * 2]  # a subset, as the smoke subset would be
        strength = strength_a if run.startswith("a_f") else strength_b
        w.probs[run] = {}
        for s in seeds:
            name = f"{run}_s{s}"
            st = unit_status.get(name, "done")
            if st == "missing":
                continue
            p = cnn_probs(meta, ids, s, strength) if len(ids) else np.empty(0)
            w.probs[run][s] = (ids, p)
            write_unit(paths, run, s, ids, p, chash, status=st)
    if baselines:
        write_baselines(paths, meta)
    sheets = stage2_sheets(cfg, smoke)
    w.targets = synth_targets(sheets)
    paths.derived.mkdir(parents=True, exist_ok=True)
    w.targets.to_parquet(paths.polygons_parquet, index=False)
    frames = []
    if stage2 is not None:
        frac = {"pass": 0.85, "fail": 0.85, "not_reached": 0.3}[stage2]
        quality = {"pass": 3.0, "fail": -0.5, "not_reached": 3.0}[stage2]
        n_fp = {"pass": 4, "fail": 40, "not_reached": 4}[stage2]
        frames.append(synth_detections("cnn", sheets, w.targets, tp_frac=frac, n_fp=n_fp, quality=quality, seed=1))
    if comparators:
        frames.append(synth_detections("slope_susc_lr", sheets, w.targets, tp_frac=0.7, n_fp=20, quality=0.3,
                                       seed=2))
        frames.append(synth_detections("random", sheets, w.targets, tp_frac=0.7, n_fp=30, quality=0.0, seed=3))
    paths.stage2.mkdir(parents=True, exist_ok=True)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(paths.stage2 / "detections.parquet", index=False)
    return w
