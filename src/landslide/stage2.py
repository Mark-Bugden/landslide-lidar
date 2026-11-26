"""Stage 2: sliding-window scores on the 20 reserved sheets, NMS, object matching and P@R50 (task P1-12).

BRIEF.md Appendix A §9 and Appendix B `stage2`; PLAN.md D22, D23 and the §4.3 contracts. Every choice below is
recorded in `docs/decisions/P1-12.md`.

**Lattice (D22).** Window centres are the pixel corners (r, c) of the sheet raster with r ≡ c ≡ 0 (mod `stride_px`)
from the raster origin, kept if the centre lies in the sheet polygon (closed: the boundary counts). A window is the
150 px square centred on the corner (`landslide.dtm` convention: rows r − 75 … r + 74). A centre whose window plus
the 28 px channel context is not fully valid (or leaves the raster) is dropped and counted; on the 20 reserved sheets
there is none, because their rasters are fully valid and every centre is ≥ 125 px (the 250 m margin) from the raster
edge. Rows are in row-major (r, c) order. Per-window statistics come from summed-area tables: slope mean and std
(ddof 0), and the mean and high-class fraction of the susceptibility code (0/1/2) over the 150 x 150 window.

**Scorers.** `slope_only` (window-mean slope), `susceptibility` (window-mean class), `slope_susc_lr` (logistic
regression on slope mean, slope std, susceptibility mean and high fraction, fitted on the stage-1 positives vs the
background sample of `meta.csv`), `random` (U[0, 1) from the `random_baseline` stream) and `cnn` (the s2 model's
seeds: channels 0–5 from `channels.full_channels`, detr per window with `channels.detrend`, cast to float16 like
`patches.npy`, normalised with each seed's `norm.json`, 8-view TTA with `predict.tta_predict`, then the mean of the
seed probabilities).

**NMS (D23).** Greedy over the lattice of one sheet in descending jittered score, where jittered = score + `scale` x
U[0, 1) (`replication.nms_jitter_scale` = 1e-9, `nms_jitter` stream). A selected cell suppresses every cell whose
centre is within `nms_radius_m` (d ≤ 150 m, compared exactly on integer pixel offsets). Every cell ends selected or
suppressed. Exact ties left after the jitter are broken by (sheet, row, col).

**Matching (Appendix A §9, D23).** The detections of all evaluated sheets are processed together in descending
jittered score (ties: sheet, row, col). For each detection, in order: TP if its centre is within `match_buffer_m`
(≤ 50 m, exact GEOS distance) of a still-unmatched target (the lowest objectid if several; that target becomes
matched); duplicate if it is within 50 m of a target that is already matched; ignored if it is within
`ignore_buffer_m` (≤ 150 m) of any register polygon (all 21,356, targets included; invalid ones repaired as in P1-08);
FP otherwise. Targets are the eligible polygons assigned to the evaluated sheets (465 for the 20).

**Metrics.** `metrics.pr_sweep` on the jittered scores (so its order is the matching order): P@R50 (primary),
interpolated precision and max recall, pooled and per stratum (flysch / non-flysch sheets, each matched on its own),
plus the post hoc "ring counted as FP" variant (ignored → fp; duplicates stay ignored). 50 % recall never reached is
status "not_reached", a FAIL.

**Random streams.** Per sheet, the child `SeedSequence([replication.seed, stream_id], spawn_key=(sheet code,))` of the
registered streams `nms_jitter` and `random_baseline` (as P1-09's per-sheet streams), so a sheet's draws do not depend
on which other sheets are evaluated (the smoke test uses one sheet). Every scorer uses the same jitter on a sheet.

**Outputs** (under `Paths.stage2`, i.e. `data/stage2/` or `data/smoke/stage2/`): `<scorer>/<sheet>.npz` (row, col, x,
y, score), `lattice/<sheet>.npz` (the lattice and its window statistics), `s2_infer/<sheet>.npz` (per-seed CNN
probabilities, for resume), `detections.parquet` (scorer, sheet, x, y, score, outcome, target_objectid, plus
score_jittered, row, col, rank) and `summary.json`. The tracked `results/stage2_baselines.csv` has one row per
baseline scorer. Nothing here draws a map or writes sheet geometry.

No CNN may see real data before the freeze (gate G3): `run_cnn` is only called by the runner (smoke test, phase 2).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely

from landslide import sat
from landslide.channels import WINDOW_CHANNEL, channel_params, detrend, full_channels
from landslide.config import config_hash
from landslide.dtm import read_dtm
from landslide.geo import repair
from landslide.logs import get_logger, progress
from landslide.metrics import PRSweep, pr_sweep
from landslide.paths import REPO_ROOT, Paths
from landslide.sampling import CornerGrid, grid_of_sheet, sheet_rng
from landslide.splits import stage2_sheets

if TYPE_CHECKING:
    from torch import nn

    from landslide.predict import Normaliser

log = get_logger("stage2")

# ---- scorers ----
CNN = "cnn"
#: Baseline scorers in CSV order, with their `kind` (Appendix A §10 terminology: only unfitted scores are baselines).
BASELINE_KINDS: dict[str, str] = {
    "slope_only": "non_ml_baseline",
    "susceptibility": "non_ml_baseline",
    "slope_susc_lr": "classical_ml",
    "random": "chance",
}
SCORER_KINDS: dict[str, str] = {**BASELINE_KINDS, CNN: "cnn"}
#: Scorers that need only the DTM, the susceptibility raster and the polygons (run in phase 1).
UNFITTED_BASELINES = ("slope_only", "susceptibility", "random")
LR_SCORER = "slope_susc_lr"
#: The LR comparator's features, in `meta.csv` column names (PLAN.md §4.3, D19).
LR_FEATURES = ("slope_mean", "slope_std", "susc_mean", "susc_high_frac")
STATS = LR_FEATURES
STRATA = ("flysch", "nonflysch")
NO_TARGET = -1
#: Windows assembled per CNN chunk (performance only: 512 windows x 7 x 150 x 150 float16 = 161 MB).
CNN_CHUNK_WINDOWS = 512
#: P1-06's measured 8-view TTA throughput on the RTX 3060 (images per second), for the cost estimate only.
P1_06_TTA_IMAGES_PER_S = 4600.0
DETECTION_COLUMNS = ("scorer", "sheet", "x", "y", "score", "outcome", "target_objectid",
                     "score_jittered", "row", "col", "rank")


# ======================================================================================================================
# Parameters
# ======================================================================================================================


@dataclass(frozen=True)
class Stage2Params:
    cell_m: float
    px: int
    context_px: int
    stride_px: int
    nms_radius_m: float
    match_buffer_m: float
    ignore_buffer_m: float
    recall_level: float
    precision_min: float
    jitter_scale: float
    seed: int
    model_seeds: tuple[int, ...]
    batch_size: int

    @property
    def half(self) -> int:
        """Half the window (75 px): the window at corner (r, c) covers rows r - half .. r + half - 1."""
        return self.px // 2

    @property
    def valid_half(self) -> int:
        """Half-size of the box that must be valid: the window plus the channel context (103 px)."""
        return self.half + self.context_px

    @property
    def nms_radius_px2(self) -> float:
        """(NMS radius in pixels)^2: a cell at integer offset (dr, dc) is suppressed iff dr^2 + dc^2 <= this."""
        return (self.nms_radius_m / self.cell_m) ** 2


def stage2_params(cfg: Mapping[str, Any]) -> Stage2Params:
    cp = channel_params(cfg)
    s2, th, rep = cfg["stage2"], cfg["eval"]["thresholds"], cfg["replication"]
    p = Stage2Params(
        cell_m=cp.cell_m, px=cp.patch_px, context_px=cp.context_px, stride_px=int(s2["stride_px"]),
        nms_radius_m=float(s2["nms_radius_m"]), match_buffer_m=float(s2["match_buffer_m"]),
        ignore_buffer_m=float(s2["ignore_buffer_m"]), recall_level=float(th["stage2_recall"]),
        precision_min=float(th["stage2_precision_min"]), jitter_scale=float(rep["nms_jitter_scale"]),
        seed=int(rep["seed"]), model_seeds=tuple(int(s) for s in cfg["train"]["seeds"]),
        batch_size=int(cfg["train"]["batch_size"]))
    if p.stride_px < 1 or p.nms_radius_m <= 0 or p.match_buffer_m < 0 or p.ignore_buffer_m < 0:
        raise ValueError("stage2 stride, NMS radius and buffers must be positive")
    if not 0 < p.jitter_scale < 1e-3:
        raise ValueError(f"nms_jitter_scale {p.jitter_scale} is not a small tie-breaking jitter")
    return p


def cnn_cost_estimate(n_windows: int, n_seeds: int = 3, n_views: int = 8,
                      images_per_s: float = P1_06_TTA_IMAGES_PER_S) -> dict[str, float]:
    """Forward passes (windows x views x seeds) and the GPU time at a given throughput."""
    images = int(n_windows) * int(n_views) * int(n_seeds)
    sec = images / float(images_per_s)
    return {"n_windows": int(n_windows), "n_views": int(n_views), "n_seeds": int(n_seeds), "images": images,
            "images_per_s": float(images_per_s), "seconds": sec, "minutes": sec / 60.0}


# ======================================================================================================================
# Inputs
# ======================================================================================================================


@dataclass
class Inputs:
    """Everything stage 2 reads. `sheets` holds the evaluated sheets (sorted by sheet number) with `sheet`, `flysch`,
    `grid_xmin`, `grid_ymax`, `grid_w`, `grid_h` and `geometry`; `polygons` all register polygons with `objectid`,
    `eligible`, `sheet` and `geometry` (EPSG:5514)."""

    sheets: pd.DataFrame
    polygons: pd.DataFrame
    dtm_path: Callable[[str], Path]
    susc_path: Callable[[str], Path]
    meta_path: Path | None = None

    @property
    def sheet_ids(self) -> list[str]:
        return list(self.sheets["sheet"])

    def flysch_of(self) -> dict[str, bool]:
        return {str(s): bool(f) for s, f in zip(self.sheets["sheet"], self.sheets["flysch"], strict=True)}


def load_inputs(cfg: Mapping[str, Any], paths: Paths, sheets: Sequence[str] | None = None) -> Inputs:
    """The stage-2 sheets of `sheets.parquet` (all 20, or the subset `sheets`) and the register polygons.

    Checks that `is_stage2` marks exactly the config's 20 sheets and that each sheet's flysch flag matches its list.
    """
    import geopandas as gpd

    lists = stage2_sheets(cfg)
    expected = {s: (k == "flysch") for k, v in lists.items() for s in v}
    sh = gpd.read_parquet(paths.sheets_parquet)
    marked = set(sh.loc[sh["is_stage2"].astype(bool), "sheet"])
    if marked != set(expected):
        raise ValueError(f"sheets.parquet is_stage2 marks {sorted(marked ^ set(expected))} differently from the config")
    want = sorted(expected) if sheets is None else sorted(set(sheets))
    unknown = [s for s in want if s not in expected]
    if unknown:
        raise ValueError(f"not stage-2 sheets: {unknown}")
    sel = sh.loc[sh["sheet"].isin(want)].sort_values("sheet").reset_index(drop=True)
    bad = [s for s, f in zip(sel["sheet"], sel["flysch"], strict=True) if bool(f) != expected[s]]
    if bad:
        raise ValueError(f"flysch flag of {bad} differs from its stage-2 list")
    poly = gpd.read_parquet(paths.polygons_parquet)
    return Inputs(sheets=pd.DataFrame(sel), polygons=pd.DataFrame(poly), dtm_path=paths.dtm,
                  susc_path=paths.susceptibility, meta_path=paths.meta_csv)


def targets_of(inputs: Inputs, sheets: Sequence[str] | None = None) -> pd.DataFrame:
    """Eligible polygons assigned to the given sheets (default: all evaluated sheets), sorted by objectid."""
    want = set(inputs.sheet_ids if sheets is None else sheets)
    p = inputs.polygons
    t = p.loc[p["eligible"].astype(bool) & p["sheet"].isin(want), ["objectid", "sheet", "geometry"]]
    return t.sort_values("objectid").reset_index(drop=True)


# ======================================================================================================================
# Lattice (D22) and window statistics
# ======================================================================================================================


@dataclass
class Lattice:
    sheet: str
    row: np.ndarray
    col: np.ndarray
    x: np.ndarray
    y: np.ndarray
    n_inside: int
    n_invalid: int
    stats: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.row.size)


def lattice_corners(geom: Any, grid: CornerGrid, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Corners (r, c) with r, c ≡ 0 (mod stride) from the raster origin whose position lies in `geom` (closed),
    row-major."""
    r = np.arange(0, grid.n_rows, stride, dtype=np.int64)
    c = np.arange(0, grid.n_cols, stride, dtype=np.int64)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    x, y = grid.xy(rr, cc)
    shapely.prepare(geom)
    keep = shapely.intersects_xy(geom, x, y)
    return rr[keep], cc[keep]


def windows_valid(valid: np.ndarray, row: np.ndarray, col: np.ndarray, half: int) -> np.ndarray:
    """True where the corner-centred box of half-size `half` lies inside the array and every pixel is valid."""
    H, W = valid.shape
    inside = (row - half >= 0) & (col - half >= 0) & (row + half <= H) & (col + half <= W)
    out = np.zeros(row.shape, dtype=bool)
    if inside.any():
        S = sat.integral(valid, dtype=np.int64)
        n = sat.corner_box_sums(S, row[inside], col[inside], half)
        out[inside] = n == (2 * half) ** 2
    return out


def window_stats(slope: np.ndarray, susc: np.ndarray | None, row: np.ndarray, col: np.ndarray,
                 half: int) -> dict[str, np.ndarray]:
    """Per-window statistics over the (2 half)^2 window from summed-area tables (D19's definitions).

    slope_mean, slope_std (ddof 0, from centred sums of squares); susc_mean (mean class code 0/1/2) and
    susc_high_frac (share of code 2), exact integer sums. The windows must be fully valid.
    """
    n = float((2 * half) ** 2)
    s = np.asarray(slope, dtype=np.float64)
    ok = np.isfinite(s)
    off = float(s[ok].mean()) if ok.any() else 0.0
    d = np.where(ok, s - off, 0.0)
    cnt = sat.corner_box_sums(sat.integral(ok, dtype=np.int64), row, col, half)
    if np.any(cnt != n):
        raise ValueError("a window holds invalid slope cells")
    s1 = sat.corner_box_sums(sat.integral(d), row, col, half) / n
    s2 = sat.corner_box_sums(sat.integral(d * d), row, col, half) / n
    out = {"slope_mean": off + s1, "slope_std": np.sqrt(np.maximum(s2 - s1 * s1, 0.0))}
    if susc is not None:
        code = np.asarray(susc)
        if not np.isin(code, (0, 1, 2)).all():
            raise ValueError("susceptibility codes must be 0, 1 or 2")
        code = code.astype(np.int64)
        out["susc_mean"] = sat.corner_box_sums(sat.integral(code), row, col, half) / n
        out["susc_high_frac"] = sat.corner_box_sums(sat.integral(code == 2, dtype=np.int64), row, col, half) / n
    return out


def build_lattice(sheet: str, geom: Any, grid: CornerGrid, valid: np.ndarray, slope: np.ndarray,
                  susc: np.ndarray | None, p: Stage2Params) -> Lattice:
    """The D22 lattice of one sheet with its window statistics. `valid` is the DTM validity mask."""
    if valid.shape != (grid.n_rows - 1, grid.n_cols - 1):
        raise ValueError(f"raster shape {valid.shape} does not match the grid")
    r, c = lattice_corners(geom, grid, p.stride_px)
    ok = windows_valid(valid, r, c, p.valid_half)
    r, c = r[ok], c[ok]
    x, y = grid.xy(r, c)
    stats = window_stats(slope, susc, r, c, p.half) if r.size else {k: np.zeros(0) for k in STATS}
    return Lattice(sheet=sheet, row=r, col=c, x=x, y=y, n_inside=int(ok.size), n_invalid=int((~ok).sum()),
                   stats=stats)


@dataclass
class SheetRaster:
    """One sheet in memory: its DTM (float64, invalid NaN), the corner grid of `sheets.parquet` and the class codes."""

    sheet: str
    z: np.ndarray
    grid: CornerGrid
    geom: Any
    susc: np.ndarray | None


def read_sheet(inputs: Inputs, sheet: str, cfg: Mapping[str, Any], with_susc: bool = True) -> SheetRaster:
    """Read one sheet's DTM (and susceptibility raster), checking both against the grid in `sheets.parquet`."""
    import rasterio

    row = inputs.sheets.loc[inputs.sheets["sheet"] == sheet].iloc[0]
    cell = float(cfg["dtm"]["cell_m"])
    grid = grid_of_sheet(row, cell)
    dtm = read_dtm(inputs.dtm_path(sheet), cfg)
    H, W = dtm.shape
    if not CornerGrid.from_transform(dtm.transform, H, W).same_lattice(grid):
        raise ValueError(f"{sheet}: the DTM raster grid differs from sheets.parquet")
    susc = None
    if with_susc:
        with rasterio.open(inputs.susc_path(sheet), mode="r") as src:
            same = (src.height, src.width) == (H, W) and                 CornerGrid.from_transform(src.transform, H, W).same_lattice(grid)
            if not same:
                raise ValueError(f"{sheet}: the susceptibility raster is not on the DTM grid")
            susc = src.read(1)
    return SheetRaster(sheet=sheet, z=dtm.z, grid=grid, geom=row["geometry"], susc=susc)


# ======================================================================================================================
# Random streams
# ======================================================================================================================


def sheet_stream(cfg: Mapping[str, Any], name: str, sheet: str) -> np.random.Generator:
    """The sheet's child of the registered stream `name`, seeded with `replication.seed`."""
    return sheet_rng(cfg, int(cfg["replication"]["seed"]), name, sheet)


def jitter(cfg: Mapping[str, Any], sheet: str, n: int) -> np.ndarray:
    """The tie-breaking jitter of a sheet's n lattice cells (row-major): scale x U[0, 1)."""
    return float(cfg["replication"]["nms_jitter_scale"]) * sheet_stream(cfg, "nms_jitter", sheet).random(int(n))


def random_scores(cfg: Mapping[str, Any], sheet: str, n: int) -> np.ndarray:
    """The random baseline: U[0, 1) per lattice cell (row-major) from the sheet's `random_baseline` stream."""
    return sheet_stream(cfg, "random_baseline", sheet).random(int(n))


# ======================================================================================================================
# Baseline scorers
# ======================================================================================================================


def make_lr(cfg: Mapping[str, Any]):
    """StandardScaler + LogisticRegression (sklearn defaults, fixed random_state = `replication.seed`)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(random_state=int(cfg["replication"]["seed"])))


def lr_training_rows(meta: pd.DataFrame) -> pd.DataFrame:
    """The LR comparator's training rows: every stage-1 positive and the unmatched background sample."""
    pos = meta["source"] == "positive"
    bg = meta["source"] == "background"
    if not (meta.loc[pos, "label"] == 1).all() or not (meta.loc[bg, "label"] == 0).all():
        raise ValueError("meta.csv: positives must have label 1 and background rows label 0")
    rows = meta.loc[pos | bg]
    if pos.sum() == 0 or bg.sum() == 0:
        raise ValueError("meta.csv holds no positives or no background rows")
    missing = [c for c in LR_FEATURES if c not in rows.columns]
    if missing:
        raise ValueError(f"meta.csv lacks the LR features {missing}")
    if not np.isfinite(rows[list(LR_FEATURES)].to_numpy(dtype=np.float64)).all():
        raise ValueError("non-finite LR features in meta.csv")
    return rows


def fit_lr_comparator(meta: pd.DataFrame, cfg: Mapping[str, Any]):
    """Fit `make_lr` on (slope mean, slope std, susc mean, susc high fraction): stage-1 positives vs background."""
    rows = lr_training_rows(meta)
    model = make_lr(cfg)
    model.fit(rows[list(LR_FEATURES)].to_numpy(dtype=np.float64), rows["label"].to_numpy(dtype=np.int64))
    return model


def lr_scores(model: Any, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    X = np.column_stack([np.asarray(stats[k], dtype=np.float64) for k in LR_FEATURES])
    return model.predict_proba(X)[:, 1].astype(np.float64)


def baseline_scores(scorer: str, lat: Lattice, cfg: Mapping[str, Any], lr_model: Any = None) -> np.ndarray:
    """The lattice scores of a baseline scorer (higher = more landslide-like)."""
    if scorer == "slope_only":
        return np.asarray(lat.stats["slope_mean"], dtype=np.float64)
    if scorer == "susceptibility":
        return np.asarray(lat.stats["susc_mean"], dtype=np.float64)
    if scorer == "random":
        return random_scores(cfg, lat.sheet, lat.n)
    if scorer == LR_SCORER:
        if lr_model is None:
            raise ValueError("slope_susc_lr needs a fitted model (fit_lr_comparator on meta.csv)")
        return lr_scores(lr_model, lat.stats)
    raise ValueError(f"unknown baseline scorer {scorer!r}")


# ======================================================================================================================
# CNN scorer
# ======================================================================================================================


@dataclass
class SeedModel:
    seed: int
    model: nn.Module
    norm: Normaliser


def load_s2_models(cfg: Mapping[str, Any], paths: Paths, device: Any = None, run: str = "s2",
                   seeds: Sequence[int] | None = None) -> list[SeedModel]:
    """The s2 run's finished seeds: `best.pt` (`model.load_checkpoint`) and `norm.json` per seed, in eval mode.

    A seed without `done.json`, `best.pt` or `norm.json`, or trained under another config hash, raises: a missing
    seed makes stage 2 INCOMPLETE, never a smaller ensemble.
    """
    import torch

    from landslide.model import load_checkpoint
    from landslide.predict import Normaliser

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chash = config_hash(dict(cfg))
    out = []
    for seed in (seeds if seeds is not None else cfg["train"]["seeds"]):
        d = paths.run_dir(run, int(seed))
        for f in ("done.json", "best.pt", "norm.json"):
            if not (d / f).is_file():
                raise FileNotFoundError(f"stage 2 needs {d / f} (unit {run}_s{seed} not finished)")
        model, meta = load_checkpoint(d / "best.pt", dict(cfg), map_location=dev)
        if meta.get("config_hash") != chash:
            raise ValueError(f"{d / 'best.pt'} was trained under config {meta.get('config_hash')}, not {chash}")
        out.append(SeedModel(seed=int(seed), model=model, norm=Normaliser.from_json(d / "norm.json")))
    return out


def window_channels(terrain: np.ndarray, z: np.ndarray, row: np.ndarray, col: np.ndarray,
                    cfg: Mapping[str, Any], dtype: Any = np.float64) -> np.ndarray:
    """(n, 7, px, px) windows in config channel order: channels 0–5 cut from the full-raster `terrain`
    (`full_channels` order) and detr = `detrend` of each window's z, computed in float64 and cast to `dtype`."""
    p = channel_params(dict(cfg))
    h = p.patch_px // 2
    if terrain.shape != (len(p.terrain_names), *z.shape):
        raise ValueError(f"terrain has shape {terrain.shape}, expected {(len(p.terrain_names), *z.shape)}")
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    H, W = z.shape
    if np.any(row - h < 0) or np.any(col - h < 0) or np.any(row + h > H) or np.any(col + h > W):
        raise ValueError("a window leaves the raster")
    n = row.size
    zw = np.empty((n, p.patch_px, p.patch_px))
    for k in range(n):
        zw[k] = z[row[k] - h:row[k] + h, col[k] - h:col[k] + h]
    detr = detrend(zw) if n else zw
    out = np.empty((n, len(p.names), p.patch_px, p.patch_px), dtype=dtype)
    ti = {name: i for i, name in enumerate(p.terrain_names)}
    for j, name in enumerate(p.names):
        if name == WINDOW_CHANNEL:
            out[:, j] = detr
        else:
            t = terrain[ti[name]]
            for k in range(n):
                out[k, j] = t[row[k] - h:row[k] + h, col[k] - h:col[k] + h]
    return out


def cnn_window_probs(models: Sequence[SeedModel], terrain: np.ndarray, z: np.ndarray, row: np.ndarray,
                     col: np.ndarray, cfg: Mapping[str, Any], *, chunk_windows: int = CNN_CHUNK_WINDOWS,
                     batch_size: int | None = None, show_progress: bool = False,
                     desc: str | None = None) -> np.ndarray:
    """8-view TTA probabilities of every window for every seed model, shape (n_models, n), float64.

    Windows are built in chunks, cast to float16 (as `patches.npy` stores them; non-finite values raise) and passed
    to `predict.tta_predict` with the model's `norm.json` and chunks of `train.batch_size` samples (8 x that many
    images per forward pass), as in training.
    """
    import torch

    from landslide.predict import tta_predict

    bs = int(batch_size) if batch_size is not None else int(cfg["train"]["batch_size"])
    n = int(np.asarray(row).size)
    out = np.empty((len(models), n), dtype=np.float64)
    # The float16 cast is elementwise, so casting the whole raster once equals casting each window (and is faster).
    t16 = np.asarray(terrain).astype(np.float16)
    bar = progress(total=n, desc=desc or "stage-2 windows", unit="win", leave=False, disable=not show_progress)
    for a in range(0, n, int(chunk_windows)):
        b = min(n, a + int(chunk_windows))
        x = window_channels(t16, z, row[a:b], col[a:b], cfg, dtype=np.float16)
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite window values (or float16 overflow) in windows {a}..{b - 1}")
        xt = torch.from_numpy(x)
        for i, sm in enumerate(models):
            out[i, a:b] = tta_predict(sm.model, xt, batch_size=bs, norm=sm.norm).numpy()
        bar.update(b - a)
    bar.close()
    return out


def cnn_scores(probs: np.ndarray) -> np.ndarray:
    """The seed ensemble: the arithmetic mean of the seed probabilities (Appendix A §7, §9)."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] < 1:
        raise ValueError("expected (n_seeds, n) probabilities")
    return probs.mean(axis=0)


# ======================================================================================================================
# NMS (D23)
# ======================================================================================================================


@dataclass
class NMSResult:
    selected: np.ndarray      # bool per cell
    suppressed_by: np.ndarray  # index of the selected cell that suppressed it; itself for selected cells
    order: np.ndarray         # processing order (indices)


def processing_order(score_j: np.ndarray, *keys: np.ndarray) -> np.ndarray:
    """Indices in descending jittered score; exact ties by the given keys (ascending, first key first)."""
    s = np.asarray(score_j, dtype=np.float64)
    return np.lexsort(tuple(np.asarray(k) for k in reversed(keys)) + (-s,))


def nms(row: np.ndarray, col: np.ndarray, score_j: np.ndarray, radius_px2: float) -> NMSResult:
    """Greedy NMS on lattice cells at integer pixel positions (row, col).

    Cells are taken in descending `score_j` (ties by row, then col). An unsuppressed cell is selected and suppresses
    every cell with (dr^2 + dc^2) <= radius_px2, i.e. distance <= the radius (exact in integers). Every cell ends
    selected or suppressed.
    """
    from scipy.spatial import cKDTree

    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    s = np.asarray(score_j, dtype=np.float64)
    n = row.size
    if s.shape != (n,) or col.shape != (n,):
        raise ValueError("row, col and scores must have the same length")
    if not np.isfinite(s).all():
        raise ValueError("NMS scores contain NaN or inf")
    order = processing_order(s, row, col)
    sup = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return NMSResult(selected=np.zeros(0, dtype=bool), suppressed_by=sup, order=order)
    pts = np.column_stack([row, col]).astype(np.float64)
    nbrs = cKDTree(pts).query_ball_point(pts, r=math.sqrt(radius_px2) + 1.0)
    selected = np.zeros(n, dtype=bool)
    for i in order:
        if sup[i] >= 0:
            continue
        selected[i] = True
        sup[i] = i
        nb = np.asarray(nbrs[i], dtype=np.int64)
        d2 = (row[nb] - row[i]) ** 2 + (col[nb] - col[i]) ** 2
        nb = nb[(d2 <= radius_px2) & (sup[nb] < 0)]
        sup[nb] = i
    if np.any(sup < 0):  # pragma: no cover - every cell is reached by the loop
        raise AssertionError("a lattice cell was neither selected nor suppressed")
    return NMSResult(selected=selected, suppressed_by=sup, order=order)


# ======================================================================================================================
# Object matching (Appendix A §9, D23)
# ======================================================================================================================


def _within(points: np.ndarray, geoms: np.ndarray, dist: float) -> tuple[np.ndarray, np.ndarray]:
    """(point index, geometry index) of every pair at exact GEOS distance <= dist."""
    if not len(geoms) or not len(points):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    tree = shapely.STRtree(geoms)
    ip, ig = tree.query(points, predicate="dwithin", distance=dist + max(1e-6, 1e-9 * dist))
    if ip.size:
        keep = shapely.distance(points[ip], geoms[ig]) <= dist
        ip, ig = ip[keep], ig[keep]
    return ip.astype(np.int64), ig.astype(np.int64)


def match_detections(x: np.ndarray, y: np.ndarray, order: np.ndarray, target_oid: np.ndarray,
                     target_geom: np.ndarray, register_geom: np.ndarray, match_buffer_m: float,
                     ignore_buffer_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Outcome ("tp", "duplicate", "ignored", "fp") and target objectid (-1 if none) of every detection.

    Detections are processed in `order`. TP: within `match_buffer_m` of a still-unmatched target (lowest objectid
    first), which becomes matched. Duplicate: within `match_buffer_m` of a target, all of which are already
    matched (target_objectid = the lowest of them). Ignored: within `ignore_buffer_m` of any register polygon.
    FP: anything else. Distances are exact (<=).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    pts = shapely.points(x, np.asarray(y, dtype=np.float64))
    toid = np.asarray(target_oid, dtype=np.int64)
    if np.unique(toid).size != toid.size:
        raise ValueError("target objectids must be unique")
    tgeom = np.asarray(target_geom, dtype=object)
    ip, ig = _within(pts, tgeom, float(match_buffer_m))
    cand: dict[int, np.ndarray] = {}
    if ip.size:
        o = np.lexsort((toid[ig], ip))
        ip, ig = ip[o], ig[o]
        starts = np.flatnonzero(np.r_[True, ip[1:] != ip[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], ip.size], strict=True):
            cand[int(ip[a])] = ig[a:b]
    jp, _ = _within(pts, np.asarray(register_geom, dtype=object), float(ignore_buffer_m))
    near = np.zeros(n, dtype=bool)
    near[jp] = True
    outcome = np.empty(n, dtype=object)
    tid = np.full(n, NO_TARGET, dtype=np.int64)
    matched = np.zeros(toid.size, dtype=bool)
    order = np.asarray(order, dtype=np.int64)
    if np.sort(order).tolist() != list(range(n)):
        raise ValueError("order must be a permutation of the detections")
    for i in order:
        c = cand.get(int(i))
        if c is not None:
            free = c[~matched[c]]
            if free.size:
                matched[free[0]] = True
                outcome[i], tid[i] = "tp", toid[free[0]]
            else:
                outcome[i], tid[i] = "duplicate", toid[c[0]]
        elif near[i]:
            outcome[i] = "ignored"
        else:
            outcome[i] = "fp"
    return outcome.astype(str), tid


def ring_as_fp(outcome: np.ndarray) -> np.ndarray:
    """The post hoc variant: detections in the 150 m ring (ignored) count as FP; duplicates stay ignored."""
    o = np.asarray(outcome).astype(str)
    return np.where(o == "ignored", "fp", o)


# ======================================================================================================================
# Evaluation of one scorer
# ======================================================================================================================


def detections_of_sheet(cfg: Mapping[str, Any], p: Stage2Params, sheet: str, row: np.ndarray, col: np.ndarray,
                        x: np.ndarray, y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """NMS of one sheet's score lattice; the selected cells with their raw and jittered scores."""
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError(f"{sheet}: non-finite lattice scores")
    sj = score + jitter(cfg, sheet, score.size)
    res = nms(row, col, sj, p.nms_radius_px2)
    k = res.selected
    return pd.DataFrame({"sheet": sheet, "row": np.asarray(row)[k], "col": np.asarray(col)[k],
                         "x": np.asarray(x)[k], "y": np.asarray(y)[k], "score": score[k], "score_jittered": sj[k]})


def match_pooled(det: pd.DataFrame, targets: pd.DataFrame, register_geom: np.ndarray, p: Stage2Params,
                 sheet_rank: Mapping[str, int]) -> pd.DataFrame:
    """Match the detections of several sheets together (descending jittered score; ties by sheet, row, col).
    Returns a copy sorted in processing order, with `rank`, `outcome` and `target_objectid`."""
    det = det.reset_index(drop=True)
    sk = det["sheet"].map(sheet_rank).to_numpy(dtype=np.int64)
    order = processing_order(det["score_jittered"].to_numpy(), sk, det["row"].to_numpy(), det["col"].to_numpy())
    outcome, tid = match_detections(det["x"].to_numpy(), det["y"].to_numpy(), order,
                                    targets["objectid"].to_numpy(), targets["geometry"].to_numpy(),
                                    register_geom, p.match_buffer_m, p.ignore_buffer_m)
    out = det.assign(outcome=outcome, target_objectid=tid).iloc[order].reset_index(drop=True)
    out["rank"] = np.arange(len(out), dtype=np.int64)
    return out


def sweep(det: pd.DataFrame, n_targets: int, p: Stage2Params, ring_fp: bool = False) -> PRSweep:
    outcome = det["outcome"].to_numpy()
    if ring_fp:
        outcome = ring_as_fp(outcome)
    return pr_sweep(det["score_jittered"].to_numpy(), outcome, n_targets, p.recall_level)


@dataclass
class ScorerResult:
    scorer: str
    detections: pd.DataFrame
    pooled: PRSweep
    ring_fp: PRSweep
    strata: dict[str, PRSweep]
    strata_ring_fp: dict[str, PRSweep]
    n_targets: dict[str, int]
    n_windows: int
    n_sheets: int
    strata_consistent: bool
    residual_ties: int

    def summary(self, precision_min: float) -> dict[str, Any]:
        d = {"scorer": self.scorer, "kind": SCORER_KINDS.get(self.scorer, "other"), "n_sheets": self.n_sheets,
             "n_windows": self.n_windows, "n_detections": int(len(self.detections)), "n_targets": self.n_targets,
             "pooled": self.pooled.to_dict(), "ring_as_fp": self.ring_fp.to_dict(),
             "strata": {k: v.to_dict() for k, v in self.strata.items()},
             "strata_ring_as_fp": {k: v.to_dict() for k, v in self.strata_ring_fp.items()},
             "meets_threshold": bool(self.pooled.passes(precision_min)),
             "strata_match_pooled_outcomes": self.strata_consistent,
             "residual_tied_detection_scores": self.residual_ties}
        return d


def evaluate_scorer(cfg: Mapping[str, Any], scorer: str, lattices: Mapping[str, pd.DataFrame], inputs: Inputs,
                    register_geom: np.ndarray | None = None) -> ScorerResult:
    """NMS per sheet, pooled matching and the metrics of one scorer.

    `lattices[sheet]` holds the scorer's lattice (row, col, x, y, score) for every evaluated sheet.
    """
    p = stage2_params(cfg)
    sheets = inputs.sheet_ids
    missing = [s for s in sheets if s not in lattices]
    if missing:
        raise ValueError(f"{scorer}: no lattice scores for {missing}")
    flysch = inputs.flysch_of()
    rank = {s: i for i, s in enumerate(sheets)}
    if register_geom is None:
        register_geom = repair(inputs.polygons["geometry"].to_numpy())
    cols = ("row", "col", "x", "y", "score")
    det = pd.concat([detections_of_sheet(cfg, p, s, *(lattices[s][k].to_numpy() for k in cols)) for s in sheets],
                    ignore_index=True)
    targets = targets_of(inputs)
    pooled_det = match_pooled(det, targets, register_geom, p, rank)
    pooled_det.insert(0, "scorer", scorer)
    n_t = {"pooled": int(len(targets))}
    strata, strata_fp, consistent = {}, {}, True
    for name in STRATA:
        sh = [s for s in sheets if flysch[s] == (name == "flysch")]
        if not sh:
            continue
        t = targets_of(inputs, sh)
        n_t[name] = int(len(t))
        d = match_pooled(det.loc[det["sheet"].isin(sh)], t, register_geom, p, rank)
        ref = pooled_det.loc[pooled_det["sheet"].isin(sh)].reset_index(drop=True)
        consistent &= bool(np.array_equal(d["outcome"].to_numpy(), ref["outcome"].to_numpy())
                           and np.array_equal(d["target_objectid"].to_numpy(), ref["target_objectid"].to_numpy()))
        if len(t):
            strata[name], strata_fp[name] = sweep(d, len(t), p), sweep(d, len(t), p, ring_fp=True)
    sj = pooled_det["score_jittered"].to_numpy()
    ties = int(len(sj) - np.unique(sj).size)
    n_windows = int(sum(len(lattices[s]) for s in sheets))
    return ScorerResult(
        scorer=scorer, detections=pooled_det[list(DETECTION_COLUMNS)],
        pooled=sweep(pooled_det, len(targets), p), ring_fp=sweep(pooled_det, len(targets), p, ring_fp=True),
        strata=strata, strata_ring_fp=strata_fp, n_targets=n_t, n_windows=n_windows, n_sheets=len(sheets),
        strata_consistent=consistent, residual_ties=ties)


# ======================================================================================================================
# Files
# ======================================================================================================================


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def scores_path(paths: Paths, scorer: str, sheet: str) -> Path:
    return paths.stage2 / scorer / f"{sheet}.npz"


def lattice_path(paths: Paths, sheet: str) -> Path:
    return paths.stage2 / "lattice" / f"{sheet}.npz"


def infer_path(paths: Paths, sheet: str) -> Path:
    return paths.stage2 / "s2_infer" / f"{sheet}.npz"


def save_lattice(paths: Paths, lat: Lattice, chash: str) -> None:
    save_npz(lattice_path(paths, lat.sheet), row=lat.row, col=lat.col, x=lat.x, y=lat.y,
             n_inside=np.int64(lat.n_inside), n_invalid=np.int64(lat.n_invalid), config_hash=np.str_(chash),
             **{k: np.asarray(v, dtype=np.float64) for k, v in lat.stats.items()})


def load_lattice(paths: Paths, sheet: str) -> Lattice:
    d = load_npz(lattice_path(paths, sheet))
    return Lattice(sheet=sheet, row=d["row"], col=d["col"], x=d["x"], y=d["y"], n_inside=int(d["n_inside"]),
                   n_invalid=int(d["n_invalid"]), stats={k: d[k] for k in STATS if k in d})


def save_scores(paths: Paths, scorer: str, lat: Lattice, score: np.ndarray) -> None:
    """`<scorer>/<sheet>.npz` with the §4.3 arrays row, col, x, y, score."""
    score = np.asarray(score, dtype=np.float64)
    if score.shape != (lat.n,) or not np.isfinite(score).all():
        raise ValueError(f"{scorer}/{lat.sheet}: need {lat.n} finite scores")
    save_npz(scores_path(paths, scorer, lat.sheet), row=lat.row, col=lat.col, x=lat.x, y=lat.y, score=score)


def load_scores(paths: Paths, scorer: str, sheet: str) -> pd.DataFrame:
    d = load_npz(scores_path(paths, scorer, sheet))
    return pd.DataFrame({k: d[k] for k in ("row", "col", "x", "y", "score")})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def upsert_detections(paths: Paths, results: Sequence[ScorerResult]) -> Path:
    """Replace the rows of the given scorers in `detections.parquet`, keeping every other scorer's rows."""
    path = paths.stage2 / "detections.parquet"
    new = pd.concat([r.detections for r in results], ignore_index=True)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old.loc[~old["scorer"].isin([r.scorer for r in results])], new], ignore_index=True)
    new = new.sort_values(["scorer", "rank"], kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    new.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=REPO_ROOT, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def upsert_summary(paths: Paths, cfg: Mapping[str, Any], results: Sequence[ScorerResult],
                   lattice_info: Mapping[str, Any] | None = None) -> Path:
    """`summary.json`: per-scorer metrics (replacing the given scorers) and the lattice summary."""
    path = paths.stage2 / "summary.json"
    data: dict[str, Any] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    p = stage2_params(cfg)
    data.setdefault("scorers", {})
    for r in results:
        data["scorers"][r.scorer] = r.summary(p.precision_min)
    if lattice_info is not None:
        data["lattice"] = lattice_info
    data["config_hash"] = config_hash(dict(cfg))
    data["git_commit"] = git_commit()
    data["written"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path


def _sweep_cols(prefix: str, s: PRSweep | None, with_counts: bool = False) -> dict[str, Any]:
    nan = float("nan")
    d = {f"{prefix}status": s.status if s else "", f"{prefix}p_at_r50": nan if s is None or
         s.precision_at_recall is None else s.precision_at_recall,
         f"{prefix}interp_precision": nan if s is None or s.interpolated_precision is None
         else s.interpolated_precision, f"{prefix}max_recall": nan if s is None else s.max_recall}
    if with_counts:
        d.update({f"{prefix}n_tp": s.n_tp if s else 0, f"{prefix}n_fp": s.n_fp if s else 0})
    return d


def results_row(r: ScorerResult, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """One row of `results/stage2_baselines.csv`."""
    p = stage2_params(cfg)
    s = r.pooled
    row: dict[str, Any] = {
        "scorer": r.scorer, "kind": SCORER_KINDS.get(r.scorer, "other"), "n_sheets": r.n_sheets,
        "n_windows": r.n_windows, "n_detections": int(len(r.detections)), "n_targets": r.n_targets["pooled"],
        "n_tp": s.n_tp, "n_fp": s.n_fp, "n_ignored": s.n_ignored, "n_duplicate": s.n_duplicate,
        **_sweep_cols("", s),
        "threshold_at_r50": float("nan") if s.threshold_at_recall is None else s.threshold_at_recall,
        "meets_threshold": bool(s.passes(p.precision_min)),
    }
    for name in STRATA:
        row[f"{name}_n_targets"] = r.n_targets.get(name, 0)
        row.update(_sweep_cols(f"{name}_", r.strata.get(name)))
    row.update(_sweep_cols("ringfp_", r.ring_fp, with_counts=True))
    for name in STRATA:
        row.update(_sweep_cols(f"ringfp_{name}_", r.strata_ring_fp.get(name)))
    row["config_hash"] = config_hash(dict(cfg))
    return row


def upsert_results_csv(paths: Paths, cfg: Mapping[str, Any], results: Sequence[ScorerResult]) -> Path:
    """Replace the given baseline scorers' rows in `results/stage2_baselines.csv` (baselines only, CSV order)."""
    path = paths.results / "stage2_baselines.csv"
    rows = [results_row(r, cfg) for r in results if r.scorer in BASELINE_KINDS]
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old.loc[~old["scorer"].isin(new["scorer"])], new], ignore_index=True)
    order = {s: i for i, s in enumerate(BASELINE_KINDS)}
    new = new.sort_values("scorer", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)
    _write_text(path, new.to_csv(index=False, lineterminator="\n", float_format="%.6g"))
    return path


# ======================================================================================================================
# Drivers
# ======================================================================================================================


def lattice_summary(lattices: Mapping[str, Lattice], flysch: Mapping[str, bool], cfg: Mapping[str, Any]) -> dict:
    p = stage2_params(cfg)
    per = {s: {"n_windows": lat.n, "n_centres_inside": lat.n_inside, "n_invalid_dropped": lat.n_invalid,
               "flysch": bool(flysch[s])} for s, lat in sorted(lattices.items())}
    n = sum(v["n_windows"] for v in per.values())
    return {"stride_px": p.stride_px, "window_px": p.px, "valid_box_px": 2 * p.valid_half, "per_sheet": per,
            "n_windows": n, "n_invalid_dropped": sum(v["n_invalid_dropped"] for v in per.values()),
            "cnn_cost_estimate": cnn_cost_estimate(n, n_seeds=len(p.model_seeds))}


def run_baselines(cfg: Mapping[str, Any], paths: Paths, scorers: Sequence[str], inputs: Inputs | None = None,
                  meta: pd.DataFrame | None = None, show_progress: bool = False,
                  write_results: bool = True) -> dict[str, ScorerResult]:
    """Build the lattices, score them with the given baselines, evaluate and write every output.

    One sheet raster in memory at a time. `meta` (the LR's training table) is read from `meta.csv` if the LR is
    requested and none is given.
    """
    bad = [s for s in scorers if s not in BASELINE_KINDS]
    if bad:
        raise ValueError(f"not baseline scorers: {bad}")
    p = stage2_params(cfg)
    inputs = inputs if inputs is not None else load_inputs(cfg, paths)
    chash = config_hash(dict(cfg))
    lr_model = None
    if LR_SCORER in scorers:
        if meta is None:
            if inputs.meta_path is None or not Path(inputs.meta_path).is_file():
                raise FileNotFoundError(f"{LR_SCORER} needs meta.csv ({inputs.meta_path})")
            meta = pd.read_csv(inputs.meta_path)
        lr_model = fit_lr_comparator(meta, cfg)
    t0 = time.perf_counter()
    lattices: dict[str, Lattice] = {}
    scores: dict[str, dict[str, pd.DataFrame]] = {s: {} for s in scorers}
    for sheet in progress(inputs.sheet_ids, desc="stage-2 sheets", disable=not show_progress):
        sr = read_sheet(inputs, sheet, cfg)
        slope = full_channels(sr.z, dict(cfg), names=["slope"])[0]
        lat = build_lattice(sheet, sr.geom, sr.grid, np.isfinite(sr.z), slope, sr.susc, p)
        del sr, slope
        save_lattice(paths, lat, chash)
        lattices[sheet] = lat
        for s in scorers:
            sc = baseline_scores(s, lat, cfg, lr_model)
            save_scores(paths, s, lat, sc)
            scores[s][sheet] = pd.DataFrame({"row": lat.row, "col": lat.col, "x": lat.x, "y": lat.y, "score": sc})
        log.info("%s: %d windows (%d centres inside, %d dropped as invalid)", sheet, lat.n, lat.n_inside,
                 lat.n_invalid)
    log.info("lattices and baseline scores: %.1f s", time.perf_counter() - t0)
    register = repair(inputs.polygons["geometry"].to_numpy())
    results = {s: evaluate_scorer(cfg, s, scores[s], inputs, register) for s in scorers}
    for r in results.values():
        log.info("%s: %s", r.scorer, json.dumps(results_row(r, cfg), default=str))
    upsert_detections(paths, list(results.values()))
    upsert_summary(paths, cfg, list(results.values()), lattice_summary(lattices, inputs.flysch_of(), cfg))
    if write_results:
        upsert_results_csv(paths, cfg, list(results.values()))
    return results


def run_cnn(cfg: Mapping[str, Any], paths: Paths, inputs: Inputs | None = None,
            models: Sequence[SeedModel] | None = None, device: Any = None, show_progress: bool = False,
            overwrite: bool = False, chunk_windows: int = CNN_CHUNK_WINDOWS) -> dict[str, Any]:
    """Score every evaluated sheet with the s2 seed models and write `s2_infer/` and `cnn/` (resumable per sheet).

    Only the runner calls this (smoke test, phase 2): no CNN sees real data before the freeze (gate G3).
    """
    from landslide.train import deterministic_mode, ensure_cublas_workspace

    ensure_cublas_workspace()  # before the models start CUDA
    p = stage2_params(cfg)
    inputs = inputs if inputs is not None else load_inputs(cfg, paths)
    models = list(models) if models is not None else load_s2_models(cfg, paths, device)
    seeds = np.asarray([m.seed for m in models], dtype=np.int64)
    chash = config_hash(dict(cfg))
    info: dict[str, Any] = {"sheets": {}, "seeds": seeds.tolist()}
    t_all = time.perf_counter()
    with deterministic_mode():
        for sheet in inputs.sheet_ids:
            ip = infer_path(paths, sheet)
            if ip.exists() and not overwrite:
                d = load_npz(ip)
                if np.array_equal(d["seeds"], seeds) and str(d["config_hash"]) == chash \
                        and scores_path(paths, CNN, sheet).exists():
                    info["sheets"][sheet] = {"status": "skipped", "n_windows": int(d["row"].size)}
                    continue
            t0 = time.perf_counter()
            sr = read_sheet(inputs, sheet, cfg)
            terrain = full_channels(sr.z, dict(cfg))
            slope = terrain[channel_params(dict(cfg)).terrain_names.index("slope")]
            lat = build_lattice(sheet, sr.geom, sr.grid, np.isfinite(sr.z), slope, sr.susc, p)
            lp = lattice_path(paths, sheet)
            if lp.exists():
                old = load_lattice(paths, sheet)
                if not (np.array_equal(old.row, lat.row) and np.array_equal(old.col, lat.col)):
                    raise ValueError(f"{sheet}: the lattice differs from {lp}")
            else:
                save_lattice(paths, lat, chash)
            probs = cnn_window_probs(models, terrain, sr.z, lat.row, lat.col, cfg, chunk_windows=chunk_windows,
                                     show_progress=show_progress, desc=f"stage-2 windows {sheet}")
            del terrain, sr
            save_npz(ip, row=lat.row, col=lat.col, seeds=seeds, probs=probs, config_hash=np.str_(chash))
            save_scores(paths, CNN, lat, cnn_scores(probs))
            sec = time.perf_counter() - t0
            info["sheets"][sheet] = {"status": "done", "n_windows": lat.n, "seconds": round(sec, 2),
                                     "windows_per_s": round(lat.n / sec, 2) if sec > 0 else None}
            log.info("%s: CNN scored %d windows x %d seeds in %.1f s", sheet, lat.n, len(models), sec)
    info["seconds"] = round(time.perf_counter() - t_all, 2)
    return info


def evaluate_scorers(cfg: Mapping[str, Any], paths: Paths, scorers: Sequence[str], inputs: Inputs | None = None,
                     write_results: bool = True) -> dict[str, ScorerResult]:
    """Evaluate scorers from their saved lattices (`<scorer>/<sheet>.npz`) and write the outputs.

    The CNN's metrics go to `detections.parquet` and `summary.json` only; baselines also update the tracked CSV.
    """
    inputs = inputs if inputs is not None else load_inputs(cfg, paths)
    register = repair(inputs.polygons["geometry"].to_numpy())
    results = {}
    for s in scorers:
        lat = {}
        for sheet in inputs.sheet_ids:
            df = load_scores(paths, s, sheet)
            lp = lattice_path(paths, sheet)
            if lp.exists():
                ref = load_lattice(paths, sheet)
                if not (np.array_equal(ref.row, df["row"]) and np.array_equal(ref.col, df["col"])):
                    raise ValueError(f"{s}/{sheet}: the score lattice differs from {lp}")
            lat[sheet] = df
        results[s] = evaluate_scorer(cfg, s, lat, inputs, register)
    upsert_detections(paths, list(results.values()))
    upsert_summary(paths, cfg, list(results.values()))
    if write_results:
        upsert_results_csv(paths, cfg, list(results.values()))
    return results
