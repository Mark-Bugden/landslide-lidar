"""Patch extraction, `meta.csv`, the dataset summary and the dataset check (task P1-10).

BRIEF.md Appendix A §5 "Storage" and §6; PLAN.md §4.3 (contracts of `meta.csv` and `patches.npy`) and D19;
`docs/decisions/P1-10.md` records every reading.

**Extraction.** For every row i of P1-09's `samples.parquet`, `patches[i]` holds the seven channels of the 150 x 150
patch centred on pixel corner (row, col) of the sample's sheet raster, in `cfg["channels"]` order and physical units,
as float16. The channels come from `landslide.channels.patch_channels` (the patch plus its 28 px context, the code
path shared with stage 2). The build streams sheet by sheet: each sheet's DTM is read once with
`landslide.dtm.read_dtm` over the window that holds every patch plus context of that sheet, and its susceptibility
class raster once over the same window. Each patch is asserted NaN-free in float64, its statistics
(`landslide.features`, D19) are computed in float64, its float64 range is checked against the float16 maximum, and
only then is it cast and written into `patches.npy` (`numpy.lib.format.open_memmap`).

**Row order.** `meta.csv` row i, `patches[i]` and `samples.parquet` row i describe the same sample (`id` = i).

**meta.csv.** The PLAN.md §4.3 columns, plus P1-09's `source_layer` and `source_objectid`, then the 14 statistics.
`slope_bin` is P1-09's stored bin and `slope_mean` its stored mean (the values matching used); the mean recomputed
here from the float64 patch must equal it to `SLOPE_MEAN_TOL`. Floats are written as their shortest round-trip repr;
read the file back with `read_meta` (fixed dtypes). The build re-reads it and asserts every value is bit-exact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from landslide.channels import channel_params, patch_channels
from landslide.config import make_rng
from landslide.dtm import extract_block, read_dtm
from landslide.features import STAT_COLUMNS, patch_stats
from landslide.logs import get_logger, progress
from landslide.sampling import COLUMNS as SAMPLE_COLUMNS
from landslide.sampling import (MATCH_LEVELS, SOURCES, CornerGrid, grid_of_sheet, sampling_params,
                                slope_bins)

log = get_logger("patches")

#: The patches' storage dtype (PLAN.md §4.3) and its largest finite value (65504).
PATCH_DTYPE = np.dtype(np.float16)
FLOAT16_MAX = float(np.finfo(np.float16).max)
#: |slope_mean recomputed from the float64 patch - P1-09's stored slope_mean| must stay below this (degrees). Both are
#: means of the same slope values (P1-04 D18c: bitwise equal per pixel); only the summation order differs (~1e-12).
SLOPE_MEAN_TOL = 1e-9
#: Relative tolerance when a statistic is recomputed and compared with `meta.csv` (verification).
STAT_RTOL = 1e-9
#: Physical bounds checked on the stored patches (verification): slope in degrees, the sky-view factor.
CHANNEL_BOUNDS: dict[str, tuple[float, float]] = {"slope": (0.0, 90.0), "svf": (0.0, 1.0)}

BUILD_INFO_NAME = "build_info.json"
SUMMARY_NAME = "dataset_summary.json"
FIGURE_NAME = "slope_distributions.png"

#: The sample columns of `meta.csv`: PLAN.md §4.3 identity, location, context, sample type, splits and matching, with
#: P1-09's `source_layer`, `source_objectid` after `polygon_objectid`. The statistics follow (`STAT_COLUMNS`).
META_SAMPLE_COLUMNS: tuple[str, ...] = (
    "id", "label", "x", "y", "row", "col", "sheet", "kraj", "flysch", "source", "polygon_objectid", "source_layer",
    "source_objectid", "fold", "group_b", "block", "stratum", "slope_bin", "match_level", "pair_id",
)
META_COLUMNS: tuple[str, ...] = META_SAMPLE_COLUMNS + STAT_COLUMNS
_INT_COLUMNS = ("id", "label", "row", "col", "polygon_objectid", "source_layer", "source_objectid", "fold",
                "slope_bin", "match_level", "pair_id")
_STR_COLUMNS = ("sheet", "kraj", "source", "group_b", "block", "stratum")

#: The seeded rows the verification re-extracts are drawn from SeedSequence([replication.seed, stream id]). The
#: stream `verify_sample` is requested for `replication.streams`; until it is there, this id (the next free one) is
#: used.
VERIFY_STREAM = "verify_sample"
VERIFY_STREAM_ID_DEFAULT = 11


class PatchError(RuntimeError):
    """A patch fails an extraction assertion (NaN, float16 range, misaligned grid, window outside the raster)."""


# ======================================================================================================================
# Geometry of a patch
# ======================================================================================================================


@dataclass(frozen=True)
class PatchParams:
    names: tuple[str, ...]
    px: int
    context_px: int

    @property
    def half(self) -> int:
        return self.px // 2

    @property
    def ctx_half(self) -> int:
        """Half-size of the patch plus its channel context (75 + 28 = 103 px)."""
        return self.half + self.context_px

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.names), self.px, self.px)


def patch_params(cfg: Mapping[str, Any]) -> PatchParams:
    p = channel_params(cfg)
    return PatchParams(names=p.names, px=p.patch_px, context_px=p.context_px)


def window_for(rows: np.ndarray, cols: np.ndarray, half: int) -> tuple[int, int, int, int]:
    """(row_off, col_off, height, width) of the smallest window holding every corner-centred box of half-size
    `half` at corners (rows, cols)."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if not rows.size:
        raise ValueError("no corners")
    r0, c0 = int(rows.min()) - half, int(cols.min()) - half
    return r0, c0, int(rows.max()) + half - r0, int(cols.max()) + half - c0


# ======================================================================================================================
# One sheet: the DTM and susceptibility windows
# ======================================================================================================================


@dataclass(frozen=True)
class SheetSource:
    """Where a sheet's rasters are, and the grid they must have (`sheets.parquet`, as P1-08 read it from the DTM)."""

    sheet: str
    dtm_path: Path
    susc_path: Path
    grid: CornerGrid


@dataclass
class SheetWindow:
    """A window of one sheet: the DTM (float64, invalid cells NaN) and the susceptibility codes (uint8), both on the
    same pixels, starting at pixel (row_off, col_off) of the sheet raster."""

    sheet: str
    z: np.ndarray
    susc: np.ndarray
    row_off: int = 0
    col_off: int = 0

    def __post_init__(self) -> None:
        if self.z.shape != self.susc.shape:
            raise PatchError(f"{self.sheet}: DTM window {self.z.shape} and susceptibility window "
                             f"{self.susc.shape} differ")


def _lattice_of(transform: Any, height: int, width: int, what: str) -> CornerGrid:
    try:
        return CornerGrid.from_transform(transform, height, width)
    except ValueError as e:
        raise PatchError(f"{what}: {e}") from e


def read_sheet_window(src: SheetSource, rows: np.ndarray, cols: np.ndarray, cfg: Mapping[str, Any]) -> SheetWindow:
    """Read the window holding every patch plus context at corners (rows, cols) of one sheet: the DTM with
    `read_dtm` (one open) and the susceptibility class raster with rasterio (one open).

    Checks: the window lies inside the sheet's grid; the DTM window and the whole susceptibility raster are on the
    sheet's grid (national lattice indices and size, P1-09's `same_lattice`); the class raster is one uint8 band.
    """
    import rasterio
    from rasterio.windows import Window

    pp = patch_params(cfg)
    r0, c0, h, w = window_for(rows, cols, pp.ctx_half)
    H, W = src.grid.n_rows - 1, src.grid.n_cols - 1            # pixels
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        raise PatchError(f"{src.sheet}: a patch plus its {pp.context_px} px context leaves the {H} x {W} raster")
    dtm = read_dtm(src.dtm_path, cfg, window=(r0, c0, h, w))
    gi0, gj0 = src.grid.lattice_offset()
    if _lattice_of(dtm.transform, h, w, f"{src.sheet} DTM").lattice_offset() != (gi0 + c0, gj0 - r0):
        raise PatchError(f"{src.sheet}: the DTM raster is not on the grid of sheets.parquet")
    with rasterio.open(src.susc_path, mode="r") as ds:
        if ds.count != 1 or ds.dtypes[0] != "uint8":
            raise PatchError(f"{src.susc_path}: expected one uint8 band, got {ds.count} x {ds.dtypes}")
        if not _lattice_of(ds.transform, ds.height, ds.width, f"{src.sheet} susceptibility").same_lattice(src.grid):
            raise PatchError(f"{src.sheet}: the susceptibility raster is not on the sheet's DTM grid")
        susc = ds.read(1, window=Window(c0, r0, w, h))
    return SheetWindow(sheet=src.sheet, z=dtm.z, susc=susc, row_off=r0, col_off=c0)


# ======================================================================================================================
# One patch
# ======================================================================================================================


def extract_patch(z: np.ndarray, susc: np.ndarray, r: int, c: int,
                  cfg: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    """The float64 channels (7, px, px) of the patch at corner (r, c) of `z`, and its D19 statistics.

    Raises `PatchError` if any channel value is NaN or inf (the patch plus context must be fully valid, Appendix A §5
    "Storage"), or if the patch leaves the susceptibility array.
    """
    pp = patch_params(cfg)
    ch = patch_channels(z, r, c, cfg)
    if not np.isfinite(ch).all():
        bad = [n for n, a in zip(pp.names, ch) if not np.isfinite(a).all()]
        raise PatchError(f"patch at corner ({r}, {c}) has NaN or inf in {bad}: its context is not fully valid")
    h = pp.half
    if r - h < 0 or c - h < 0 or r + h > susc.shape[0] or c + h > susc.shape[1]:
        raise PatchError(f"patch at corner ({r}, {c}) leaves the susceptibility window {susc.shape}")
    z_ring = extract_block(np.asarray(z), r - h - 1, c - h - 1, r + h + 1, c + h + 1)
    stats = patch_stats(ch, pp.names, z_ring, susc[r - h:r + h, c - h:c + h])
    return ch, stats


def to_float16(ch: np.ndarray) -> np.ndarray:
    """Cast float64 channels to float16 after checking the float64 range: every value finite and |v| <= 65504, the
    float16 maximum (a larger value would round to 65504 or overflow to inf)."""
    ch = np.asarray(ch, dtype=np.float64)
    if not np.isfinite(ch).all():
        raise PatchError("NaN or inf before the float16 cast")
    amax = float(np.abs(ch).max(initial=0.0))
    if amax > FLOAT16_MAX:
        raise PatchError(f"float16 overflow: |value| up to {amax:.6g} > {FLOAT16_MAX:g}")
    out = ch.astype(PATCH_DTYPE)
    if not np.isfinite(out).all():  # pragma: no cover - excluded by the range check
        raise PatchError("float16 cast produced inf")
    return out


# ======================================================================================================================
# The whole dataset
# ======================================================================================================================


@dataclass
class Extraction:
    """Per-row statistics (in `STAT_COLUMNS` order) and aggregate numbers of the build (no coordinates)."""

    stats: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)


def check_samples_table(samples: pd.DataFrame) -> None:
    """The P1-09 table must have its columns, ids 0..N-1 on a RangeIndex, and finite slope means."""
    missing = [k for k in SAMPLE_COLUMNS if k not in samples.columns]
    if missing:
        raise ValueError(f"samples table lacks columns {missing}")
    if not isinstance(samples.index, pd.RangeIndex) or samples.index.start != 0 or samples.index.step != 1:
        raise ValueError("samples need a 0..N-1 RangeIndex")
    if not (samples["id"].to_numpy() == np.arange(len(samples))).all():
        raise ValueError("samples ids are not 0..N-1 in row order")
    if not np.isfinite(samples["slope_mean"].to_numpy(dtype=np.float64)).all():
        raise ValueError("samples have a non-finite slope_mean")


def extract_dataset(cfg: Mapping[str, Any], samples: pd.DataFrame,
                    open_window: Callable[[str, np.ndarray, np.ndarray], SheetWindow], out: np.ndarray,
                    workers: int = 1, show_progress: bool = False) -> Extraction:
    """Fill `out[i]` (float16, (N, 7, px, px), e.g. an `open_memmap`) with the patch of samples row i, sheet by sheet.

    `open_window(sheet, rows, cols)` returns a `SheetWindow` holding every patch plus context of the sheet's samples
    (`read_sheet_window` for the real files). Patches of one sheet run on `workers` threads; the result does not
    depend on it. Asserts: every patch NaN-free, within the float16 range, and its recomputed slope mean equal to the
    stored `slope_mean` within `SLOPE_MEAN_TOL`.
    """
    check_samples_table(samples)
    pp = patch_params(cfg)
    N = len(samples)
    if tuple(out.shape) != (N, *pp.shape) or out.dtype != PATCH_DTYPE:
        raise ValueError(f"output array {out.shape} {out.dtype} != {(N, *pp.shape)} {PATCH_DTYPE}")
    n_ch = len(pp.names)
    stats = np.full((N, len(STAT_COLUMNS)), np.nan)
    done = np.zeros(N, dtype=bool)
    ch_min = np.full(n_ch, np.inf)
    ch_max = np.full(n_ch, -np.inf)
    f16_err = np.zeros(n_ch)
    stored = samples["slope_mean"].to_numpy(dtype=np.float64)
    groups = list(samples.groupby("sheet", sort=True))
    bar = progress(groups, desc="patches", unit="sheet") if show_progress else groups
    ex = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for sheet, g in bar:
            idx = g.index.to_numpy()
            rows, cols = g["row"].to_numpy(dtype=np.int64), g["col"].to_numpy(dtype=np.int64)
            win = open_window(str(sheet), rows, cols)
            lr, lc = rows - win.row_off, cols - win.col_off

            def one(k: int, win: SheetWindow = win, lr: np.ndarray = lr, lc: np.ndarray = lc,
                    idx: np.ndarray = idx):
                try:
                    ch, st = extract_patch(win.z, win.susc, int(lr[k]), int(lc[k]), cfg)
                    f16 = to_float16(ch)
                except PatchError as e:
                    raise PatchError(f"{win.sheet}, sample id {int(idx[k])}: {e}") from e
                err = np.abs(f16.astype(np.float64) - ch).max(axis=(1, 2))
                return f16, st, ch.min(axis=(1, 2)), ch.max(axis=(1, 2)), err

            results = ex.map(one, range(len(idx))) if ex is not None else map(one, range(len(idx)))
            for i, (f16, st, lo, hi, err) in zip(idx, results):
                d = abs(st["slope_mean"] - stored[i])
                if not d < SLOPE_MEAN_TOL:          # fail fast, before the rest of the build
                    raise PatchError(f"slope_mean of sample id {int(i)} recomputed from the patch differs from "
                                     f"samples.parquet by {d:.3g} deg (tolerance {SLOPE_MEAN_TOL:g})")
                out[i] = f16
                stats[i] = [st[k] for k in STAT_COLUMNS]
                done[i] = True
                np.minimum(ch_min, lo, out=ch_min)
                np.maximum(ch_max, hi, out=ch_max)
                np.maximum(f16_err, err, out=f16_err)
            del win
    finally:
        if ex is not None:
            ex.shutdown()
    if not done.all():  # pragma: no cover - every row belongs to one sheet group
        raise AssertionError(f"{int((~done).sum())} rows were not extracted")
    st = pd.DataFrame(stats, columns=list(STAT_COLUMNS))
    max_diff = float(np.abs(st["slope_mean"].to_numpy() - stored).max(initial=0.0))
    bin_deg = float(cfg["sampling"]["slope_bin_deg"])
    flips = int((slope_bins(st["slope_mean"].to_numpy(), bin_deg) != samples["slope_bin"].to_numpy()).sum())
    info = {
        "n_patches": N,
        "n_sheets": len(groups),
        "channels": list(pp.names),
        "channel_min": {n: float(v) for n, v in zip(pp.names, ch_min)},
        "channel_max": {n: float(v) for n, v in zip(pp.names, ch_max)},
        "float16_max_abs_error": {n: float(v) for n, v in zip(pp.names, f16_err)},
        "float16_headroom": float(FLOAT16_MAX - max(np.abs(ch_min).max(initial=0), np.abs(ch_max).max(initial=0))),
        "max_slope_mean_diff_deg": max_diff,
        "slope_bin_flips_recomputed_vs_stored": flips,
    }
    return Extraction(stats=st, info=info)


# ======================================================================================================================
# meta.csv
# ======================================================================================================================


def build_meta(samples: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """The `meta.csv` table: the samples' columns (`META_SAMPLE_COLUMNS`) and the statistics, row i = sample i.

    `slope_mean` is P1-09's stored value (consistent with `slope_bin` and the matching); the other 13 statistics are
    the recomputed ones. `extract_dataset` has asserted that the recomputed slope mean equals the stored one.
    """
    check_samples_table(samples)
    if len(stats) != len(samples) or tuple(stats.columns) != STAT_COLUMNS:
        raise ValueError("statistics do not match the samples")
    meta = samples[list(META_SAMPLE_COLUMNS)].copy()
    for k in STAT_COLUMNS:
        meta[k] = samples["slope_mean"].to_numpy(dtype=np.float64) if k == "slope_mean" else stats[k].to_numpy()
    for k in _INT_COLUMNS:
        meta[k] = meta[k].astype(np.int64)
    meta["flysch"] = meta["flysch"].astype(bool)
    return meta.reset_index(drop=True)


def write_meta(meta: pd.DataFrame, path: Path) -> None:
    """UTF-8 CSV, "\\n" line ends, floats as their shortest round-trip repr. Written to a temporary file, then
    renamed."""
    if tuple(meta.columns) != META_COLUMNS:
        raise ValueError("meta columns are not META_COLUMNS")
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    meta.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    tmp.replace(path)


def read_meta(path: Path | str) -> pd.DataFrame:
    """Read `meta.csv` with fixed dtypes. pandas 3's default float parser is exact on the shortest-repr floats the
    build writes (the build re-reads the file and asserts bit equality; `float_precision` is deprecated in 3.1)."""
    dtype: dict[str, Any] = {k: "int64" for k in _INT_COLUMNS}
    dtype.update({k: str for k in _STR_COLUMNS})
    dtype["flysch"] = bool
    dtype.update({k: "float64" for k in ("x", "y", *STAT_COLUMNS)})
    return pd.read_csv(path, dtype=dtype, encoding="utf-8")


# ======================================================================================================================
# The npy file
# ======================================================================================================================


def npy_header(path: Path | str) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    """(shape, fortran_order, dtype, data offset in bytes) from a `.npy` header."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"{path}: unsupported .npy version {version}")
        return tuple(shape), bool(fortran), np.dtype(dtype), f.tell()


def sha256_file(path: Path | str, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ======================================================================================================================
# Summary (aggregate counts only) and the figure
# ======================================================================================================================


def _kind(meta: pd.DataFrame) -> pd.Series:
    """positive | negative (matched) | background."""
    return pd.Series(np.where(meta["label"] == 1, "positive", np.where(meta["pair_id"] >= 0, "negative",
                                                                       "background")), index=meta.index)


def _counts(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(s.value_counts(dropna=False).items(), key=lambda kv: str(kv[0]))}


def _by(meta: pd.DataFrame, col: str) -> dict[str, dict[str, int]]:
    kind = _kind(meta)
    out: dict[str, dict[str, int]] = {}
    for key, g in meta.groupby(col, sort=True, dropna=False):
        k = kind.loc[g.index]
        out[str(key)] = {"positive": int((k == "positive").sum()), "negative": int((k == "negative").sum()),
                         "background": int((k == "background").sum()), "total": int(len(g))}
    return out


def test_set_masks(meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Masks of the matched patches in each test set: (a) the flysch folds pooled (fold >= 0), (b) the transfer test
    group, and (b)'s training group. The background (pair_id -1) belongs to no test set (Appendix A §5)."""
    matched = meta["pair_id"] >= 0
    return {"a_flysch_folds_pooled": matched & (meta["fold"] >= 0),
            "b_test": matched & (meta["group_b"] == "test"),
            "b_train": matched & (meta["group_b"] == "train")}


def summarise(meta: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Counts by label, kind, source, kraj, flysch flag, fold, group_b, stratum and match level (aggregate only: no
    coordinates, no per-sheet numbers, no susceptibility)."""
    kind = _kind(meta)
    neg = meta[kind == "negative"]
    pos = meta[kind == "positive"].set_index("pair_id")
    nn = neg.set_index("pair_id").loc[pos.index]
    d = np.abs(nn["slope_mean"].to_numpy() - pos["slope_mean"].to_numpy())
    pp = patch_params(cfg)
    sets = {}
    for name, m in test_set_masks(meta).items():
        k = kind[m]
        sets[name] = {"positive": int((k == "positive").sum()), "negative": int((k == "negative").sum()),
                      "sheets": int(meta.loc[m, "sheet"].nunique())}
    return {
        "n_patches": int(len(meta)),
        "patch_shape": list(pp.shape),
        "channels": list(pp.names),
        "by_label": _counts(meta["label"]),
        "by_kind": _counts(kind),
        "by_source": _counts(meta["source"]),
        "by_kraj": _by(meta, "kraj"),
        "by_flysch": _by(meta, "flysch"),
        "by_fold": _by(meta, "fold"),
        "by_group_b": _by(meta, "group_b"),
        "by_stratum": _by(meta, "stratum"),
        "by_match_level": _counts(neg["match_level"]),
        "by_match_level_and_source": {str(k): _counts(g["source"]) for k, g in neg.groupby("match_level", sort=True)},
        "test_sets": sets,
        "pairs": {"n": int(len(pos)),
                  "same_sheet": int((nn["sheet"].to_numpy() == pos["sheet"].to_numpy()).sum()),
                  "same_slope_bin": int((nn["slope_bin"].to_numpy() == pos["slope_bin"].to_numpy()).sum()),
                  "abs_slope_mean_diff_deg": ({"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                                               "max": float(d.max())} if d.size else None)},
        "sheets_with_patches": int(meta["sheet"].nunique()),
    }


#: Figure colours: categorical slots 1 and 2 of the dataviz reference palette (light), text and grid inks.
_FIG = {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "grid": "#e4e3df", "positive": "#2a78d6",
        "negative": "#eb6834"}


def plot_slope_distributions(meta: pd.DataFrame, cfg: Mapping[str, Any], path: Path | str) -> None:
    """Histograms of patch-mean slope, positives vs their matched negatives, for test set (a) (the flysch folds
    pooled) and test set (b) (the transfer test group). 2-degree bins = the matching bins (`slope_bin_deg`); the two
    classes are drawn as side-by-side bars in each bin. Histograms only: no maps, no susceptibility."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    bin_deg = float(cfg["sampling"]["slope_bin_deg"])
    sets = test_set_masks(meta)
    panels = [("(a) held-out flysch: 5 folds pooled", sets["a_flysch_folds_pooled"]),
              ("(b) transfer: test kraje", sets["b_test"])]
    top = float(meta.loc[meta["pair_id"] >= 0, "slope_mean"].max()) if (meta["pair_id"] >= 0).any() else bin_deg
    edges = np.arange(0.0, (math.floor(top / bin_deg) + 2) * bin_deg, bin_deg)
    fig = Figure(figsize=(11, 4.2), dpi=150, facecolor=_FIG["surface"])
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, sharey=False)
    w = bin_deg * 0.42
    for ax, (title, m) in zip(axes, panels):
        sub = meta[m]
        ax.set_facecolor(_FIG["surface"])
        for k, (lab, col, off) in enumerate((("positive", _FIG["positive"], 0.06), ("negative", _FIG["negative"],
                                                                                     0.52))):
            v = sub.loc[sub["label"] == (1 if lab == "positive" else 0), "slope_mean"].to_numpy()
            cnt, _ = np.histogram(v, bins=edges)
            ax.bar(edges[:-1] + off * bin_deg, cnt, width=w, align="edge", color=col, linewidth=0,
                   label=f"{lab}s (n = {len(v):,})", zorder=2)
        ax.set_title(title, loc="left", fontsize=11, color=_FIG["text"])
        ax.set_xlabel("patch-mean slope (degrees, 2° matching bins)", color=_FIG["text2"], fontsize=9)
        ax.set_ylabel("patches", color=_FIG["text2"], fontsize=9)
        ax.set_xlim(edges[0], edges[-1])
        ax.grid(axis="y", color=_FIG["grid"], linewidth=1, zorder=0)
        ax.tick_params(colors=_FIG["text2"], labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(_FIG["grid"])
        ax.legend(frameon=False, fontsize=9, labelcolor=_FIG["text"])
    fig.suptitle("Slope matching: positives vs matched negatives per test set", x=0.01, ha="left", fontsize=12,
                 color=_FIG["text"])
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_FIG["surface"])


# ======================================================================================================================
# Verification (scripts/verify_dataset.py)
# ======================================================================================================================


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    level: str = "fail"          # "fail": a failure fails the dataset; "warn": reported only

    def line(self) -> str:
        tag = "PASS" if self.ok else ("FAIL" if self.level == "fail" else "WARN")
        return f"{tag:4}  {self.name:<20} {self.detail}"


def verify_rng(cfg: Mapping[str, Any]) -> np.random.Generator:
    """SeedSequence([replication.seed, stream id of `verify_sample`]) (D02)."""
    rep = cfg["replication"]
    streams = {k: int(v) for k, v in rep.get("streams", {}).items()}
    if VERIFY_STREAM in streams:
        sid = streams[VERIFY_STREAM]
    else:
        sid = VERIFY_STREAM_ID_DEFAULT
        if sid in streams.values():
            raise ValueError(f"default stream id {sid} for {VERIFY_STREAM!r} is already registered")
    return make_rng(int(rep["seed"]), sid)


def verify_rows(cfg: Mapping[str, Any], n: int, n_check: int) -> np.ndarray:
    """The rows the verification re-extracts: the first, the last and a seeded uniform sample, sorted."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(max(n_check, 0), n)
    pick = verify_rng(cfg).choice(n, size=k, replace=False) if k else np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate([pick, [0, n - 1]]).astype(np.int64))


def _close(a: np.ndarray, b: np.ndarray, rtol: float) -> np.ndarray:
    return np.abs(a - b) <= rtol * np.maximum(1.0, np.abs(b))


def verify_dataset(cfg: Mapping[str, Any], patches_path: Path, meta_path: Path, sheets: pd.DataFrame,
                   sources: Mapping[str, SheetSource], *, samples: pd.DataFrame | None = None,
                   build_info: Mapping[str, Any] | None = None, summary: Mapping[str, Any] | None = None,
                   config_hash: str | None = None, n_check: int = 200, full_hash: bool = True,
                   require_build_info: bool = False, chunk: int = 256) -> list[Check]:
    """Every check of the P1-10 card. Returns one `Check` per item; never raises for a data problem.

    `sheets` is `sheets.parquet` (sheet, geometry, grid_*, is_stage1, is_stage2); `sources` gives each sheet's
    rasters for the re-extraction.
    """
    checks: list[Check] = []

    def run(name: str, f: Callable[[], str], level: str = "fail") -> bool:
        try:
            detail = f()
            checks.append(Check(name, True, detail, level))
            return True
        except Exception as e:  # noqa: BLE001 - every problem becomes a reported failure
            checks.append(Check(name, False, f"{type(e).__name__}: {e}", level))
            return False

    pp = patch_params(cfg)
    sp = sampling_params(cfg)
    state: dict[str, Any] = {}

    def files() -> str:
        for p in (patches_path, meta_path):
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        state["meta"] = read_meta(meta_path)
        return f"{Path(patches_path).name}, {Path(meta_path).name} ({len(state['meta']):,} rows)"

    if not run("files", files):
        return checks
    meta: pd.DataFrame = state["meta"]
    N = len(meta)

    def header() -> str:
        shape, fortran, dtype, offset = npy_header(patches_path)
        state["offset"] = offset
        if dtype != PATCH_DTYPE or fortran:
            raise AssertionError(f"dtype {dtype}, fortran_order {fortran}; expected {PATCH_DTYPE}, C order")
        if shape != (N, *pp.shape):
            raise AssertionError(f"shape {shape} != (len(meta), channels, px, px) = {(N, *pp.shape)}")
        return f"{shape} {dtype}"

    header_ok = run("shape_dtype", header)

    def file_size() -> str:
        size = Path(patches_path).stat().st_size
        want = state["offset"] + N * int(np.prod(pp.shape)) * PATCH_DTYPE.itemsize
        if size != want:
            raise AssertionError(f"{size:,} bytes, expected header {state['offset']} + data = {want:,}")
        if build_info is not None and int(build_info["patches"]["bytes"]) != size:
            raise AssertionError(f"{size:,} bytes, build_info.json recorded {build_info['patches']['bytes']:,}")
        return f"{size:,} bytes ({size / 2**30:.2f} GiB)"

    if header_ok:
        run("file_size", file_size)

    def values() -> str:
        mm = np.load(patches_path, mmap_mode="r")
        lo = np.full(len(pp.names), np.inf)
        hi = np.full(len(pp.names), -np.inf)
        bad_rows: list[int] = []
        for i in range(0, N, chunk):
            a = np.asarray(mm[i:i + chunk], dtype=np.float32)
            fin = np.isfinite(a).all(axis=(1, 2, 3))
            if not fin.all():
                bad_rows += (np.flatnonzero(~fin) + i).tolist()
                a = np.where(np.isfinite(a), a, np.nan)
            with np.errstate(invalid="ignore"):
                np.fmin(lo, np.nanmin(a, axis=(0, 2, 3)), out=lo)
                np.fmax(hi, np.nanmax(a, axis=(0, 2, 3)), out=hi)
        del mm
        state["ranges"] = (lo, hi)              # over the finite values
        if bad_rows:
            raise AssertionError(f"{len(bad_rows)} patches with NaN or inf, first ids {bad_rows[:5]}")
        return "every value finite"

    def overflow() -> str:
        lo, hi = state["ranges"]
        amax = float(max(np.abs(lo).max(), np.abs(hi).max()))
        if not amax < FLOAT16_MAX:
            raise AssertionError(f"|value| reaches {amax:g}, the float16 maximum (saturated or overflowed)")
        return f"max finite |value| {amax:.4g} < {FLOAT16_MAX:g}"

    def bounds() -> str:
        lo, hi = state["ranges"]
        out = []
        for n, (a, b) in CHANNEL_BOUNDS.items():
            k = pp.names.index(n)
            if not (lo[k] >= a and hi[k] <= b):
                raise AssertionError(f"{n} spans [{lo[k]:g}, {hi[k]:g}], outside [{a:g}, {b:g}]")
            out.append(f"{n} [{lo[k]:.3g}, {hi[k]:.3g}]")
        return ", ".join(out)

    if header_ok:
        run("nan_inf", values)
        if "ranges" in state:
            run("float16_overflow", overflow)
            run("channel_bounds", bounds)

    def columns() -> str:
        if tuple(meta.columns) != META_COLUMNS:
            raise AssertionError(f"columns differ from META_COLUMNS: {list(meta.columns)}")
        bad = [k for k in STAT_COLUMNS if not np.isfinite(meta[k].to_numpy()).all()]
        if bad:
            raise AssertionError(f"non-finite statistics in {bad}")
        return f"{len(META_COLUMNS)} columns, statistics finite"

    run("meta_columns", columns)

    def ids() -> str:
        if not meta["id"].is_unique:
            raise AssertionError(f"{int(meta['id'].duplicated().sum())} duplicate ids")
        if not (meta["id"].to_numpy() == np.arange(N)).all():
            raise AssertionError("id is not the row number (row i must describe patches[i])")
        return f"{N:,} unique ids = row numbers"

    run("unique_ids", ids)

    def labels() -> str:
        src = meta["source"]
        if not src.isin(SOURCES).all():
            raise AssertionError(f"unknown sources {sorted(set(src) - set(SOURCES))}")
        if not ((meta["label"] == 1) == (src == "positive")).all():
            raise AssertionError("label 1 is not the same as source 'positive'")
        if not ((meta["pair_id"] < 0) == (src == "background")).all() or not (meta.loc[meta["pair_id"] < 0,
                                                                                         "pair_id"] == -1).all():
            raise AssertionError("pair_id -1 is not the same as source 'background'")
        neg = (meta["label"] == 0) & (meta["pair_id"] >= 0)
        if not (meta.loc[~neg, "match_level"] == 0).all() or not meta.loc[neg, "match_level"].isin(
                list(MATCH_LEVELS)).all():
            raise AssertionError("match_level must be 0 for positives and background, 1-6 for negatives")
        return "label, source, pair_id and match_level consistent"

    run("labels_sources", labels)

    def pairs() -> str:
        m = meta[meta["pair_id"] >= 0]
        c = m.groupby(["pair_id", "label"]).size().unstack(fill_value=0)
        for lab in (0, 1):
            if lab not in c.columns:
                c[lab] = 0
        bad = c[(c[0] != 1) | (c[1] != 1)]
        if len(bad):
            raise AssertionError(f"{len(bad)} pair_ids without exactly one positive and one negative, e.g. "
                                 f"{bad.index[:5].tolist()}")
        pos = m[m["label"] == 1].set_index("pair_id")
        neg = m[m["label"] == 0].set_index("pair_id").loc[pos.index]
        if not (neg["stratum"].to_numpy() == pos["stratum"].to_numpy()).all():
            raise AssertionError("a negative outside its positive's stratum")
        want = np.array([MATCH_LEVELS[int(v)][1] for v in neg["match_level"]], dtype=np.int64)
        if not (np.abs(neg["slope_bin"].to_numpy() - pos["slope_bin"].to_numpy()) == want).all():
            raise AssertionError("the slope-bin rule fails for some match level")
        return f"{len(pos):,} pairs, one negative each, same stratum, bin rule per level"

    run("pairs", pairs)

    if samples is not None:
        def vs_samples() -> str:
            if len(samples) != N:
                raise AssertionError(f"{len(samples)} samples vs {N} meta rows")
            for k in META_SAMPLE_COLUMNS + ("slope_mean",):
                a, b = meta[k].to_numpy(), samples[k].to_numpy()
                if not (a == b).all():
                    raise AssertionError(f"column {k} differs from samples.parquet (first row "
                                         f"{int(np.flatnonzero(a != b)[0])})")
            return "every sample column equals samples.parquet, row by row"

        run("meta_vs_samples", vs_samples)

    def patch_rule() -> str:
        import shapely

        sh = sheets.set_index("sheet")
        unknown = sorted(set(meta["sheet"]) - set(sh.index))
        if unknown:
            raise AssertionError(f"sheets not in sheets.parquet: {unknown[:5]}")
        if not sh.loc[meta["sheet"].unique(), "is_stage1"].astype(bool).all():
            raise AssertionError("a patch on a sheet that is not a stage-1 sheet")
        h = sp.half_m
        for s, g in meta.groupby("sheet", sort=False):
            grid = grid_of_sheet(sh.loc[s], sp.cell_m)
            x, y = grid.xy(g["row"].to_numpy(), g["col"].to_numpy())
            if not ((x == g["x"].to_numpy()).all() and (y == g["y"].to_numpy()).all()):
                raise AssertionError(f"{s}: x, y are not the pixel corners (row, col)")
            r, c = g["row"].to_numpy(), g["col"].to_numpy()
            H, W = grid.n_rows - 1, grid.n_cols - 1
            if ((r - pp.ctx_half < 0) | (r + pp.ctx_half > H) | (c - pp.ctx_half < 0) | (c + pp.ctx_half > W)).any():
                raise AssertionError(f"{s}: a patch plus context leaves the raster")
        sq = shapely.box(meta["x"] - h, meta["y"] - h, meta["x"] + h, meta["y"] + h)
        inside = shapely.covered_by(sq, sh.loc[meta["sheet"], "geometry"].to_numpy())
        if not inside.all():
            raise AssertionError(f"{int((~inside).sum())} patch squares leave their sheet, first id "
                                 f"{int(meta['id'].to_numpy()[~inside][0])}")
        s2 = sh.loc[sh["is_stage2"].astype(bool), "geometry"].to_numpy()
        dmin = math.inf
        if len(s2):
            tree = shapely.STRtree(s2)
            ip, _ = tree.query(sq, predicate="dwithin", distance=sp.stage2_buffer_m)
            if ip.size:
                raise AssertionError(f"{len(np.unique(ip))} patch squares within {sp.stage2_buffer_m:g} m of a "
                                     f"stage-2 sheet, first id {int(meta['id'].to_numpy()[ip[0]])}")
            _, dd = tree.query_nearest(sq, return_distance=True, all_matches=False)
            dmin = float(np.min(dd))
        return (f"{N:,} squares inside their stage-1 sheet; nearest stage-2 sheet {dmin:.1f} m "
                f"(> {sp.stage2_buffer_m:g}); corners on the raster grid")

    run("patch_rule", patch_rule)

    def alignment() -> str:
        if not header_ok:
            raise AssertionError("the patch file header is not valid")
        rows = verify_rows(cfg, N, n_check)
        mm = np.load(patches_path, mmap_mode="r")
        worst = 0.0
        for s, g in meta.iloc[rows].groupby("sheet", sort=True):
            src = sources[str(s)]
            for i, r, c in zip(g.index, g["row"].to_numpy(), g["col"].to_numpy()):
                win = read_sheet_window(src, np.array([r]), np.array([c]), cfg)   # this patch's own window
                ch, st = extract_patch(win.z, win.susc, int(r - win.row_off), int(c - win.col_off), cfg)
                f16 = to_float16(ch)
                stored = np.asarray(mm[i])
                if not np.array_equal(f16.view(np.uint16), stored.view(np.uint16)):
                    raise AssertionError(f"patches[{i}] differs from the re-extracted patch of meta row {i}")
                got = np.array([st[k] for k in STAT_COLUMNS])
                want = meta.loc[i, list(STAT_COLUMNS)].to_numpy(dtype=np.float64)
                tol = np.where(np.array(STAT_COLUMNS) == "slope_mean", SLOPE_MEAN_TOL, STAT_RTOL)
                ok = np.abs(got - want) <= tol * np.maximum(1.0, np.abs(want))
                if not ok.all():
                    bad = [k for k, o in zip(STAT_COLUMNS, ok) if not o]
                    raise AssertionError(f"meta row {i}: statistics {bad} differ from the re-extracted patch")
                worst = max(worst, float(np.max(np.abs(got - want))))
        del mm
        return f"{len(rows)} seeded rows re-extracted: patches bit-identical, statistics within {worst:.2g}"

    run("alignment", alignment)

    if build_info is None and require_build_info:
        checks.append(Check("build_info", False, f"{BUILD_INFO_NAME} is missing (the build did not finish)"))
    if build_info is not None:
        def info() -> str:
            if build_info["meta"]["sha256"] != sha256_file(meta_path):
                raise AssertionError("meta.csv sha256 differs from build_info.json")
            if int(build_info["meta"]["rows"]) != N or list(build_info["patches"]["shape"]) != [N, *pp.shape]:
                raise AssertionError("build_info.json rows/shape differ")
            if full_hash and build_info["patches"].get("sha256") != sha256_file(patches_path):
                raise AssertionError("patches.npy sha256 differs from build_info.json")
            return "meta.csv" + (" and patches.npy" if full_hash else "") + " sha256 as recorded"

        run("build_info", info)

        if config_hash is not None:
            def chash() -> str:
                if build_info.get("config_hash") != config_hash:
                    raise AssertionError(f"built with config {str(build_info.get('config_hash'))[:12]}, current "
                                         f"{config_hash[:12]}")
                return f"config {config_hash[:12]}"

            run("config_hash", chash, level="warn")

    if summary is not None:
        def summ() -> str:
            if int(summary["n_patches"]) != N:
                raise AssertionError(f"dataset_summary.json n_patches {summary['n_patches']} != {N}")
            if build_info is not None and summary.get("config_hash") != build_info.get("config_hash"):
                raise AssertionError("dataset_summary.json and build_info.json have different config hashes")
            return "n_patches and config_hash agree"

        run("summary", summ)

    return checks


def failures(checks: Sequence[Check]) -> list[Check]:
    return [c for c in checks if not c.ok and c.level == "fail"]


def dump_json(obj: Any, path: Path) -> None:
    """UTF-8 JSON with a trailing newline, written to a temporary file and renamed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
