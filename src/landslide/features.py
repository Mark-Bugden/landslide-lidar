"""Per-patch terrain and susceptibility statistics (task P1-10; PLAN.md D19; BRIEF.md Appendix A §5 "Storage", §10).

Every statistic is computed in float64 over the 150 x 150 patch (never the 28 px context), from the float64 channels
of `landslide.channels.patch_channels` before they are cast to float16, so the baselines see exactly the terrain the
CNN sees, at full precision.

| column            | definition (over the patch's px x px pixels)                                           |
| ----------------- | -------------------------------------------------------------------------------------- |
| slope_mean        | mean of the slope channel (degrees)                                                    |
| slope_std         | standard deviation of slope (population, ddof = 0)                                     |
| slope_p90         | 90th percentile of slope (numpy's default linear interpolation)                        |
| lrm_s_std         | std of lrm_s (m)                                                                       |
| lrm_l_std         | std of lrm_l (m)                                                                       |
| curv_prof_std     | std of curv_prof (asinh-scaled, dimensionless)                                         |
| curv_plan_std     | std of curv_plan                                                                       |
| svf_mean          | mean of svf                                                                            |
| svf_std           | std of svf                                                                             |
| detr_std          | std of detr (m)                                                                        |
| detr_range_2_98   | 98th minus 2nd percentile of detr (m)                                                  |
| tri               | mean over the patch pixels of the mean |z - z_n| over the 8 neighbours n of each pixel (m) |
| susc_mean         | mean of the susceptibility class code (0 low/no data, 1 mid, 2 high)                   |
| susc_high_frac    | fraction of patch pixels with code 2                                                   |

The first twelve are Appendix A §10's "12 simple statistics"; the last two are the slope_susc_lr inputs.

**TRI.** D19 defines it as the mean over the patch of each pixel's mean absolute elevation difference to its eight
neighbours, on the DTM z (not smoothed). A patch-edge pixel's neighbours lie one pixel into the channel context,
exactly as for slope (central differences), so the caller passes z for the patch plus a 1 px ring.

**Susceptibility** is the uint8 class raster of P1-07 on the sheet's DTM grid; its statistics use the same px x px
pixels as the patch. Licence (BRIEF.md "Constraints"): row 5 appears only as these per-patch numbers inside `data/`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: The statistics columns of `meta.csv`, in the order of PLAN.md §4.3.
STAT_COLUMNS: tuple[str, ...] = (
    "slope_mean", "slope_std", "slope_p90", "lrm_s_std", "lrm_l_std", "curv_prof_std", "curv_plan_std",
    "svf_mean", "svf_std", "detr_std", "detr_range_2_98", "tri", "susc_mean", "susc_high_frac",
)
#: Appendix A §10's twelve terrain statistics (terrain_stats_lr / terrain_stats_gbm).
TERRAIN_STATS: tuple[str, ...] = STAT_COLUMNS[:12]
#: The susceptibility statistics (slope_susc_lr, stage-2 comparator).
SUSC_STATS: tuple[str, ...] = STAT_COLUMNS[12:]

#: Susceptibility class codes (P1-07): 0 low or no data, 1 mid, 2 high.
SUSC_CODES = (0, 1, 2)
SUSC_HIGH = 2

#: The eight neighbour offsets (drow, dcol) of the TRI.
NEIGHBOURS: tuple[tuple[int, int], ...] = tuple((di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                                                if (di, dj) != (0, 0))


def tri(z_ring: np.ndarray) -> float:
    """Mean over the interior pixels of the mean |z - z_n| over the 8 neighbours (D19).

    `z_ring` is the DTM over the patch plus a 1 px ring, shape (px + 2, px + 2); the result averages over the px x px
    interior. NaN anywhere in the ring gives NaN.
    """
    z = np.asarray(z_ring, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 3 or z.shape[1] < 3:
        raise ValueError(f"need the patch plus a 1 px ring (at least 3 x 3), got shape {z.shape}")
    H, W = z.shape
    c = z[1:H - 1, 1:W - 1]
    acc = np.zeros(c.shape)
    for di, dj in NEIGHBOURS:
        acc += np.abs(c - z[1 + di:H - 1 + di, 1 + dj:W - 1 + dj])
    acc /= len(NEIGHBOURS)
    return float(acc.mean())


def susc_stats(susc: np.ndarray) -> tuple[float, float]:
    """(mean class code, fraction of high-class pixels) of a patch of the class raster. Raises on unknown codes."""
    s = np.asarray(susc)
    if s.ndim != 2 or s.size == 0:
        raise ValueError(f"need a 2-D susceptibility patch, got shape {s.shape}")
    if not np.isin(s, SUSC_CODES).all():
        bad = sorted(set(np.unique(s).tolist()) - set(SUSC_CODES))
        raise ValueError(f"unexpected susceptibility codes {bad} (expected {SUSC_CODES})")
    s64 = s.astype(np.float64)
    return float(s64.mean()), float((s == SUSC_HIGH).mean())


def patch_stats(ch: np.ndarray, names: Sequence[str], z_ring: np.ndarray, susc: np.ndarray) -> dict[str, float]:
    """The D19 statistics of one patch, in `STAT_COLUMNS` order.

    `ch` is the (7, px, px) float64 output of `patch_channels`, with channel names `names` (config order); `z_ring`
    the DTM over the patch plus a 1 px ring, (px + 2, px + 2); `susc` the class codes over the patch, (px, px).
    Raises if any input is not finite or the shapes disagree.
    """
    ch = np.asarray(ch)
    names = tuple(names)
    if ch.dtype != np.float64:
        raise TypeError(f"statistics are computed from the float64 channels, got {ch.dtype}")
    if ch.ndim != 3 or ch.shape[0] != len(names) or ch.shape[1] != ch.shape[2]:
        raise ValueError(f"channels of shape {ch.shape} do not match {len(names)} names")
    px = ch.shape[1]
    if np.asarray(z_ring).shape != (px + 2, px + 2) or np.asarray(susc).shape != (px, px):
        raise ValueError(f"z_ring {np.asarray(z_ring).shape} / susc {np.asarray(susc).shape} do not fit a "
                         f"{px} px patch")
    if not np.isfinite(ch).all():
        raise ValueError("channels contain NaN or inf")
    if not np.isfinite(z_ring).all():
        raise ValueError("the DTM ring around the patch contains NaN")
    c = {n: ch[i] for i, n in enumerate(names)}
    slope, detr = c["slope"], c["detr"]
    p2, p98 = np.percentile(detr, [2.0, 98.0])
    s_mean, s_high = susc_stats(susc)
    out = {
        "slope_mean": float(slope.mean()),
        "slope_std": float(slope.std()),
        "slope_p90": float(np.percentile(slope, 90.0)),
        "lrm_s_std": float(c["lrm_s"].std()),
        "lrm_l_std": float(c["lrm_l"].std()),
        "curv_prof_std": float(c["curv_prof"].std()),
        "curv_plan_std": float(c["curv_plan"].std()),
        "svf_mean": float(c["svf"].mean()),
        "svf_std": float(c["svf"].std()),
        "detr_std": float(detr.std()),
        "detr_range_2_98": float(p98 - p2),
        "tri": tri(z_ring),
        "susc_mean": s_mean,
        "susc_high_frac": s_high,
    }
    assert tuple(out) == STAT_COLUMNS
    return out
