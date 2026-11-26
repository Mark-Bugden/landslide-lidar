"""Sampling and slope matching (task P1-09): positives, random and look-alike negatives, 1:1 slope matching and the
unmatched background sample. BRIEF.md Appendix A §4-§5 and §12 (deviations 1, 5, 11, 12); PLAN.md D11-D17;
`docs/decisions/P1-09.md` records every reading.

Order of operations (Appendix A §5, literally):

1. **Positive feasibility** (pass 1, geometry only). A pixel corner of a stage-1 sheet is *patch-feasible* when the
   300 m square centred on it lies inside the sheet and does not touch any stage-2 sheet buffered by 150 m (§4 patch
   rule, D11). For each eligible stage-1 polygon, the patch-feasible corners strictly inside it are enumerated
   exactly (scanline on the corner lattice); a polygon with none is *infeasible* (§12 deviation 12).
2. **Positive points** (D12): k = clip(floor(area / 40,000 m²), 1, 3) corners per polygon, drawn at random and
   >= 100 m apart (greedy), fewer if there isn't room.
3. **Water-filling** (D13): the 4,000-point budget is split between the flysch and non-flysch pools by their share of
   eligible stage-1 polygons. In each pool every sheet takes its polygons whole, in seeded random order, up to a common
   per-sheet cap that counts points; the cap is the largest integer whose total fits the pool budget.
4. **Validity** (pass 2, one sheet DTM at a time): selected positives whose patch plus 28 px context is not fully
   valid are dropped and counted, before matching. There is no top-up (§12 deviation 11).
5. **Negative candidates** (pass 2): random candidates on a jittered 60 m grid (D14) and look-alike candidates (D15).
   Every candidate is patch-feasible, fully valid (validity applies at candidate level) and has its centre >= 150 m
   from every register polygon and every ZABAGED landslide/scree polygon (exact distances).
6. **Matching** (D15, D16): look-alikes first (round-robin over types, exact bin, levels 1, 2, 5) up to
   round(12 % of the positives), then random candidates through levels 1-6. Selected negatives are >= 150 m apart.
   Unmatched positives are dropped and counted.
7. **Background** (D17): 4,000 unused random candidates in seeded random order, >= 150 m from the selected negatives
   and from each other.

Geometry is exact on the selected samples: membership tests are scanline intervals on the corner lattice (exact up
to floating-point rounding of edge crossings) and exact GEOS distances; `check_samples` re-checks every rule on the
final table with shapely squares and exact distances.

Coordinates: corner (r, c) of a sheet raster is at `landslide.dtm.corner_xy(transform, r, c)`. All rasters lie on the
national DMR 5G lattice (x = 0.4 + 2i, y = -0.12 + 2j), so every corner also has integer lattice indices (gi, gj),
and distances between corners are tested exactly in integers.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shapely
from rasterio.transform import Affine
from scipy.spatial import cKDTree

from landslide.channels import channel_params, full_channels, terrain_channels
from landslide.config import stream_id
from landslide.dtm import Dtm, corner_xy
from landslide.geo import assign_to_sheets, repair
from landslide.logs import get_logger, progress
from landslide.sat import ValueSat, corner_box_sums, integral
from landslide.sheets import GRID_ORIGIN

log = get_logger("sampling")

#: `source` values of `samples.parquet` (PLAN.md §4.3).
SOURCES = ("positive", "random", "quarry", "spoil", "rock", "cut", "background")
RANDOM = "random"
#: Match levels (Appendix A §5): scope and bin offset (0 = exact bin, 1 = adjacent bin).
MATCH_LEVELS: dict[int, tuple[str, int]] = {1: ("sheet", 0), 2: ("block", 0), 3: ("sheet", 1), 4: ("block", 1),
                                            5: ("stratum", 0), 6: ("stratum", 1)}
#: Look-alikes are matched in the exact bin only, so at levels 1, 2 and 5 (D15).
LOOKALIKE_LEVELS = (1, 2, 5)
#: ZABAGED layer ids (BRIEF.md data-source row 6) by role. Read from `replication.zabaged_layers` if the config has it
#: (key requested from P1-02/P1-16); these are the brief's ids.
ZABAGED_LAYERS_DEFAULT: dict[str, int] = {"quarry": 118, "spoil": 119, "rock": 130, "landslide": 129, "road": 79,
                                          "railway": 75}
#: Roles whose geometries give look-alike candidates by type; roads and railways give the "cut" type.
LOOKALIKE_AREA_ROLES = ("quarry", "spoil", "rock")
CUT_ROLES = ("road", "railway")

_EMPTY = np.zeros(0, dtype=np.int64)


# ======================================================================================================================
# Parameters and random streams
# ======================================================================================================================


@dataclass(frozen=True)
class SamplingParams:
    """Every value this module uses, from the config (PLAN.md §3 rule 6)."""

    seed: int
    cell_m: float
    patch_px: int
    half_px: int
    half_m: float
    context_px: int
    max_pos: int
    max_neg: int
    pts_area_unit_m2: float
    max_pts_per_poly: int
    min_pt_spacing_m: float
    neg_min_dist_to_polygon_m: float
    neg_exclude_zabaged_landslide_m: float
    neg_candidate_spacing_m: float
    neg_jitter_m: float
    neg_min_spacing_m: float
    slope_bin_deg: float
    lookalike_frac: float
    lookalike_types: tuple[str, ...]
    lookalike_max_per_polygon: int
    cut_point_step_m: float
    cut_window_px: int
    cut_slope_p90_deg: float
    stage2_buffer_m: float
    background_n: int
    zabaged_layers: dict[str, int]

    @property
    def context_half_px(self) -> int:
        """Half-size of the patch plus its channel context (75 + 28 = 103 px)."""
        return self.half_px + self.context_px

    @property
    def grid_step_px(self) -> int:
        return int(round(self.neg_candidate_spacing_m / self.cell_m))


def sampling_params(cfg: Mapping[str, Any]) -> SamplingParams:
    """Collect and check the sampling values of the config."""
    s = cfg["sampling"]
    rep = cfg.get("replication", {})
    cell = float(cfg["dtm"]["cell_m"])
    px = int(cfg["patch"]["px"])
    size_m = float(cfg["patch"]["size_m"])
    if px % 2 or not math.isclose(px * cell, size_m):
        raise ValueError(f"patch.px ({px}) must be even and equal patch.size_m / dtm.cell_m ({size_m} / {cell})")
    context = channel_params(cfg).context_px
    types = tuple(str(t) for t in s["lookalike_types"])
    if set(types) - {"quarry", "spoil", "rock", "cut"} or len(set(types)) != len(types):
        raise ValueError(f"unknown or repeated look-alike types {types}")
    p = SamplingParams(
        seed=int(s["seed"]), cell_m=cell, patch_px=px, half_px=px // 2, half_m=size_m / 2.0, context_px=context,
        max_pos=int(s["max_pos"]), max_neg=int(s["max_neg"]),
        pts_area_unit_m2=float(s["pts_area_unit_m2"]), max_pts_per_poly=int(s["max_pts_per_poly"]),
        min_pt_spacing_m=float(s["min_pt_spacing_m"]),
        neg_min_dist_to_polygon_m=float(s["neg_min_dist_to_polygon_m"]),
        neg_exclude_zabaged_landslide_m=float(s["neg_exclude_zabaged_landslide_m"]),
        neg_candidate_spacing_m=float(s["neg_candidate_spacing_m"]),
        neg_jitter_m=float(rep["negative_jitter_m"]),
        neg_min_spacing_m=float(s["neg_min_spacing_m"]),
        slope_bin_deg=float(s["slope_bin_deg"]),
        lookalike_frac=float(s["lookalike_frac"]), lookalike_types=types,
        lookalike_max_per_polygon=int(s["lookalike_max_per_polygon"]),
        cut_point_step_m=float(s["cut_point_step_m"]), cut_window_px=int(s["cut_window_px"]),
        cut_slope_p90_deg=float(s["cut_slope_p90_deg"]),
        # Appendix A §4: "never touch any stage-2 sheet buffered by 150 m" -- half a stage-2 window, which is a patch.
        stage2_buffer_m=float(rep.get("sampling", {}).get("stage2_buffer_m", size_m / 2.0)),
        background_n=int(cfg["stage2"]["background_n"]),
        zabaged_layers={**ZABAGED_LAYERS_DEFAULT, **{k: int(v) for k, v in rep.get("zabaged_layers", {}).items()}},
    )
    step = p.neg_candidate_spacing_m / cell
    if abs(step - round(step)) > 1e-9:
        raise ValueError(f"neg_candidate_spacing_m ({p.neg_candidate_spacing_m}) must be a multiple of the cell")
    if p.cut_window_px % 2 == 0:
        raise ValueError(f"cut_window_px must be odd (centred on a pixel), got {p.cut_window_px}")
    return p


def sheet_code(sheet: str) -> int:
    """`14-34-07` -> 143407: a stable integer per sheet, used as a spawn key."""
    digits = str(sheet).replace("-", "")
    if not digits.isdigit():
        raise ValueError(f"not a sheet number: {sheet!r}")
    return int(digits)


def stream_seq(cfg: Mapping[str, Any], seed: int, name: str) -> np.random.SeedSequence:
    """`SeedSequence([seed, stream_id])` of a registered stream (D02, as `landslide.config.rng_for`)."""
    return np.random.SeedSequence([int(seed), stream_id(cfg, name)])


def sheet_rng(cfg: Mapping[str, Any], seed: int, name: str, sheet: str) -> np.random.Generator:
    """The per-sheet child of stream `name`: `SeedSequence([seed, stream_id], spawn_key=(sheet code,))`.

    Per-sheet steps (positive points, the water-filling order, the negative jitter, look-alike points) draw from the
    child of their sheet, so a sheet's draws don't depend on which other sheets are processed, or in what order.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream_id(cfg, name)],
                                                        spawn_key=(sheet_code(sheet),)))


def stream_rng(cfg: Mapping[str, Any], seed: int, name: str) -> np.random.Generator:
    """The plain stream `name` (global steps: matching, background)."""
    return np.random.default_rng(stream_seq(cfg, seed, name))


def round_half_up(x: float) -> int:
    """floor(x + 0.5): "rounded", without Python's round-half-to-even."""
    return int(math.floor(x + 0.5))


def slope_bins(slope_mean: np.ndarray, bin_deg: float) -> np.ndarray:
    """2° bins of patch-mean slope: floor(mean / bin)."""
    return np.floor(np.asarray(slope_mean, dtype=np.float64) / bin_deg).astype(np.int64)


# ======================================================================================================================
# The corner lattice of a sheet raster
# ======================================================================================================================


@dataclass(frozen=True)
class CornerGrid:
    """The pixel corners of one north-up raster: corner (r, c), 0 <= r < n_rows, 0 <= c < n_cols, sits at
    `corner_xy(transform, r, c)` = (x0 + cell * c, y0 - cell * r). A raster of h x w pixels has (h+1) x (w+1) corners.
    """

    x0: float
    y0: float
    cell: float
    n_rows: int
    n_cols: int

    @classmethod
    def from_transform(cls, transform: Affine, height: int, width: int) -> CornerGrid:
        if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0 or transform.a != -transform.e:
            raise ValueError(f"need a north-up raster with square pixels, got {transform}")
        return cls(float(transform.c), float(transform.f), float(transform.a), int(height) + 1, int(width) + 1)

    @property
    def transform(self) -> Affine:
        return Affine(self.cell, 0.0, self.x0, 0.0, -self.cell, self.y0)

    def xy(self, r, c) -> tuple[np.ndarray, np.ndarray]:
        return corner_xy(self.transform, r, c)

    def lattice_offset(self) -> tuple[int, int]:
        """(gi, gj) of corner (0, 0) on the national lattice x = x_origin + cell * gi, y = y_origin + cell * gj."""
        fi = (self.x0 - GRID_ORIGIN[0]) / self.cell
        fj = (self.y0 - GRID_ORIGIN[1]) / self.cell
        gi, gj = round(fi), round(fj)
        if abs(fi - gi) > 1e-6 or abs(fj - gj) > 1e-6:
            raise ValueError(f"raster origin ({self.x0}, {self.y0}) is not on the DMR 5G lattice")
        return int(gi), int(gj)

    def global_ij(self, r, c) -> tuple[np.ndarray, np.ndarray]:
        """National lattice indices of corners (r, c): gi grows east, gj grows north."""
        gi0, gj0 = self.lattice_offset()
        return gi0 + np.asarray(c, dtype=np.int64), gj0 - np.asarray(r, dtype=np.int64)

    def snap(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """The nearest corner (r, c) to world (x, y) (may lie outside the grid)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return (np.rint((self.y0 - y) / self.cell).astype(np.int64),
                np.rint((x - self.x0) / self.cell).astype(np.int64))

    def inside(self, r, c) -> np.ndarray:
        r = np.asarray(r)
        c = np.asarray(c)
        return (r >= 0) & (r < self.n_rows) & (c >= 0) & (c < self.n_cols)


# ======================================================================================================================
# Exact geometry on the corner lattice
# ======================================================================================================================


@dataclass(frozen=True)
class Intervals:
    """Inclusive column intervals of corners per row: corners (row[k], lo[k] .. hi[k]). Sorted by (row, lo) and
    disjoint (the even-odd pairs of a scanline never overlap)."""

    row: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def empty(cls) -> Intervals:
        return cls(_EMPTY, _EMPTY, _EMPTY)

    @property
    def n(self) -> int:
        """Number of intervals."""
        return int(self.row.size)

    def count(self) -> int:
        """Number of corners."""
        return int((self.hi - self.lo + 1).sum())


def ring_edges(geom: Any) -> np.ndarray:
    """All ring edges (exteriors and holes, every part) of a polygonal geometry as rows (x1, y1, x2, y2)."""
    parts = shapely.get_parts(geom)
    parts = parts[shapely.get_type_id(parts) == 3]  # polygons only
    if not len(parts):
        return np.zeros((0, 4))
    rings = shapely.get_rings(parts)
    xy, idx = shapely.get_coordinates(rings, return_index=True)
    same = idx[1:] == idx[:-1]
    return np.column_stack([xy[:-1][same], xy[1:][same]])


def row_intervals(geom: Any, grid: CornerGrid, max_cells: int = 4_000_000) -> Intervals:
    """The corners of `grid` strictly inside a polygonal geometry, as row intervals (even-odd scanline).

    For each corner row y, the crossings of the line with every non-horizontal ring edge (half-open in y, so a vertex
    on the line counts once) are sorted; consecutive pairs (x_a, x_b) bound the inside, and the corners with
    x_a < x < x_b are inside. This is exact up to the rounding of the crossing abscissae (~1e-10 m), and agrees with
    GEOS point-in-polygon (tested).
    """
    e = ring_edges(geom)
    e = e[e[:, 1] != e[:, 3]]
    if not len(e):
        return Intervals.empty()
    x1, y1, x2, y2 = e.T
    ylo, yhi = np.minimum(y1, y2), np.maximum(y1, y2)
    inv = (x2 - x1) / (y2 - y1)
    cell, x0, y0 = grid.cell, grid.x0, grid.y0
    r_first = max(0, math.ceil((y0 - yhi.max()) / cell))
    r_last = min(grid.n_rows - 1, math.floor((y0 - ylo.min()) / cell))
    if r_last < r_first:
        return Intervals.empty()
    rows_per_chunk = max(1, max_cells // len(e))
    out_r, out_lo, out_hi = [], [], []
    for ra in range(r_first, r_last + 1, rows_per_chunk):
        rr = np.arange(ra, min(r_last + 1, ra + rows_per_chunk), dtype=np.int64)
        y = y0 + (-cell) * rr.astype(np.float64)          # the same arithmetic as corner_xy
        sel = (yhi >= y[-1]) & (ylo <= y[0])
        if not sel.any():
            continue
        yy = y[:, None]
        active = (ylo[sel] <= yy) & (yy < yhi[sel])
        xi = np.where(active, x1[sel] + (yy - y1[sel]) * inv[sel], np.inf)
        xi.sort(axis=1)
        n_act = active.sum(axis=1)
        if np.any(n_act % 2):
            raise ValueError("odd number of edge crossings on a scanline: the rings are not closed")
        n_pairs = int(n_act.max()) // 2
        if n_pairs == 0:
            continue
        xa = xi[:, 0:2 * n_pairs:2]
        xb = xi[:, 1:2 * n_pairs:2]
        ok = np.arange(n_pairs)[None, :] < (n_act // 2)[:, None]
        rows = np.broadcast_to(rr[:, None], xa.shape)[ok]
        xa, xb = xa[ok], xb[ok]
        lo = np.floor((xa - x0) / cell).astype(np.int64) + 1
        hi = np.ceil((xb - x0) / cell).astype(np.int64) - 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, grid.n_cols - 1)
        keep = lo <= hi
        out_r.append(rows[keep])
        out_lo.append(lo[keep])
        out_hi.append(hi[keep])
    if not out_r:
        return Intervals.empty()
    r, lo, hi = np.concatenate(out_r), np.concatenate(out_lo), np.concatenate(out_hi)
    o = np.lexsort((lo, r))
    return Intervals(r[o], lo[o], hi[o])


def intersect_intervals(a: Intervals, b: Intervals) -> Intervals:
    """Corners in both interval sets."""
    if not a.n or not b.n:
        return Intervals.empty()
    start = np.searchsorted(b.row, a.row, side="left")
    end = np.searchsorted(b.row, a.row, side="right")
    n = end - start
    if not n.any():
        return Intervals.empty()
    ia = np.repeat(np.arange(a.n), n)
    first = np.repeat(np.cumsum(n) - n, n)
    ib = np.repeat(start, n) + (np.arange(int(n.sum())) - first)
    lo = np.maximum(a.lo[ia], b.lo[ib])
    hi = np.minimum(a.hi[ia], b.hi[ib])
    keep = lo <= hi
    r, lo, hi = a.row[ia][keep], lo[keep], hi[keep]
    o = np.lexsort((lo, r))
    return Intervals(r[o], lo[o], hi[o])


def expand_intervals(iv: Intervals) -> tuple[np.ndarray, np.ndarray]:
    """The corners (r, c) of an interval set, row-major."""
    n = iv.hi - iv.lo + 1
    if not n.size:
        return _EMPTY, _EMPTY
    r = np.repeat(iv.row, n)
    first = np.repeat(np.cumsum(n) - n, n)
    c = np.repeat(iv.lo, n) + (np.arange(int(n.sum())) - first)
    return r, c


def in_intervals(iv: Intervals, r, c) -> np.ndarray:
    """For each corner (r, c), whether it lies in one of the intervals."""
    r = np.asarray(r, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if not iv.n or not r.size:
        return np.zeros(r.shape, dtype=bool)
    key_iv = iv.row * (1 << 32) + iv.lo
    k = np.searchsorted(key_iv, r * (1 << 32) + c, side="right") - 1
    ok = k >= 0
    kk = np.where(ok, k, 0)
    return ok & (iv.row[kk] == r) & (iv.hi[kk] >= c)


def square_sweep(geom: Any, h: float) -> Any:
    """∂geom ⊕ [-h, h]²: the union over the ring edges of the hexagon swept by the square along each edge (the convex
    hull of the square at both ends). Exact polygonal Minkowski sum."""
    e = ring_edges(geom)
    if not len(e):
        return shapely.Polygon()
    off = np.array([(-h, -h), (h, -h), (h, h), (-h, h)])
    pts = np.concatenate([e[:, None, 0:2] + off[None], e[:, None, 2:4] + off[None]], axis=1)  # (E, 8, 2)
    hulls = shapely.convex_hull(shapely.multipoints(pts))
    return shapely.union_all(hulls)


def dilate_square(geom: Any, h: float) -> Any:
    """geom ⊕ [-h, h]² = geom ∪ (∂geom ⊕ square), exactly (a point of the sum outside geom lies in a square placed on
    the boundary)."""
    return shapely.union(geom, square_sweep(geom, h))


def erode_square(geom: Any, h: float) -> Any:
    """{c : c + [-h, h]² inside geom}, up to the boundary: geom minus (∂geom ⊕ square). For a convex polygon this is
    the intersection of the four translates (tested)."""
    return shapely.difference(geom, square_sweep(geom, h))


class Exclusion:
    """The rule "the centre is >= d from every geometry of a set", tested exactly (GEOS distances)."""

    def __init__(self, geoms: Sequence[Any] | np.ndarray, dist_m: float, name: str = ""):
        g = np.asarray(geoms, dtype=object)
        self.geoms = g[~shapely.is_empty(g)] if len(g) else g
        self.dist = float(dist_m)
        self.name = name
        self.tree = shapely.STRtree(self.geoms) if len(self.geoms) else None

    def far(self, x, y) -> np.ndarray:
        """True where the point (x, y) is at least `dist` from every geometry."""
        x = np.asarray(x, dtype=np.float64)
        out = np.ones(x.shape, dtype=bool)
        if self.tree is None or not x.size:
            return out
        pts = shapely.points(x, np.asarray(y, dtype=np.float64))
        ip, ig = self.tree.query(pts, predicate="dwithin", distance=self.dist)
        if ip.size:
            d = shapely.distance(pts[ip], self.geoms[ig])
            out[ip[d < self.dist]] = False
        return out

    def min_distance(self, x, y) -> np.ndarray:
        """The exact distance from each point to the nearest geometry (inf if the set is empty)."""
        x = np.asarray(x, dtype=np.float64)
        if self.tree is None or not x.size:
            return np.full(x.shape, np.inf)
        pts = shapely.points(x, np.asarray(y, dtype=np.float64))
        ip, ig = self.tree.query_nearest(pts, all_matches=False)
        d = np.full(x.shape, np.inf)
        d[ip] = shapely.distance(pts[ip], self.geoms[ig])
        return d


# ======================================================================================================================
# Per-sheet geometry: the patch rule (D11)
# ======================================================================================================================


@dataclass
class SheetGeometry:
    """The patch rule of one stage-1 sheet on its corner lattice.

    - `inside`: corners whose square (half-size h) lies inside the sheet: the scanline of `sheet ⊖ square`.
    - `zone`: the union of `Q ⊕ square` over the stage-2 sheets Q within reach; a corner whose square comes within
      `stage2_buffer_m` of a stage-2 sheet is a corner within that distance of `zone` (dist(c + S, Q) =
      dist(c, Q ⊕ S)), tested exactly with GEOS.
    """

    sheet: str
    geom: Any
    grid: CornerGrid
    inside: Intervals
    zone: Any | None
    zone_dist: float

    def near_stage2(self, r, c) -> np.ndarray:
        r = np.asarray(r, dtype=np.int64)
        out = np.zeros(r.shape, dtype=bool)
        if self.zone is None or not r.size:
            return out
        x, y = self.grid.xy(r, c)
        xmin, ymin, xmax, ymax = self.zone.bounds
        d = self.zone_dist
        cand = (x >= xmin - d) & (x <= xmax + d) & (y >= ymin - d) & (y <= ymax + d)
        if cand.any():
            out[cand] = shapely.dwithin(shapely.points(x[cand], y[cand]), self.zone, d)
        return out

    def patch_ok(self, r, c) -> np.ndarray:
        """The patch rule at corners (r, c)."""
        r = np.asarray(r, dtype=np.int64)
        c = np.asarray(c, dtype=np.int64)
        ok = in_intervals(self.inside, r, c)
        if ok.any():
            ok[ok] = ~self.near_stage2(r[ok], c[ok])
        return ok

    def feasible_in(self, geom: Any) -> tuple[np.ndarray, np.ndarray]:
        """The patch-feasible corners strictly inside `geom` (row-major)."""
        iv = intersect_intervals(row_intervals(geom, self.grid), self.inside)
        r, c = expand_intervals(iv)
        if r.size and self.zone is not None:
            keep = ~self.near_stage2(r, c)
            r, c = r[keep], c[keep]
        return r, c


def grid_of_sheet(row: Any, cell_m: float) -> CornerGrid:
    """The corner grid of a sheet raster from its `sheets.parquet` row (P1-08 read the grid from the raster)."""
    t = Affine(float(cell_m), 0.0, float(row.grid_xmin), 0.0, -float(cell_m), float(row.grid_ymax))
    return CornerGrid.from_transform(t, int(row.grid_h), int(row.grid_w))


def stage2_dilated(stage2_geoms: Sequence[Any], h: float) -> list[Any]:
    """Q ⊕ [-h, h]² for each stage-2 sheet."""
    return [dilate_square(g, h) for g in stage2_geoms]


def sheet_geometry(sheet: str, geom: Any, grid: CornerGrid, s2_dilated: Sequence[Any],
                   p: SamplingParams) -> SheetGeometry:
    inner = erode_square(geom, p.half_m)
    near = [q for q in s2_dilated if shapely.dwithin(q, geom, p.stage2_buffer_m)]
    zone = shapely.union_all(near) if near else None
    if zone is not None:
        shapely.prepare(zone)
    return SheetGeometry(sheet=sheet, geom=geom, grid=grid, inside=row_intervals(inner, grid), zone=zone,
                         zone_dist=p.stage2_buffer_m)


# ======================================================================================================================
# Positives (D12, D13)
# ======================================================================================================================


def points_per_polygon(area_m2: float, p: SamplingParams) -> int:
    """k = clip(floor(area / pts_area_unit_m2), 1, max_pts_per_poly) (Appendix A §5)."""
    return int(min(max(math.floor(float(area_m2) / p.pts_area_unit_m2), 1), p.max_pts_per_poly))


def greedy_spaced(r: np.ndarray, c: np.ndarray, k: int, min_dist_m: float, cell_m: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Up to k indices into (r, c): each draw is uniform among the corners still >= min_dist_m from every corner drawn
    so far (greedy random). Fewer than k if there isn't room."""
    cand = np.arange(r.size, dtype=np.int64)
    chosen: list[int] = []
    lim = (min_dist_m / cell_m) ** 2
    while len(chosen) < k and cand.size:
        j = int(cand[rng.integers(cand.size)])
        chosen.append(j)
        dr = r[cand] - r[j]
        dc = c[cand] - c[j]
        cand = cand[(dr * dr + dc * dc).astype(np.float64) >= lim]
    return np.asarray(chosen, dtype=np.int64)


def positive_points_sheet(cfg: Mapping[str, Any], p: SamplingParams, sg: SheetGeometry,
                          polys: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pass 1 for one sheet: the patch-feasible corners of each eligible polygon and its points (D12).

    `polys` has `objectid`, `area_m2`, `geom` (repaired where invalid), sorted by objectid. Returns per-polygon
    records and per-point records.
    """
    rng = sheet_rng(cfg, p.seed, "positive_points", sg.sheet)
    prow: list[dict[str, Any]] = []
    pts: list[dict[str, Any]] = []
    for oid, area, geom in zip(polys["objectid"], polys["area_m2"], polys["geom"]):
        r, c = sg.feasible_in(geom)
        k_target = points_per_polygon(area, p)
        pick = greedy_spaced(r, c, k_target, p.min_pt_spacing_m, p.cell_m, rng) if r.size else _EMPTY
        prow.append({"sheet": sg.sheet, "objectid": int(oid), "n_feasible": int(r.size), "k_target": k_target,
                     "k": int(pick.size)})
        for i, j in enumerate(pick):
            pts.append({"sheet": sg.sheet, "objectid": int(oid), "point": i, "row": int(r[j]), "col": int(c[j])})
    return prow, pts


def fill_sheet(ks: Sequence[int], cap: int) -> np.ndarray:
    """Polygons (in order, k points each) taken whole while the sheet's points stay <= cap; a polygon that doesn't fit
    is skipped and the next one is tried (D13)."""
    taken = np.zeros(len(ks), dtype=bool)
    acc = 0
    for i, k in enumerate(ks):
        if acc + k <= cap:
            taken[i] = True
            acc += k
    return taken


@dataclass
class PoolFill:
    pool: str
    budget: int
    n_eligible: int
    cap: int
    total: int
    totals_by_cap: dict[int, int]
    n_sheets: int
    n_polygons_available: int
    n_points_available: int


def waterfill(cfg: Mapping[str, Any], p: SamplingParams, poly: pd.DataFrame, sheet_pool: Mapping[str, str],
              n_eligible_by_pool: Mapping[str, int]) -> tuple[np.ndarray, dict[str, PoolFill]]:
    """D13. `poly` has `sheet`, `objectid`, `k` (points, 0 if infeasible). Returns a mask of the polygons taken and the
    per-pool record.

    Budgets: the flysch pool gets round(max_pos x its share of eligible stage-1 polygons), the other pool the rest.
    Order: each sheet's polygons with k >= 1, sorted by objectid, permuted by the sheet's child of the `waterfill`
    stream. Cap: every integer K from 1 to the largest per-sheet point total is tried, and the largest K whose total
    is <= the pool budget is used.
    """
    pools = sorted(set(n_eligible_by_pool))
    if pools != ["flysch", "nonflysch"]:
        raise ValueError(f"expected pools flysch and nonflysch, got {pools}")
    n_all = sum(int(v) for v in n_eligible_by_pool.values())
    b_f = round_half_up(p.max_pos * int(n_eligible_by_pool["flysch"]) / n_all) if n_all else 0
    budgets = {"flysch": b_f, "nonflysch": p.max_pos - b_f}
    if not isinstance(poly.index, pd.RangeIndex) or poly.index.start != 0 or poly.index.step != 1:
        raise ValueError("poly needs a 0..n-1 RangeIndex")
    taken = np.zeros(len(poly), dtype=bool)
    pool_of = poly["sheet"].map(sheet_pool)
    if pool_of.isna().any():
        raise ValueError("a polygon's sheet has no pool")
    out: dict[str, PoolFill] = {}
    for pool in pools:
        sub = poly.loc[(pool_of.to_numpy() == pool) & (poly["k"].to_numpy() >= 1)]
        orders: list[tuple[np.ndarray, list[int]]] = []
        for sheet, g in sub.groupby("sheet", sort=True):
            g = g.sort_values("objectid", kind="stable")
            perm = sheet_rng(cfg, p.seed, "waterfill", str(sheet)).permutation(len(g))
            orders.append((g.index.to_numpy()[perm], [int(v) for v in g["k"].to_numpy()[perm]]))
        k_max = max((sum(ks) for _, ks in orders), default=0)
        totals: dict[int, int] = {}
        for cap in range(1, k_max + 1):
            totals[cap] = int(sum(int(np.asarray(ks)[fill_sheet(ks, cap)].sum()) for _, ks in orders))
        fits = [cap for cap, t in totals.items() if t <= budgets[pool]]
        cap = max(fits) if fits else 0
        for idx, ks in orders:
            taken[idx[fill_sheet(ks, cap)]] = True
        out[pool] = PoolFill(pool=pool, budget=budgets[pool], n_eligible=int(n_eligible_by_pool[pool]), cap=cap,
                             total=totals.get(cap, 0), totals_by_cap=totals, n_sheets=len(orders),
                             n_polygons_available=int(len(sub)), n_points_available=int(sub["k"].sum()))
    return taken, out


# ======================================================================================================================
# Pass 2: one sheet DTM at a time
# ======================================================================================================================


@dataclass
class SheetRaster:
    """Validity and slope tables of one sheet raster (corner-centred boxes, PLAN.md D11)."""

    shape: tuple[int, int]
    valid_sat: np.ndarray
    slope: np.ndarray
    slope_sat: ValueSat

    def context_valid(self, r, c, half: int) -> np.ndarray:
        """True where every pixel of the corner-centred box of half-size `half` is inside the raster and valid."""
        r = np.asarray(r, dtype=np.int64)
        c = np.asarray(c, dtype=np.int64)
        H, W = self.shape
        out = np.zeros(r.shape, dtype=bool)
        fit = (r - half >= 0) & (r + half <= H) & (c - half >= 0) & (c + half <= W)
        if fit.any():
            out[fit] = corner_box_sums(self.valid_sat, r[fit], c[fit], half) == (2 * half) ** 2
        return out

    def slope_mean(self, r, c, half: int) -> np.ndarray:
        r = np.asarray(r, dtype=np.int64)
        if not r.size:
            return np.zeros(0)
        return self.slope_sat.mean(r, np.asarray(c, dtype=np.int64), half)

    def slope_p90(self, i, j, window_px: int) -> np.ndarray:
        """90th percentile (numpy's default linear interpolation) of slope over the window_px square centred on pixel
        (i, j); NaN if the window leaves the raster or holds an invalid value."""
        i = np.asarray(i, dtype=np.int64)
        j = np.asarray(j, dtype=np.int64)
        h = window_px // 2
        H, W = self.shape
        out = np.full(i.shape, np.nan)
        fit = (i - h >= 0) & (i + h < H) & (j - h >= 0) & (j + h < W)
        if fit.any():
            d = np.arange(-h, h + 1)
            ii = i[fit][:, None, None] + d[None, :, None]
            jj = j[fit][:, None, None] + d[None, None, :]
            win = self.slope[ii, jj].reshape(int(fit.sum()), -1)
            out[fit] = np.percentile(win, 90, axis=1)
        return out


def sheet_raster(z: np.ndarray, cfg: Mapping[str, Any]) -> SheetRaster:
    """Validity SAT of z and the slope (the patch code's slope, `full_channels`) with its SAT."""
    valid = np.isfinite(z)
    vs = integral(valid, dtype=np.int32 if z.size < 2**31 else np.int64)
    slope = full_channels(z, cfg, names=["slope"])[0]
    return SheetRaster(shape=z.shape, valid_sat=vs, slope=slope, slope_sat=ValueSat(slope))


def random_grid(grid: CornerGrid, p: SamplingParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """D14: the 60 m grid anchored at the raster origin (corner (0, 0)), row-major over the whole raster, each point
    moved by U(-15, 15) m per axis and snapped to the nearest corner. Returns all grid points (r, c), possibly outside
    the grid; the jitter is drawn for every grid point, so the draws don't depend on the filters."""
    step = p.grid_step_px
    rows = np.arange(0, grid.n_rows, step, dtype=np.int64)
    cols = np.arange(0, grid.n_cols, step, dtype=np.int64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    j = rng.uniform(-p.neg_jitter_m, p.neg_jitter_m, size=(rr.size, 2))   # (east, north) metres
    x, y = grid.xy(rr, cc)
    return grid.snap(x + j[:, 0], y + j[:, 1])


@dataclass
class Rules:
    """The exclusion rules shared by all negative candidates."""

    register: Exclusion
    landslide: Exclusion


def exclusion_rules(p: SamplingParams, polygons: pd.DataFrame, zabaged: Mapping[str, pd.DataFrame]) -> Rules:
    """Centre >= neg_min_dist_to_polygon_m from every register polygon (all of them, any group; repaired copies of
    the invalid ones) and >= neg_exclude_zabaged_landslide_m from every ZABAGED landslide/scree feature."""
    ls = zabaged.get("landslide")
    if ls is None:
        raise KeyError("ZABAGED layer 'landslide' (129) is missing")
    return Rules(register=Exclusion(repair(np.asarray(polygons["geometry"].to_numpy(), dtype=object)),
                                    p.neg_min_dist_to_polygon_m, "register"),
                 landslide=Exclusion(repair(np.asarray(ls["geometry"].to_numpy(), dtype=object)),
                                     p.neg_exclude_zabaged_landslide_m, "zabaged_landslide"))


def candidate_filter(sg: SheetGeometry, sr: SheetRaster, rules: Rules, p: SamplingParams, r: np.ndarray,
                     c: np.ndarray, counts: Counter | None = None, prefix: str = "") -> np.ndarray:
    """Mask of the corners that pass, in order: inside the grid, the patch rule (inside the sheet, away from stage-2
    sheets), full validity of the patch plus context, centre >= 150 m from every register polygon, centre >= 150 m
    from every ZABAGED landslide/scree polygon. Counts the survivors of each step under `prefix`."""
    r = np.asarray(r, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    ok = sg.grid.inside(r, c)
    if counts is not None:
        counts[prefix + "points"] += int(r.size)
        counts[prefix + "in_grid"] += int(ok.sum())

    def step(name: str, f: Callable[[np.ndarray], np.ndarray]) -> None:
        idx = np.flatnonzero(ok)
        if idx.size:
            ok[idx] = f(idx)
        if counts is not None:
            counts[prefix + name] += int(ok.sum())

    def far(excl: Exclusion) -> Callable[[np.ndarray], np.ndarray]:
        def f(i: np.ndarray) -> np.ndarray:
            x, y = sg.grid.xy(r[i], c[i])
            return excl.far(x, y)
        return f

    step("in_sheet", lambda i: in_intervals(sg.inside, r[i], c[i]))
    step("clear_of_stage2", lambda i: ~sg.near_stage2(r[i], c[i]))
    step("valid", lambda i: sr.context_valid(r[i], c[i], p.context_half_px))
    step("far_from_register", far(rules.register))
    step("far_from_zabaged_landslide", far(rules.landslide))
    return ok


def dedupe_corners(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Indices of the first occurrence of each corner, in input order."""
    if not r.size:
        return _EMPTY
    key = r * (1 << 32) + c
    _, first = np.unique(key, return_index=True)
    return np.sort(first)


POLYGONAL = (3, 6)   # shapely type ids: Polygon, MultiPolygon
PUNCTUAL = (0, 4)    # Point, MultiPoint


@dataclass
class LookalikeSources:
    """Look-alike source features, prepared once.

    - `areas`: the quarry, spoil and rock features (`type`, `layer`, `objectid`, `geom`, `sheet`), polygonal or point,
      each assigned to one sheet: a polygon to the sheet holding the largest part of its area (the Appendix A §4 rule
      for register polygons; ties to the lowest sheet number), a point feature to the sheet containing its
      representative point. Only features assigned to a stage-1 sheet are kept, sorted by (sheet, type, objectid).
    - `cut`: points every `cut_point_step_m` along every road and railway part (`layer`, `objectid`, `x`, `y`,
      `sheet`: the stage-1 sheet containing the point), in (layer role, objectid, arc length) order.
    - `stats`: feature counts by type and geometry kind.
    """

    areas: pd.DataFrame
    cut: pd.DataFrame
    stats: dict[str, Any]


def cut_points(lines: pd.DataFrame, step_m: float) -> pd.DataFrame:
    """Points at arc length 0, step, 2 step, ... (< length) along every LineString part of every feature, in feature
    order. `lines` has `layer`, `objectid`, `geom`. Returns `layer`, `objectid`, `x`, `y`."""
    if not len(lines):
        return pd.DataFrame({"layer": _EMPTY, "objectid": _EMPTY, "x": np.zeros(0), "y": np.zeros(0)})
    parts, idx = shapely.get_parts(np.asarray(lines["geom"].to_numpy(), dtype=object), return_index=True)
    is_line = np.isin(shapely.get_type_id(parts), [1, 2])  # LineString, LinearRing
    parts, idx = parts[is_line], idx[is_line]
    length = shapely.length(parts)
    n = np.ceil(length / step_m).astype(np.int64)
    n[~(length > 0)] = 0
    pi = np.repeat(np.arange(parts.size), n)
    d = (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)) * step_m
    xy = shapely.get_coordinates(shapely.line_interpolate_point(parts[pi], d))
    fi = idx[pi]
    return pd.DataFrame({"layer": lines["layer"].to_numpy()[fi].astype(np.int64),
                         "objectid": lines["objectid"].to_numpy()[fi].astype(np.int64),
                         "x": xy[:, 0], "y": xy[:, 1]})


def _containing_sheet(x: np.ndarray, y: np.ndarray, sheet_geoms: np.ndarray, sheet_ids: Sequence[str]) -> np.ndarray:
    """The sheet containing each point (on a shared edge, the first sheet in input order), or None."""
    out = np.full(np.asarray(x).size, None, dtype=object)
    if not out.size or not len(sheet_geoms):
        return out
    tree = shapely.STRtree(sheet_geoms)
    ip, js = tree.query(shapely.points(x, y), predicate="intersects")
    if ip.size:
        first = pd.Series(js).groupby(ip).min()
        out[first.index.to_numpy()] = np.asarray(list(sheet_ids), dtype=object)[first.to_numpy()]
    return out


def prepare_lookalikes(zabaged: Mapping[str, pd.DataFrame], sheets: pd.DataFrame,
                       p: SamplingParams) -> LookalikeSources:
    """Assign the look-alike source features to sheets and build the cut points (D15)."""
    stage1 = set(sheets.loc[sheets["is_stage1"].astype(bool), "sheet"].astype(str))
    sheet_geoms = sheets["geometry"].to_numpy()
    sheet_ids = sheets["sheet"].astype(str).tolist()
    stats: dict[str, Any] = {}
    rows = []
    for t in LOOKALIKE_AREA_ROLES:
        if t not in p.lookalike_types:
            continue
        df = zabaged.get(t)
        if df is None:
            raise KeyError(f"ZABAGED layer for look-alike type {t!r} is missing")
        g = repair(np.asarray(df["geometry"].to_numpy(), dtype=object))
        tid = shapely.get_type_id(g)
        poly, pt = np.isin(tid, POLYGONAL), np.isin(tid, PUNCTUAL)
        sheet = np.full(len(g), None, dtype=object)
        if poly.any():
            sheet[poly], _ = assign_to_sheets(g[poly], sheet_geoms, sheet_ids)
        if pt.any():
            rp = shapely.get_coordinates(shapely.point_on_surface(g[pt]))
            sheet[pt] = _containing_sheet(rp[:, 0], rp[:, 1], sheet_geoms, sheet_ids)
        use = (poly | pt) & np.isin(sheet.astype(str), list(stage1))
        stats[t] = {"features": int(len(g)), "polygonal": int(poly.sum()), "point": int(pt.sum()),
                    "other_geometry": int((~poly & ~pt).sum()), "in_stage1_sheets": int(use.sum())}
        rows.append(pd.DataFrame({"type": t, "layer": int(p.zabaged_layers[t]),
                                  "objectid": df["objectid"].to_numpy()[use].astype(np.int64),
                                  "geom": g[use], "sheet": sheet[use]}))
    if rows:
        areas = pd.concat(rows, ignore_index=True)
        areas["_t"] = areas["type"].map({t: i for i, t in enumerate(LOOKALIKE_AREA_ROLES)})
        areas = areas.sort_values(["sheet", "_t", "objectid"], kind="stable").drop(columns="_t").reset_index(drop=True)
    else:
        areas = pd.DataFrame({"type": pd.Series(dtype=object), "layer": _EMPTY, "objectid": _EMPTY,
                              "geom": pd.Series(dtype=object), "sheet": pd.Series(dtype=object)})
    lines = []
    if "cut" in p.lookalike_types:
        for role in CUT_ROLES:
            df = zabaged.get(role)
            if df is None:
                raise KeyError(f"ZABAGED layer {role!r} (cuts) is missing")
            lines.append(pd.DataFrame({"layer": int(p.zabaged_layers[role]),
                                       "objectid": df["objectid"].to_numpy().astype(np.int64),
                                       "geom": np.asarray(df["geometry"].to_numpy(), dtype=object)}))
            stats[role] = {"features": int(len(df))}
    cut = cut_points(pd.concat(lines, ignore_index=True) if lines else pd.DataFrame(), p.cut_point_step_m)
    st1 = sheets.loc[sheets["is_stage1"].astype(bool)]
    cut["sheet"] = _containing_sheet(cut["x"].to_numpy(), cut["y"].to_numpy(), st1["geometry"].to_numpy(),
                                     st1["sheet"].astype(str).tolist())
    stats["cut_points"] = int(len(cut))
    stats["cut_points_in_stage1_sheets"] = int(cut["sheet"].notna().sum())
    cut = cut.loc[cut["sheet"].notna()].reset_index(drop=True)
    return LookalikeSources(areas=areas, cut=cut, stats=stats)


def lookalike_candidates_sheet(cfg: Mapping[str, Any], p: SamplingParams, sg: SheetGeometry, sr: SheetRaster,
                               rules: Rules, areas: pd.DataFrame, cuts: pd.DataFrame,
                               counts: Counter) -> pd.DataFrame:
    """D15 for one sheet.

    - Area features (in (type, objectid) order): the corners strictly inside a polygon, or the snapped points of a
      point feature, that pass every candidate rule; up to `lookalike_max_per_polygon` of them drawn uniformly
      without replacement (the sheet's child of the `lookalike_points` stream).
    - Cut points: snapped to the nearest corner; kept if every candidate rule passes and the 90th-percentile slope
      over the `cut_window_px` window centred on the pixel holding the point is >= `cut_slope_p90_deg`; one candidate
      per corner (the first point in order).
    """
    rng = sheet_rng(cfg, p.seed, "lookalike_points", sg.sheet)
    out = []
    for t, layer, oid, geom in zip(areas["type"], areas["layer"], areas["objectid"], areas["geom"]):
        counts[f"{t}:features"] += 1
        if shapely.get_type_id(geom) in POLYGONAL:
            r, c = expand_intervals(row_intervals(geom, sg.grid))
        else:
            xy = shapely.get_coordinates(geom)
            r, c = sg.grid.snap(xy[:, 0], xy[:, 1])
            keep = dedupe_corners(r, c)
            r, c = r[keep], c[keep]
        idx = np.flatnonzero(candidate_filter(sg, sr, rules, p, r, c, counts, prefix=f"{t}:"))
        if not idx.size:
            continue
        counts[f"{t}:features_with_candidates"] += 1
        pick = idx[rng.choice(idx.size, size=min(p.lookalike_max_per_polygon, idx.size), replace=False)]
        counts[f"{t}:candidates"] += int(pick.size)
        out.append(pd.DataFrame({"row": r[pick], "col": c[pick], "source": t, "source_layer": int(layer),
                                 "source_objectid": int(oid)}))
    if len(cuts):
        x, y = cuts["x"].to_numpy(), cuts["y"].to_numpy()
        r, c = sg.grid.snap(x, y)
        idx = np.flatnonzero(candidate_filter(sg, sr, rules, p, r, c, counts, prefix="cut:"))
        if idx.size:
            i = np.floor((sg.grid.y0 - y[idx]) / sg.grid.cell).astype(np.int64)   # the pixel holding the point
            j = np.floor((x[idx] - sg.grid.x0) / sg.grid.cell).astype(np.int64)
            idx = idx[sr.slope_p90(i, j, p.cut_window_px) >= p.cut_slope_p90_deg]
            counts["cut:steep"] += int(idx.size)
            idx = idx[dedupe_corners(r[idx], c[idx])]
            counts["cut:candidates"] += int(idx.size)
            out.append(pd.DataFrame({"row": r[idx], "col": c[idx], "source": "cut",
                                     "source_layer": cuts["layer"].to_numpy()[idx],
                                     "source_objectid": cuts["objectid"].to_numpy()[idx]}))
    if not out:
        return pd.DataFrame({"row": _EMPTY, "col": _EMPTY, "source": pd.Series(dtype=object),
                             "source_layer": _EMPTY, "source_objectid": _EMPTY})
    return pd.concat(out, ignore_index=True)


# ======================================================================================================================
# Matching (D15, D16) and the background (D17)
# ======================================================================================================================


class Spacing:
    """The >= neg_min_spacing_m rule among selected negatives (and the background), exact on the lattice.

    `alive[i]` is False once candidate i is selected or lies < spacing from a selected one.
    """

    def __init__(self, gi: np.ndarray, gj: np.ndarray, spacing_m: float, cell_m: float):
        self.gi = np.asarray(gi, dtype=np.int64)
        self.gj = np.asarray(gj, dtype=np.int64)
        self.alive = np.ones(self.gi.size, dtype=bool)
        self.lim = (spacing_m / cell_m) ** 2
        self.radius = spacing_m / cell_m
        self.tree = cKDTree(np.column_stack([self.gi, self.gj]).astype(np.float64)) if self.gi.size else None

    def take(self, i: int) -> None:
        near = np.asarray(self.tree.query_ball_point([float(self.gi[i]), float(self.gj[i])], r=self.radius),
                          dtype=np.int64)
        if near.size:
            di = self.gi[near] - self.gi[i]
            dj = self.gj[near] - self.gj[i]
            self.alive[near[(di * di + dj * dj).astype(np.float64) < self.lim]] = False
        self.alive[i] = False


class Matcher:
    """Candidate lookup by (scope, bin, class) and the spacing state."""

    def __init__(self, pos: pd.DataFrame, cand: pd.DataFrame, p: SamplingParams):
        self.pos = pos
        self.cand = cand
        self.p = p
        self.spacing = Spacing(cand["gi"].to_numpy(), cand["gj"].to_numpy(), p.neg_min_spacing_m, p.cell_m)
        self.groups: dict[tuple[str, Any, int, str], np.ndarray] = {}
        for kind in ("sheet", "block", "stratum"):
            for key, idx in cand.groupby([kind, "slope_bin", "cls"], sort=True).indices.items():
                self.groups[(kind, key[0], int(key[1]), str(key[2]))] = np.sort(np.asarray(idx, dtype=np.int64))
        self.p_attr = {k: pos[k].to_numpy() for k in ("sheet", "block", "stratum")}
        self.p_bin = pos["slope_bin"].to_numpy().astype(np.int64)

    def options(self, i: int, level: int, cls: str) -> np.ndarray:
        kind, off = MATCH_LEVELS[level]
        key = self.p_attr[kind][i]
        b = int(self.p_bin[i])
        bins = (b,) if off == 0 else (b - 1, b + 1)
        arrs = [self.groups.get((kind, key, bb, cls), _EMPTY) for bb in bins]
        idx = arrs[0] if len(arrs) == 1 else np.concatenate(arrs)
        return idx[self.spacing.alive[idx]]


@dataclass
class MatchResult:
    neg: np.ndarray          # candidate index matched to each positive (-1 if unmatched)
    level: np.ndarray        # match level (0 if unmatched)
    lookalike_quota: int
    background: np.ndarray   # candidate indices of the background sample, in draw order
    background_available: int


def match(cfg: Mapping[str, Any], p: SamplingParams, pos: pd.DataFrame, cand: pd.DataFrame) -> MatchResult:
    """Look-alikes first, then random candidates (D15, D16); then the background (D17).

    `pos` needs sheet, block, stratum, slope_bin (row order = positive index); `cand` needs sheet, block, stratum,
    slope_bin, cls ("random" or a look-alike type), gi, gj.

    Look-alike phase, for level in (1, 2, 5) while fewer than the quota are matched: the still-unmatched positives are
    put in random order; the look-alike types take turns (config order); on its turn a type is matched to the first
    positive in that order that has an available candidate of the type in the level's scope and exact bin, chosen at
    random among them; a type with no such positive leaves the rotation for this level.
    Random phase, for level 1..6: the unmatched positives in random order each take a random available random
    candidate in the level's scope and bin(s).
    "Available" = not selected and >= neg_min_spacing_m from every selected negative.
    """
    rng = stream_rng(cfg, p.seed, "matching")
    n = len(pos)
    neg = np.full(n, -1, dtype=np.int64)
    level_of = np.zeros(n, dtype=np.int64)
    M = Matcher(pos, cand, p)
    quota = round_half_up(p.lookalike_frac * n)
    n_la = 0
    types = [t for t in p.lookalike_types]
    for level in LOOKALIKE_LEVELS:
        if n_la >= quota:
            break
        order = rng.permutation(np.flatnonzero(neg < 0))
        active = list(types)
        ptr = {t: 0 for t in types}
        k = 0
        while n_la < quota and active:
            k %= len(active)
            t = active[k]
            found = None
            while ptr[t] < order.size:
                i = int(order[ptr[t]])
                if neg[i] < 0:
                    opts = M.options(i, level, t)
                    if opts.size:
                        found = (i, opts)
                        break
                ptr[t] += 1
            if found is None:
                del active[k]
                continue
            i, opts = found
            ci = int(opts[rng.integers(opts.size)])
            neg[i], level_of[i] = ci, level
            M.spacing.take(ci)
            n_la += 1
            k += 1
    for level in sorted(MATCH_LEVELS):
        for i in rng.permutation(np.flatnonzero(neg < 0)):
            opts = M.options(int(i), level, RANDOM)
            if opts.size:
                ci = int(opts[rng.integers(opts.size)])
                neg[i], level_of[i] = ci, level
                M.spacing.take(ci)
    # background (D17)
    brng = stream_rng(cfg, p.seed, "background")
    is_random = cand["cls"].to_numpy() == RANDOM
    pool = np.flatnonzero(is_random & M.spacing.alive)
    bg: list[int] = []
    for ci in brng.permutation(pool):
        if len(bg) >= p.background_n:
            break
        if M.spacing.alive[ci]:
            bg.append(int(ci))
            M.spacing.take(int(ci))
    return MatchResult(neg=neg, level=level_of, lookalike_quota=quota, background=np.asarray(bg, dtype=np.int64),
                       background_available=int(pool.size))


# ======================================================================================================================
# The whole build
# ======================================================================================================================


@dataclass
class Inputs:
    """Everything the build reads.

    - `sheets`: `sheets.parquet` (sheet, geometry, grid_xmin, grid_ymax, grid_w, grid_h, flysch, kraj, group_b,
      stratum, block, fold_a, is_stage1, is_stage2).
    - `polygons`: `polygons.parquet` (objectid, eligible, area_m2, sheet, geometry): all register polygons.
    - `zabaged`: role -> frame with `objectid` and `geometry` (EPSG:5514); roles in `ZABAGED_LAYERS_DEFAULT`.
    - `read_dtm(sheet)`: the sheet's DTM with the invalid rule applied (`landslide.dtm.read_dtm`).
    """

    sheets: pd.DataFrame
    polygons: pd.DataFrame
    zabaged: Mapping[str, pd.DataFrame]
    read_dtm: Callable[[str], Dtm]


@dataclass
class SamplingResult:
    samples: pd.DataFrame
    summary: dict[str, Any]
    polygons: pd.DataFrame = field(repr=False)      # per eligible stage-1 polygon: feasibility, points, taken
    positives: pd.DataFrame = field(repr=False)     # every selected positive point, with validity and slope
    candidates: pd.DataFrame = field(repr=False)    # every negative candidate
    match: MatchResult | None = field(repr=False, default=None)


SHEET_ATTRS = ("kraj", "flysch", "group_b", "block", "stratum", "fold")


def _sheet_table(sheets: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"sheet": sheets["sheet"].astype(str).to_numpy(),
                      "kraj": sheets["kraj"].to_numpy(dtype=object),
                      "flysch": sheets["flysch"].astype(bool).to_numpy(),
                      "group_b": sheets["group_b"].to_numpy(dtype=object),
                      "block": sheets["block"].to_numpy(dtype=object),
                      "stratum": sheets["stratum"].to_numpy(dtype=object),
                      "fold": pd.array(sheets["fold_a"], dtype="Int64").fillna(-1).to_numpy(dtype=np.int64)})
    return t.set_index("sheet")


def build_samples(cfg: Mapping[str, Any], inputs: Inputs, show_progress: bool = False) -> SamplingResult:
    """The P1-09 build (module docstring). Deterministic: every random step draws from a registered stream."""
    p = sampling_params(cfg)
    sheets = inputs.sheets.sort_values("sheet", kind="stable").reset_index(drop=True)
    stage1 = sheets.loc[sheets["is_stage1"].astype(bool)].reset_index(drop=True)
    stage2 = sheets.loc[sheets["is_stage2"].astype(bool)]
    attrs = _sheet_table(sheets)
    s2d = stage2_dilated(list(stage2["geometry"]), p.half_m)
    polys_all = inputs.polygons.sort_values("objectid", kind="stable").reset_index(drop=True)
    st1_set = set(stage1["sheet"])
    elig = polys_all.loc[polys_all["eligible"].astype(bool) & polys_all["sheet"].isin(st1_set)].copy()
    elig["geom"] = repair(elig["geometry"].to_numpy())
    bar = (lambda it, **kw: progress(it, **kw)) if show_progress else (lambda it, **kw: it)
    log.info("sampling: %d stage-1 sheets, %d stage-2 sheets, %d eligible stage-1 polygons", len(stage1),
             len(stage2), len(elig))

    # ---- pass 1: positive feasibility and points (geometry only) ----
    geoms: dict[str, SheetGeometry] = {}
    prows: list[dict[str, Any]] = []
    ppts: list[dict[str, Any]] = []
    by_sheet = {s: g for s, g in elig.groupby("sheet", sort=True)}
    for row in bar(list(stage1.itertuples(index=False)), desc="pass 1 (positives)", unit="sheet"):
        sg = sheet_geometry(row.sheet, row.geometry, grid_of_sheet(row, p.cell_m), s2d, p)
        geoms[row.sheet] = sg
        g = by_sheet.get(row.sheet)
        if g is None:
            continue
        a, b = positive_points_sheet(cfg, p, sg, g.sort_values("objectid", kind="stable"))
        prows += a
        ppts += b
    poly = pd.DataFrame(prows, columns=["sheet", "objectid", "n_feasible", "k_target", "k"])
    poly = poly.sort_values(["sheet", "objectid"], kind="stable").reset_index(drop=True)
    pts = pd.DataFrame(ppts, columns=["sheet", "objectid", "point", "row", "col"])
    if len(poly) != len(elig):
        raise AssertionError("pass 1 lost polygons")

    # ---- water-filling ----
    sheet_pool = {s: ("flysch" if f else "nonflysch") for s, f in zip(stage1["sheet"], stage1["flysch"].astype(bool))}
    pool_of_poly = poly["sheet"].map(sheet_pool)
    n_elig = {"flysch": int((pool_of_poly == "flysch").sum()), "nonflysch": int((pool_of_poly == "nonflysch").sum())}
    taken, fills = waterfill(cfg, p, poly, sheet_pool, n_elig)
    poly["taken"] = taken
    sel = poly.loc[poly["taken"], ["sheet", "objectid"]]
    pos = pts.merge(sel, on=["sheet", "objectid"], how="inner").sort_values(["sheet", "objectid", "point"],
                                                                            kind="stable").reset_index(drop=True)

    # ---- pass 2: validity of the positives, negative candidates (one sheet DTM at a time) ----
    rules = exclusion_rules(p, polys_all, inputs.zabaged)
    look = prepare_lookalikes(inputs.zabaged, sheets, p)
    log.info("pass 1: %d infeasible of %d polygons; water-fill caps %s; %d positives selected",
             int((poly["n_feasible"] == 0).sum()), len(poly), {k: f.cap for k, f in fills.items()}, len(pos))
    area_by_sheet = {s: g for s, g in look.areas.groupby("sheet", sort=True)}
    cut_by_sheet = {s: g for s, g in look.cut.groupby("sheet", sort=True)}
    pos_by_sheet = {s: g.index.to_numpy() for s, g in pos.groupby("sheet", sort=True)}
    pos_valid = np.zeros(len(pos), dtype=bool)
    pos_slope = np.full(len(pos), np.nan)
    counts: Counter = Counter()
    cands = []
    for row in bar(list(stage1.itertuples(index=False)), desc="pass 2 (DTM)", unit="sheet"):
        sg = geoms[row.sheet]
        dtm = inputs.read_dtm(row.sheet)
        if CornerGrid.from_transform(dtm.transform, *dtm.shape) != sg.grid:
            raise AssertionError(f"{row.sheet}: the raster grid differs from sheets.parquet")
        sr = sheet_raster(dtm.z, cfg)
        del dtm
        ip = pos_by_sheet.get(row.sheet)
        if ip is not None:
            r, c = pos["row"].to_numpy()[ip], pos["col"].to_numpy()[ip]
            v = sr.context_valid(r, c, p.context_half_px)
            pos_valid[ip] = v
            pos_slope[ip[v]] = sr.slope_mean(r[v], c[v], p.half_px)
        rr, cc = random_grid(sg.grid, p, sheet_rng(cfg, p.seed, "negative_jitter", row.sheet))
        ok = candidate_filter(sg, sr, rules, p, rr, cc, counts, prefix="random:")
        rnd = pd.DataFrame({"row": rr[ok], "col": cc[ok], "source": RANDOM, "source_layer": -1,
                            "source_objectid": -1})
        la = lookalike_candidates_sheet(cfg, p, sg, sr, rules,
                                        area_by_sheet.get(row.sheet, look.areas.iloc[0:0]),
                                        cut_by_sheet.get(row.sheet, look.cut.iloc[0:0]), counts)
        cs = pd.concat([rnd, la], ignore_index=True)
        if len(cs):
            cs["slope_mean"] = sr.slope_mean(cs["row"].to_numpy(), cs["col"].to_numpy(), p.half_px)
            cs["sheet"] = row.sheet
            cands.append(cs)
        del sr
    cand = (pd.concat(cands, ignore_index=True) if cands else
            pd.DataFrame(columns=["row", "col", "source", "source_layer", "source_objectid", "slope_mean", "sheet"]))
    pos["valid"] = pos_valid
    pos["slope_mean"] = pos_slope
    if pos.loc[pos["valid"], "slope_mean"].isna().any() or cand["slope_mean"].isna().any():
        raise AssertionError("a fully valid patch has a NaN slope mean")

    # ---- matching ----
    kept = pos.loc[pos["valid"]].reset_index(drop=True)
    kept["slope_bin"] = slope_bins(kept["slope_mean"].to_numpy(), p.slope_bin_deg)
    for k in ("block", "stratum"):
        kept[k] = kept["sheet"].map(attrs[k])
    cand["slope_bin"] = slope_bins(cand["slope_mean"].to_numpy(), p.slope_bin_deg)
    for k in ("block", "stratum"):
        cand[k] = cand["sheet"].map(attrs[k])
    cand["cls"] = cand["source"]
    gi = np.zeros(len(cand), dtype=np.int64)
    gj = np.zeros(len(cand), dtype=np.int64)
    for s, g in cand.groupby("sheet", sort=False):
        gi[g.index], gj[g.index] = geoms[s].grid.global_ij(g["row"].to_numpy(), g["col"].to_numpy())
    cand["gi"], cand["gj"] = gi, gj
    log.info("pass 2: %d of %d positives fail validity; %d candidates (%s)", int((~pos["valid"]).sum()), len(pos),
             len(cand), dict(sorted(Counter(cand["source"]).items())))
    m = match(cfg, p, kept, cand)
    log.info("matching: %d of %d positives matched (look-alikes %d, quota %d); background %d",
             int((m.neg >= 0).sum()), len(kept), int((cand["cls"].to_numpy()[m.neg[m.neg >= 0]] != RANDOM).sum()),
             m.lookalike_quota, len(m.background))

    samples = assemble(p, kept, cand, m, geoms, attrs)
    summary = summarise(p, poly, pos, kept, cand, m, fills, counts, samples, look, attrs)
    return SamplingResult(samples=samples, summary=summary, polygons=poly, positives=pos, candidates=cand, match=m)


COLUMNS = ["id", "label", "x", "y", "row", "col", "sheet", "kraj", "flysch", "source", "polygon_objectid",
           "source_layer", "source_objectid", "fold", "group_b", "block", "stratum", "slope_mean", "slope_bin",
           "match_level", "pair_id"]


def assemble(p: SamplingParams, kept: pd.DataFrame, cand: pd.DataFrame, m: MatchResult,
             geoms: Mapping[str, SheetGeometry], attrs: pd.DataFrame) -> pd.DataFrame:
    """The samples table (PLAN.md §4.3 columns, plus `source_layer` and `source_objectid` for look-alikes).

    Pairs are numbered in (sheet, row, col, objectid) order of their positive; each positive is followed by its
    negative; the background follows, in (sheet, row, col) order. `id` is the row number.
    """
    matched = np.flatnonzero(m.neg >= 0)
    P = kept.iloc[matched].copy()
    P["_i"] = matched
    P = P.sort_values(["sheet", "row", "col", "objectid", "point"], kind="stable").reset_index(drop=True)
    P["pair_id"] = np.arange(len(P), dtype=np.int64)
    pos_t = pd.DataFrame({"label": 1, "row": P["row"], "col": P["col"], "sheet": P["sheet"], "source": "positive",
                          "polygon_objectid": P["objectid"].astype(np.int64), "source_layer": -1,
                          "source_objectid": -1, "slope_mean": P["slope_mean"], "slope_bin": P["slope_bin"],
                          "match_level": 0, "pair_id": P["pair_id"], "_k": 0})
    ni = m.neg[P["_i"].to_numpy()]
    N = cand.iloc[ni].reset_index(drop=True)
    neg_t = pd.DataFrame({"label": 0, "row": N["row"], "col": N["col"], "sheet": N["sheet"], "source": N["source"],
                          "polygon_objectid": -1, "source_layer": N["source_layer"].astype(np.int64),
                          "source_objectid": N["source_objectid"].astype(np.int64), "slope_mean": N["slope_mean"],
                          "slope_bin": N["slope_bin"], "match_level": m.level[P["_i"].to_numpy()],
                          "pair_id": P["pair_id"].to_numpy(), "_k": 1})
    B = cand.iloc[m.background].sort_values(["sheet", "row", "col"], kind="stable").reset_index(drop=True)
    bg_t = pd.DataFrame({"label": 0, "row": B["row"], "col": B["col"], "sheet": B["sheet"], "source": "background",
                         "polygon_objectid": -1, "source_layer": -1, "source_objectid": -1,
                         "slope_mean": B["slope_mean"], "slope_bin": B["slope_bin"], "match_level": 0, "pair_id": -1,
                         "_k": 2})
    S = pd.concat([pos_t, neg_t, bg_t], ignore_index=True)
    S["_pid"] = np.where(S["pair_id"] >= 0, S["pair_id"], np.iinfo(np.int64).max)
    S = S.sort_values(["_pid", "_k"], kind="stable").reset_index(drop=True)
    x = np.empty(len(S))
    y = np.empty(len(S))
    for s, g in S.groupby("sheet", sort=False):
        x[g.index], y[g.index] = geoms[s].grid.xy(g["row"].to_numpy(), g["col"].to_numpy())
    S["x"], S["y"] = x, y
    for k in SHEET_ATTRS:
        S[k] = S["sheet"].map(attrs[k]).to_numpy()
    S["id"] = np.arange(len(S), dtype=np.int64)
    out = S[COLUMNS].copy()
    for k in ("label", "row", "col", "polygon_objectid", "source_layer", "source_objectid", "fold", "slope_bin",
              "match_level", "pair_id"):
        out[k] = out[k].astype(np.int64)
    out["flysch"] = out["flysch"].astype(bool)
    for k in ("sheet", "kraj", "source", "group_b", "block", "stratum"):
        out[k] = out[k].astype(object)
    return out


# ======================================================================================================================
# Summary (aggregate counts only: no coordinates)
# ======================================================================================================================


def _vc(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(s.value_counts(dropna=False).items(), key=lambda kv: str(kv[0]))}


def _crosstab(df: pd.DataFrame, by: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, g in df.groupby(by, sort=True, dropna=False):
        out[str(key)] = {"positive": int((g["label"] == 1).sum()),
                         "negative": int(((g["label"] == 0) & (g["pair_id"] >= 0)).sum()),
                         "background": int((g["pair_id"] < 0).sum())}
    return out


def summarise(p: SamplingParams, poly: pd.DataFrame, pos: pd.DataFrame, kept: pd.DataFrame, cand: pd.DataFrame,
              m: MatchResult, fills: Mapping[str, PoolFill], counts: Counter, samples: pd.DataFrame,
              look: LookalikeSources, attrs: pd.DataFrame) -> dict[str, Any]:
    """Aggregate counts of every step (no coordinates, no per-sheet geometry)."""
    neg = samples[(samples["label"] == 0) & (samples["pair_id"] >= 0)]
    infeasible = poly["n_feasible"] == 0
    pos = pos.assign(pool=np.where(pos["sheet"].map(attrs["flysch"]).astype(bool), "flysch", "nonflysch"),
                     stratum=pos["sheet"].map(attrs["stratum"]))
    kept = kept.assign(matched=m.neg >= 0)
    pools = {}
    for name, f in fills.items():
        pp = pos[pos["pool"] == name]
        pools[name] = {"eligible_polygons": f.n_eligible, "budget": f.budget,
                       "sheets_with_feasible_polygons": f.n_sheets, "feasible_polygons": f.n_polygons_available,
                       "feasible_points": f.n_points_available, "cap": f.cap, "selected_points": f.total,
                       "total_at_cap_plus_1": f.totals_by_cap.get(f.cap + 1),
                       "selected_polygons": int(poly.loc[poly["taken"], "sheet"].map(attrs["flysch"]).astype(bool)
                                                .eq(name == "flysch").sum()),
                       "validity_dropped": int((~pp["valid"]).sum()), "kept": int(pp["valid"].sum()),
                       "totals_by_cap": {str(k): v for k, v in f.totals_by_cap.items()}}
    by_stratum = {}
    for s, g in pos.groupby("stratum", sort=True):
        kk = kept[kept["stratum"] == s]
        by_stratum[str(s)] = {"selected": int(len(g)), "validity_dropped": int((~g["valid"]).sum()),
                              "kept": int(len(kk)), "matched": int(kk["matched"].sum()),
                              "unmatched": int((~kk["matched"]).sum())}
    unmatched = int((m.neg < 0).sum())
    sel_poly = poly[poly["taken"]]
    return {
        "positives": {
            "eligible_stage1_polygons": int(len(poly)),
            "infeasible_polygons": int(infeasible.sum()),
            "feasible_polygons": int((~infeasible).sum()),
            "polygons_by_k_target": _vc(poly.loc[~infeasible, "k_target"]),
            "polygons_by_k": _vc(poly.loc[~infeasible, "k"]),
            "polygons_with_k_reduced_for_room": int(((poly["k"] < poly["k_target"]) & ~infeasible).sum()),
            "feasible_points": int(poly["k"].sum()),
            "pools": pools,
            "selected_polygons": int(len(sel_poly)),
            "selected_sheets": int(sel_poly["sheet"].nunique()),
            "selected_points": int(len(pos)),
            "validity_dropped": int((~pos["valid"]).sum()),
            "validity_dropped_whole_polygons": int((pos.groupby(["sheet", "objectid"])["valid"].sum() == 0).sum()),
            "kept": int(len(kept)),
            "matched": int(len(kept) - unmatched),
            "unmatched": unmatched,
            "by_stratum": by_stratum,
        },
        "candidates": {
            "random_steps": {k.split(":", 1)[1]: int(v) for k, v in sorted(counts.items()) if k.startswith("random:")},
            "lookalike_steps": {k: int(v) for k, v in sorted(counts.items()) if not k.startswith("random:")},
            "lookalike_sources": look.stats,
            "by_source": _vc(cand["source"]),
            "by_stratum": _vc(cand["stratum"]),
        },
        "matching": {
            "lookalike_quota": m.lookalike_quota,
            "lookalike_matched": int((neg["source"] != RANDOM).sum()),
            "lookalike_matched_by_type": _vc(neg.loc[neg["source"] != RANDOM, "source"]),
            "by_level": _vc(neg["match_level"]),
            "by_source": _vc(neg["source"]),
            "by_level_and_source": {str(k): _vc(g["source"]) for k, g in neg.groupby("match_level", sort=True)},
            "unmatched_positives": unmatched,
        },
        "background": {"n": int(len(m.background)), "target": p.background_n,
                       "available_random_candidates": m.background_available},
        "samples": {
            "n": int(len(samples)),
            "by_label": _vc(samples["label"]),
            "by_source": _vc(samples["source"]),
            "by_stratum": _crosstab(samples, "stratum"),
            "by_fold": _crosstab(samples, "fold"),
            "by_group_b": _crosstab(samples, "group_b"),
            "by_kraj": _crosstab(samples, "kraj"),
            "by_flysch": _crosstab(samples, "flysch"),
            "sheets_with_samples": int(samples["sheet"].nunique()),
        },
    }


# ======================================================================================================================
# Checks on the final table (asserted on the real build)
# ======================================================================================================================


def check_samples(cfg: Mapping[str, Any], samples: pd.DataFrame, inputs: Inputs,
                  recheck_dtm: bool = True) -> dict[str, Any]:
    """Re-check every rule on the final table, independently of the build's screens. Raises AssertionError on the
    first failure; returns the measured margins."""
    p = sampling_params(cfg)
    S = samples
    out: dict[str, Any] = {}
    sheets = inputs.sheets.set_index("sheet")
    pos = S[S["label"] == 1]
    neg = S[(S["label"] == 0) & (S["pair_id"] >= 0)]
    bg = S[S["pair_id"] < 0]
    nonpos = S[S["label"] == 0]

    # 1:1 pairs, strata and the bin rule per level
    assert (S["id"].to_numpy() == np.arange(len(S))).all(), "ids are not 0..N-1"
    assert pos["pair_id"].is_unique and neg["pair_id"].is_unique, "a pair_id is used twice"
    assert set(pos["pair_id"]) == set(neg["pair_id"]), "pairs without a positive or a negative"
    assert len(neg) <= p.max_neg, "more negatives than sampling.max_neg"
    pp = pos.set_index("pair_id")
    nn = neg.set_index("pair_id").loc[pp.index]
    assert (nn["stratum"].to_numpy() == pp["stratum"].to_numpy()).all(), "a negative outside its positive's stratum"
    assert (pos["match_level"] == 0).all() and (bg["match_level"] == 0).all(), "match_level set on a non-negative"
    lv = nn["match_level"].to_numpy()
    assert np.isin(lv, list(MATCH_LEVELS)).all(), "a negative without a match level"
    db = np.abs(nn["slope_bin"].to_numpy() - pp["slope_bin"].to_numpy())
    want = np.array([MATCH_LEVELS[int(v)][1] for v in lv])
    assert (db == want).all(), "the slope-bin rule fails for some match level"
    for kind, lvls in (("sheet", (1, 3)), ("block", (2, 4))):
        sel = np.isin(lv, lvls)
        assert (nn[kind].to_numpy()[sel] == pp[kind].to_numpy()[sel]).all(), f"level {lvls} outside the same {kind}"
    la = nn["source"].to_numpy() != RANDOM
    assert np.isin(lv[la], LOOKALIKE_LEVELS).all(), "a look-alike matched outside the exact bin"
    assert (S["slope_bin"].to_numpy() == slope_bins(S["slope_mean"].to_numpy(), p.slope_bin_deg)).all()
    out["n_pairs"] = int(len(pp))
    out["match_levels"] = {str(k): int(v) for k, v in sorted(Counter(lv.tolist()).items())}

    # spacing: negatives and background >= neg_min_spacing_m apart (exact, on the national lattice)
    gi, gj = _lattice(nonpos)
    if len(nonpos) > 1:
        tree = cKDTree(np.column_stack([gi, gj]).astype(np.float64))
        d, _ = tree.query(np.column_stack([gi, gj]).astype(np.float64), k=2)
        min_sp = float(d[:, 1].min()) * p.cell_m
        assert min_sp >= p.neg_min_spacing_m, f"two negatives {min_sp:.3f} m apart"
        out["min_negative_spacing_m"] = min_sp

    # centres >= 150 m from every register polygon and every ZABAGED landslide/scree polygon (exact)
    reg = Exclusion(repair(inputs.polygons["geometry"].to_numpy()), p.neg_min_dist_to_polygon_m, "register")
    ls_df = inputs.zabaged.get("landslide")
    ls = Exclusion(repair(np.asarray(ls_df["geometry"].to_numpy() if ls_df is not None else [], dtype=object)),
                   p.neg_exclude_zabaged_landslide_m, "zabaged_landslide")
    x, y = nonpos["x"].to_numpy(), nonpos["y"].to_numpy()
    d_reg = reg.min_distance(x, y)
    d_ls = ls.min_distance(x, y)
    assert (d_reg >= p.neg_min_dist_to_polygon_m).all(), f"a negative centre {d_reg.min():.3f} m from a polygon"
    assert (d_ls >= p.neg_exclude_zabaged_landslide_m).all(), f"a negative {d_ls.min():.3f} m from ZABAGED 129"
    out["min_negative_distance_to_register_m"] = float(d_reg.min()) if d_reg.size else None
    out["min_negative_distance_to_zabaged_landslide_m"] = float(d_ls.min()) if np.isfinite(d_ls).any() else None

    # the patch rule, exact: the square inside the own sheet, and > 150 m from every stage-2 sheet
    h = p.half_m
    sq = shapely.box(S["x"] - h, S["y"] - h, S["x"] + h, S["y"] + h)
    own = sheets.loc[S["sheet"], "geometry"].to_numpy()
    assert shapely.covered_by(sq, own).all(), "a patch square leaves its sheet"
    assert sheets.loc[S["sheet"].unique(), "is_stage1"].astype(bool).all(), "a sample outside the stage-1 sheets"
    s2 = sheets.loc[sheets["is_stage2"].astype(bool), "geometry"].to_numpy()
    if len(s2):
        t2 = shapely.STRtree(s2)
        ip, _ = t2.query(sq, predicate="dwithin", distance=p.stage2_buffer_m)
        assert ip.size == 0, "a patch square touches a stage-2 sheet buffered by 150 m"
        _, dd = t2.query_nearest(sq, return_distance=True, all_matches=False)
        out["min_square_distance_to_stage2_m"] = float(np.min(dd))
    # the corner convention: x, y are the corners (row, col) of the sheet rasters
    for s, g in S.groupby("sheet", sort=False):
        grid = grid_of_sheet(sheets.loc[s], p.cell_m)
        gx, gy = grid.xy(g["row"].to_numpy(), g["col"].to_numpy())
        assert (gx == g["x"].to_numpy()).all() and (gy == g["y"].to_numpy()).all(), f"{s}: x, y are not the corners"

    # positives: inside their own polygon, in the polygon's sheet, >= 100 m apart within a polygon
    polys = inputs.polygons.set_index("objectid")
    pg = polys.loc[pos["polygon_objectid"]]
    assert (pg["sheet"].to_numpy() == pos["sheet"].to_numpy()).all(), "a positive outside its polygon's sheet"
    assert pg["eligible"].astype(bool).all(), "a positive from an ineligible polygon"
    inside = shapely.contains_xy(repair(pg["geometry"].to_numpy()), pos["x"].to_numpy(), pos["y"].to_numpy())
    assert inside.all(), "a positive outside its polygon"
    gi, gj = _lattice(pos)
    pos_ = pos.assign(_gi=gi, _gj=gj)
    min_pp = math.inf
    for _, g in pos_.groupby("polygon_objectid"):
        if len(g) > 1:
            a = np.column_stack([g["_gi"], g["_gj"]]).astype(np.float64)
            dd = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
            min_pp = min(min_pp, float(dd[np.triu_indices(len(g), 1)].min()) * p.cell_m)
    assert min_pp >= p.min_pt_spacing_m, f"two points of one polygon {min_pp:.3f} m apart"
    out["min_point_spacing_within_polygon_m"] = None if math.isinf(min_pp) else min_pp

    # validity and slope, recomputed from the raster windows (independent of the SAT screens)
    if recheck_dtm:
        out.update(recheck_windows(cfg, S, inputs))
    return out


def _lattice(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fi = (df["x"].to_numpy() - GRID_ORIGIN[0]) / 2.0
    fj = (df["y"].to_numpy() - GRID_ORIGIN[1]) / 2.0
    gi, gj = np.rint(fi), np.rint(fj)
    assert np.abs(fi - gi).max(initial=0) < 1e-6 and np.abs(fj - gj).max(initial=0) < 1e-6, "off-lattice samples"
    return gi.astype(np.int64), gj.astype(np.int64)


def recheck_windows(cfg: Mapping[str, Any], S: pd.DataFrame, inputs: Inputs) -> dict[str, Any]:
    """For every sample, read the patch plus context from the sheet raster, assert it is fully valid, recompute the
    slope with the per-patch code path (`terrain_channels` on the window, cropped) and compare the mean with
    `slope_mean`. Bin flips at a bin edge are counted, not failed (they need |difference| < 1e-9 deg anyway)."""
    p = sampling_params(cfg)
    m = p.context_px
    hh = p.context_half_px
    max_diff = 0.0
    flips = 0
    for s, g in S.groupby("sheet", sort=True):
        z = inputs.read_dtm(s).z
        for idx, r, c in zip(g.index, g["row"].to_numpy(), g["col"].to_numpy()):
            w = z[r - hh:r + hh, c - hh:c + hh]
            assert w.shape == (2 * hh, 2 * hh) and np.isfinite(w).all(), f"{s} ({r}, {c}): invalid patch context"
            sl = terrain_channels(w, cfg, names=["slope"], margin=m)[0]
            mean = float(sl.mean())
            max_diff = max(max_diff, abs(mean - float(S.at[idx, "slope_mean"])))
            flips += int(math.floor(mean / p.slope_bin_deg) != int(S.at[idx, "slope_bin"]))
        del z
    assert max_diff < 1e-9, f"slope mean differs from the per-patch computation by {max_diff:.3g} deg"
    return {"max_slope_mean_diff_deg": max_diff, "slope_bin_flips_vs_per_patch": flips,
            "windows_rechecked": int(len(S))}


# ======================================================================================================================
# ZABAGED loading (P1-07's cache format)
# ======================================================================================================================


def load_zabaged(cfg: Mapping[str, Any], paths: Any) -> dict[str, pd.DataFrame]:
    """The six ZABAGED layers this task uses, by role (`ZABAGED_LAYERS_DEFAULT`), each a GeoDataFrame with `objectid`
    and `geometry` in EPSG:5514.

    PROVISIONAL until P1-07 is merged: reads `paths.zabaged("<layer id>")` (`data/cache/zabaged/<id>.gpkg`) with
    geopandas. This is the only function that knows the cache format.
    """
    import geopandas as gpd

    p = sampling_params(cfg)
    out: dict[str, pd.DataFrame] = {}
    for role, layer in p.zabaged_layers.items():
        path = paths.zabaged(str(layer))
        if not path.exists():
            raise FileNotFoundError(f"ZABAGED layer {layer} ({role}) not found at {path}")
        df = gpd.read_file(path)
        if df.crs is None or df.crs.to_epsg() != 5514:
            raise ValueError(f"{path}: expected EPSG:5514, got {df.crs}")
        oid = next((c for c in df.columns if c.lower() == "objectid"), None)
        if oid is None:
            raise ValueError(f"{path}: no objectid column")
        out[role] = gpd.GeoDataFrame({"objectid": df[oid].astype("int64").to_numpy()},
                                     geometry=df.geometry.to_numpy(), crs=df.crs)
    return out
