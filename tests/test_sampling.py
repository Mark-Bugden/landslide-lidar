"""Unit tests for `landslide.sampling` (P1-09), on synthetic geometry and arrays only.

Every exact-geometry routine is checked against shapely/GEOS brute force: the scanline corner sets against
`contains_xy`, the square erosion against `covered_by` of the actual squares, the stage-2 rule against
`dwithin` of the actual squares. Matching is checked on hand-built cases with known answers.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest
import shapely
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely import affinity

from landslide import sampling as S
from landslide.config import load_config

CFG = load_config()
P = S.sampling_params(CFG)
# a corner lattice on the DMR 5G lattice (x = 0.4 + 2i, y = -0.12 + 2j)
X0, Y0 = 0.4 + 2 * -350_000, -0.12 + 2 * -550_000


def grid(n_rows: int = 301, n_cols: int = 351) -> S.CornerGrid:
    return S.CornerGrid.from_transform(Affine(2.0, 0.0, X0, 0.0, -2.0, Y0), n_rows - 1, n_cols - 1)


def all_corners(g: S.CornerGrid) -> tuple[np.ndarray, np.ndarray]:
    r, c = np.meshgrid(np.arange(g.n_rows), np.arange(g.n_cols), indexing="ij")
    return r.ravel(), c.ravel()


def mask_of(iv: S.Intervals, g: S.CornerGrid) -> np.ndarray:
    m = np.zeros((g.n_rows, g.n_cols), dtype=bool)
    r, c = S.expand_intervals(iv)
    m[r, c] = True
    return m


def rot_rect(cx: float, cy: float, w: float, h: float, deg: float) -> Polygon:
    return affinity.rotate(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), deg, origin=(cx, cy))


G = grid()
GX0, GY0 = X0, Y0                      # top-left corner of the test grid
CX, CY = GX0 + 350.0, GY0 - 300.0      # its centre (grid is 700 m x 600 m)


# ======================================================================================================================
# Parameters, streams, lattice
# ======================================================================================================================


def test_params_from_config():
    assert (P.half_px, P.half_m, P.context_px, P.context_half_px) == (75, 150.0, 28, 103)
    assert P.grid_step_px == 30 and P.neg_jitter_m == 15.0 and P.stage2_buffer_m == 150.0
    assert P.lookalike_types == ("quarry", "spoil", "rock", "cut")
    assert P.zabaged_layers == {"quarry": 118, "spoil": 119, "rock": 130, "landslide": 129, "road": 79, "railway": 75}
    bad = copy.deepcopy(CFG)
    bad["patch"]["size_m"] = 302
    with pytest.raises(ValueError):
        S.sampling_params(bad)


def test_sheet_rng_is_per_sheet_and_reproducible():
    a = S.sheet_rng(CFG, 11, "negative_jitter", "14-34-07").random(4)
    b = S.sheet_rng(CFG, 11, "negative_jitter", "14-34-07").random(4)
    c = S.sheet_rng(CFG, 11, "negative_jitter", "14-34-08").random(4)
    d = S.sheet_rng(CFG, 11, "positive_points", "14-34-07").random(4)
    assert (a == b).all() and not np.allclose(a, c) and not np.allclose(a, d)
    assert S.sheet_code("02-41-01") == 24101
    with pytest.raises(KeyError):
        S.sheet_rng(CFG, 11, "no_such_stream", "14-34-07")


def test_round_half_up_and_bins():
    assert [S.round_half_up(v) for v in (2346.5, 2346.49, 430.08, 0.5)] == [2347, 2346, 430, 1]
    assert S.slope_bins(np.array([0.0, 1.999999, 2.0, 3.9, 21.0]), 2.0).tolist() == [0, 0, 1, 1, 10]


def test_corner_grid_lattice_and_snap():
    gi, gj = G.global_ij([0, 10], [0, 5])
    x, y = G.xy([0, 10], [0, 5])
    assert np.allclose(x, 0.4 + 2 * gi, atol=1e-6) and np.allclose(y, -0.12 + 2 * gj, atol=1e-6)
    r, c = G.snap(x + 0.9, y - 0.9)
    assert r.tolist() == [0, 10] and c.tolist() == [0, 5]
    off = S.CornerGrid.from_transform(Affine(2.0, 0.0, X0 + 1.0, 0.0, -2.0, Y0), 10, 10)
    with pytest.raises(ValueError):
        off.lattice_offset()
    # a raster transform that differs from the stored (rounded) corner in the last bits is the same grid
    a = S.CornerGrid.from_transform(Affine(2.0, 0.0, X0, 0.0, -2.0, Y0), 300, 350)
    b = S.CornerGrid.from_transform(Affine(2.0, 0.0, X0 + 3e-10, 0.0, -2.0, Y0 - 2e-10), 300, 350)
    c = S.CornerGrid.from_transform(Affine(2.0, 0.0, X0 + 2.0, 0.0, -2.0, Y0), 300, 350)
    assert a != b and a.same_lattice(b) and not a.same_lattice(c)
    assert not a.same_lattice(S.CornerGrid.from_transform(Affine(2.0, 0.0, X0, 0.0, -2.0, Y0), 300, 351))


# ======================================================================================================================
# Scanline corners, intervals
# ======================================================================================================================


SHAPES = {
    "rotated_rect": rot_rect(CX, CY, 420.3, 250.7, 17.0),
    "star": Polygon([(CX + (220 if k % 2 == 0 else 90) * math.cos(k * math.pi / 7),
                      CY + (220 if k % 2 == 0 else 90) * math.sin(k * math.pi / 7)) for k in range(14)]),
    "with_hole": rot_rect(CX, CY, 500.1, 400.3, 8).difference(rot_rect(CX + 20, CY, 150.2, 120.9, 33)),
    "multi": MultiPolygon([rot_rect(CX - 200, CY, 120.5, 90.3, 5), rot_rect(CX + 180, CY + 60, 200.7, 60.1, -40)]),
    "partly_outside": rot_rect(GX0 + 30, GY0 - 40, 300.3, 200.9, 25),
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_row_intervals_equal_geos_contains(name):
    geom = SHAPES[name]
    r, c = all_corners(G)
    x, y = G.xy(r, c)
    ref = shapely.contains_xy(geom, x, y).reshape(G.n_rows, G.n_cols)
    for max_cells in (4_000_000, 50):    # one chunk, many chunks
        got = mask_of(S.row_intervals(geom, G, max_cells=max_cells), G)
        assert (got == ref).all(), f"{name}: {int((got != ref).sum())} corners differ"
    assert ref.sum() > 100


def test_interval_ops():
    a = S.row_intervals(SHAPES["star"], G)
    b = S.row_intervals(SHAPES["rotated_rect"], G)
    both = mask_of(S.intersect_intervals(a, b), G)
    assert (both == (mask_of(a, G) & mask_of(b, G))).all()
    assert S.intersect_intervals(a, b).count() == int(both.sum())
    r, c = all_corners(G)
    assert (S.in_intervals(a, r, c).reshape(G.n_rows, G.n_cols) == mask_of(a, G)).all()
    assert S.intersect_intervals(a, S.Intervals.empty()).n == 0
    assert S.row_intervals(box(GX0 - 900, GY0 + 100, GX0 - 800, GY0 + 200), G).n == 0   # off the grid


# ======================================================================================================================
# The patch rule: square erosion and the stage-2 zone, against the actual squares
# ======================================================================================================================


@pytest.mark.parametrize("shape", ["convex", "l_shape"])
def test_erode_square_equals_square_inside(shape):
    h = 37.3   # a small square so the test grid holds interesting cases
    if shape == "convex":
        P_ = rot_rect(CX, CY, 500.3, 350.9, 6.0)
    else:
        P_ = Polygon([(CX - 250.37, CY - 200.37), (CX + 240.37, CY - 200.37), (CX + 240.37, CY - 20.37),
                      (CX - 30.37, CY - 20.37), (CX - 30.37, CY + 230.37), (CX - 250.37, CY + 230.37)])
    E = S.erode_square(P_, h)
    r, c = all_corners(G)
    x, y = G.xy(r, c)
    ref = shapely.covered_by(shapely.box(x - h, y - h, x + h, y + h), P_)
    got = S.in_intervals(S.row_intervals(E, G), r, c)
    assert (got == ref).all(), int((got != ref).sum())
    assert ref.sum() > 1000
    if shape == "convex":   # = the intersection of the four translates
        tr = [affinity.translate(P_, -dx, -dy) for dx in (-h, h) for dy in (-h, h)]
        assert math.isclose(shapely.intersection_all(tr).area, E.area, rel_tol=1e-9)


def test_stage2_zone_equals_square_distance():
    h, d = 40.0, 30.0
    Q = rot_rect(CX + 260, CY, 200.3, 500.7, 7.0)
    QS = S.dilate_square(Q, h)
    rng = np.random.default_rng(0)
    x = rng.uniform(CX - 100, CX + 400, 4000)
    y = rng.uniform(CY - 300, CY + 300, 4000)
    ref = shapely.dwithin(shapely.box(x - h, y - h, x + h, y + h), Q, d)
    got = shapely.dwithin(shapely.points(x, y), QS, d)
    assert (got == ref).all() and 200 < ref.sum() < 3800


def test_sheet_geometry_patch_ok_brute_force():
    p = S.sampling_params(CFG)
    g = grid(500, 600)                                   # 1.2 km x 1.0 km
    sheet = rot_rect(GX0 + 600, GY0 - 500, 900.3, 760.1, 5.0)
    s2 = rot_rect(GX0 + 600 + 900, GY0 - 500 + 60, 900.0, 760.0, 5.0)   # a stage-2 neighbour to the east
    sg = S.sheet_geometry("90-01-01", sheet, g, S.stage2_dilated([s2], p.half_m), p)
    assert sg.zone is not None
    r, c = all_corners(g)
    x, y = g.xy(r, c)
    sq = shapely.box(x - 150, y - 150, x + 150, y + 150)
    ref = shapely.covered_by(sq, sheet) & ~shapely.dwithin(sq, s2, 150)
    got = sg.patch_ok(r, c)
    assert (got == ref).all(), int((got != ref).sum())
    assert ref.sum() > 1000 and (shapely.covered_by(sq, sheet) & ~ref).sum() > 1000   # the stage-2 rule bites
    # feasible_in = corners strictly inside a polygon AND patch-feasible
    poly = rot_rect(GX0 + 700, GY0 - 450, 300.0, 120.0, 30.0)
    rr, cc = sg.feasible_in(poly)
    ref_in = ref & shapely.contains_xy(poly, x, y)
    assert set(zip(rr.tolist(), cc.tolist())) == set(zip(r[ref_in].tolist(), c[ref_in].tolist()))


def test_exclusion_is_exact_at_the_threshold():
    ex = S.Exclusion([box(0.0, 0.0, 10.0, 10.0), Point(1000.0, 0.0)], 150.0)
    far = ex.far(np.array([160.0, 159.999, 5.0, 1150.0, 1149.0, 500.0]), np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0]))
    assert far.tolist() == [True, False, False, True, False, True]
    d = ex.min_distance(np.array([160.0, 500.0]), np.array([5.0, 0.0]))
    assert np.allclose(d, [150.0, 490.0])
    assert S.Exclusion([], 150.0).far(np.array([0.0]), np.array([0.0])).all()


# ======================================================================================================================
# Positives
# ======================================================================================================================


def test_points_per_polygon():
    got = [S.points_per_polygon(a, P) for a in (3000, 39_999, 79_999.9, 80_000, 119_999, 120_000, 8.6e6)]
    assert got == [1, 1, 1, 2, 2, 3, 3]


def test_greedy_spaced():
    rng = np.random.default_rng(1)
    r, c = np.meshgrid(np.arange(0, 200), np.arange(0, 40), indexing="ij")
    r, c = r.ravel(), c.ravel()
    pick = S.greedy_spaced(r, c, 3, 100.0, 2.0, rng)
    assert pick.size == 3
    pr, pc = r[pick], c[pick]
    d2 = (pr[:, None] - pr[None]) ** 2 + (pc[:, None] - pc[None]) ** 2
    assert (d2[np.triu_indices(3, 1)] * 4.0 >= 100.0 ** 2).all()
    # no room for a second point: every corner within 40 m of every other
    r2, c2 = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    assert S.greedy_spaced(r2.ravel(), c2.ravel(), 3, 100.0, 2.0, rng).size == 1
    assert S.greedy_spaced(np.zeros(0, int), np.zeros(0, int), 3, 100.0, 2.0, rng).size == 0
    # exactly 100 m apart is allowed
    pk = S.greedy_spaced(np.array([0, 50]), np.array([0, 0]), 2, 100.0, 2.0, np.random.default_rng(0))
    assert pk.size == 2


def test_fill_sheet_takes_polygons_whole_and_skips():
    assert S.fill_sheet([3, 1, 2], 3).tolist() == [True, False, False]
    assert S.fill_sheet([3, 1, 2], 4).tolist() == [True, True, False]
    assert S.fill_sheet([3, 1, 2], 2).tolist() == [False, True, False]
    assert S.fill_sheet([3, 1, 2], 6).tolist() == [True, True, True]


def test_waterfill_budget_split_and_largest_cap():
    cfg = copy.deepcopy(CFG)
    cfg["sampling"]["max_pos"] = 20
    p = S.sampling_params(cfg)
    rng = np.random.default_rng(5)
    rows = []
    for s in ("10-00-01", "10-00-02", "10-00-03", "20-00-01", "20-00-02"):
        for oid in range(1, 9):
            rows.append({"sheet": s, "objectid": oid + 100 * len(rows), "k": int(rng.integers(0, 4))})
    poly = pd.DataFrame(rows)
    pool = {s: ("flysch" if s.startswith("10") else "nonflysch") for s in poly["sheet"].unique()}
    n_elig = {"flysch": 30, "nonflysch": 10}      # share 0.75 -> budget 15 / 5
    taken, fills = S.waterfill(cfg, p, poly, pool, n_elig)
    assert fills["flysch"].budget == 15 and fills["nonflysch"].budget == 5
    for name, f in fills.items():
        assert f.total <= f.budget
        assert all(t > f.budget for cap, t in f.totals_by_cap.items() if cap > f.cap)
        sel = poly[taken & (poly["sheet"].map(pool) == name)]
        assert int(sel["k"].sum()) == f.total
        assert (sel.groupby("sheet")["k"].sum() <= f.cap).all()
    assert not taken[poly["k"].to_numpy() == 0].any()
    # the order is the sheet's own permutation: reproducible
    taken2, _ = S.waterfill(cfg, p, poly, pool, n_elig)
    assert (taken == taken2).all()


# ======================================================================================================================
# Negatives: the grid, cut points, the raster tables
# ======================================================================================================================


def test_random_grid_is_jittered_snapped_and_anchored():
    g = grid(400, 500)
    rr, cc = S.random_grid(g, P, np.random.default_rng(3))
    n_r, n_c = math.ceil(400 / 30), math.ceil(500 / 30)
    assert rr.size == n_r * n_c
    base_r = np.repeat(np.arange(n_r) * 30, n_c)
    base_c = np.tile(np.arange(n_c) * 30, n_r)
    dr, dc = rr - base_r, cc - base_c
    assert dr.min() >= -7 and dr.max() <= 7 and dc.min() >= -7 and dc.max() <= 7
    assert len(set(dr.tolist())) == 15 and len(set(dc.tolist())) == 15
    r2, c2 = S.random_grid(g, P, np.random.default_rng(3))
    assert (rr == r2).all() and (cc == c2).all()


def test_cut_points_every_step_along_each_part():
    lines = pd.DataFrame({"layer": [79, 75, 79], "objectid": [1, 2, 3],
                          "geom": [LineString([(0, 0), (250, 0)]),
                                   shapely.MultiLineString([[(0, 10), (0, 110)], [(5, 5), (5, 6)]]),
                                   LineString([(0, 0), (0, 0)])]})
    pts = S.cut_points(lines, 100.0)
    got =[(int(o), round(x, 6), round(y, 6)) for o, x, y in pts[["objectid", "x", "y"]].itertuples(index=False)]
    assert got == [(1, 0.0, 0.0), (1, 100.0, 0.0), (1, 200.0, 0.0), (2, 0.0, 10.0), (2, 5.0, 5.0)]
    assert pts["layer"].tolist() == [79, 79, 79, 75, 75]


def test_sheet_raster_tables_brute_force():
    rng = np.random.default_rng(4)
    z = 300 + np.cumsum(rng.normal(size=(260, 280)), axis=0) + np.cumsum(rng.normal(size=(260, 280)), axis=1)
    z[200:, 250:] = np.nan
    z[40, 60] = np.nan
    sr = S.sheet_raster(z, CFG)
    r = np.array([103, 120, 150, 157, 160, 102, 140])
    c = np.array([103, 140, 140, 176, 177, 120, 103])
    for h in (103, 75):
        got = sr.context_valid(r, c, h)
        for k in range(r.size):
            r0, c0 = r[k] - h, c[k] - h
            ok = r0 >= 0 and c0 >= 0 and r[k] + h <= z.shape[0] and c[k] + h <= z.shape[1]
            want = ok and bool(np.isfinite(z[r0:r[k] + h, c0:c[k] + h]).all())
            assert got[k] == want, (h, k)
    v = sr.context_valid(r, c, 103)
    m = sr.slope_mean(r[v], c[v], 75)
    for k, (rk, ck) in enumerate(zip(r[v], c[v])):
        assert abs(m[k] - sr.slope[rk - 75:rk + 75, ck - 75:ck + 75].mean()) < 1e-10
    i, j = np.array([20, 100, 255]), np.array([20, 100, 100])
    p90 = sr.slope_p90(i, j, 21)
    assert abs(p90[1] - np.percentile(sr.slope[90:111, 90:111], 90)) < 1e-12
    assert np.isnan(p90[0]) or abs(p90[0] - np.percentile(sr.slope[10:31, 10:31], 90)) < 1e-12
    assert np.isnan(p90[2])                                     # the window leaves the raster


# ======================================================================================================================
# Matching (hand-built)
# ======================================================================================================================


def _pos(rows):
    return pd.DataFrame(rows, columns=["sheet", "block", "stratum", "slope_bin"])


def _cand(rows):
    return pd.DataFrame(rows, columns=["sheet", "block", "stratum", "slope_bin", "cls", "gi", "gj"])


def _cfg(frac=0.12, bg=0):
    cfg = copy.deepcopy(CFG)
    cfg["sampling"]["lookalike_frac"] = frac
    cfg["stage2"]["background_n"] = bg
    return cfg, S.sampling_params(cfg)


def test_match_levels_scope_and_bins():
    cfg, p = _cfg(frac=0.0)
    pos = _pos([("A", "b1", "F", 10),    # level 1 candidate exists
                ("A", "b1", "F", 12),    # only adjacent bin in the same sheet -> level 3
                ("A", "b1", "F", 14),    # exact bin only in another sheet of the block -> level 2
                ("A", "b1", "F", 16),    # exact bin only in another block of the stratum -> level 5
                ("A", "b1", "F", 18),    # adjacent bin in another block -> level 6
                ("A", "b1", "F", 24)])   # only candidates in another stratum -> unmatched
    cand = _cand([("A", "b1", "F", 10, "random", 0, 0),
                  ("A", "b1", "F", 13, "random", 1000, 0),
                  ("B", "b1", "F", 14, "random", 2000, 0),
                  ("C", "b2", "F", 16, "random", 3000, 0),
                  ("C", "b2", "F", 19, "random", 4000, 0),
                  ("D", "b3", "N", 24, "random", 5000, 0)])
    m = S.match(cfg, p, pos, cand)
    assert m.neg.tolist() == [0, 1, 2, 3, 4, -1]
    assert m.level.tolist() == [1, 3, 2, 5, 6, 0]


def test_match_spacing_kills_close_candidates():
    cfg, p = _cfg(frac=0.0)
    pos = _pos([("A", "b1", "F", 10), ("A", "b1", "F", 10)])
    # two same-bin candidates 148 m apart (74 lattice units), one far away in another sheet of the block
    cand = _cand([("A", "b1", "F", 10, "random", 0, 0), ("A", "b1", "F", 10, "random", 74, 0),
                  ("B", "b1", "F", 10, "random", 5000, 0)])
    m = S.match(cfg, p, pos, cand)
    assert sorted(m.level.tolist()) == [1, 2]
    assert 2 in m.neg.tolist()
    # exactly 150 m (75 units) apart is allowed
    cand2 = _cand([("A", "b1", "F", 10, "random", 0, 0), ("A", "b1", "F", 10, "random", 75, 0)])
    m2 = S.match(cfg, p, pos, cand2)
    assert sorted(m2.neg.tolist()) == [0, 1] and m2.level.tolist() == [1, 1]


def test_match_lookalikes_round_robin_exact_bin_first():
    cfg, p = _cfg(frac=0.5)                 # 8 positives -> quota 4
    pos = _pos([("A", "b1", "F", 10)] * 8)
    rows = []
    for k in range(6):
        rows.append(("A", "b1", "F", 10, "quarry", 200 * k, 0))
        rows.append(("A", "b1", "F", 10, "cut", 200 * k, 5000))
        rows.append(("A", "b1", "F", 10, "random", 200 * k, 10_000))
    rows.append(("A", "b1", "F", 10, "rock", 0, 20_000))
    rows.append(("A", "b1", "F", 11, "spoil", 0, 30_000))   # adjacent bin: never used for a look-alike
    cand = _cand(rows)
    m = S.match(cfg, p, pos, cand)
    assert m.lookalike_quota == 4
    cls = cand["cls"].to_numpy()[m.neg]
    # turns: quarry, spoil (none in the exact bin -> leaves), rock, cut, quarry -> quota reached
    assert sorted(cls[np.isin(cls, ["quarry", "spoil", "rock", "cut"])].tolist()) == ["cut", "quarry", "quarry", "rock"]
    assert (m.level == 1).all()
    assert (cls == "random").sum() == 4


def test_match_lookalikes_never_at_adjacent_levels():
    cfg, p = _cfg(frac=1.0)
    pos = _pos([("A", "b1", "F", 10), ("A", "b1", "F", 10)])
    cand = _cand([("A", "b1", "F", 11, "quarry", 0, 0), ("C", "b2", "F", 10, "rock", 900, 0),
                  ("A", "b1", "F", 11, "random", 3000, 0)])
    m = S.match(cfg, p, pos, cand)
    cls = cand["cls"].to_numpy()
    got = sorted((int(lv), str(cls[n])) for n, lv in zip(m.neg, m.level) if n >= 0)
    assert got == [(3, "random"), (5, "rock")]


def test_background_spacing_and_count():
    cfg, p = _cfg(frac=0.0, bg=5)
    pos = _pos([("A", "b1", "F", 10)])
    cand = _cand([("A", "b1", "F", 10, "random", 0, 0)]
                 + [("A", "b1", "F", 3, "random", 40 * k, 1000) for k in range(20)]
                 + [("A", "b1", "F", 3, "quarry", 40 * k, 3000) for k in range(20)])
    m = S.match(cfg, p, pos, cand)
    assert m.neg.tolist() == [0]
    assert m.background.size == 5
    g = cand.iloc[m.background]
    assert (g["cls"] == "random").all()
    xy = g[["gi", "gj"]].to_numpy().astype(float)
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))[np.triu_indices(5, 1)]
    assert (d * 2.0 >= 150.0).all()
    assert 0 not in m.background.tolist()
