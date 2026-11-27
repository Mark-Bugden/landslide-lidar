"""landslide.stage2 (P1-12): lattice, window statistics, NMS, object matching, metrics, outputs, LR comparator and the
CNN scorer. Synthetic data only, except the last test (needs_data), which reads one real DTM and never runs a CNN.

The CNN tests use randomly initialised models on toy rasters (gate G3: no CNN sees real data before the freeze).
"""

from __future__ import annotations

import copy
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
import torch
from rasterio.transform import Affine
from shapely.geometry import Polygon, box

from landslide import stage2 as S2
from landslide.channels import full_channels, patch_channels
from landslide.config import config_hash, load_config
from landslide.metrics import NOT_REACHED, REACHED
from landslide.model import build_model
from landslide.paths import Paths
from landslide.predict import Normaliser, tta_predict
from landslide.sampling import CornerGrid, grid_of_sheet
from landslide.sheets import grid_origin
from landslide.sheets import dtm_grid
from terrain_synth import synthetic_dtm

CFG = load_config()
P = S2.stage2_params(CFG)


# ======================================================================================================================
# helpers
# ======================================================================================================================


def toy_cfg(px: int = 64, stride: int = 16) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["patch"]["px"] = px
    cfg["stage2"]["stride_px"] = stride
    return cfg


def rotated_square(cx: float, cy: float, side: float, deg: float = 3.0) -> Polygon:
    t = math.radians(deg)
    h = side / 2.0
    pts = [(-h, -h), (h, -h), (h, h), (-h, h)]
    return Polygon([(cx + u * math.cos(t) - v * math.sin(t), cy + u * math.sin(t) + v * math.cos(t)) for u, v in pts])


def sheet_row(sheet: str, geom: Polygon, cfg: dict, flysch: bool) -> dict:
    g = dtm_grid(geom.bounds, cfg)
    return {"sheet": sheet, "flysch": flysch, "grid_xmin": g.xmin, "grid_ymax": g.ymax, "grid_w": g.w,
            "grid_h": g.h, "geometry": geom}


def write_tif(path, arr: np.ndarray, row: dict, cell: float = 2.0) -> None:
    t = Affine(cell, 0.0, row["grid_xmin"], 0.0, -cell, row["grid_ymax"])
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype=arr.dtype.name, crs="EPSG:5514", transform=t) as dst:
        dst.write(arr, 1)


def lattice_frame(rows, cols, scores, x0=-600_000.4, y0=-1_150_000.12, cell=2.0) -> pd.DataFrame:
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    return pd.DataFrame({"row": r, "col": c, "x": x0 + cell * c, "y": y0 - cell * r,
                         "score": np.asarray(scores, dtype=np.float64)})


def brute_nms(row, col, s, r2):
    order = np.lexsort((col, row, -s))
    alive = np.ones(len(s), dtype=bool)
    sel = np.zeros(len(s), dtype=bool)
    for i in order:
        if not alive[i]:
            continue
        sel[i] = True
        for j in range(len(s)):
            if alive[j] and (row[j] - row[i]) ** 2 + (col[j] - col[i]) ** 2 <= r2:
                alive[j] = False
    return sel


# ======================================================================================================================
# parameters
# ======================================================================================================================


def test_params_from_config():
    assert (P.px, P.half, P.context_px, P.valid_half, P.stride_px) == (150, 75, 28, 103, 25)
    assert P.nms_radius_px2 == 75.0 ** 2
    assert (P.match_buffer_m, P.ignore_buffer_m, P.recall_level, P.precision_min) == (50.0, 150.0, 0.5, 0.3)
    assert P.jitter_scale == 1e-9 and P.model_seeds == (0, 1, 2) and P.batch_size == 64


def test_cnn_cost_estimate():
    e = S2.cnn_cost_estimate(146_622)
    assert e["images"] == 146_622 * 8 * 3
    assert e["seconds"] == pytest.approx(e["images"] / 4600.0)


# ======================================================================================================================
# lattice (D22) and window statistics
# ======================================================================================================================


def test_lattice_corners_mod_stride_and_inside():
    grid = CornerGrid(x0=-600_000.4, y0=-1_150_000.12, cell=2.0, n_rows=401, n_cols=451, origin=grid_origin(CFG))
    geom = rotated_square(-600_000.4 + 450, -1_150_000.12 - 400, 400.0, deg=7.0)
    r, c = S2.lattice_corners(geom, grid, 25)
    assert r.size and np.all(r % 25 == 0) and np.all(c % 25 == 0)
    rr, cc = np.meshgrid(np.arange(0, 401, 25), np.arange(0, 451, 25), indexing="ij")
    x, y = grid.xy(rr.ravel(), cc.ravel())
    want = [geom.intersects(shapely.Point(a, b)) for a, b in zip(x, y)]
    assert r.size == sum(want)
    np.testing.assert_array_equal(np.column_stack([r, c]), np.column_stack([rr.ravel(), cc.ravel()])[want])
    assert np.all(np.diff(r * 10_000 + c) > 0)  # row-major


def test_windows_valid_drops_invalid_and_outside():
    valid = np.ones((300, 300), dtype=bool)
    valid[250, 20] = False
    r = np.array([103, 103, 150, 150, 102, 197, 150, 100])
    c = np.array([103, 197, 150, 124, 150, 103, 123, 100])
    got = S2.windows_valid(valid, r, c, 103)
    # (103,103) fits exactly (rows 0..205); (103,197) fits (197 + 103 = 300); (150,150) and (150,124) reach row 250
    # but not col 20; (102,150) leaves the array; (197,103) and (150,123) (cols 20..225) touch the bad cell;
    # (100,100) leaves the array.
    assert got.tolist() == [True, True, True, True, False, False, False, False]


def test_window_stats_equal_brute_force():
    rng = np.random.default_rng(3)
    slope = rng.uniform(0, 45, size=(260, 270))
    susc = rng.integers(0, 3, size=(260, 270)).astype(np.uint8)
    r = np.array([75, 100, 185, 130])
    c = np.array([75, 120, 195, 75])
    st = S2.window_stats(slope, susc, r, c, 75)
    for k in range(r.size):
        w = slope[r[k] - 75:r[k] + 75, c[k] - 75:c[k] + 75]
        s = susc[r[k] - 75:r[k] + 75, c[k] - 75:c[k] + 75].astype(np.int64)
        assert st["slope_mean"][k] == pytest.approx(w.mean(), abs=1e-10)
        assert st["slope_std"][k] == pytest.approx(w.std(), abs=1e-9)
        assert st["susc_mean"][k] == s.mean() and st["susc_high_frac"][k] == (s == 2).mean()
    with pytest.raises(ValueError, match="codes"):
        S2.window_stats(slope, np.full((260, 270), 3, dtype=np.uint8), r, c, 75)
    bad = slope.copy()
    bad[100, 120] = np.nan
    with pytest.raises(ValueError, match="invalid slope"):
        S2.window_stats(bad, susc, r, c, 75)


def test_build_lattice_counts_and_coordinates():
    cfg = CFG
    geom = rotated_square(-600_000.4 + 700, -1_150_000.12 - 650, 700.0, deg=4.0)
    row = sheet_row("90-00-01", geom, cfg, False)
    grid = grid_of_sheet(pd.Series(row), 2.0, grid_origin(cfg))
    H, W = grid.n_rows - 1, grid.n_cols - 1
    z = synthetic_dtm((H, W), seed=1, border=False, holes=False, lake=False)
    slope = full_channels(z, cfg, names=["slope"])[0]
    valid = np.isfinite(z)
    lat = S2.build_lattice("90-00-01", geom, grid, valid, slope, None, P)
    assert lat.n_invalid == 0 and lat.n == lat.n_inside > 0
    assert min(lat.row.min(), lat.col.min(), H - lat.row.max(), W - lat.col.max()) >= 125  # the 250 m margin
    x, y = grid.xy(lat.row, lat.col)
    np.testing.assert_array_equal(lat.x, x)
    assert set(lat.stats) == {"slope_mean", "slope_std"}
    # A hole inside one window's context drops exactly the windows whose 206 px box reaches it.
    k = lat.n // 2
    valid2 = valid.copy()
    valid2[lat.row[k] + 90, lat.col[k] - 100] = False
    lat2 = S2.build_lattice("90-00-01", geom, grid, valid2, slope, None, P)
    gone = (np.abs(lat.row - (lat.row[k] + 90) - 0.5) < 103) & (np.abs(lat.col - (lat.col[k] - 100) - 0.5) < 103)
    assert lat2.n_invalid == int(gone.sum()) >= 1 and lat2.n == lat.n - lat2.n_invalid
    with pytest.raises(ValueError, match="does not match"):
        S2.build_lattice("90-00-01", geom, grid, valid[:-1], slope, None, P)


# ======================================================================================================================
# random streams
# ======================================================================================================================


def test_jitter_and_random_streams():
    a = S2.jitter(CFG, "25-41-05", 100)
    assert np.array_equal(a, S2.jitter(CFG, "25-41-05", 100))
    assert a.min() >= 0 and a.max() < 1e-9 and np.unique(a).size == 100
    assert not np.array_equal(a, S2.jitter(CFG, "24-43-19", 100))
    np.testing.assert_array_equal(S2.jitter(CFG, "25-41-05", 50), a[:50])  # a prefix: independent of other sheets
    r = S2.random_scores(CFG, "25-41-05", 100)
    assert np.array_equal(r, S2.random_scores(CFG, "25-41-05", 100))
    assert not np.allclose(r, a / 1e-9, rtol=1e-6, atol=0)  # a different stream from the jitter
    cfg2 = copy.deepcopy(CFG)
    cfg2["replication"]["seed"] += 1
    assert not np.array_equal(S2.jitter(cfg2, "25-41-05", 100), a)


# ======================================================================================================================
# NMS (D23)
# ======================================================================================================================


def test_nms_150m_edge():
    r2 = P.nms_radius_px2
    # 75 px = 150 m exactly: suppressed (d <= 150 m).
    res = S2.nms(np.array([0, 0]), np.array([0, 75]), np.array([1.0, 0.9]), r2)
    assert res.selected.tolist() == [True, False] and res.suppressed_by.tolist() == [0, 0]
    # (25, 75) px = 158.1 m: both selected.
    res = S2.nms(np.array([0, 25]), np.array([0, 75]), np.array([1.0, 0.9]), r2)
    assert res.selected.tolist() == [True, True]
    # The 29 lattice offsets within 150 m (50 m lattice) are suppressed, the rest of a 7 x 7 block is not.
    rr, cc = np.meshgrid(np.arange(-3, 4) * 25, np.arange(-3, 4) * 25, indexing="ij")
    rr, cc = rr.ravel() + 100, cc.ravel() + 100
    s = np.where((rr == 100) & (cc == 100), 2.0, 1.0)
    res = S2.nms(rr, cc, s, r2)
    centre = int(np.flatnonzero(s == 2.0)[0])
    near = ((rr - 100) ** 2 + (cc - 100) ** 2) <= 75 ** 2
    assert near.sum() == 29
    assert np.all(res.suppressed_by[near] == centre)
    assert not np.any(res.suppressed_by[~near] == centre)


def test_nms_ties_broken_by_jitter():
    rr, cc = np.meshgrid(np.arange(6) * 25, np.arange(6) * 25, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    s = np.full(rr.size, 0.5)
    sj = s + S2.jitter(CFG, "25-41-05", rr.size)
    res = S2.nms(rr, cc, sj, P.nms_radius_px2)
    assert res.selected[int(np.argmax(sj))]
    np.testing.assert_array_equal(res.selected, brute_nms(rr, cc, sj, P.nms_radius_px2))
    # Without jitter the exact ties fall back to (row, col): the first cell wins.
    res0 = S2.nms(rr, cc, s, P.nms_radius_px2)
    assert res0.selected[0] and res0.order[0] == 0
    np.testing.assert_array_equal(S2.nms(rr, cc, sj, P.nms_radius_px2).selected, res.selected)  # deterministic


@pytest.mark.parametrize("seed", range(5))
def test_nms_matches_brute_force_and_covers_every_cell(seed):
    rng = np.random.default_rng(seed)
    rr, cc = np.meshgrid(np.arange(14) * 25, np.arange(12) * 25, indexing="ij")
    keep = rng.random(rr.size) < 0.8
    rr, cc = rr.ravel()[keep], cc.ravel()[keep]
    s = np.round(rng.random(rr.size), 1)  # many exact ties
    sj = s + 1e-9 * rng.random(rr.size)
    res = S2.nms(rr, cc, sj, P.nms_radius_px2)
    np.testing.assert_array_equal(res.selected, brute_nms(rr, cc, sj, P.nms_radius_px2))
    sel = np.flatnonzero(res.selected)
    assert np.all(res.suppressed_by >= 0) and np.all(res.selected[res.suppressed_by])
    for i, j in itertools.combinations(sel, 2):  # detections are more than 150 m apart
        assert (rr[i] - rr[j]) ** 2 + (cc[i] - cc[j]) ** 2 > 75 ** 2
    pos = np.empty(rr.size, dtype=np.int64)
    pos[res.order] = np.arange(rr.size)
    for i in np.flatnonzero(~res.selected):  # suppressed by an earlier, nearby selected cell
        k = res.suppressed_by[i]
        assert pos[k] < pos[i] and (rr[i] - rr[k]) ** 2 + (cc[i] - cc[k]) ** 2 <= 75 ** 2


# ======================================================================================================================
# matching (Appendix A §9, D23)
# ======================================================================================================================

T_OID = np.array([5, 3, 8])
T_GEOM = np.array([box(0, 0, 100, 100), box(120, 0, 220, 100), box(400, 0, 500, 100)], dtype=object)
REGISTER = np.concatenate([T_GEOM, np.array([box(1000, 0, 1100, 100)], dtype=object)])
DETS = [  # (x, y, expected outcome, expected target) in processing order
    (110, 50, "tp", 3),          # in the buffers of 5 and 3, both unmatched: lowest objectid
    (111, 50, "tp", 5),          # 3 is matched, 5 is not
    (50, 50, "duplicate", 5),    # inside 5, already matched
    (160, 150, "duplicate", 3),  # exactly 50 m from 3 (matched): in its buffer
    (450, 150, "tp", 8),         # exactly 50 m from 8: TP (d <= 50)
    (450, 151, "ignored", -1),   # 51 m from 8 (matched): not in the buffer, but in the 150 m ring
    (1050, 250, "ignored", -1),  # exactly 150 m from a non-target register polygon
    (1050, 250.5, "fp", -1),     # 150.5 m
    (700, 50, "fp", -1),         # 200 m from 8, 300 m from the register polygon
]


def test_matching_every_outcome():
    x = np.array([d[0] for d in DETS], dtype=float)
    y = np.array([d[1] for d in DETS], dtype=float)
    out, tid = S2.match_detections(x, y, np.arange(len(DETS)), T_OID, T_GEOM, REGISTER, 50.0, 150.0)
    assert out.tolist() == [d[2] for d in DETS]
    assert tid.tolist() == [d[3] for d in DETS]


def test_matching_follows_the_order():
    x = np.array([110, 111, 50], dtype=float)
    y = np.array([50, 50, 50], dtype=float)
    out, tid = S2.match_detections(x, y, np.array([2, 0, 1]), T_OID, T_GEOM, REGISTER, 50.0, 150.0)
    # (50, 50) first takes 5; (110, 50) then takes 3; (111, 50) finds both matched: duplicate of the lowest (3).
    assert out.tolist() == ["tp", "duplicate", "tp"] and tid.tolist() == [3, 3, 5]
    with pytest.raises(ValueError, match="permutation"):
        S2.match_detections(x, y, np.array([0, 0, 1]), T_OID, T_GEOM, REGISTER, 50.0, 150.0)
    with pytest.raises(ValueError, match="unique"):
        S2.match_detections(x, y, np.arange(3), np.array([5, 5, 8]), T_GEOM, REGISTER, 50.0, 150.0)


def test_ring_as_fp_keeps_duplicates():
    o = np.array(["tp", "duplicate", "ignored", "fp"])
    assert S2.ring_as_fp(o).tolist() == ["tp", "duplicate", "fp", "fp"]


# ======================================================================================================================
# evaluation of a scorer: pooled matching, P@R50, strata, ring as FP, "not reached" = FAIL
# ======================================================================================================================


def _inputs_two_sheets(cfg=CFG) -> S2.Inputs:
    x0, y0 = -600_000.4, -1_150_000.12
    g1 = box(x0, y0 - 2000, x0 + 2000, y0)               # flysch sheet: rows/cols 0..1000
    g2 = box(x0 + 10_000, y0 - 2000, x0 + 12_000, y0)    # non-flysch sheet, 8 km away
    sheets = pd.DataFrame([{"sheet": "90-00-01", "flysch": True, "geometry": g1},
                           {"sheet": "90-00-02", "flysch": False, "geometry": g2}])
    X = lambda c: x0 + 2.0 * c  # noqa: E731
    Y = lambda r: y0 - 2.0 * r  # noqa: E731
    polys = [  # objectid, eligible, sheet, geometry
        (11, True, "90-00-01", box(X(95), Y(105), X(105), Y(95))),       # target around corner (100, 100)
        (12, True, "90-00-01", box(X(395), Y(105), X(405), Y(95))),      # target around (100, 400)
        (13, True, "90-00-01", box(X(695), Y(705), X(705), Y(695))),     # target around (700, 700)
        (14, False, None, box(X(395), Y(405), X(405), Y(395))),          # register only, around (400, 400)
        (21, True, "90-00-02", box(X(5095), Y(105), X(5105), Y(95))),    # target on sheet 2 around (100, 5100)
        (99, True, "77-00-01", box(X(-3000), Y(105), X(-2990), Y(95))),  # eligible, another sheet
    ]
    poly = pd.DataFrame(polys, columns=["objectid", "eligible", "sheet", "geometry"])
    return S2.Inputs(sheets=sheets, polygons=poly, dtm_path=lambda s: None, susc_path=lambda s: None)


def test_evaluate_pooled_strata_and_ring():
    inp = _inputs_two_sheets()
    # Sheet 1: four isolated cells (> 150 m apart) with distinct scores; sheet 2: one TP and one FP.
    l1 = lattice_frame([100, 100, 400, 900], [100, 400, 400, 900], [0.9, 0.5, 0.8, 0.7])
    l2 = lattice_frame([100, 900], [5100, 5900], [0.85, 0.6])
    res = S2.evaluate_scorer(CFG, "slope_only", {"90-00-01": l1, "90-00-02": l2}, inp)
    d = res.detections
    assert list(d.columns) == list(S2.DETECTION_COLUMNS) and (d["scorer"] == "slope_only").all()
    assert d["score"].tolist() == [0.9, 0.85, 0.8, 0.7, 0.6, 0.5] and d["rank"].tolist() == list(range(6))
    assert d["outcome"].tolist() == ["tp", "tp", "ignored", "fp", "fp", "tp"]
    assert d["target_objectid"].tolist() == [11, 21, -1, -1, -1, 12]
    assert res.n_targets == {"pooled": 4, "flysch": 3, "nonflysch": 1}
    # Pooled: TP 1, 2 (recall 0.5 reached at 0.85, precision 2/2); interp max over recall >= 0.5.
    assert res.pooled.status == REACHED and res.pooled.precision_at_recall == 1.0
    assert res.pooled.interpolated_precision == 1.0 and res.pooled.max_recall == 0.75
    assert res.pooled.threshold_at_recall == pytest.approx(0.85, abs=1e-8)
    # Ring as FP: the ignored detection at 0.8 is an FP; P@R50 unchanged (reached before it).
    assert res.ring_fp.precision_at_recall == 1.0 and res.ring_fp.n_fp == 3
    # Flysch: targets 11, 12, 13; TP at 0.9 and 0.5: recall 2/3 reached only at 0.5 with 2 TP, 1 FP (0.7).
    fl = res.strata["flysch"]
    assert fl.status == REACHED and fl.precision_at_recall == pytest.approx(2 / 3)
    assert res.strata_ring_fp["flysch"].precision_at_recall == pytest.approx(2 / 4)
    nf = res.strata["nonflysch"]
    assert nf.precision_at_recall == 1.0 and nf.max_recall == 1.0
    assert res.strata_consistent and res.n_windows == 6 and res.n_sheets == 2
    row = S2.results_row(res, CFG)
    assert row["p_at_r50"] == 1.0 and row["flysch_p_at_r50"] == pytest.approx(2 / 3) and row["meets_threshold"]
    assert row["ringfp_flysch_p_at_r50"] == pytest.approx(0.5) and row["kind"] == "non_ml_baseline"


def test_fifty_percent_recall_never_reached_is_fail():
    inp = _inputs_two_sheets()
    l1 = lattice_frame([100, 900], [100, 900], [0.9, 0.8])  # one TP of 3 flysch targets
    l2 = lattice_frame([900], [5900], [0.6])                # no TP on sheet 2
    res = S2.evaluate_scorer(CFG, "random", {"90-00-01": l1, "90-00-02": l2}, inp)
    assert res.pooled.status == NOT_REACHED and res.pooled.max_recall == 0.25
    assert res.pooled.precision_at_recall is None and not res.pooled.passes(0.0)
    row = S2.results_row(res, CFG)
    assert row["status"] == "not_reached" and math.isnan(row["p_at_r50"]) and math.isnan(row["interp_precision"])
    assert row["meets_threshold"] is False and row["max_recall"] == 0.25
    assert row["nonflysch_status"] == "not_reached" and row["nonflysch_max_recall"] == 0.0


def test_evaluate_applies_nms_per_sheet_with_jitter():
    inp = _inputs_two_sheets()
    rr, cc = np.meshgrid(np.arange(0, 1001, 25), np.arange(0, 1001, 25), indexing="ij")
    flat = lattice_frame(rr.ravel(), cc.ravel(), np.zeros(rr.size))  # all tied: the jitter decides
    l2 = lattice_frame([100], [5100], [0.0])
    res = S2.evaluate_scorer(CFG, "susceptibility", {"90-00-01": flat, "90-00-02": l2}, inp)
    d1 = res.detections.loc[res.detections["sheet"] == "90-00-01"].sort_values(["row", "col"])
    sj = S2.jitter(CFG, "90-00-01", rr.size)
    want = brute_nms(rr.ravel(), cc.ravel(), sj, P.nms_radius_px2)
    got = np.zeros(rr.size, dtype=bool)
    got[(d1["row"].to_numpy() // 25) * 41 + d1["col"].to_numpy() // 25] = True
    np.testing.assert_array_equal(got, want)
    assert np.all(np.diff(res.detections["score_jittered"].to_numpy()) <= 0)
    assert res.residual_ties == 0


# ======================================================================================================================
# LR comparator
# ======================================================================================================================


def _synthetic_meta(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    src = np.array(["positive"] * n + ["random"] * n + ["background"] * n)
    lab = np.r_[np.ones(n), np.zeros(2 * n)].astype(int)
    slope = np.r_[rng.normal(22, 6, n), rng.normal(22, 6, n), rng.normal(12, 7, n)].clip(0, 60)
    return pd.DataFrame({"id": np.arange(3 * n), "label": lab, "source": src, "slope_mean": slope,
                         "slope_std": rng.gamma(4, 1.5, 3 * n), "susc_mean": rng.uniform(0, 2, 3 * n),
                         "susc_high_frac": rng.uniform(0, 1, 3 * n)})


def test_lr_comparator_fits_positives_vs_background():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    meta = _synthetic_meta()
    model = S2.fit_lr_comparator(meta, CFG)
    rows = meta[meta["source"].isin(["positive", "background"])]
    ref = make_pipeline(StandardScaler(), LogisticRegression(random_state=CFG["replication"]["seed"]))
    ref.fit(rows[list(S2.LR_FEATURES)].to_numpy(), rows["label"].to_numpy())
    stats = {k: np.linspace(meta[k].min(), meta[k].max(), 7) for k in S2.LR_FEATURES}
    got = S2.lr_scores(model, stats)
    np.testing.assert_allclose(got, ref.predict_proba(np.column_stack([stats[k] for k in S2.LR_FEATURES]))[:, 1],
                               rtol=0, atol=1e-12)
    assert np.all(np.diff(got[:3]) != 0) and np.all((got > 0) & (got < 1))
    # The matched negatives play no part.
    meta2 = meta.copy()
    meta2.loc[meta2["source"] == "random", "slope_mean"] = 0.0
    np.testing.assert_allclose(S2.lr_scores(S2.fit_lr_comparator(meta2, CFG), stats), got, rtol=0, atol=1e-12)
    bad = meta.copy()
    bad.loc[bad["source"] == "background", "label"] = 1
    with pytest.raises(ValueError, match="label"):
        S2.fit_lr_comparator(bad, CFG)
    with pytest.raises(ValueError, match="fitted model"):
        S2.baseline_scores("slope_susc_lr", S2.Lattice("x", *(np.zeros(0),) * 4, 0, 0), CFG)


# ======================================================================================================================
# end to end on a synthetic world: run_baselines, the files, upserts, evaluate_scorers
# ======================================================================================================================


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.delenv("LANDSLIDE_DATA_DIR", raising=False)
    paths = Paths(root=tmp_path)
    x0, y0 = -600_000.4, -1_150_000.12
    rows, polys, oid = [], [], 100
    rng = np.random.default_rng(7)
    for k, (sheet, flysch) in enumerate([("90-00-01", True), ("90-00-02", False)]):
        geom = rotated_square(x0 + 1500 + 8000 * k, y0 - 1200, 1300.0, deg=3.0 + k)
        row = sheet_row(sheet, geom, CFG, flysch)
        rows.append(row)
        H, W = row["grid_h"], row["grid_w"]
        z = synthetic_dtm((H, W), seed=10 + k, border=False, holes=False, lake=False).astype(np.float32)
        code = np.digitize(z, np.quantile(z, [0.5, 0.8])).astype(np.uint8)
        (tmp_path / "dtm").mkdir(exist_ok=True)
        write_tif(tmp_path / "dtm" / f"{sheet}.tif", z, row)
        write_tif(tmp_path / "dtm" / f"{sheet}_susc.tif", code, row)
        cx, cy = geom.centroid.x, geom.centroid.y
        for _ in range(6):
            px, py = cx + rng.uniform(-550, 550), cy + rng.uniform(-550, 550)
            g = box(px - 40, py - 40, px + 40, py + 40)
            elig = bool(rng.random() < 0.7)
            polys.append((oid, elig, sheet if elig else None, g))
            oid += 1
    inputs = S2.Inputs(sheets=pd.DataFrame(rows), polygons=pd.DataFrame(polys, columns=["objectid", "eligible",
                                                                                          "sheet", "geometry"]),
                       dtm_path=lambda s: tmp_path / "dtm" / f"{s}.tif",
                       susc_path=lambda s: tmp_path / "dtm" / f"{s}_susc.tif")
    return paths, inputs


def test_run_baselines_end_to_end(world):
    paths, inputs = world
    res = S2.run_baselines(CFG, paths, list(S2.UNFITTED_BASELINES), inputs)
    assert set(res) == set(S2.UNFITTED_BASELINES)
    for sheet in inputs.sheet_ids:
        lat = S2.load_lattice(paths, sheet)
        assert lat.n > 100 and lat.n_invalid == 0
        for s in S2.UNFITTED_BASELINES:
            d = S2.load_npz(S2.scores_path(paths, s, sheet))
            assert sorted(d) == ["col", "row", "score", "x", "y"]
            np.testing.assert_array_equal(d["row"], lat.row)
        sl = S2.load_scores(paths, "slope_only", sheet)["score"].to_numpy()
        np.testing.assert_array_equal(sl, lat.stats["slope_mean"])
        np.testing.assert_array_equal(S2.load_scores(paths, "susceptibility", sheet)["score"], lat.stats["susc_mean"])
        np.testing.assert_array_equal(S2.load_scores(paths, "random", sheet)["score"],
                                      S2.random_scores(CFG, sheet, lat.n))
        # window statistics against the per-patch slope of the channel library
        with rasterio.open(inputs.dtm_path(sheet)) as src:
            z = src.read(1).astype(np.float64)
        for k in (0, lat.n // 2, lat.n - 1):
            pc = patch_channels(z, int(lat.row[k]), int(lat.col[k]), CFG)[0]
            assert lat.stats["slope_mean"][k] == pytest.approx(pc.mean(), abs=1e-9)
            assert lat.stats["slope_std"][k] == pytest.approx(pc.std(), abs=1e-9)
    det = pd.read_parquet(paths.stage2 / "detections.parquet")
    assert list(det.columns) == list(S2.DETECTION_COLUMNS)
    assert set(det["scorer"]) == set(S2.UNFITTED_BASELINES)
    assert set(det["outcome"]) <= {"tp", "fp", "ignored", "duplicate"}
    csv = pd.read_csv(paths.results / "stage2_baselines.csv")
    assert csv["scorer"].tolist() == ["slope_only", "susceptibility", "random"]
    assert csv["n_targets"].nunique() == 1 and csv["n_targets"].iloc[0] == (inputs.polygons["eligible"]).sum()
    summ = json.loads((paths.stage2 / "summary.json").read_text(encoding="utf-8"))
    assert summ["lattice"]["n_invalid_dropped"] == 0 and summ["config_hash"] == config_hash(CFG)
    assert set(summ["scorers"]) == set(S2.UNFITTED_BASELINES)

    # Adding the LR later keeps the other rows; re-evaluating from the saved lattices reproduces the detections.
    res2 = S2.run_baselines(CFG, paths, ["slope_susc_lr"], inputs, meta=_synthetic_meta())
    csv2 = pd.read_csv(paths.results / "stage2_baselines.csv")
    assert csv2["scorer"].tolist() == ["slope_only", "susceptibility", "slope_susc_lr", "random"]
    pd.testing.assert_frame_equal(csv2.loc[csv2["scorer"] != "slope_susc_lr"].reset_index(drop=True), csv)
    assert res2["slope_susc_lr"].n_windows == res["slope_only"].n_windows
    ev = S2.evaluate_scorers(CFG, paths, ["slope_only"], inputs)
    pd.testing.assert_frame_equal(ev["slope_only"].detections, res["slope_only"].detections)
    det2 = pd.read_parquet(paths.stage2 / "detections.parquet")
    assert set(det2["scorer"]) == set(S2.UNFITTED_BASELINES) | {"slope_susc_lr"}


def test_smoke_paths_never_write_results(world, tmp_path):
    _, inputs = world
    paths = Paths(smoke=True, root=tmp_path)
    S2.run_baselines(CFG, paths, ["random"], inputs)
    assert (tmp_path / "data" / "smoke" / "results" / "stage2_baselines.csv").exists()
    assert not (tmp_path / "results").exists()
    assert (tmp_path / "data" / "smoke" / "stage2" / "random").is_dir()


# ======================================================================================================================
# CNN scorer: toy raster, random-init models (never real data)
# ======================================================================================================================


def _toy_sheet(cfg, seed=4):
    geom = rotated_square(-600_000.4 + 300, -1_150_000.12 - 300, 160.0, deg=2.0)
    row = sheet_row("90-00-03", geom, cfg, True)
    grid = grid_of_sheet(pd.Series(row), 2.0, grid_origin(cfg))
    z = synthetic_dtm((grid.n_rows - 1, grid.n_cols - 1), seed=seed, border=False, holes=False, lake=False)
    return row, grid, geom, z


def _seed_models(cfg, seeds=(0, 1), device="cpu"):
    out = []
    for s in seeds:
        torch.manual_seed(100 + s)
        m = build_model(cfg, None).to(device).eval()
        # plausible per-channel statistics in config order (slope, lrm_s, lrm_l, curv_prof, curv_plan, svf, detr)
        norm = Normaliser([12.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0], [8.0, 0.5, 2.0, 1.0, 1.0, 0.05, 1.5])
        out.append(S2.SeedModel(seed=s, model=m, norm=norm))
    return out


def test_window_channels_equal_patch_channels():
    cfg = toy_cfg()
    _, grid, geom, z = _toy_sheet(cfg)
    p = S2.stage2_params(cfg)
    r, c = S2.lattice_corners(geom, grid, p.stride_px)
    terrain = full_channels(z, cfg)
    w = S2.window_channels(terrain, z, r, c, cfg)
    assert w.shape == (r.size, 7, 64, 64) and r.size >= 4
    for k in range(r.size):
        np.testing.assert_allclose(w[k], patch_channels(z, int(r[k]), int(c[k]), cfg), rtol=0, atol=5e-9)
    w16 = S2.window_channels(terrain, z, r, c, cfg, dtype=np.float16)
    assert w16.dtype == np.float16 and np.array_equal(w16, w.astype(np.float16))
    with pytest.raises(ValueError, match="leaves"):
        S2.window_channels(terrain, z, np.array([10]), np.array([200]), cfg)


@pytest.mark.parametrize("px", [64, 150])
def test_cnn_scorer_on_toy_raster(px):
    cfg = toy_cfg(px=px, stride=16 if px == 64 else 25)
    _, grid, geom, z = _toy_sheet(cfg)
    p = S2.stage2_params(cfg)
    r, c = S2.lattice_corners(geom, grid, p.stride_px)
    if px == 150:
        r, c = r[:3], c[:3]
    terrain = full_channels(z, cfg)
    models = _seed_models(cfg)
    probs = S2.cnn_window_probs(models, terrain, z, r, c, cfg, chunk_windows=5)
    assert probs.shape == (2, r.size) and probs.dtype == np.float64 and np.all((probs > 0) & (probs < 1))
    # Reference: the per-patch channels, cast to float16 like patches.npy, through tta_predict.
    x_ref = np.stack([patch_channels(z, int(a), int(b), cfg) for a, b in zip(r, c)]).astype(np.float16)
    for i, sm in enumerate(models):
        ref = tta_predict(sm.model, torch.from_numpy(x_ref), batch_size=cfg["train"]["batch_size"],
                          norm=sm.norm).numpy()
        np.testing.assert_allclose(probs[i], ref, rtol=0, atol=1e-6)
    assert not np.allclose(probs[0], probs[1])
    np.testing.assert_array_equal(S2.cnn_scores(probs), probs.mean(axis=0))
    # Chunking does not change the result.
    np.testing.assert_allclose(S2.cnn_window_probs(models, terrain, z, r, c, cfg, chunk_windows=512), probs,
                               rtol=0, atol=1e-6)


def _write_run_dirs(paths: Paths, cfg, models, chash=None):
    for sm in models:
        d = paths.run_dir("s2", sm.seed)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": sm.model.state_dict(), "unit": f"s2_s{sm.seed}", "run": "s2", "seed": sm.seed,
                    "epoch": 1, "val_auc": 0.5, "config_hash": chash or config_hash(cfg), "arch": "resnet18",
                    "in_channels": 7}, d / "best.pt")
        (d / "norm.json").write_text(json.dumps(sm.norm.to_dict()), encoding="utf-8")
        (d / "done.json").write_text("{}", encoding="utf-8")


def test_load_s2_models_and_run_cnn(tmp_path, monkeypatch):
    monkeypatch.delenv("LANDSLIDE_DATA_DIR", raising=False)
    cfg = toy_cfg()
    cfg["train"]["seeds"] = [0, 1]
    paths = Paths(root=tmp_path)
    models = _seed_models(cfg)
    _write_run_dirs(paths, cfg, models)
    loaded = S2.load_s2_models(cfg, paths, device="cpu")
    assert [m.seed for m in loaded] == [0, 1]
    for a, b in zip(loaded, models):
        assert all(torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items())
        assert np.array_equal(a.norm.mean, b.norm.mean) and not a.model.training

    row, grid, geom, z = _toy_sheet(cfg)
    code = np.zeros(z.shape, dtype=np.uint8)
    write_tif(tmp_path / "toy.tif", z.astype(np.float32), row)
    write_tif(tmp_path / "toy_susc.tif", code, row)
    inputs = S2.Inputs(sheets=pd.DataFrame([row]), polygons=pd.DataFrame(columns=["objectid", "eligible", "sheet",
                                                                                   "geometry"]),
                       dtm_path=lambda s: tmp_path / "toy.tif", susc_path=lambda s: tmp_path / "toy_susc.tif")
    info = S2.run_cnn(cfg, paths, inputs, models=loaded)
    assert info["sheets"]["90-00-03"]["status"] == "done"
    inf = S2.load_npz(S2.infer_path(paths, "90-00-03"))
    sc = S2.load_scores(paths, "cnn", "90-00-03")
    z32 = z.astype(np.float32).astype(np.float64)
    want = S2.cnn_window_probs(loaded, full_channels(z32, cfg), z32, inf["row"], inf["col"], cfg)
    np.testing.assert_allclose(inf["probs"], want, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sc["score"].to_numpy(), inf["probs"].mean(axis=0))
    assert S2.run_cnn(cfg, paths, inputs, models=loaded)["sheets"]["90-00-03"]["status"] == "skipped"

    (paths.run_dir("s2", 1) / "done.json").unlink()
    with pytest.raises(FileNotFoundError, match="s2_s1"):
        S2.load_s2_models(cfg, paths, device="cpu")
    _write_run_dirs(paths, cfg, models, chash="0" * 64)
    with pytest.raises(ValueError, match="trained under config"):
        S2.load_s2_models(cfg, paths, device="cpu")


# ======================================================================================================================
# real data: one reserved sheet's lattice (no CNN)
# ======================================================================================================================


@pytest.mark.needs_data
@pytest.mark.skipif(not Paths().sheets_parquet.exists() or not Paths().susceptibility_dir.exists(),
                    reason="data/derived or the susceptibility cache is not available")
def test_real_lattice_of_one_reserved_sheet():
    paths = Paths()
    inputs = S2.load_inputs(CFG, paths, ["25-41-05"])
    sr = S2.read_sheet(inputs, "25-41-05", CFG)
    slope = full_channels(sr.z, CFG, names=["slope"])[0]
    lat = S2.build_lattice("25-41-05", sr.geom, sr.grid, np.isfinite(sr.z), slope, sr.susc, P)
    H, W = sr.z.shape
    assert lat.n_invalid == 0 and lat.n > 7000
    assert min(lat.row.min(), lat.col.min(), H - lat.row.max(), W - lat.col.max()) >= P.valid_half
    assert set(lat.stats) == set(S2.LR_FEATURES) and all(np.isfinite(v).all() for v in lat.stats.values())
    assert len(S2.targets_of(inputs)) == 108
