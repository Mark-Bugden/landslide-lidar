"""End-to-end tests of the P1-09 build on a synthetic world (`sampling_synth.py`): the order of operations, brute-force
cross-checks of every screen, the final checks, and determinism."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import shapely

from landslide import sampling as S
from landslide.geo import repair
from sampling_synth import make_world, synth_config


@pytest.fixture(scope="module")
def world():
    inputs, cfg = make_world()
    return inputs, cfg, S.build_samples(cfg, inputs)


def _brute_rules(inputs, cfg, sheet, r, c, valid_only=False):
    """Every candidate rule, by brute force on the actual squares and raster windows."""
    p = S.sampling_params(cfg)
    sh = inputs.sheets.set_index("sheet")
    g = S.grid_of_sheet(sh.loc[sheet], p.cell_m)
    x, y = g.xy(r, c)
    h = p.half_m
    sq = shapely.box(x - h, y - h, x + h, y + h)
    ok = shapely.covered_by(sq, sh.loc[sheet, "geometry"])
    for q in sh.loc[sh["is_stage2"].astype(bool), "geometry"]:
        ok &= ~shapely.dwithin(sq, q, p.stage2_buffer_m)
    z = inputs.read_dtm(sheet).z
    hh = p.context_half_px
    for k in np.flatnonzero(ok):
        w = z[max(r[k] - hh, 0):r[k] + hh, max(c[k] - hh, 0):c[k] + hh]
        ok[k] = w.shape == (2 * hh, 2 * hh) and bool(np.isfinite(w).all())
    if valid_only:
        return ok
    pts = shapely.points(x, y)
    reg = repair(inputs.polygons["geometry"].to_numpy())
    ls = inputs.zabaged["landslide"]["geometry"].to_numpy()
    d_reg = np.min(shapely.distance(pts[:, None], reg[None, :]), axis=1)
    d_ls = np.min(shapely.distance(pts[:, None], ls[None, :]), axis=1)
    return ok & (d_reg >= p.neg_min_dist_to_polygon_m) & (d_ls >= p.neg_exclude_zabaged_landslide_m)


def test_final_checks_pass(world):
    inputs, cfg, res = world
    chk = S.check_samples(cfg, res.samples, inputs, recheck_dtm=True)
    assert chk["n_pairs"] == int((res.samples["label"] == 1).sum()) > 20
    assert chk["max_slope_mean_diff_deg"] < 1e-9
    assert list(res.samples.columns) == S.COLUMNS


def test_positive_feasibility_brute_force(world):
    """n_feasible of every eligible stage-1 polygon = corners strictly inside it whose square is inside its sheet and
    > 150 m from the stage-2 sheet (actual squares, GEOS)."""
    inputs, cfg, res = world
    p = S.sampling_params(cfg)
    sh = inputs.sheets.set_index("sheet")
    s2 = sh.loc[sh["is_stage2"].astype(bool), "geometry"].to_numpy()
    polys = inputs.polygons.set_index("objectid")
    for row in res.polygons.itertuples(index=False):
        geom = polys.loc[row.objectid, "geometry"]
        g = S.grid_of_sheet(sh.loc[row.sheet], p.cell_m)
        xmin, ymin, xmax, ymax = geom.bounds
        c = np.arange(int(np.floor((xmin - g.x0) / 2)) - 1, int(np.ceil((xmax - g.x0) / 2)) + 2)
        r = np.arange(int(np.floor((g.y0 - ymax) / 2)) - 1, int(np.ceil((g.y0 - ymin) / 2)) + 2)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        x, y = g.xy(rr.ravel(), cc.ravel())
        inside = shapely.contains_xy(geom, x, y)
        sq = shapely.box(x - 150, y - 150, x + 150, y + 150)
        ok = inside & shapely.covered_by(sq, sh.loc[row.sheet, "geometry"])
        for q in s2:
            ok &= ~shapely.dwithin(sq, q, p.stage2_buffer_m)
        assert int(ok.sum()) == row.n_feasible, row
    poly = res.polygons
    assert (poly["n_feasible"] == 0).sum() >= 3            # edge-straddling and stage-2-adjacent polygons
    assert ((poly["n_feasible"] > 0) & (poly["k"] == 0)).sum() == 0


def test_positive_points_rules(world):
    inputs, cfg, res = world
    p = S.sampling_params(cfg)
    pts = res.positives
    polys = inputs.polygons.set_index("objectid")
    poly = res.polygons.set_index(["sheet", "objectid"])
    for (s, oid), g in pts.groupby(["sheet", "objectid"]):
        k_target = S.points_per_polygon(polys.loc[oid, "area_m2"], p)
        assert len(g) <= k_target and poly.loc[(s, oid), "k"] == len(g)
        rc = g[["row", "col"]].to_numpy()
        d = np.sqrt(((rc[:, None] - rc[None]) ** 2).sum(-1)) * p.cell_m
        assert (d[np.triu_indices(len(g), 1)] >= p.min_pt_spacing_m).all()
    assert (res.polygons["k_target"] == 3).any() and (res.polygons["k"] >= 2).any()


def test_waterfill_and_validity_order(world):
    """Water-filling selects whole polygons under the cap; validity drops selected points afterwards, without top-up;
    only kept positives are matched."""
    inputs, cfg, res = world
    poly, pos, S_ = res.polygons, res.positives, res.samples
    summ = res.summary["positives"]
    taken = poly[poly["taken"]]
    assert int(taken["k"].sum()) == len(pos) == summ["selected_points"]
    for name, f in summ["pools"].items():
        flysch = name == "flysch"
        sheets = inputs.sheets.set_index("sheet")["flysch"].astype(bool)
        per_sheet = taken[taken["sheet"].map(sheets) == flysch].groupby("sheet")["k"].sum()
        assert (per_sheet <= f["cap"]).all()
        assert f["selected_points"] <= f["budget"] < (f["total_at_cap_plus_1"] or 10**9)
    assert summ["validity_dropped"] == int((~pos["valid"]).sum()) > 0
    assert summ["kept"] == summ["selected_points"] - summ["validity_dropped"]
    assert summ["matched"] + summ["unmatched"] == summ["kept"]
    used = S_.loc[S_["label"] == 1, ["sheet", "row", "col"]].apply(tuple, axis=1)
    valid = pos.loc[pos["valid"], ["sheet", "row", "col"]].apply(tuple, axis=1)
    invalid = pos.loc[~pos["valid"], ["sheet", "row", "col"]].apply(tuple, axis=1)
    assert set(used) <= set(valid) and not set(used) & set(invalid)
    # the budget split follows the eligible-polygon shares
    n_f = summ["pools"]["flysch"]["eligible_polygons"]
    n_n = summ["pools"]["nonflysch"]["eligible_polygons"]
    assert summ["pools"]["flysch"]["budget"] == S.round_half_up(cfg["sampling"]["max_pos"] * n_f / (n_f + n_n))


def test_every_candidate_passes_the_rules_brute_force(world):
    inputs, cfg, res = world
    cand = res.candidates
    for s, g in cand.groupby("sheet"):
        ok = _brute_rules(inputs, cfg, s, g["row"].to_numpy(), g["col"].to_numpy())
        assert ok.all(), (s, g[~ok].head())


def test_random_candidates_equal_brute_force(world):
    """The random candidates of each sheet are exactly the jittered grid points that pass every rule."""
    inputs, cfg, res = world
    p = S.sampling_params(cfg)
    sh = inputs.sheets.set_index("sheet")
    cand = res.candidates
    for s in sh.index[sh["is_stage1"].astype(bool)]:
        g = S.grid_of_sheet(sh.loc[s], p.cell_m)
        rr, cc = S.random_grid(g, p, S.sheet_rng(cfg, p.seed, "negative_jitter", s))
        inside = g.inside(rr, cc)
        rr, cc = rr[inside], cc[inside]
        ok = _brute_rules(inputs, cfg, s, rr, cc)
        want = set(zip(rr[ok].tolist(), cc[ok].tolist()))
        got = cand[(cand["sheet"] == s) & (cand["source"] == "random")]
        assert set(zip(got["row"].tolist(), got["col"].tolist())) == want, s


def test_lookalike_candidates(world):
    inputs, cfg, res = world
    p = S.sampling_params(cfg)
    cand = res.candidates
    la = cand[cand["source"] != "random"]
    assert set(la["source"]) == {"quarry", "spoil", "rock", "cut"}
    # at most lookalike_max_per_polygon per area feature
    area = la[la["source"] != "cut"]
    assert (area.groupby(["source", "source_objectid"]).size() <= p.lookalike_max_per_polygon).all()
    # a cut candidate is steep: p90 slope over the 21 px window >= 20 deg (recomputed from the sheet raster)
    cuts = la[la["source"] == "cut"]
    assert len(cuts) and not cuts.duplicated(["sheet", "row", "col"]).any()
    zab = pd.concat([inputs.zabaged["road"].assign(layer=79), inputs.zabaged["railway"].assign(layer=75)])
    pts = S.cut_points(zab.rename(columns={"geometry": "geom"}), p.cut_point_step_m)
    sh = inputs.sheets.set_index("sheet")
    for s, g in cuts.groupby("sheet"):
        grid = S.grid_of_sheet(sh.loc[s], p.cell_m)
        sr = S.sheet_raster(inputs.read_dtm(s).z, cfg)
        r, c = grid.snap(pts["x"], pts["y"])
        for row, col in zip(g["row"], g["col"]):
            k = np.flatnonzero((r == row) & (c == col))[0]          # the first point snapping there
            i = int(np.floor((grid.y0 - pts["y"].iloc[k]) / 2))
            j = int(np.floor((pts["x"].iloc[k] - grid.x0) / 2))
            assert np.percentile(sr.slope[i - 10:i + 11, j - 10:j + 11], 90) >= p.cut_slope_p90_deg
    # the polygon straddling sheets (0,0)|(1,0) gives candidates in one sheet only
    straddle = inputs.zabaged["spoil"]["objectid"].iloc[-1]
    assert area.loc[area["source_objectid"] == straddle, "sheet"].nunique() <= 1


def test_matching_rules_in_summary(world):
    inputs, cfg, res = world
    m = res.summary["matching"]
    n_kept = res.summary["positives"]["kept"]
    assert m["lookalike_quota"] == S.round_half_up(cfg["sampling"]["lookalike_frac"] * n_kept)
    assert m["lookalike_matched"] <= m["lookalike_quota"]
    assert res.summary["background"]["n"] == cfg["stage2"]["background_n"]
    bg = res.samples[res.samples["source"] == "background"]
    assert (bg["pair_id"] == -1).all() and (bg["match_level"] == 0).all()
    # background is drawn from unused random candidates
    neg = res.samples[(res.samples["label"] == 0) & (res.samples["pair_id"] >= 0)]
    key = lambda d: set(zip(d["sheet"], d["row"], d["col"]))  # noqa: E731
    assert not key(bg) & key(neg)


def test_summary_is_aggregate_and_json(world):
    _, _, res = world
    txt = json.dumps(res.summary, ensure_ascii=False)
    for bad in ('"x"', '"y"', '"row"', '"col"', '"geometry"', 'grid_xmin', 'bounds'):
        assert bad not in txt


def test_build_is_deterministic(world):
    inputs, cfg, res = world
    inputs2, cfg2 = make_world()
    res2 = S.build_samples(cfg2, inputs2)
    pd.testing.assert_frame_equal(res.samples, res2.samples, check_exact=True)
    assert json.dumps(res.summary, sort_keys=True) == json.dumps(res2.summary, sort_keys=True)


def test_per_sheet_draws_do_not_depend_on_other_sheets(world):
    """Removing one stage-1 sheet changes the water-filling, not the other sheets' points or candidates."""
    inputs, cfg, res = world
    drop = "90-03-02"
    inputs2, _ = make_world()
    inputs2.sheets.loc[inputs2.sheets["sheet"] == drop, "is_stage1"] = False
    res2 = S.build_samples(cfg, inputs2)
    keep = lambda d: d[d["sheet"] != drop].reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(keep(res.polygons.drop(columns="taken")), res2.polygons.drop(columns="taken"))
    cols = ["sheet", "row", "col", "source", "source_objectid", "slope_mean"]
    pd.testing.assert_frame_equal(keep(res.candidates[cols]), res2.candidates[cols].reset_index(drop=True))


def test_check_samples_catches_violations(world):
    inputs, cfg, res = world
    bad = res.samples.copy()
    i = bad.index[(bad["label"] == 0) & (bad["pair_id"] >= 0)][0]
    bad.loc[i, "slope_bin"] += 5                       # breaks the bin rule
    bad.loc[i, "slope_mean"] = bad.loc[i, "slope_bin"] * 2.0 + 0.5
    with pytest.raises(AssertionError):
        S.check_samples(cfg, bad, inputs, recheck_dtm=False)
    bad = res.samples.copy()
    j = bad.index[bad["source"] == "background"][0]
    k = bad.index[bad["source"] == "background"][1]
    bad.loc[k, ["x", "y", "row", "col", "sheet"]] = bad.loc[j, ["x", "y", "row", "col", "sheet"]].to_numpy()
    with pytest.raises(AssertionError):                 # two background samples 0 m apart
        S.check_samples(cfg, bad, inputs, recheck_dtm=False)


def test_tight_budget_and_unmatched_positives():
    """A small budget gives a small cap; positives with no candidate in their stratum are dropped and counted."""
    inputs, cfg = make_world(cfg=synth_config(max_pos=10, background_n=5))
    res = S.build_samples(cfg, inputs)
    summ = res.summary["positives"]
    assert summ["selected_points"] <= 10
    assert summ["matched"] == int((res.samples["label"] == 1).sum())
    S.check_samples(cfg, res.samples, inputs, recheck_dtm=False)
