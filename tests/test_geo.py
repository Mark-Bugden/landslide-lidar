"""Synthetic tests for `landslide.geo`: kraj1, in_mapped, eligibility, polygon -> sheet assignment, the sheet-kraj
rule (D06) with its tie rules, the flysch closing (D07), area fractions and the DTM validity scan."""

import json

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from landslide import geo as G

CFG = {"labels": {"skupina": 1, "in_mapped": True, "min_area_m2": 3000},
       "dtm": {"cell_m": 2.0, "invalid_rule": "z <= 0 (ImageServer returns 0.0 outside the border) or z > 2000 -> NaN"},
       "flysch": {"closing_m": 2000, "subregions": ["A", "B"], "min_sheet_frac": 0.5}}


def arr(*geoms):
    return np.array(list(geoms), dtype=object)


# ---- kraj1 ----

@pytest.mark.parametrize("value, want", [("Jihomoravský, Zlínský", "Jihomoravský"), ("Zlínský", "Zlínský"),
                                         (" Olomoucký ,Zlínský", "Olomoucký"), (None, None), (float("nan"), None),
                                         ("", None)])
def test_first_kraj(value, want):
    assert G.first_kraj(value) == want


# ---- in_mapped and eligibility ----

def test_in_mapped_uses_representative_point_in_union():
    union = shapely.union_all([box(0, 0, 10, 10), box(10, 0, 20, 10)])
    polys = arr(box(1, 1, 3, 3),            # inside the first sheet
                box(9, 4, 11, 6),           # across the shared edge: still inside the union
                box(30, 30, 32, 32),        # outside
                box(19, 4, 25, 5),          # mostly outside the union: its representative point (22, 4.5) is outside
                box(15, 4, 21, 5))          # mostly inside: representative point (18, 4.5) inside
    assert G.in_mapped(polys, union).tolist() == [True, True, False, False, True]


def test_eligible_mask():
    sk = np.array([1, 1, 1, 2, 1])
    inm = np.array([True, True, False, True, True])
    area = np.array([3000.0, 2999.9, 5000.0, 5000.0, 1e6])
    assert G.eligible_mask(sk, inm, area, CFG).tolist() == [True, False, False, False, True]


def test_repair_only_touches_invalid():
    bow = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # self-intersecting bow tie
    ok = box(0, 0, 1, 1)
    out = G.repair(arr(ok, bow))
    assert out[0] is ok
    assert shapely.is_valid(out[1]) and out[1].area > 0 and out[1].geom_type in ("Polygon", "MultiPolygon")


# ---- polygon -> sheet ----

SHEETS = arr(box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10))
IDS = ["01-01-01", "01-01-02", "01-01-03"]


def test_assign_largest_share():
    polys = arr(box(7, 0, 17, 1),     # 3 in A, 7 in B
                box(1, 1, 2, 2),      # wholly in A
                box(8, 0, 22, 1),     # 2 in A, 10 in B, 2 in C
                box(50, 50, 51, 51))  # in no sheet
    sheet, share = G.assign_to_sheets(polys, SHEETS, IDS)
    assert sheet.tolist() == ["01-01-02", "01-01-01", "01-01-02", None]
    assert share[0] == pytest.approx(0.7) and share[1] == pytest.approx(1.0)
    assert share[2] == pytest.approx(10 / 14) and share[3] == 0.0


def test_assign_exact_tie_goes_to_lowest_sheet_id():
    polys = arr(box(5, 0, 15, 1))  # 5 in A, 5 in B
    sheet, share = G.assign_to_sheets(polys, SHEETS, IDS)
    assert sheet.tolist() == ["01-01-01"] and share[0] == pytest.approx(0.5)
    # Order of the sheet arrays must not matter: the id decides, not the position.
    sheet, _ = G.assign_to_sheets(polys, SHEETS[::-1], IDS[::-1])
    assert sheet.tolist() == ["01-01-01"]


def test_assign_touching_only_is_not_an_overlap():
    polys = arr(box(10, 0, 12, 1))  # wholly in B, touches A along x = 10
    sheet, share = G.assign_to_sheets(polys, SHEETS, IDS)
    assert sheet.tolist() == ["01-01-02"] and share[0] == pytest.approx(1.0)


def test_assign_invalid_polygon_does_not_fail():
    bow = Polygon([(8, 0), (14, 6), (14, 0), (8, 6)])  # bow tie across the A/B edge, more of it in B
    sheet, share = G.assign_to_sheets(arr(bow), SHEETS, IDS)
    assert sheet.tolist() == ["01-01-02"] and 0.5 < share[0] <= 1.0


# ---- sheet kraj (D06) ----

def test_majority_kraj_rules():
    assert G.majority_kraj({"Zlínský": 3, "Jihomoravský": 2}, {"Jihomoravský": 1e9}) == ("Zlínský", "majority")
    assert G.majority_kraj({"Zlínský": 2, "Jihomoravský": 2}, {"Zlínský": 5.0, "Jihomoravský": 7.0}) == \
        ("Jihomoravský", "area")
    assert G.majority_kraj({"Zlínský": 2, "Jihomoravský": 2, "Olomoucký": 1},
                           {"Zlínský": 5.0, "Jihomoravský": 5.0, "Olomoucký": 99.0}) == ("Jihomoravský", "alphabetical")
    assert G.majority_kraj({}, {}) == (None, "none")


def test_sheet_kraj_majority_tie_and_first_listed():
    sheets = arr(box(0, 0, 10, 10), box(10, 0, 20, 10), box(100, 100, 110, 110))
    polys = arr(
        # sheet 0: two small Zlínský against one big Jihomoravský -> count wins
        box(1, 1, 2, 2), box(3, 3, 4, 4), box(5, 0, 9.5, 9),
        # sheet 1: one each, the Olomoucký polygon covers more of the sheet -> area decides the tie
        box(11, 1, 12, 2), box(13, 1, 19, 9),
    )
    kraj = ["Zlínský", "Zlínský, Jihomoravský", "Jihomoravský", "Moravskoslezský", "Olomoucký, Zlínský"]
    out = G.sheet_kraj(sheets, polys, kraj)
    assert out["kraj"].tolist() == ["Zlínský", "Olomoucký", None]
    assert out["kraj_rule"].tolist() == ["majority", "area", "none"]
    assert out["kraj_n_polygons"].tolist() == [3, 2, 0]
    assert json.loads(out.loc[0, "kraj_votes"]) == {"Jihomoravský": 1, "Zlínský": 2}


def test_sheet_kraj_counts_polygons_that_only_touch():
    # "all register polygons that intersect the sheet": a polygon sharing an edge intersects.
    sheets = arr(box(0, 0, 10, 10))
    polys = arr(box(10, 0, 12, 2), box(10, 4, 12, 6), box(1, 1, 2, 2))
    out = G.sheet_kraj(sheets, polys, ["Zlínský", "Zlínský", "Jihomoravský"])
    assert out.loc[0, "kraj"] == "Zlínský" and out.loc[0, "kraj_n_polygons"] == 3


# ---- flysch outline (D07) ----

def test_closing_bridges_gaps_narrower_than_twice_the_buffer():
    a, b = box(0, 0, 5000, 5000), box(6000, 0, 11000, 5000)  # 1 km gap
    u = shapely.union_all([a, b])
    out, st = G.close_fill_drop(u, 2000)
    assert st["final"]["n_parts"] == 1 and out.area > u.area
    out, st = G.close_fill_drop(u, 400)  # 0.8 km structuring element: the 1 km gap stays open
    assert st["final"]["n_parts"] == 2 and out.area == pytest.approx(u.area, rel=1e-4)  # corners: chords


def test_closing_fills_holes_and_drops_small_parts():
    ring = box(0, 0, 20000, 20000).difference(box(5000, 5000, 15000, 15000))  # 10 km hole: not closed by 2 km
    small = box(100000, 0, 100800, 1000)  # 0.8 km² -> dropped
    keep = box(200000, 0, 201200, 1000)   # 1.2 km² -> kept
    u = shapely.union_all([ring, small, keep])
    out, st = G.close_fill_drop(u, 2000)
    assert st["closed"]["n_holes"] == 1 and st["filled"]["n_holes"] == 0
    assert st["final"]["n_parts"] == 2 and st["filled"]["n_parts"] == 3
    assert out.area == pytest.approx(20000 ** 2 + 1.2e6, rel=1e-4)
    out, st = G.close_fill_drop(u, 2000, fill_holes=False)
    assert st["final"]["n_holes"] == 1


def test_small_part_threshold_is_inclusive():
    u = shapely.union_all([box(0, 0, 1000, 1000), box(5000, 0, 5999, 1000)])  # 1 km² and 0.999 km²
    out, st = G.close_fill_drop(u, 0.0)
    assert st["final"]["n_parts"] == 1 and out.area == pytest.approx(1e6)


def test_flysch_outline_selects_subregions_and_uses_config_buffer():
    geoms = arr(box(0, 0, 5000, 5000), box(6000, 0, 11000, 5000), box(0, 6000, 5000, 11000))
    sub = ["A", "B", "C"]  # C is not a flysch subregion
    out, st = G.flysch_outline(geoms, sub, CFG)
    assert st["n_geology_polygons"] == 2 and st["buffer_m"] == 2000
    assert not out.intersects(box(1000, 8000, 2000, 9000))
    out1, st1 = G.flysch_outline(geoms, sub, CFG, buffer_m=1000)
    assert st1["buffer_m"] == 1000 and out1.area <= out.area


def test_area_fraction_inside():
    region = box(0, 0, 10, 10)
    frac = G.area_fraction_inside(arr(box(5, 0, 15, 10), box(0, 0, 2, 2), box(20, 20, 21, 21)), region)
    assert frac.tolist() == pytest.approx([0.5, 1.0, 0.0])


# ---- DTM validity ----

def test_invalid_bounds_parse_and_reject():
    assert G.invalid_bounds(CFG) == (0.0, 2000.0)
    with pytest.raises(ValueError):
        G.invalid_bounds({"dtm": {"invalid_rule": "z below zero is invalid"}})


def test_dtm_validity_flags():
    z = np.full((4, 5), 300.0, dtype=np.float32)
    v = G.dtm_validity(z, CFG)
    assert v["dtm_all_valid"] and not v["has_zero_cells"] and v["n_invalid"] == 0
    z[0, 0] = 0.0
    v = G.dtm_validity(z, CFG)
    assert v["has_zero_cells"] and not v["dtm_all_valid"] and v["n_zero"] == 1 and v["n_invalid"] == 1
    z[0, 0] = 300.0
    z[1, 1] = np.nan
    v = G.dtm_validity(z, CFG)
    assert not v["has_zero_cells"] and not v["dtm_all_valid"] and v["n_nonfinite"] == 1
    z[1, 1] = 2000.5
    v = G.dtm_validity(z, CFG)
    assert not v["dtm_all_valid"] and v["n_gt_max"] == 1
    z[1, 1] = 2000.0  # the rule is z > 2000
    assert G.dtm_validity(z, CFG)["dtm_all_valid"]
    z[2, 2] = -1.0
    v = G.dtm_validity(z, CFG)
    assert not v["dtm_all_valid"] and not v["has_zero_cells"] and v["n_le_min"] == 1
    z[2, 2] = 7.0
    v = G.dtm_validity(z, CFG, nodata=7.0)
    assert not v["dtm_all_valid"] and v["n_nodata"] == 1


def test_scan_dtm_reads_grid_from_raster(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    z = np.full((6, 8), 250.0, dtype=np.float32)
    z[5, 7] = 0.0
    xmin, ymax = 0.4 + 2 * -303112, -0.12 + 2 * -543510
    p = tmp_path / "x.tif"
    with rasterio.open(p, "w", driver="GTiff", width=8, height=6, count=1, dtype="float32", crs="EPSG:5514",
                       transform=from_origin(xmin, ymax, 2.0, 2.0), nodata=float("nan")) as ds:
        ds.write(z, 1)
    r = G.scan_dtm(p, CFG)
    assert (r["grid_xmin"], r["grid_ymax"], r["grid_w"], r["grid_h"]) == (round(xmin, 6), round(ymax, 6), 8, 6)
    assert r["has_zero_cells"] and not r["dtm_all_valid"] and r["n_zero"] == 1
