"""Synthetic tests for `landslide.sheets`: grid snapping, grid comparison, DTM value comparison, the provenance
draw, the manifest check and the ArcGIS pagination. One `needs_data` test checks the real 860 grids."""

import hashlib
import json

import numpy as np
import pytest

from landslide import sheets as S
from landslide.config import load_config
from landslide.paths import Paths

CFG = {"crs": "EPSG:5514", "dtm": {"cell_m": 2.0, "margin_m": 250,
                                   "service": "https://ags.cuzk.gov.cz/arcgis2/rest/services/dmr5g/ImageServer (x)"},
       "sources": {"dtm": load_config()["sources"]["dtm"]},
       "replication": {"seed": 2611, "streams": {"provenance_sample": 1}, "provenance": {"n_sheets": 12}}}
O = S.grid_origin(CFG)  # (0.4, -0.12): the config's sources.dtm.grid_origin_m
VR = S.package_valid_range(CFG)  # (-100, 2000): sources.dtm.package_valid_range_m


def compare_dtm(*args, **kw):
    return S.compare_dtm(*args, valid_range=VR, **kw)


def test_source_values():
    assert O == (0.4, -0.12) and VR == (-100.0, 2000.0)


def _on_x(v):
    return abs(((v - 0.4) / 2) - round((v - 0.4) / 2)) < 1e-9


def _on_y(v):
    return abs(((v + 0.12) / 2) - round((v + 0.12) / 2)) < 1e-9


# ---- snapping ----

def test_snap_known_sheet():
    # Hand-over sheet 14-34-07: fetch log xmin -606223.6, ymax -1087020.12, 2848 x 2394.
    g = S.snap_out(-606223.1, -1091807.5, -600528.3, -1087021.0, origin=O)
    assert (g.xmin, g.ymax, g.w, g.h) == (-606223.6, -1087020.12, 2848, 2394)
    assert (g.xmax, g.ymin) == (-600527.6, -1091808.12)


@pytest.mark.parametrize("seed", range(5))
def test_snap_contains_box_and_is_minimal(seed):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        x, y = rng.uniform(-900_000, -430_000), rng.uniform(-1_230_000, -935_000)
        w, h = rng.uniform(1, 7000), rng.uniform(1, 7000)
        g = S.snap_out(x, y, x + w, y + h, origin=O)
        assert g.xmin <= x and g.ymin <= y and g.xmax >= x + w and g.ymax >= y + h
        assert x - g.xmin < 2 and y - g.ymin < 2 and g.xmax - (x + w) < 2 and g.ymax - (y + h) < 2
        assert all(_on_x(v) for v in (g.xmin, g.xmax)) and all(_on_y(v) for v in (g.ymin, g.ymax))
        assert g.w == round((g.xmax - g.xmin) / 2) and g.h == round((g.ymax - g.ymin) / 2)


def test_snap_on_lattice_stays_put():
    # Edges already on the lattice (with float noise from the decimal origin) are not pushed out by a pixel.
    g = S.snap_out(-606223.6, -1091808.12, -600527.6, -1087020.12, origin=O)
    assert (g.xmin, g.ymin, g.xmax, g.ymax) == (-606223.6, -1091808.12, -600527.6, -1087020.12)
    assert (g.w, g.h) == (2848, 2394)
    g = S.snap_out(0.4, -0.12, 2.4, 1.88, origin=O)
    assert (g.i0, g.j1, g.w, g.h) == (0, 1, 1, 1)


def test_snap_just_off_lattice_goes_out():
    g = S.snap_out(0.4 + 1e-4, -0.12 + 1e-4, 2.4 - 1e-4, 1.88 - 1e-4, origin=O)
    assert (g.i0, g.j1, g.w, g.h) == (0, 1, 1, 1)
    g = S.snap_out(0.4 - 1e-4, -0.12 - 1e-4, 2.4 + 1e-4, 1.88 + 1e-4, origin=O)
    assert (g.i0, g.j1, g.w, g.h) == (-1, 2, 3, 3)


def test_snap_rejects_empty_box():
    with pytest.raises(ValueError):
        S.snap_out(1, 1, 1, 2, origin=O)


def test_dtm_grid_adds_margin():
    g = S.dtm_grid((-605973.1, -1091557.5, -600778.3, -1087271.0), CFG)
    assert g.key() == S.snap_out(-606223.1, -1091807.5, -600528.3, -1087021.0, origin=O).key()


def test_edge_clearance():
    # Lattice lines: x = 0.4 + 2k (..., -999.6, ..., 1000.4), y = -0.12 + 2k (..., -1000.12, ..., 1999.88).
    # With the 250 m margin the box below becomes x -999.59 .. 1000.2, y -999.82 .. 1999.83: clearances
    # 0.01, 0.2, 0.3 and 0.05 m.
    c = S.edge_clearance((-749.59, -750.12 + 0.3, 750.4 - 0.2, 1749.88 - 0.05), CFG)
    assert c == pytest.approx(0.01, abs=1e-6)
    assert S.edge_clearance((-749.6, -750.12, 750.4, 1749.88), CFG) == pytest.approx(0.0, abs=1e-6)


# ---- grid from corner / transform, and comparison ----

def test_grid_from_corner_roundtrip_and_offgrid():
    g = S.snap_out(-606223.1, -1091807.5, -600528.3, -1087021.0, origin=O)
    assert S.grid_from_corner(g.xmin, g.ymax, g.w, g.h, origin=O).key() == g.key()
    assert S.grid_from_transform(g.transform, g.w, g.h, origin=O).key() == g.key()
    with pytest.raises(S.OffGrid):
        S.grid_from_corner(g.xmin + 0.01, g.ymax, g.w, g.h, origin=O)
    with pytest.raises(S.OffGrid):
        S.grid_from_transform((2.0, 0.0, g.xmin, 0.0, 2.0, g.ymax), g.w, g.h, origin=O)  # south-up
    with pytest.raises(S.OffGrid):
        S.grid_from_transform((1.0, 0.0, g.xmin, 0.0, -1.0, g.ymax), g.w, g.h, origin=O)  # wrong cell


def test_compare_grids_counts_and_mismatches():
    a = S.snap_out(0.4, -0.12, 100.4, 99.88, origin=O)
    b = S.snap_out(1000.4, -0.12, 1100.4, 99.88, origin=O)
    recipe = {"01-01-01": a, "01-01-02": b}
    log = {"01-01-01": {"xmin": a.xmin, "ymax": a.ymax, "w": a.w, "h": a.h},
           "01-01-02": {"xmin": b.xmin + 2.0, "ymax": b.ymax, "w": b.w - 1, "h": b.h}}
    rasters = {"01-01-01": {"transform": a.transform, "width": a.w, "height": a.h},
               "01-01-02": {"transform": b.transform, "width": b.w, "height": b.h}}
    res = S.compare_grids(recipe, log, rasters, 2.0, origin=O)
    assert res["equal"] == {"recipe_vs_fetch_log": 1, "recipe_vs_raster": 2, "fetch_log_vs_raster": 1}
    assert res["n_mismatched_sheets"] == 1
    m = res["mismatches"][0]
    assert m["sheet"] == "01-01-02" and m["recipe_vs_fetch_log"] == {"d_i0": 1, "d_w": -1}


def test_compare_grids_missing_entries():
    a = S.snap_out(0.4, -0.12, 100.4, 99.88, origin=O)
    res = S.compare_grids({"01-01-01": a}, {}, {"01-01-01": {"error": "missing"}}, 2.0, origin=O)
    assert res["n_mismatched_sheets"] == 1
    assert res["mismatches"][0]["fetch_log"] == "missing" and res["mismatches"][0]["raster"] == "missing"


# ---- export parameters ----

def test_export_params_follow_recipe_without_outsr():
    g = S.snap_out(-606223.1, -1091807.5, -600528.3, -1087021.0, origin=O)
    p = S.export_image_params(g, 5514)
    assert p == {"bbox": "-606223.6,-1091808.12,-600527.6,-1087020.12", "bboxSR": 5514, "imageSR": 5514,
                 "size": "2848,2394", "format": "tiff", "pixelType": "F32", "interpolation": "RSP_NearestNeighbor",
                 "f": "image"}
    assert not any(k.lower() == "outsr" for k in p)
    assert S.dtm_service_url(CFG) == "https://ags.cuzk.gov.cz/arcgis2/rest/services/dmr5g/ImageServer"


# ---- DTM value comparison ----

def _pair(seed=0, shape=(40, 50)):
    rng = np.random.default_rng(seed)
    dl = rng.uniform(200, 900, shape).astype(np.float32)
    return dl.copy(), dl


def test_compare_identical():
    pkg, dl = _pair()
    r = compare_dtm(pkg, dl)
    assert r["ok"] and r["n_compared"] == pkg.size == r["n_bitwise_equal"] and r["diff"]["n"] == 0


def test_compare_counts_value_differences_with_magnitudes():
    pkg, dl = _pair()
    dl[0, 0] = np.nextafter(dl[0, 0], np.float32(1e9))  # one ulp
    dl[1, 1] += np.float32(0.5)
    dl[2, 2] += np.float32(25.0)
    r = compare_dtm(pkg, dl)
    assert not r["ok"]
    d = r["diff"]
    assert d["n"] == 3 and d["max_abs"] == pytest.approx(25.0, abs=1e-3)
    assert d["histogram_abs"]["[0, 1e-06)"] == 0 and d["histogram_abs"]["[1e-06, 0.001)"] == 1
    assert d["histogram_abs"]["[0.1, 1)"] == 1 and d["histogram_abs"]["[10, inf)"] == 1
    assert [e[:2] for e in d["examples_row_col_package_download"]] == [[0, 0], [1, 1], [2, 2]]


def test_compare_is_bitwise_signed_zero():
    pkg, dl = _pair()
    pkg[3, 3], dl[3, 3] = np.float32(0.0), np.float32(-0.0)
    r = compare_dtm(pkg, dl)
    assert r["diff"]["n"] == 1 and r["diff"]["max_abs"] == 0.0 and not r["ok"]


def test_compare_nan_explained_by_nodata_and_range():
    pkg, dl = _pair()
    dl[0, :5] = np.nan            # NoData as NaN
    dl[1, :4] = -9999.0           # nodata tag value
    dl[2, :3] = -100.5            # below range
    dl[3, :2] = 2000.25           # above range
    dl[4, 0] = pkg[4, 0] = -100.0         # the range is closed: these stay values
    dl[4, 1] = pkg[4, 1] = 2000.0
    pkg[0, :5] = pkg[1, :4] = pkg[2, :3] = pkg[3, :2] = np.nan
    r = compare_dtm(pkg, dl, download_nodata=-9999.0)
    assert r["ok"], r
    assert r["n_package_nan"] == 14
    assert r["package_nan_explained"] == {"nodata": 9, "below_range": 3, "above_range": 2}
    assert r["n_package_nan_unexplained"] == 0 and r["n_compared"] == pkg.size - 14


def test_compare_flags_unexplained_nan():
    pkg, dl = _pair()
    pkg[5, 5] = np.nan
    r = compare_dtm(pkg, dl)
    assert not r["ok"] and r["n_package_nan_unexplained"] == 1
    assert r["unexplained_examples_row_col_download"][0][:2] == [5, 5]


def test_compare_flags_invalid_value_kept_in_package():
    pkg, dl = _pair()
    dl[6, 6] = pkg[6, 6] = np.float32(2500.0)  # bitwise equal, but the package should have NaN here
    r = compare_dtm(pkg, dl)
    assert r["diff"]["n"] == 0 and r["n_download_invalid_but_package_valued"] == 1 and not r["ok"]


def test_compare_zero_border_is_valid():
    pkg, dl = _pair()
    dl[:, :3] = pkg[:, :3] = 0.0  # the service's 0 m outside the border is inside [-100, 2000]
    r = compare_dtm(pkg, dl)
    assert r["ok"] and r["download"]["n_zero"] == 120 and r["package_n_zero"] == 120


def test_compare_shape_mismatch_and_dtype():
    pkg, dl = _pair()
    assert compare_dtm(pkg, dl[:, :-1])["ok"] is False
    with pytest.raises(TypeError):
        compare_dtm(pkg.astype(np.float64), dl.astype(np.float64))


def test_to_package_array_matches_compare():
    _, dl = _pair()
    dl[0, 0], dl[1, 1], dl[2, 2] = np.nan, -500.0, 3000.0
    pkg = S.to_package_array(dl, None, VR)
    assert np.isnan(pkg[[0, 1, 2], [0, 1, 2]]).all()
    assert compare_dtm(pkg, dl)["ok"]


def test_package_stats_and_fetch_log_comparison():
    z = np.full((10, 10), 416.26, np.float32)
    z[0, :3] = 0.0
    z[1, 0] = 585.2133
    z[9, 9] = np.nan
    st = S.package_stats(z)
    assert (st["n_nan"], st["n_zero"], st["n_negative"]) == (1, 3, 0)
    assert st["zmin"] == 0.0 and st["zmax"] == float(np.float32(585.2133))
    rec = {"invalid_frac": 0.01, "zmin": 0.0, "zmax": float(np.float32(585.2133))}
    assert all(S.compare_fetch_log_stats(st, rec).values())
    assert S.compare_fetch_log_stats(st, dict(rec, invalid_frac=0.0)) == {"invalid_frac": False, "zmin": True,
                                                                         "zmax": True}
    assert not S.compare_fetch_log_stats(st, dict(rec, zmax=585.2134))["zmax"]


def test_to_jsonable_keeps_nan_visible():
    assert S.to_jsonable({"a": float("nan"), "b": np.float32(1.5), "c": (1, 2)}) == {"a": "NaN", "b": 1.5,
                                                                                    "c": [1, 2]}


def test_geotiff_georef_roundtrip(tmp_path):
    import rasterio
    from rasterio.transform import Affine

    g = S.snap_out(-606223.1, -1091807.5, -606123.3, -1091707.0, origin=O)
    z = np.full((g.h, g.w), 300.0, np.float32)
    paths = []
    for name, extra in (("a.tif", {}), ("b.tif", {"compress": "deflate", "nodata": float("nan")})):
        p = tmp_path / name
        with rasterio.open(p, "w", driver="GTiff", width=g.w, height=g.h, count=1, dtype="float32",
                           crs="EPSG:5514", transform=Affine(*g.transform), **extra) as ds:
            ds.write(z, 1)
        paths.append(p)
    a, b = S.geotiff_georef(paths[0]), S.geotiff_georef(paths[1])
    assert a["geokeys"]["ProjectedCSType"] == 5514 and a["geokeys"]["GTRasterType"] == 1
    assert a["tiepoint"][3:5] == [g.xmin, g.ymax] and a["pixel_scale"][:2] == [2.0, 2.0]
    assert S.compare_georef(a, b)["equal"]
    b2 = dict(b, transform=[2.0, 0.0, g.xmin + 2, 0.0, -2.0, g.ymax])
    assert not S.compare_georef(a, b2)["equal"]


def test_parse_geokeys():
    d = (1, 1, 0, 3, 1024, 0, 1, 1, 2049, 34737, 7, 27, 3072, 0, 1, 5514)
    assert S.parse_geokeys(d) == {1024: 1, 3072: 5514}


# ---- provenance draw ----

def test_draw_is_deterministic_distinct_and_order_independent():
    ids = [f"{a:02d}-{b:02d}-{c:02d}" for a in range(2, 6) for b in range(10, 25) for c in range(1, 15)]
    d1 = S.draw_provenance_sample(CFG, ids)
    d2 = S.draw_provenance_sample(CFG, list(reversed(ids)))
    assert d1 == d2 and len(d1) == 12 == len(set(d1)) and set(d1) <= set(ids)
    assert S.draw_provenance_sample(CFG, ids, seed=1) != d1
    assert S.draw_provenance_sample(CFG, ids, n=3) == S.draw_provenance_sample(CFG, ids, n=3)
    with pytest.raises(ValueError):
        S.draw_provenance_sample(CFG, ids[:5])


# ---- manifest ----

def test_check_manifest(tmp_path):
    (tmp_path / "dmr5g").mkdir()
    files = {"a.gpkg": b"alpha", "dmr5g/x.tif": b"x" * (3 << 20), "dmr5g/y.tif": b"why"}
    for rel, b in files.items():
        (tmp_path / rel).write_bytes(b)
    (tmp_path / "dmr5g" / "fetch_log.jsonl").write_text("{}\n")
    rows = [{"file": rel, "sha256": hashlib.sha256(b).hexdigest()} for rel, b in files.items()]
    rows[2]["sha256"] = "0" * 64
    rows.append({"file": "dmr5g/z.tif", "sha256": "1" * 64})
    seen = []
    r = S.check_manifest(tmp_path, rows, on_chunk=seen.append)
    assert (r["n_listed"], r["n_ok"], r["n_mismatch"], r["n_missing"]) == (4, 2, 1, 1) and not r["ok"]
    assert r["mismatches"][0]["file"] == "dmr5g/y.tif" and r["missing"] == ["dmr5g/z.tif"]
    assert [u["file"] for u in r["unlisted"]] == ["dmr5g/fetch_log.jsonl"]
    assert sum(seen) == sum(len(b) for b in files.values())
    assert S.sha256_file(tmp_path / "dmr5g/x.tif", chunk_bytes=1000) == hashlib.sha256(b"x" * (3 << 20)).hexdigest()


# ---- ArcGIS helpers (fake client) ----

class _Resp:
    def __init__(self, body):
        self.content = json.dumps(body).encode("utf-8")


class _FakeLayer:
    """Answers returnIdsOnly, returnCountOnly and object-id range queries for ids 1..n minus a few."""

    def __init__(self, ids):
        self.ids = sorted(ids)
        self.calls = []

    def _range(self, where):
        lo = int(where.split(">=")[1].split()[0])
        hi = int(where.split("<=")[1].split()[0])
        return [i for i in self.ids if lo <= i <= hi]

    def get_json(self, url, params=None):
        self.calls.append(params)
        if params.get("returnIdsOnly") == "true":
            return {"objectIds": list(reversed(self.ids))}
        if params.get("returnCountOnly") == "true":
            return {"count": len(self.ids)}
        raise AssertionError(params)

    def get(self, url, params=None):
        self.calls.append(params)
        assert "outSR" not in params
        feats = [{"attributes": {"objectid": i}, "geometry": {"rings": [[[0, 0], [0, 1], [1, 1], [0, 0]]]}}
                 for i in self._range(params["where"])]
        return _Resp({"spatialReference": {"wkid": 102067, "latestWkid": 5514}, "features": feats})


def test_query_pages_paginates_by_object_id(tmp_path):
    ids = [i for i in range(1, 2502) if i % 97]
    layer = _FakeLayer(ids)
    assert S.count_features(layer, "u") == len(ids)
    res = S.query_pages(layer, "u", "1=1", "*", "json", 1000, "objectid", tmp_path)
    assert res["n_ids"] == len(ids) and res["n_pages"] == 3
    assert [p["n"] for p in res["pages"]] == [1000, 1000, len(ids) - 2000]
    got = [f["attributes"]["objectid"] for p in res["pages"]
           for f in json.loads((tmp_path / p["file"]).read_text())["features"]]
    assert got == ids


def test_query_pages_detects_short_page(tmp_path):
    layer = _FakeLayer(range(1, 11))
    orig = layer.get

    def short(url, params=None):
        r = orig(url, params)
        body = json.loads(r.content)
        body["features"] = body["features"][:-1]
        return _Resp(body)

    layer.get = short
    with pytest.raises(RuntimeError):
        S.query_pages(layer, "u", "1=1", "*", "json", 4, "objectid", tmp_path)


def test_esri_pages_to_frame(tmp_path):
    page = {"spatialReference": {"wkid": 102067, "latestWkid": 5514}, "geometryType": "esriGeometryPolygon",
            "fields": [{"name": "objectid", "type": "esriFieldTypeOID"},
                       {"name": "subregion", "type": "esriFieldTypeString", "length": 50}],
            "features": [{"attributes": {"objectid": 7, "subregion": "Vulkanity vnějších Karpat"},
                          "geometry": {"rings": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]],
                                                 [[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]]}}]}
    p = tmp_path / "page_0000.json"
    p.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
    gdf = S.esri_pages_to_frame([p], 5514)
    assert gdf.crs.to_epsg() == 5514 and len(gdf) == 1
    assert gdf.geometry.iloc[0].area == pytest.approx(100 - 36)
    assert gdf["subregion"].iloc[0] == "Vulkanity vnějších Karpat"


def test_sql_in_quotes():
    assert S.sql_in("subregion", ["a", "b'c"]) == "subregion IN ('a', 'b''c')"


# ---- redaction of the committed results (licence of row 3) ----

def test_redact_coordinates_removes_sheet_grids():
    import importlib.util

    spec = importlib.util.spec_from_file_location("provenance_check",
                                                  Paths().root / "scripts" / "provenance_check.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    g = S.snap_out(-606223.1, -1091807.5, -600528.3, -1087021.0, origin=O)
    meta = {"extent": {"xmin": -874935.2}, "maxRecordCount": 1000}
    result = {"dtm": {"per_sheet": [{"sheet": "14-34-07", "grid": g.as_dict(),
                                     "request": {"url": "u", "params": S.export_image_params(g, 5514)}}]},
              "grids": {"mismatches": [{"sheet": "14-34-07", "recipe_vs_fetch_log": {"d_w": 1},
                                        "grid_recipe": g.as_dict(), "grid_fetch_log": g.as_dict(),
                                        "fetch_log": "corner (-606223.61, -1087020.12) is 0.01 m off the lattice"}]},
              "sources": {"sheets": {"metadata": dict(meta)}, "geology": {"metadata": dict(meta)},
                          "register": {"metadata": dict(meta)}}}
    out = json.dumps(pc.redact_coordinates(result))
    assert "606223" not in out and "1087020" not in out and "874935" in out  # register extent (CC BY) stays
    assert result["dtm"]["per_sheet"][0]["grid"]["xmin"] == g.xmin  # the input is not modified
    red = json.loads(out)
    assert red["dtm"]["per_sheet"][0]["grid"] == {"w": g.w, "h": g.h}
    assert red["grids"]["mismatches"][0]["recipe_vs_fetch_log"] == {"d_w": 1}
    assert "extent" not in red["sources"]["sheets"]["metadata"] and "extent" not in red["sources"]["geology"]["metadata"]


# ---- real data (skipped without data/raw_data) ----

@pytest.mark.needs_data
@pytest.mark.skipif(not Paths().dtm_fetch_log.exists(), reason="data/raw_data is not available")
def test_recipe_matches_fetch_log_for_all_sheets():
    from landslide.config import load_config

    cfg = load_config()
    p = Paths()
    gdf = S.load_sheets(p.surveyed_sheets)
    log = S.read_fetch_log(p.dtm_fetch_log)
    assert len(gdf) == 860 and set(gdf["sheet"]) == set(log)
    for r in gdf.itertuples():
        rec = log[r.sheet]
        want = S.grid_from_corner(rec["xmin"], rec["ymax"], rec["w"], rec["h"], origin=S.grid_origin(cfg))
        assert S.dtm_grid(r.geometry.bounds, cfg).key() == want.key(), r.sheet
