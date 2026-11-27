"""Synthetic tests for `landslide.susceptibility` (row 5): the export request, PNG -> class decoding, the mid/high
overlap rule, the class raster on the DTM grid, the service checks and the inspection helpers. No network."""

import importlib.util
import io
import json

import numpy as np
import pytest
from PIL import Image

from landslide import sheets as S
from landslide import susceptibility as SU
from landslide.config import load_config
from landslide.paths import Paths

CFG = load_config()
SRC = SU.source(CFG)  # BRIEF.md note 5, from the config's sources.susceptibility block
ORIGIN = S.grid_origin(CFG)
ALPHA = SRC.alpha_min
GRID = S.snap_out(-514749.6, -1146860.12, -514749.6 + 2 * 7, -1146860.12 + 2 * 5, origin=ORIGIN)  # 7 x 5 px


def _png(rgba: np.ndarray, mode: str = "RGBA") -> bytes:
    buf = io.BytesIO()
    im = Image.fromarray(rgba, "RGBA")
    if mode != "RGBA":
        im = im.convert(mode)
    im.save(buf, format="PNG")
    return buf.getvalue()


def _rgba(alpha: np.ndarray, colour=(255, 255, 0)) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.uint8)
    out = np.zeros((*a.shape, 4), np.uint8)
    out[..., :3] = colour
    out[..., 3] = a
    return out


# ---- request ----

def test_source_is_note_5():
    assert SRC.service == "https://mapy.geology.cz/arcgis/rest/services/Geohazardy/sesuvna_nachylnost/MapServer"
    assert SRC.export_url == SRC.service + "/export" and SRC.layer_url == SRC.service + "/0"
    assert SRC.classes == {"mid": (1, "Třída střední%"), "high": (2, "Třída vysoké%")}
    assert (SRC.low_code, SRC.low_pattern, SRC.alpha_min, SRC.dpi, SRC.image_format) == (0, "Třída nízké%", 128, 96,
                                                                                         "png32")


def test_export_params_follow_note_5_without_outsr():
    p = SU.export_params(SRC, GRID, 5514, "mid")
    assert p == {"bbox": "-514749.6,-1146860.12,-514735.6,-1146850.12", "bboxSR": 5514, "imageSR": 5514,
                 "size": "7,5", "format": "png32", "transparent": "true", "layers": "show:0", "dpi": 96,
                 "layerDefs": '{"0": "struktura LIKE \'Třída střední%\'"}', "f": "image"}
    assert json.loads(SU.export_params(SRC, GRID, 5514, "high")["layerDefs"]) == \
        {"0": "struktura LIKE 'Třída vysoké%'"}
    assert SU.export_params(SRC, GRID, 5514, "high", f="json")["f"] == "json"
    assert not any(k.lower() == "outsr" for k in p)


def test_check_export_json():
    body = {"width": 7, "height": 5, "scale": 7559.04,
            "extent": {"xmin": GRID.xmin, "ymin": GRID.ymin, "xmax": GRID.xmax, "ymax": GRID.ymax,
                       "spatialReference": {"wkid": 102067, "latestWkid": 5514}}}
    assert SU.check_export_json(body, GRID, 5514)["ok"]
    shifted = json.loads(json.dumps(body))
    shifted["extent"]["xmin"] += 2.0
    r = SU.check_export_json(shifted, GRID, 5514)
    assert not r["ok"] and not r["extent_equal"] and r["extent_minus_request_m"][0] == 2.0
    assert not SU.check_export_json(dict(body, width=8), GRID, 5514)["ok"]


# ---- PNG -> class ----

def test_decode_png_roundtrip_and_threshold():
    alpha = np.array([[0, 64, 127, 128, 191, 254, 255]] * 5, np.uint8)
    rgba = SU.decode_png(_png(_rgba(alpha)), (7, 5))
    assert rgba.shape == (5, 7, 4) and rgba.dtype == np.uint8
    assert np.array_equal(rgba[..., 3], alpha)
    assert SU.class_mask(rgba[..., 3], ALPHA)[0].tolist() == [False, False, False, True, True, True, True]


def test_decode_png_rejects_wrong_size_mode_and_non_png():
    body = _png(_rgba(np.zeros((5, 7))))
    with pytest.raises(SU.UnexpectedResponse, match="size"):
        SU.decode_png(body, (5, 7))  # (width, height) swapped
    with pytest.raises(SU.UnexpectedResponse, match="RGBA"):
        SU.decode_png(_png(_rgba(np.zeros((5, 7))), mode="RGB"), (7, 5))
    with pytest.raises(SU.UnexpectedResponse, match="not a PNG"):
        SU.decode_png(b'{"error": {"code": 500, "message": "Error exporting map"}}', (7, 5))


def test_alpha_histogram_and_colour_check():
    alpha = np.array([[0, 0, 128, 255], [64, 191, 255, 255]], np.uint8)
    assert SU.alpha_histogram(alpha) == {"0": 2, "64": 1, "128": 1, "191": 1, "255": 3}
    rgba = _rgba(alpha, colour=(255, 0, 0))
    ok = SU.colour_check(rgba, (255, 0, 0), ALPHA)
    assert ok["n_class"] == 5 and ok["n_other_colour"] == 0
    rgba[0, 2, :3] = (56, 168, 0)  # a "low" pixel drawn: the filter was not applied
    bad = SU.colour_check(rgba, (255, 0, 0), ALPHA)
    assert bad["n_other_colour"] == 1 and "56,168,0" in bad["colours"]
    rgba[1, 0, :3] = (1, 2, 3)  # alpha 64: not a class pixel, so its colour does not matter
    assert SU.colour_check(rgba, (255, 0, 0), ALPHA)["n_other_colour"] == 1


# ---- overlap rule ----

def test_combine_codes_and_overlap_rule():
    mid = np.array([[255, 255, 128, 0, 0, 191]], np.uint8)
    high = np.array([[0, 255, 128, 255, 127, 128]], np.uint8)
    codes, st = SU.combine(SRC, {"mid": mid, "high": high})
    assert codes.dtype == np.uint8
    assert codes.tolist() == [[1, 2, 2, 2, 0, 2]]  # in both renders -> 2 (high wins)
    assert st["n_overlap"] == 3 and st["n_mid_render"] == 4 and st["n_high_render"] == 4
    assert st["overlap_alpha_pairs"] == {"128,128": 1, "191,128": 1, "255,255": 1}
    assert st["n_code"] == {"0": 1, "1": 1, "2": 4} and st["n_px"] == 6


def test_combine_does_not_depend_on_dict_order_and_checks_shapes():
    mid = np.full((2, 2), 255, np.uint8)
    high = np.full((2, 2), 255, np.uint8)
    a, _ = SU.combine(SRC, {"mid": mid, "high": high})
    b, _ = SU.combine(SRC, {"high": high, "mid": mid})
    assert np.array_equal(a, b) and (a == 2).all()
    with pytest.raises(ValueError):
        SU.combine(SRC, {"mid": mid, "high": high[:1]})


def test_combine_threshold_is_128():
    mid = np.array([[127, 128]], np.uint8)
    codes, st = SU.combine(SRC, {"mid": mid, "high": np.zeros_like(mid)})
    assert codes.tolist() == [[0, 1]] and st["n_overlap"] == 0


# ---- class raster on the DTM grid ----

def _dtm(tmp_path, grid):
    import rasterio
    from rasterio.transform import Affine

    p = tmp_path / "dtm.tif"
    with rasterio.open(p, "w", driver="GTiff", width=grid.w, height=grid.h, count=1, dtype="float32",
                       crs="EPSG:5514", transform=Affine(*grid.transform), nodata=float("nan")) as ds:
        ds.write(np.full((grid.h, grid.w), 300.0, np.float32), 1)
    return p


def test_read_dtm_grid_and_write_class_raster(tmp_path):
    import rasterio

    dtm = SU.read_dtm_grid(_dtm(tmp_path, GRID), 2.0, ORIGIN)
    assert dtm.grid.key() == GRID.key() and (dtm.width, dtm.height) == (7, 5)
    codes = np.arange(35, dtype=np.uint8).reshape(5, 7) % 3
    out = tmp_path / "s.tif"
    SU.write_class_raster(out, codes, dtm, tags={"SHEET": "25-14-17"})
    with rasterio.open(out) as ds:
        assert ds.dtypes == ("uint8",) and ds.count == 1 and ds.nodata is None
        assert ds.profile["compress"] == "deflate"
        assert ds.transform == dtm.transform and ds.crs == dtm.crs and (ds.width, ds.height) == (7, 5)
        assert np.array_equal(ds.read(1), codes)
        assert ds.tags()["SHEET"] == "25-14-17" and "2=high" in ds.tags()["CLASSES"]
    assert not out.with_name(out.name + ".part").exists()
    with pytest.raises(ValueError):
        SU.write_class_raster(out, codes[:4], dtm)
    with pytest.raises(ValueError):
        SU.write_class_raster(out, codes.astype(np.int16), dtm)


def test_read_dtm_grid_rejects_off_lattice(tmp_path):
    import rasterio
    from rasterio.transform import Affine

    p = tmp_path / "off.tif"
    with rasterio.open(p, "w", driver="GTiff", width=3, height=3, count=1, dtype="float32", crs="EPSG:5514",
                       transform=Affine(2.0, 0, GRID.xmin + 0.3, 0, -2.0, GRID.ymax)) as ds:
        ds.write(np.zeros((3, 3), np.float32), 1)
    with pytest.raises(S.OffGrid):
        SU.read_dtm_grid(p, 2.0, ORIGIN)


# ---- one sheet with a fake client ----

class _Resp:
    def __init__(self, body: bytes, ctype: str = "image/png"):
        self.content = body
        self.headers = {"Content-Type": ctype}
        self.status_code = 200


class _FakeExport:
    def __init__(self, planes, ctype="image/png"):
        self.planes = planes
        self.ctype = ctype
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        cls = "mid" if "střední" in params["layerDefs"] else "high"
        colour = (255, 255, 0) if cls == "mid" else (255, 0, 0)
        return _Resp(_png(_rgba(self.planes[cls], colour)), self.ctype)


def test_render_sheet_end_to_end(tmp_path):
    import rasterio

    dtm_path = _dtm(tmp_path, GRID)
    mid = np.zeros((5, 7), np.uint8)
    high = np.zeros((5, 7), np.uint8)
    mid[:, :3] = 255
    high[:, 3:5] = 255
    mid[0, 3] = high[0, 3] = 128  # a boundary tie
    client = _FakeExport({"mid": mid, "high": high})
    colours = {"mid": (255, 255, 0), "high": (255, 0, 0)}
    rec = SU.render_sheet(SRC, client, "25-14-17", dtm_path, tmp_path / "out" / "25_14_17.tif", tmp_path / "png",
                          5514, 2.0, ORIGIN, colours)
    assert [c[1]["size"] for c in client.calls] == ["7,5", "7,5"]
    assert all(c[0] == SRC.export_url for c in client.calls)
    assert rec["n_overlap"] == 1 and rec["overlap_alpha_pairs"] == {"128,128": 1}
    assert rec["n_code"] == {"0": 10, "1": 15, "2": 10}
    assert (tmp_path / "png" / "25_14_17_mid.png").read_bytes()[:8] == SU.PNG_SIGNATURE
    with rasterio.open(tmp_path / "out" / "25_14_17.tif") as ds:
        codes = ds.read(1)
    assert codes[0, 3] == 2 and (codes[:, :3] == 1).all() and (codes[1:, 3:5] == 2).all() and (codes[:, 5:] == 0).all()


def test_render_sheet_stops_on_size_mismatch_error_body_and_colour(tmp_path):
    dtm_path = _dtm(tmp_path, GRID)
    colours = {"mid": (255, 255, 0), "high": (255, 0, 0)}
    small = {"mid": np.zeros((4, 7), np.uint8), "high": np.zeros((4, 7), np.uint8)}
    with pytest.raises(SU.UnexpectedResponse, match="size"):
        SU.render_sheet(SRC, _FakeExport(small), "s", dtm_path, tmp_path / "a.tif", tmp_path / "png", 5514, 2.0,
                        ORIGIN, colours)
    ok = {"mid": np.zeros((5, 7), np.uint8), "high": np.zeros((5, 7), np.uint8)}
    with pytest.raises(SU.UnexpectedResponse, match="image/png"):
        SU.render_sheet(SRC, _FakeExport(ok, ctype="text/html"), "s", dtm_path, tmp_path / "b.tif",
                        tmp_path / "png", 5514, 2.0, ORIGIN, colours)
    drawn = {"mid": np.full((5, 7), 255, np.uint8), "high": np.zeros((5, 7), np.uint8)}
    with pytest.raises(SU.UnexpectedResponse, match="colour"):
        SU.render_sheet(SRC, _FakeExport(drawn), "s", dtm_path, tmp_path / "c.tif", tmp_path / "png", 5514, 2.0,
                        ORIGIN, {"mid": (255, 0, 0), "high": (255, 0, 0)})
    assert not any((tmp_path / n).exists() for n in ("a.tif", "b.tif", "c.tif"))


# ---- service checks ----

LAYER_META = {"drawingInfo": {"renderer": {"uniqueValueInfos": [
    {"value": "Třída nízké náchylnosti – a", "symbol": {"color": [56, 168, 0, 255]}},
    {"value": "Třída střední náchylnosti – b", "symbol": {"color": [255, 255, 0, 255]}},
    {"value": "Třída vysoké náchylnosti – c", "symbol": {"color": [255, 0, 0, 255]}}]}}}


def test_class_colours_and_patterns():
    assert SU.class_colours(SRC, LAYER_META) == {"mid": (255, 255, 0), "high": (255, 0, 0)}
    vals = [i["value"] for i in LAYER_META["drawingInfo"]["renderer"]["uniqueValueInfos"]]
    m = SU.match_patterns(SRC, vals)
    assert m == {"mid": [vals[1]], "high": [vals[2]], "low": [vals[0]]}


def test_check_service_flags_problems():
    meta = {"pattern_matches": {"mid": ["a"], "high": ["b"]}, "class_colours": {"mid": [1, 1, 1], "high": [2, 2, 2]},
            "service": {"maxImageWidth": 4096, "maxImageHeight": 4096, "supportedImageFormatTypes": "PNG32,PNG",
                        "spatialReference": {"wkid": 102067, "latestWkid": 5514}}}
    assert SU.check_service(SRC, meta, 2902, 2496, 5514) == []
    assert len(SU.check_service(SRC, meta, 5000, 2496, 5514)) == 1
    bad = json.loads(json.dumps(meta))
    bad["pattern_matches"]["mid"] = []
    bad["class_colours"]["high"] = None
    bad["service"]["supportedImageFormatTypes"] = "PNG,JPG"
    assert len(SU.check_service(SRC, bad, 100, 100, 5514)) == 3


# ---- inspection helpers ----

def test_code_from_values():
    assert SU.code_from_values(SRC, []) == 0
    assert SU.code_from_values(SRC, ["Třída nízké náchylnosti – x"]) == 0
    assert SU.code_from_values(SRC, ["Třída střední náchylnosti – x"]) == 1
    assert SU.code_from_values(SRC, ["Třída vysoké náchylnosti – x", "Třída nízké náchylnosti – y"]) == 2
    assert SU.code_from_values(SRC, ["Třída střední náchylnosti", "Třída vysoké náchylnosti"]) == 2


def test_pixel_centre():
    assert SU.pixel_centre(GRID, 0, 0) == (GRID.xmin + 1.0, GRID.ymax - 1.0)
    assert SU.pixel_centre(GRID, 4, 6) == (GRID.xmin + 13.0, GRID.ymax - 9.0)


def test_inspection_points_categories():
    mid = np.zeros((30, 30), np.uint8)
    high = np.zeros((30, 30), np.uint8)
    mid[2:14, 2:14] = 255
    mid[1, 2:14] = 128  # partial edge row
    high[16:28, 16:28] = 255
    mid[20:24, 20:24] = 255  # overlapping polygons
    pts = SU.inspection_points(SRC, {"mid": mid, "high": high}, 5)
    assert set(pts) == {"mid_interior", "mid_edge", "high_interior", "high_edge", "none_interior", "none_edge",
                        "both_full"}
    for r, c in pts["both_full"]:
        assert mid[r, c] == 255 and high[r, c] == 255
    for r, c in pts["high_interior"]:
        assert (high[r - 2:r + 3, c - 2:c + 3] == 255).all() and mid[r, c] == 0
    for r, c in pts["mid_interior"]:
        assert (mid[r - 2:r + 3, c - 2:c + 3] == 255).all()
    for r, c in pts["mid_edge"]:
        assert mid[r, c] == 255 and min(mid[r - 1, c], mid[r + 1, c], mid[r, c - 1], mid[r, c + 1]) < 255
    for r, c in pts["none_edge"]:
        assert mid[r, c] == 0 and high[r, c] == 0
    assert all(0 < len(v) <= 5 for v in pts.values())
    assert pts == SU.inspection_points(SRC, {"mid": mid, "high": high}, 5)  # deterministic


# ---- the script's log handling ----

def _script():
    spec = importlib.util.spec_from_file_location("fetch_susceptibility",
                                                  Paths().root / "scripts" / "fetch_susceptibility.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_jsonl_resume_ignores_truncated_last_line(tmp_path):
    fs = _script()
    p = tmp_path / "log.jsonl"
    fs.append_jsonl(p, {"sheet": "a", "v": 1})
    fs.append_jsonl(p, {"sheet": "b", "v": 1})
    fs.append_jsonl(p, {"sheet": "a", "v": 2})
    with p.open("a", encoding="utf-8") as f:
        f.write('{"sheet": "c", "v"')  # a killed run
    recs = fs.read_jsonl(p)
    assert [r["sheet"] for r in recs] == ["a", "b", "a"]
    assert fs.latest_by_sheet(recs)["a"]["v"] == 2
    p.write_text('{"sheet": "a"}\n{bad\n{"sheet": "b"}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        fs.read_jsonl(p)


def test_update_sources_keeps_other_sections(tmp_path):
    fs = _script()
    p = tmp_path / "sources.json"
    fs.update_sources(p, "zabaged", {"n": 1})
    fs.update_sources(p, "susceptibility", {"n": 2})
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["task"] == "P1-07" and d["zabaged"] == {"n": 1} and d["susceptibility"] == {"n": 2}
