"""Synthetic tests for `scripts/fetch_zabaged.py` (row 6): envelope queries paginated by object id, deduplication of
features returned for several sheets, and the GeoPackage build. A fake layer answers the queries; no network."""

import importlib.util
import json

import numpy as np
import pytest

from landslide import sheets as S
from landslide.paths import Paths


def _script():
    spec = importlib.util.spec_from_file_location("fetch_zabaged", Paths().root / "scripts" / "fetch_zabaged.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FZ = _script()
OID = "OBJECTID"


def _square(x, y, s=10.0):
    """A clockwise (outer) Esri ring."""
    return [[x, y], [x, y + s], [x + s, y + s], [x + s, y], [x, y]]


class _Resp:
    def __init__(self, body):
        self.content = json.dumps(body, ensure_ascii=False).encode("utf-8")


class FakeLayer:
    """An ArcGIS polygon layer: features with object ids and rings; answers ids-only and object-id range queries,
    both filtered by an envelope, like `MapServer/<id>/query`."""

    def __init__(self, feats):
        self.feats = feats  # {oid: {"rings": [...], "fid_zbg": str}}
        self.calls = []

    @staticmethod
    def _bbox(rings):
        xy = np.array([p for r in rings for p in r], float)
        return xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()

    def _in_envelope(self, params):
        assert params["geometryType"] == "esriGeometryEnvelope" and params["inSR"] == 5514
        assert params["spatialRel"] == "esriSpatialRelIntersects"
        assert not any(k.lower() == "outsr" for k in params)
        ex0, ey0, ex1, ey1 = (float(v) for v in params["geometry"].split(","))
        out = []
        for oid, f in self.feats.items():
            x0, y0, x1, y1 = self._bbox(f["rings"])
            if x0 <= ex1 and x1 >= ex0 and y0 <= ey1 and y1 >= ey0:
                out.append(oid)
        return out

    def get_json(self, url, params=None):
        self.calls.append(dict(params))
        assert params.get("returnIdsOnly") == "true"
        ids = self._in_envelope(params)
        return {"objectIdFieldName": OID, "objectIds": list(reversed(ids)) or None}

    def get(self, url, params=None):
        self.calls.append(dict(params))
        w = params["where"]
        lo = int(w.split(">=")[1].split()[0])
        hi = int(w.split("<=")[1].split()[0])
        ids = [i for i in self._in_envelope(params) if lo <= i <= hi]
        return _Resp({
            "objectIdFieldName": OID, "geometryType": "esriGeometryPolygon",
            "spatialReference": {"wkid": 102067, "latestWkid": 5514},
            "fields": [{"name": OID, "type": "esriFieldTypeOID", "alias": OID},
                       {"name": "fid_zbg", "type": "esriFieldTypeString", "alias": "fid_zbg", "length": 20}],
            "features": [{"attributes": {OID: i, "fid_zbg": self.feats[i]["fid_zbg"]},
                          "geometry": {"rings": self.feats[i]["rings"]}} for i in sorted(ids)]})


def _layer(n=25, step=100.0):
    """n features on a row, 100 m apart; id 7 is a two-part polygon (MultiPolygon)."""
    feats = {}
    for k in range(n):
        oid = 3 * k + 1  # gaps in the ids
        feats[oid] = {"rings": [_square(k * step, 0.0)], "fid_zbg": f"Z{oid}"}
    feats[7]["rings"] = [_square(2 * step, 0.0), _square(2 * step + 20, 0.0, 5.0)]
    return feats


META = {"maxRecordCount": 4, "fields": [[OID, "esriFieldTypeOID"], ["fid_zbg", "esriFieldTypeString"]]}


def test_envelope_client_adds_the_filter_and_refuses_clashes():
    calls = []

    class C:
        def get(self, url, params=None):
            calls.append(params)

        def get_json(self, url, params=None):
            calls.append(params)

    env = FZ.EnvelopeClient(C(), (-1.5, -2.0, 3.25, 4.0), 5514)
    env.get("u", params={"where": "1=1"})
    env.get_json("u", params={"f": "json"})
    assert calls[0] == {"where": "1=1", "geometry": "-1.5,-2.0,3.25,4.0", "geometryType": "esriGeometryEnvelope",
                        "inSR": 5514, "spatialRel": "esriSpatialRelIntersects"}
    assert calls[1]["f"] == "json" and calls[1]["geometry"] == "-1.5,-2.0,3.25,4.0"
    with pytest.raises(ValueError):
        env.get("u", params={"geometry": "0,0,1,1"})


def test_fetch_pair_paginates_inside_the_envelope(tmp_path):
    layer = FakeLayer(_layer())
    # features 3..12 (x 300..1210) intersect; ids 10, 13, ..., 37 -> 10 ids, pages of 4
    rec = FZ.fetch_pair(layer, 130, META, "01-01-01", (305.0, -5.0, 1205.0, 5.0), 5514, tmp_path / "p")
    want = [3 * k + 1 for k in range(3, 13)]
    assert rec["n_ids"] == 10 and rec["n_pages"] == 3 and [p["n"] for p in rec["pages"]] == [4, 4, 2]
    got = [f["attributes"][OID] for p in rec["pages"]
           for f in json.loads((tmp_path / "p" / p["file"]).read_text(encoding="utf-8"))["features"]]
    assert got == want
    pages = [c for c in layer.calls if "returnIdsOnly" not in c]
    assert all(c["orderByFields"] == OID and c["f"] == "json" and c["outFields"] == "*" for c in pages)
    assert rec["layer"] == 130 and rec["sheet"] == "01-01-01"


def test_fetch_pair_empty_envelope_leaves_no_directory(tmp_path):
    rec = FZ.fetch_pair(FakeLayer(_layer()), 129, META, "01-01-01", (-500.0, -500.0, -400.0, -400.0), 5514,
                        tmp_path / "empty")
    assert rec["n_ids"] == 0 and rec["n_pages"] == 0 and rec["pages"] == []
    assert not (tmp_path / "empty").exists()


def test_fetch_pair_detects_a_short_page(tmp_path):
    layer = FakeLayer(_layer())
    orig = layer.get

    def short(url, params=None):
        body = json.loads(orig(url, params).content)
        body["features"] = body["features"][:-1]
        return _Resp(body)

    layer.get = short
    with pytest.raises(RuntimeError):
        FZ.fetch_pair(layer, 130, META, "s", (0.0, 0.0, 1000.0, 5.0), 5514, tmp_path / "p")


def _two_sheets(tmp_path):
    """Two overlapping sheet envelopes over the fake layer, fetched into the pages layout of the script."""
    layer = FakeLayer(_layer())
    root = tmp_path / "pages"
    recs = []
    for sheet, bbox in (("01-01-01", (0.0, -5.0, 1000.0, 5.0)), ("01-01-02", (600.0, -5.0, 1800.0, 5.0))):
        recs.append(FZ.fetch_pair(layer, 130, META, sheet, bbox, 5514,
                                  root / "130" / sheet.replace("-", "_")))
    return layer, root, recs


def test_dedup_and_build_layer(tmp_path):
    import geopandas as gpd

    layer, root, recs = _two_sheets(tmp_path)
    # sheet 1: features k = 0..10 (11); sheet 2: k = 6..18 (13); k = 6..10 overlap (5)
    assert [r["n_ids"] for r in recs] == [11, 13]
    st = FZ.build_layer(130, recs, root, tmp_path / "130.gpkg", 5514, OID)
    assert (st["n_raw"], st["n_unique"], st["n_duplicate_rows"], st["n_conflicting_ids"]) == (24, 19, 5, 0)
    assert st["geometry"]["types"] == {"Polygon": 18, "MultiPolygon": 1}
    assert st["roundtrip_equal"] and st["n_unique_fid_zbg"] == 19
    assert st["n_sheets_queried"] == 2 and st["n_sheets_nonempty"] == 2 and st["max_features_per_sheet"] == 13
    assert st["declared_geometry_types"] == {"esriGeometryPolygon": st["n_pages"]}
    g = gpd.read_file(tmp_path / "130.gpkg")
    assert g.crs.to_epsg() == 5514 and g[OID].tolist() == sorted(layer.feats)[:19]
    assert g.loc[g[OID] == 7].geometry.iloc[0].geom_type == "MultiPolygon"
    assert g.loc[g[OID] == 1].geometry.iloc[0].area == pytest.approx(100.0)


def test_build_layer_checks_page_hashes(tmp_path):
    _, root, recs = _two_sheets(tmp_path)
    p = root / "130" / "01_01_02" / recs[1]["pages"][0]["file"]
    p.write_bytes(p.read_bytes().replace(b'"Z', b'"Y'))
    with pytest.raises(ValueError, match="sha256"):
        FZ.build_layer(130, recs, root, tmp_path / "130.gpkg", 5514, OID)


def test_dedup_flags_conflicting_versions():
    import geopandas as gpd
    from shapely.geometry import box

    g = gpd.GeoDataFrame({OID: [5, 1, 5, 2, 2], "a": ["x", "y", "x", "z", "z2"]},
                         geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(0, 0, 1, 1), box(3, 3, 4, 4),
                                   box(3, 3, 4, 4)], crs="EPSG:5514")
    out, st = FZ.dedup(g, OID)
    assert out[OID].tolist() == [1, 2, 5]
    assert (st["n_raw"], st["n_unique"], st["n_duplicate_rows"], st["n_conflicting_ids"]) == (5, 3, 2, 1)
    assert st["conflicting_ids_examples"] == [2]
    g2 = g.copy()
    g2.loc[3, "a"] = "z2"
    g2.loc[3, "geometry"] = box(3, 3, 4, 5)
    assert FZ.dedup(g2, OID)[1]["conflicting_ids_examples"] == [2]  # same attributes, different geometry


def test_layers_and_keys():
    assert list(FZ.LAYERS) == [118, 119, 130, 129, 79, 75]
    assert Paths().zabaged(FZ.layer_key(129)).name == "129.gpkg"
    assert FZ.envelope_params(S.snap_out(0.4, -0.12, 4.4, 3.88).bbox, 5514)["geometry"] == "0.4,-0.12,4.4,3.88"


# ---- weights: the offline guard (in a subprocess, so this test process keeps its network) ----

def test_weights_offline_guard_blocks_connections():
    import subprocess
    import sys

    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('fw', r'{Paths().root / 'scripts' / 'fetch_weights.py'}')\n"
            "fw = importlib.util.module_from_spec(spec); spec.loader.exec_module(fw)\n"
            "fw.disable_network()\n"
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('127.0.0.1', 9))\n"
            "    sys.exit(2)\n"
            "except fw.NetworkDisabled:\n"
            "    pass\n"
            "sys.exit(0 if fw.network_is_disabled() else 3)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
