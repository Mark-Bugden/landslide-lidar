"""The ČGS landslide-susceptibility map (row 5), rendered per class on each sheet's DTM grid (task P1-07).

BRIEF.md note 5: the vector layer stores huge dissolved polygons, so it is not queried. Instead each class is
rendered with `MapServer/export` on the sheet's DTM grid:

- `bbox` and `size` of the DTM raster, `bboxSR = imageSR = 5514`, `format=png32`, `transparent=true`,
  `layers=show:0`, `dpi=96`, and `layerDefs={"0": "struktura LIKE 'Třída střední%'"}` (mid) or
  `'Třída vysoké%'` (high); `f=image` returns the PNG itself;
- a pixel belongs to the class if its alpha is >= 128;
- codes: 0 = low or no data, 1 = mid, 2 = high. A pixel that is in both renders gets 2 (decision in
  `docs/decisions/P1-07.md`); such pixels are counted.

The grid is read from the sheet's package DTM raster (transform and size), never recomputed (P1-03 decision D03).
The class raster is a uint8 deflate GeoTIFF with the DTM raster's transform, size and CRS.

Licence: row 5 has no stated licence. Its rasters, the PNGs and anything per sheet stay under `data/`; only
aggregate numbers may be committed.

The service URL, the class patterns, the alpha threshold and the render parameters are not keys of
`configs/landslide.json`; they are the brief's note 5, kept as module constants below.
"""

from __future__ import annotations

import io
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from landslide import sheets as S

# ---- the brief's note 5 ----

SERVICE_URL = "https://mapy.geology.cz/arcgis/rest/services/Geohazardy/sesuvna_nachylnost/MapServer"
EXPORT_URL = SERVICE_URL + "/export"
LAYER_ID = 0
LAYER_URL = f"{SERVICE_URL}/{LAYER_ID}"
CLASS_FIELD = "struktura"
#: Rendered classes in drawing order: name -> (code, LIKE pattern). Everything else is code 0.
CLASSES: dict[str, tuple[int, str]] = {"mid": (1, "Třída střední%"), "high": (2, "Třída vysoké%")}
#: The class that is not rendered (code 0 together with "no data"); used only to label point-query answers.
LOW_PATTERN = "Třída nízké%"
CODE_LOW = 0
#: A pixel belongs to the rendered class if its alpha is at least this.
ALPHA_MIN = 128
DPI = 96
IMAGE_FORMAT = "png32"
CLASS_LABELS = {0: "low or no data", 1: "mid", 2: "high"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class UnexpectedResponse(RuntimeError):
    """The service answered with something other than the expected image (an error page, a wrong size, ...)."""


# ======================================================================================================================
# Request
# ======================================================================================================================


def layer_def(pattern: str) -> str:
    """`{"0": "struktura LIKE '<pattern>'"}` as JSON (UTF-8, not escaped; requests percent-encodes it)."""
    return json.dumps({str(LAYER_ID): f"{CLASS_FIELD} LIKE '{pattern}'"}, ensure_ascii=False)


def bbox_str(grid: S.Grid) -> str:
    """The grid's xmin, ymin, xmax, ymax with the shortest decimal repr (same format as the DTM recipe)."""
    return ",".join(repr(float(round(v, 6))) for v in grid.bbox)


def export_params(grid: S.Grid, epsg: int, cls: str, f: str = "image") -> dict[str, Any]:
    """BRIEF.md note 5 for class `cls` on `grid`. `f=image` returns the PNG; `f=json` describes the render. No outSR."""
    _, pattern = CLASSES[cls]
    return {
        "bbox": bbox_str(grid),
        "bboxSR": epsg,
        "imageSR": epsg,
        "size": f"{grid.w},{grid.h}",
        "format": IMAGE_FORMAT,
        "transparent": "true",
        "layers": f"show:{LAYER_ID}",
        "dpi": DPI,
        "layerDefs": layer_def(pattern),
        "f": f,
    }


def check_export_json(body: Mapping[str, Any], grid: S.Grid, epsg: int, tol_m: float = 1e-6) -> dict[str, Any]:
    """Compare an `f=json` export answer with the request: same size, same extent, same SR. Returns the checks."""
    ext = body.get("extent") or {}
    sr = ext.get("spatialReference") or {}
    got = tuple(float(ext.get(k, math.nan)) for k in ("xmin", "ymin", "xmax", "ymax"))
    out = {
        "size_equal": (body.get("width"), body.get("height")) == (grid.w, grid.h),
        "extent_equal": all(abs(a - b) <= tol_m for a, b in zip(got, grid.bbox, strict=True)),
        "sr_equal": epsg in (sr.get("latestWkid"), sr.get("wkid")),
        "scale": body.get("scale"),
        "extent_minus_request_m": [round(a - b, 9) for a, b in zip(got, grid.bbox, strict=True)],
    }
    out["ok"] = bool(out["size_equal"] and out["extent_equal"] and out["sr_equal"])
    return out


# ======================================================================================================================
# Decoding
# ======================================================================================================================


def decode_png(body: bytes, size: tuple[int, int]) -> np.ndarray:
    """Decode a png32 response to an (H, W, 4) uint8 RGBA array. `size` is (width, height).

    Raises `UnexpectedResponse` if the body is not a PNG, not RGBA, or not the requested size.
    """
    from PIL import Image

    if body[:8] != PNG_SIGNATURE:
        raise UnexpectedResponse(f"not a PNG: {body[:300]!r}")
    with Image.open(io.BytesIO(body)) as im:
        im.load()
        if im.format != "PNG" or im.mode != "RGBA":
            raise UnexpectedResponse(f"expected an RGBA PNG, got {im.format} {im.mode}")
        if im.size != tuple(size):
            raise UnexpectedResponse(f"PNG size {im.size} != requested {tuple(size)}")
        return np.asarray(im, dtype=np.uint8).copy()


def class_mask(alpha: np.ndarray, alpha_min: int = ALPHA_MIN) -> np.ndarray:
    """Note 5: a pixel belongs to the rendered class if its alpha is >= `alpha_min`."""
    return np.asarray(alpha) >= alpha_min


def alpha_histogram(alpha: np.ndarray) -> dict[str, int]:
    """Counts of each alpha value that occurs, as {"value": count}."""
    h = np.bincount(np.asarray(alpha, dtype=np.uint8).ravel(), minlength=256)
    return {str(v): int(h[v]) for v in np.flatnonzero(h)}


def colour_check(rgba: np.ndarray, colour: tuple[int, int, int] | None,
                 alpha_min: int = ALPHA_MIN) -> dict[str, Any]:
    """The RGB values of the class pixels (alpha >= alpha_min) must all be the class's symbol colour.

    A different colour would mean the `layerDefs` filter was not applied (e.g. a cached map drawn with every class)
    or the symbology changed. Returns the number of class pixels with another colour and the colours seen.
    """
    sel = class_mask(rgba[..., 3], alpha_min)
    rgb = rgba[..., :3][sel]
    if rgb.size == 0:
        return {"n_class": 0, "n_other_colour": 0, "colours": {}}
    cols, cnt = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    seen = {",".join(str(int(v)) for v in c): int(n) for c, n in zip(cols, cnt, strict=True)}
    n_other = int(rgb.shape[0]) if colour is None else int(sum(
        n for c, n in zip(cols, cnt, strict=True) if tuple(int(v) for v in c) != tuple(colour)))
    return {"n_class": int(rgb.shape[0]), "n_other_colour": n_other, "colours": dict(list(seen.items())[:10])}


def combine(alphas: Mapping[str, np.ndarray], alpha_min: int = ALPHA_MIN) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine the per-class alpha planes into the uint8 code raster (0 low/no data, 1 mid, 2 high).

    Overlap rule (docs/decisions/P1-07.md): a pixel in both the mid and the high render gets the higher code, 2.
    The overlap is counted, with a histogram of the (mid alpha, high alpha) pairs of the overlapping pixels, which
    tells boundary ties (128, 128) apart from polygons that really overlap (255, 255).
    """
    names = sorted(CLASSES, key=lambda k: CLASSES[k][0])  # mid, high: higher codes are written last and win
    shapes = {np.asarray(alphas[k]).shape for k in names}
    if len(shapes) != 1:
        raise ValueError(f"alpha planes differ in shape: {shapes}")
    shape = shapes.pop()
    masks = {k: class_mask(alphas[k], alpha_min) for k in names}
    codes = np.full(shape, CODE_LOW, dtype=np.uint8)
    for k in names:
        codes[masks[k]] = CLASSES[k][0]
    overlap = masks["mid"] & masks["high"]
    n_overlap = int(overlap.sum())
    pairs: Counter[str] = Counter()
    if n_overlap:
        am = np.asarray(alphas["mid"])[overlap].astype(np.int32)
        ah = np.asarray(alphas["high"])[overlap].astype(np.int32)
        keys, cnt = np.unique(am * 256 + ah, return_counts=True)
        for k, n in zip(keys, cnt, strict=True):
            pairs[f"{int(k) // 256},{int(k) % 256}"] = int(n)
    counts = np.bincount(codes.ravel(), minlength=3)
    stats = {"n_px": int(codes.size), "n_mid_render": int(masks["mid"].sum()),
             "n_high_render": int(masks["high"].sum()), "n_overlap": n_overlap,
             "overlap_alpha_pairs": dict(sorted(pairs.items())),
             "n_code": {str(c): int(counts[c]) for c in range(3)}}
    return codes, stats


# ======================================================================================================================
# The DTM grid and the class raster
# ======================================================================================================================


@dataclass(frozen=True)
class DtmGrid:
    """A sheet's DTM raster grid as read from the package file: lattice grid, affine transform and CRS."""

    grid: S.Grid
    transform: Any  # rasterio Affine
    crs: Any  # rasterio CRS
    width: int
    height: int


def read_dtm_grid(path: Path | str, cell: float) -> DtmGrid:
    """The grid of a package DTM raster (header only). Raises `sheets.OffGrid` unless it is on the native lattice."""
    import rasterio

    with rasterio.open(path) as ds:
        g = S.grid_from_transform(list(ds.transform)[:6], ds.width, ds.height, cell=cell)
        return DtmGrid(grid=g, transform=ds.transform, crs=ds.crs, width=ds.width, height=ds.height)


def write_class_raster(path: Path, codes: np.ndarray, dtm: DtmGrid, tags: Mapping[str, str] | None = None) -> None:
    """Write the codes as a uint8 deflate GeoTIFF on the DTM grid (same transform, size and CRS). Atomic."""
    import rasterio

    if codes.shape != (dtm.height, dtm.width) or codes.dtype != np.uint8:
        raise ValueError(f"codes {codes.shape} {codes.dtype} do not fit the DTM grid {(dtm.height, dtm.width)}")
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 1, "width": dtm.width, "height": dtm.height,
               "transform": dtm.transform, "crs": dtm.crs, "tiled": True, "blockxsize": 256, "blockysize": 256,
               "compress": "deflate"}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(codes, 1)
        dst.update_tags(**{"CLASSES": "; ".join(f"{k}={v}" for k, v in CLASS_LABELS.items()), **(tags or {})})
    tmp.replace(path)


# ======================================================================================================================
# Service metadata
# ======================================================================================================================

SERVICE_META_KEYS = ("currentVersion", "cimVersion", "serviceDescription", "mapName", "copyrightText",
                     "spatialReference", "units", "supportedImageFormatTypes", "maxImageWidth", "maxImageHeight",
                     "maxRecordCount", "capabilities", "supportsDynamicLayers", "singleFusedMapCache", "minScale",
                     "maxScale", "documentInfo", "serviceItemId")


def class_colours(layer_meta: Mapping[str, Any]) -> dict[str, tuple[int, int, int] | None]:
    """The RGB symbol colour of each rendered class, from the layer's unique-value renderer (None if not found)."""
    infos = (((layer_meta.get("drawingInfo") or {}).get("renderer") or {}).get("uniqueValueInfos")) or []
    out: dict[str, tuple[int, int, int] | None] = {}
    for name, (_, pattern) in CLASSES.items():
        prefix = pattern.rstrip("%")
        hits = [i for i in infos if str(i.get("value", "")).startswith(prefix)]
        out[name] = tuple(int(v) for v in hits[0]["symbol"]["color"][:3]) if len(hits) == 1 else None
    return out


def renderer_summary(layer_meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    infos = (((layer_meta.get("drawingInfo") or {}).get("renderer") or {}).get("uniqueValueInfos")) or []
    return [{"value": i.get("value"), "description": i.get("description"),
             "color": (i.get("symbol") or {}).get("color")} for i in infos]


def match_patterns(values: list[str]) -> dict[str, list[str]]:
    """Which distinct `struktura` values each LIKE pattern selects (the prefix before `%`)."""
    pats = {**{k: p for k, (_, p) in CLASSES.items()}, "low": LOW_PATTERN}
    return {k: [v for v in values if v.startswith(p.rstrip("%"))] for k, p in pats.items()}


def class_statistics(client: Any) -> list[dict[str, Any]]:
    """Feature count per distinct `struktura` value (one statistics query, no geometry)."""
    stats = [{"statisticType": "count", "onStatisticField": "objectid", "outStatisticFieldName": "n"}]
    body = client.get_json(LAYER_URL + "/query", params={"where": "1=1", "groupByFieldsForStatistics": CLASS_FIELD,
                                                         "outStatistics": json.dumps(stats), "f": "json"})
    return sorted(({CLASS_FIELD: f["attributes"][CLASS_FIELD], "n_features": int(f["attributes"]["n"])}
                   for f in body.get("features", [])), key=lambda r: r[CLASS_FIELD])


def service_metadata(client: Any) -> dict[str, Any]:
    """Service and layer description, the renderer and the class statistics. Extents are left out (row 5 licence)."""
    svc = client.get_json(SERVICE_URL, params={"f": "json"})
    lyr = client.get_json(LAYER_URL, params={"f": "json"})
    meta = {
        "service": {k: svc[k] for k in SERVICE_META_KEYS if k in svc},
        "layer": {k: lyr[k] for k in ("id", "name", "type", "geometryType", "sourceSpatialReference",
                                      "currentVersion", "maxRecordCount", "minScale", "maxScale", "displayField",
                                      "capabilities", "copyrightText") if k in lyr},
        "layer_fields": [[f["name"], f["type"]] for f in lyr.get("fields", [])],
        "renderer": renderer_summary(lyr),
        "antialiasing": (svc.get("documentInfo") or {}).get("AntialiasingMode"),
    }
    stats = class_statistics(client)
    meta["class_statistics"] = stats
    meta["pattern_matches"] = match_patterns([r[CLASS_FIELD] for r in stats])
    meta["class_colours"] = {k: list(v) if v else None for k, v in class_colours(lyr).items()}
    return meta


def check_service(meta: Mapping[str, Any], max_w: int, max_h: int, epsg: int) -> list[str]:
    """Problems that must stop the run: a pattern that does not select exactly one class, a missing colour, an image
    limit smaller than the largest grid, or a service in another SR."""
    problems = []
    for k in CLASSES:
        if len(meta["pattern_matches"][k]) != 1:
            problems.append(f"pattern for {k} selects {meta['pattern_matches'][k]}")
        if meta["class_colours"].get(k) is None:
            problems.append(f"no unique renderer colour for {k}")
    svc = meta["service"]
    if max_w > int(svc.get("maxImageWidth", 0)) or max_h > int(svc.get("maxImageHeight", 0)):
        problems.append(f"largest grid {max_w}x{max_h} exceeds the service limit "
                        f"{svc.get('maxImageWidth')}x{svc.get('maxImageHeight')}")
    if S.native_wkid({"spatialReference": svc.get("spatialReference")}) != epsg:
        problems.append(f"service SR {svc.get('spatialReference')} is not EPSG:{epsg}")
    if IMAGE_FORMAT.upper() not in str(svc.get("supportedImageFormatTypes", "")).upper().split(","):
        problems.append(f"{IMAGE_FORMAT} not supported: {svc.get('supportedImageFormatTypes')}")
    return problems


# ======================================================================================================================
# One sheet
# ======================================================================================================================


def fetch_png(client: Any, grid: S.Grid, epsg: int, cls: str) -> tuple[bytes, dict[str, Any]]:
    """One `export` request. Returns the PNG body and a request record. Raises `UnexpectedResponse` on a non-PNG."""
    params = export_params(grid, epsg, cls)
    t0 = datetime.now(timezone.utc)
    resp = client.get(EXPORT_URL, params=params)
    ctype = resp.headers.get("Content-Type", "")
    body = resp.content
    if not ctype.startswith("image/png") or body[:8] != PNG_SIGNATURE:
        raise UnexpectedResponse(f"{cls}: expected image/png, got {ctype!r}: {body[:300]!r}")
    rec = {"params": params, "retrieved_utc": t0.isoformat(timespec="seconds"), "status": resp.status_code,
           "content_type": ctype, "bytes": len(body), "sha256": S.sha256_bytes(body)}
    return body, rec


def render_sheet(client: Any, sheet: str, dtm_path: Path, out_tif: Path, png_dir: Path, epsg: int, cell: float,
                 colours: Mapping[str, tuple[int, int, int] | None]) -> dict[str, Any]:
    """Render mid and high for one sheet on its DTM grid, decode them, write the class raster and both PNGs.

    Raises `UnexpectedResponse` on a non-PNG answer, a size mismatch or class pixels of an unexpected colour.
    Returns the per-sheet log record.
    """
    from landslide.paths import sheet_to_stem

    dtm = read_dtm_grid(dtm_path, cell)
    g = dtm.grid
    alphas: dict[str, np.ndarray] = {}
    reqs: dict[str, Any] = {}
    png_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        body, rec = fetch_png(client, g, epsg, cls)
        rgba = decode_png(body, (g.w, g.h))
        cc = colour_check(rgba, colours.get(cls))
        if cc["n_other_colour"]:
            raise UnexpectedResponse(f"{sheet} {cls}: {cc['n_other_colour']} class pixels have an unexpected colour "
                                     f"{cc['colours']} (expected {colours.get(cls)})")
        png = png_dir / f"{sheet_to_stem(sheet)}_{cls}.png"
        tmp = png.with_name(png.name + ".part")
        tmp.write_bytes(body)
        tmp.replace(png)
        alphas[cls] = rgba[..., 3]
        reqs[cls] = {**rec, "png": png.name, "alpha_histogram": alpha_histogram(rgba[..., 3]), "colour_check": cc}
    codes, stats = combine(alphas)
    write_class_raster(out_tif, codes, dtm, tags={
        "SOURCE": f"{EXPORT_URL} (ČGS, row 5 of BRIEF.md; no stated licence)",
        "RECIPE": f"BRIEF.md note 5; class if alpha >= {ALPHA_MIN}; overlap of mid and high -> 2",
        "RETRIEVED_UTC": reqs["mid"]["retrieved_utc"], "SHEET": sheet})
    return {"sheet": sheet, "grid": g.as_dict(), "dtm": dtm_path.name, "tif": out_tif.name,
            "tif_bytes": out_tif.stat().st_size, "requests": reqs, **stats}


# ======================================================================================================================
# Inspection: point queries against the vector layer
# ======================================================================================================================


def pixel_centre(grid: S.Grid, row: int, col: int) -> tuple[float, float]:
    return (grid.xmin + (col + 0.5) * grid.cell, grid.ymax - (row + 0.5) * grid.cell)


def inspection_points(alphas: Mapping[str, np.ndarray], n_per_category: int) -> dict[str, list[tuple[int, int]]]:
    """Pixels to check with point queries, `n` per category, spread deterministically over the raster (no RNG).

    - `<cls>_interior`: alpha 255 in the class render, and every pixel of the 5 x 5 window too; 0 in the other;
    - `<cls>_edge`: alpha 255 with a 4-neighbour whose alpha is < 255 (the first fully covered pixel at a
      boundary). If the render were shifted by a pixel, many of these would fall outside the polygon;
    - `none_interior` / `none_edge`: alpha 0 in both renders (window clear, or next to a coloured pixel);
    - `both_full`: alpha 255 in both renders, i.e. where the mid and the high polygons overlap. The service should
      return both classes there.
    """
    from scipy.ndimage import binary_erosion

    full = {k: np.asarray(a) == 255 for k, a in alphas.items()}
    zero = {k: np.asarray(a) == 0 for k, a in alphas.items()}
    both_zero = zero["mid"] & zero["high"]
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    cats: dict[str, np.ndarray] = {}
    for k in CLASSES:
        other = "high" if k == "mid" else "mid"
        cats[f"{k}_interior"] = binary_erosion(full[k], np.ones((5, 5), bool), border_value=0) & zero[other]
        cats[f"{k}_edge"] = full[k] & ~binary_erosion(full[k], cross, border_value=1) & zero[other]
    cats["none_interior"] = binary_erosion(both_zero, np.ones((5, 5), bool), border_value=0)
    cats["none_edge"] = both_zero & ~binary_erosion(both_zero, cross, border_value=1)
    cats["both_full"] = full["mid"] & full["high"]
    out = {}
    for name, m in cats.items():
        idx = np.flatnonzero(m)
        if idx.size == 0:
            out[name] = []
            continue
        pick = np.unique(np.linspace(0, idx.size - 1, min(n_per_category, idx.size)).round().astype(np.int64))
        rows, cols = np.unravel_index(idx[pick], m.shape)
        out[name] = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]
    return out


def code_from_values(values: list[str]) -> int:
    """The code a point gets from the vector layer: the highest rendered class among the polygons containing it."""
    code = CODE_LOW
    for _, (c, pattern) in CLASSES.items():
        if any(v.startswith(pattern.rstrip("%")) for v in values):
            code = max(code, c)
    return code


def point_query(client: Any, x: float, y: float, epsg: int) -> list[str]:
    """The `struktura` values of the polygons that contain the point (EPSG:5514 in, no geometry back)."""
    body = client.get_json(LAYER_URL + "/query", params={
        "geometry": f"{x!r},{y!r}", "geometryType": "esriGeometryPoint", "inSR": epsg,
        "spatialRel": "esriSpatialRelIntersects", "outFields": CLASS_FIELD, "returnGeometry": "false", "f": "json"})
    return [f["attributes"][CLASS_FIELD] for f in body.get("features", [])]
