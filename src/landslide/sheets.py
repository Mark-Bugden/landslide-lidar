"""Surveyed map sheets, their DTM grids, and the source checks for inputs 1-4 (task P1-03).

Contents:
- **Sources.** The service URLs of the brief's rows 1-4, layer metadata, `returnCountOnly` counts, and a paginated
  query that saves every response page unmodified. All requests go through `landslide.net.Client`.
- **Sheets.** The 860 surveyed sheets (row 3, EPSG:4326) reprojected locally to EPSG:5514 with pyproj (decision D03).
- **DTM grid.** The brief's note 1 recipe: sheet bounds + 250 m, snapped outwards to the native DMR 5G lattice whose
  pixel edges sit at x = 0.4 and y = -0.12 (mod 2 m). A `Grid` keeps integer pixel-edge indices, so two grids compare
  exactly, without float tolerance.
- **DTM request and comparison.** The `exportImage` parameters, the GeoTIFF georeferencing read from the tags, and the
  value comparison of a re-download with a package raster (bitwise where the package is not NaN; every package NaN
  must be NoData or outside [-100, 2000] m in the download).
- **Manifest.** Streaming sha256 of the hand-over files against `MANIFEST.csv`.

The lattice origin and the package's valid range are not keys of `configs/landslide.json`; they are taken from the
brief's note 1 (module constants below). Cell size, margin and CRS come from the config.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from landslide.config import rng_for, stream_id

# ---- constants from the brief (BRIEF.md, Data sources) ----

#: Native DMR 5G pixel edges: x = 0.4 and y = -0.12 (mod cell). BRIEF.md note 1.
GRID_ORIGIN: tuple[float, float] = (0.4, -0.12)
#: The package set NoData and values outside this closed range to NaN. BRIEF.md note 1.
PACKAGE_VALID_RANGE: tuple[float, float] = (-100.0, 2000.0)
#: Snapping tolerance in cells, so a value that should sit exactly on a pixel edge is not pushed out by float noise.
SNAP_TOL_CELLS = 1e-9

#: Rows 2-4 of the brief's data-source table. Row 1 comes from the config (`dtm.service`).
REGISTER_URL = "https://mapy.geology.cz/arcgis/rest/services/Geohazardy/svahove_nestability/MapServer/1"
SHEETS_URL = ("https://mapy.geology.cz/arcgis/rest/services/Prozkoumanost/prehled_mapovani_svahove_nestability/"
              "MapServer/0")
GEOLOGY_URL = "https://mapy.geology.cz/arcgis/rest/services/Geologie/geologicka_mapa500_CR/MapServer/7"
#: Fields of the geology package file (BRIEF.md note 4).
GEOLOGY_FIELDS = ("objectid", "region", "subregion", "hor_karto")
#: The value of `poznamka` that marks a full ZM10 sheet.
FULL_SHEET = "list celý"


def dtm_service_url(cfg: Mapping[str, Any]) -> str:
    """The DMR 5G ImageServer URL: the first token of the config's `dtm.service`."""
    return str(cfg["dtm"]["service"]).split()[0].rstrip("/")


def epsg_code(cfg: Mapping[str, Any]) -> int:
    """`"EPSG:5514"` -> 5514."""
    auth, code = str(cfg["crs"]).split(":")
    if auth.upper() != "EPSG":
        raise ValueError(f"unexpected crs {cfg['crs']!r}")
    return int(code)


# ======================================================================================================================
# DTM grid
# ======================================================================================================================


@dataclass(frozen=True)
class Grid:
    """A north-up grid on the native lattice. Pixel edges are x0 + cell*i and y0 + cell*j for integers i, j.

    `i0` indexes the left edge (xmin) and `j1` the top edge (ymax); the grid is `w` pixels wide and `h` high.
    """

    i0: int
    j1: int
    w: int
    h: int
    cell: float = 2.0
    x0: float = GRID_ORIGIN[0]
    y0: float = GRID_ORIGIN[1]

    @staticmethod
    def _coord(origin: float, cell: float, k: int) -> float:
        # Lattice coordinates have at most 2 decimals here; rounding to 1e-6 m removes float noise only.
        return round(origin + cell * k, 6)

    @property
    def xmin(self) -> float:
        return self._coord(self.x0, self.cell, self.i0)

    @property
    def xmax(self) -> float:
        return self._coord(self.x0, self.cell, self.i0 + self.w)

    @property
    def ymax(self) -> float:
        return self._coord(self.y0, self.cell, self.j1)

    @property
    def ymin(self) -> float:
        return self._coord(self.y0, self.cell, self.j1 - self.h)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL/rasterio affine coefficients (a, b, c, d, e, f)."""
        return (self.cell, 0.0, self.xmin, 0.0, -self.cell, self.ymax)

    def key(self) -> tuple[int, int, int, int]:
        return (self.i0, self.j1, self.w, self.h)

    def as_dict(self) -> dict[str, Any]:
        return {"xmin": self.xmin, "ymax": self.ymax, "w": self.w, "h": self.h, "xmax": self.xmax, "ymin": self.ymin}


def snap_out(xmin: float, ymin: float, xmax: float, ymax: float, cell: float = 2.0,
             origin: tuple[float, float] = GRID_ORIGIN, tol_cells: float = SNAP_TOL_CELLS) -> Grid:
    """The smallest lattice grid that contains the box (edges snapped outwards)."""
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"empty box {(xmin, ymin, xmax, ymax)}")
    x0, y0 = origin
    i0 = math.floor((xmin - x0) / cell + tol_cells)
    i1 = math.ceil((xmax - x0) / cell - tol_cells)
    j0 = math.floor((ymin - y0) / cell + tol_cells)
    j1 = math.ceil((ymax - y0) / cell - tol_cells)
    return Grid(i0=i0, j1=j1, w=i1 - i0, h=j1 - j0, cell=cell, x0=x0, y0=y0)


def dtm_grid(bounds: Sequence[float], cfg: Mapping[str, Any]) -> Grid:
    """BRIEF.md note 1: the sheet's EPSG:5514 bounds plus `dtm.margin_m`, snapped outwards to the native lattice."""
    m = float(cfg["dtm"]["margin_m"])
    cell = float(cfg["dtm"]["cell_m"])
    xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    return snap_out(xmin - m, ymin - m, xmax + m, ymax + m, cell=cell)


def edge_clearance(bounds: Sequence[float], cfg: Mapping[str, Any]) -> float:
    """Smallest distance (m) from an edge of bounds + margin to the nearest lattice line.

    A small value means a tiny change in the reprojected sheet corners could move that edge by one pixel.
    """
    m = float(cfg["dtm"]["margin_m"])
    cell = float(cfg["dtm"]["cell_m"])
    xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    x0, y0 = GRID_ORIGIN
    out = math.inf
    for v, o in ((xmin - m, x0), (xmax + m, x0), (ymin - m, y0), (ymax + m, y0)):
        t = (v - o) / cell
        out = min(out, abs(t - round(t)) * cell)
    return out


class OffGrid(ValueError):
    """A corner that does not sit on the native lattice."""


def grid_from_corner(xmin: float, ymax: float, w: int, h: int, cell: float = 2.0,
                     origin: tuple[float, float] = GRID_ORIGIN, tol_m: float = 1e-6) -> Grid:
    """The lattice grid with this top-left corner and size. Raises `OffGrid` if the corner is off the lattice."""
    x0, y0 = origin
    ti, tj = (xmin - x0) / cell, (ymax - y0) / cell
    i0, j1 = round(ti), round(tj)
    off = max(abs(ti - i0), abs(tj - j1)) * cell
    if off > tol_m:
        raise OffGrid(f"corner ({xmin}, {ymax}) is {off:.6g} m off the lattice x={x0}, y={y0} (mod {cell})")
    return Grid(i0=int(i0), j1=int(j1), w=int(w), h=int(h), cell=cell, x0=x0, y0=y0)


def grid_from_transform(transform: Sequence[float], width: int, height: int, cell: float = 2.0,
                        origin: tuple[float, float] = GRID_ORIGIN) -> Grid:
    """The lattice grid of a raster. Raises `OffGrid` unless it is north-up with `cell` pixels on the lattice."""
    a, b, c, d, e, f = (float(v) for v in tuple(transform)[:6])
    if (a, b, d, e) != (cell, 0.0, 0.0, -cell):
        raise OffGrid(f"transform {tuple(transform)[:6]} is not north-up with {cell} m pixels")
    return grid_from_corner(c, f, width, height, cell=cell, origin=origin)


def grid_diff(a: Grid, b: Grid) -> dict[str, int]:
    """Differences b - a in pixel units (left edge, top edge, width, height); empty if the grids are equal."""
    d = {"d_i0": b.i0 - a.i0, "d_j1": b.j1 - a.j1, "d_w": b.w - a.w, "d_h": b.h - a.h}
    return {k: v for k, v in d.items() if v}


# ======================================================================================================================
# Sheets (row 3) and reprojection (D03)
# ======================================================================================================================


def transformer_4326_to_5514():
    """D03: pyproj's default operation from EPSG:4326 to EPSG:5514, x = lon/easting, y = lat/northing."""
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)


def reproject_geometries(geoms: Iterable[Any], transformer=None) -> list[Any]:
    """Transform shapely geometries vertex by vertex with `transformer` (default: D03)."""
    import shapely

    tr = transformer if transformer is not None else transformer_4326_to_5514()

    def f(xy: np.ndarray) -> np.ndarray:
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    return [shapely.transform(g, f) for g in geoms]


def load_sheets(path: Path | str) -> Any:
    """The surveyed sheets as a GeoDataFrame in EPSG:5514: `objectid`, `sheet`, `poznamka`, `full`, `geometry`.

    The file is row 3 (EPSG:4326). It is reprojected here with `transformer_4326_to_5514` (D03).
    """
    import geopandas as gpd

    src = gpd.read_file(path)
    if src.crs is None or src.crs.to_epsg() != 4326:
        raise ValueError(f"{path}: expected EPSG:4326, got {src.crs}")
    geoms = reproject_geometries(src.geometry.values)
    out = gpd.GeoDataFrame(
        {"objectid": src["objectid"].astype("int64").values, "sheet": src["list"].astype(str).values,
         "poznamka": src["poznamka"].values, "full": (src["poznamka"] == FULL_SHEET).values},
        geometry=geoms, crs="EPSG:5514")
    if out["sheet"].duplicated().any():
        raise ValueError(f"{path}: duplicate sheet ids {sorted(out.loc[out['sheet'].duplicated(), 'sheet'])}")
    return out.sort_values("sheet").reset_index(drop=True)


def reprojection_report(lonlat: np.ndarray) -> dict[str, Any]:
    """D03 record: library versions and the PROJ operation actually used for each vertex.

    `lonlat` is an (N, 2) array of EPSG:4326 vertices. PROJ picks a Helmert variant by area of use, so each
    vertex is transformed on its own and the last-used operation is counted.
    """
    import pyproj
    from pyproj.transformer import TransformerGroup

    tr = transformer_4326_to_5514()
    used: Counter[str] = Counter()
    defs: dict[str, str] = {}
    for lon, lat in np.asarray(lonlat, dtype=float):
        tr.transform(lon, lat)
        op = tr.get_last_used_operation()
        used[op.description] += 1
        defs[op.description] = op.definition
    group = TransformerGroup("EPSG:4326", "EPSG:5514", always_xy=True)
    return {
        "call": "pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:5514', always_xy=True)",
        "pyproj": pyproj.__version__,
        "proj": pyproj.proj_version_str,
        "network_enabled": bool(pyproj.network.is_network_enabled()),
        "n_vertices": int(sum(used.values())),
        "operations_used": [{"description": d, "n_vertices": n, "definition": defs[d]}
                            for d, n in used.most_common()],
        "candidate_operations": [{"description": t.description, "accuracy_m": t.accuracy}
                                 for t in group.transformers],
        "unavailable_operations": [u.name for u in group.unavailable_operations],
    }


# ======================================================================================================================
# The hand-over fetch log
# ======================================================================================================================


def read_fetch_log(path: Path | str) -> dict[str, dict[str, Any]]:
    """`dmr5g/fetch_log.jsonl` as {sheet: record}. Raises on a duplicate sheet."""
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            s = rec["sheet"]
            if s in out:
                raise ValueError(f"{path}:{n}: duplicate sheet {s}")
            out[s] = rec
    return out


def compare_grids(recipe: Mapping[str, Grid], log: Mapping[str, Mapping[str, Any]],
                  rasters: Mapping[str, Mapping[str, Any]], cell: float) -> dict[str, Any]:
    """Compare the recipe grid of every sheet with the fetch log and with each raster's own transform and size.

    `rasters` maps sheet -> {"transform": 6 coefficients, "width", "height"} (or {"error": text}).
    Returns counts per pair and every mismatch.
    """
    sheets = sorted(set(recipe) | set(log) | set(rasters))
    pairs = {"recipe_vs_fetch_log": 0, "recipe_vs_raster": 0, "fetch_log_vs_raster": 0}
    mismatches: list[dict[str, Any]] = []
    for s in sheets:
        g_recipe = recipe.get(s)
        g_log = g_ras = None
        problems: dict[str, Any] = {}
        if s in log:
            r = log[s]
            try:
                g_log = grid_from_corner(float(r["xmin"]), float(r["ymax"]), int(r["w"]), int(r["h"]), cell=cell)
            except OffGrid as e:
                problems["fetch_log"] = str(e)
        else:
            problems["fetch_log"] = "missing"
        if s in rasters and "error" not in rasters[s]:
            r = rasters[s]
            try:
                g_ras = grid_from_transform(r["transform"], int(r["width"]), int(r["height"]), cell=cell)
            except OffGrid as e:
                problems["raster"] = str(e)
        else:
            problems["raster"] = rasters.get(s, {}).get("error", "missing")
        if g_recipe is None:
            problems["recipe"] = "sheet not in the surveyed-sheet layer"
        for name, a, b in (("recipe_vs_fetch_log", g_recipe, g_log), ("recipe_vs_raster", g_recipe, g_ras),
                           ("fetch_log_vs_raster", g_log, g_ras)):
            if a is not None and b is not None and a.key() == b.key():
                pairs[name] += 1
            elif a is not None and b is not None:
                problems[name] = grid_diff(a, b)
        if problems:
            mismatches.append({"sheet": s, **problems,
                               "grid_recipe": g_recipe.as_dict() if g_recipe else None,
                               "grid_fetch_log": g_log.as_dict() if g_log else None,
                               "grid_raster": g_ras.as_dict() if g_ras else None})
    return {"n_sheets": len(sheets), "n_recipe": len(recipe), "n_fetch_log": len(log), "n_rasters": len(rasters),
            "equal": pairs, "n_mismatched_sheets": len(mismatches), "mismatches": mismatches}


# ======================================================================================================================
# Provenance sample
# ======================================================================================================================


def draw_provenance_sample(cfg: Mapping[str, Any], sheet_ids: Iterable[str], n: int | None = None,
                           seed: int | None = None) -> list[str]:
    """Draw `n` distinct sheets (default `replication.provenance.n_sheets`) in draw order.

    The generator is `rng_for(cfg, seed, "provenance_sample")` with `seed` defaulting to `replication.seed`; the
    candidates are the sheet ids sorted, so the draw does not depend on file order.
    """
    rep = cfg["replication"]
    n = int(rep["provenance"]["n_sheets"]) if n is None else int(n)
    seed = int(rep["seed"]) if seed is None else int(seed)
    ids = sorted(set(sheet_ids))
    if n > len(ids):
        raise ValueError(f"cannot draw {n} of {len(ids)} sheets")
    rng = rng_for(dict(cfg), seed, "provenance_sample")
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[int(i)] for i in idx]


def provenance_sample_spec(cfg: Mapping[str, Any], seed: int | None = None) -> dict[str, Any]:
    rep = cfg["replication"]
    return {"seed": int(rep["seed"]) if seed is None else int(seed), "stream": "provenance_sample",
            "stream_id": stream_id(dict(cfg), "provenance_sample"), "n": int(rep["provenance"]["n_sheets"]),
            "generator": "numpy default_rng(SeedSequence([seed, stream_id])).choice(860 sorted ids, n, replace=False)"}


# ======================================================================================================================
# DTM request (row 1)
# ======================================================================================================================


def _num(v: float) -> str:
    return repr(float(round(v, 6)))


def export_image_params(grid: Grid, epsg: int) -> dict[str, Any]:
    """BRIEF.md note 1, exactly; `f=image` returns the TIFF itself. No `outSR`."""
    return {
        "bbox": ",".join(_num(v) for v in grid.bbox),
        "bboxSR": epsg,
        "imageSR": epsg,
        "size": f"{grid.w},{grid.h}",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_NearestNeighbor",
        "f": "image",
    }


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_dtm(client: Any, cfg: Mapping[str, Any], sheet: str, grid: Grid, out_dir: Path) -> dict[str, Any]:
    """Download one sheet with the recipe and save the unmodified response as `<out_dir>/<stem>.tif`.

    A sidecar `<stem>.json` records the request and response. Raises if the response is not a TIFF.
    """
    from landslide.paths import sheet_to_stem

    url = dtm_service_url(cfg) + "/exportImage"
    params = export_image_params(grid, epsg_code(cfg))
    t0 = datetime.now(timezone.utc)
    resp = client.get(url, params=params)
    ctype = resp.headers.get("Content-Type", "")
    body = resp.content
    if not ctype.startswith("image/tiff") or body[:4] not in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"{sheet}: expected a TIFF, got {ctype!r}: {body[:300]!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"{sheet_to_stem(sheet)}.tif"
    tmp = tif.with_suffix(".tif.part")
    tmp.write_bytes(body)
    tmp.replace(tif)
    rec = {"sheet": sheet, "url": url, "params": params, "request_url": resp.url,
           "retrieved_utc": t0.isoformat(timespec="seconds"), "status": resp.status_code, "content_type": ctype,
           "etag": resp.headers.get("ETag"), "bytes": len(body), "sha256": sha256_bytes(body),
           "grid": grid.as_dict(), "file": tif.name}
    (out_dir / f"{sheet_to_stem(sheet)}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# ======================================================================================================================
# GeoTIFF georeferencing and the value comparison
# ======================================================================================================================

#: GeoKeys that define the georeferencing of a projected EPSG CRS.
GEOKEYS = {1024: "GTModelType", 1025: "GTRasterType", 3072: "ProjectedCSType", 3076: "ProjLinearUnits"}


def parse_geokeys(directory: Sequence[int]) -> dict[int, int]:
    """Short-valued keys (TIFFTagLocation 0) of a GeoKeyDirectoryTag."""
    d = [int(v) for v in directory]
    n = d[3]
    out = {}
    for k in range(n):
        key, loc, count, val = d[4 + 4 * k: 8 + 4 * k]
        if loc == 0 and count == 1:
            out[key] = val
    return out


def geotiff_georef(path: Path | str) -> dict[str, Any]:
    """Size, affine transform and the raw georeferencing tags of a GeoTIFF.

    The tags (ModelTiepoint, ModelPixelScale, the GeoKeys in `GEOKEYS`) are read with Pillow, so the comparison does
    not depend on how GDAL interprets a vendor CRS string. GDAL's reading is recorded too.
    """
    import rasterio
    from PIL import Image

    with rasterio.open(path) as ds:
        info = {"width": ds.width, "height": ds.height, "count": ds.count, "dtype": ds.dtypes[0],
                "transform": list(ds.transform)[:6], "nodata": ds.nodata,
                "gdal_crs_epsg": ds.crs.to_epsg() if ds.crs else None,
                "gdal_crs_wkt": ds.crs.to_wkt() if ds.crs else None}
    with Image.open(path) as im:
        tags = im.tag_v2
        info["tiepoint"] = [float(v) for v in tags.get(33922, ())]
        info["pixel_scale"] = [float(v) for v in tags.get(33550, ())]
        keys = parse_geokeys(tags.get(34735, (1, 1, 0, 0)))
        info["geokeys"] = {name: keys.get(k) for k, name in GEOKEYS.items()}
        info["gdal_nodata_tag"] = tags.get(42113)
    return info


def compare_georef(package: Mapping[str, Any], download: Mapping[str, Any]) -> dict[str, Any]:
    """Field-by-field equality of two `geotiff_georef` results."""
    fields = {
        "size": lambda g: (g["width"], g["height"]),
        "transform": lambda g: tuple(g["transform"]),
        "tiepoint": lambda g: tuple(g["tiepoint"]),
        "pixel_scale": lambda g: tuple(g["pixel_scale"]),
        **{f"geokey_{name}": (lambda g, n=name: g["geokeys"][n]) for name in GEOKEYS.values()},
    }
    out = {k: f(package) == f(download) for k, f in fields.items()}
    out["equal"] = all(out.values())
    return out


def _bits(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        raise TypeError(f"bitwise comparison needs float32, got {a.dtype}")
    return np.ascontiguousarray(a).view(np.uint32)


def invalid_masks(z: np.ndarray, nodata: float | None,
                  valid_range: tuple[float, float] = PACKAGE_VALID_RANGE) -> tuple[np.ndarray, np.ndarray]:
    """(nodata, out_of_range) masks of a downloaded raster. NaN counts as NoData; the range is closed."""
    nod = np.isnan(z)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        nod |= z == np.float32(nodata)
    lo, hi = valid_range
    with np.errstate(invalid="ignore"):
        oor = ~nod & ((z < lo) | (z > hi))
    return nod, oor


_DIFF_BINS = (0.0, 1e-6, 1e-3, 1e-2, 0.1, 1.0, 10.0, math.inf)


def _examples(mask: np.ndarray, *arrays: np.ndarray, k: int = 10) -> list[list[float]]:
    rows, cols = np.nonzero(mask)
    out = []
    for r, c in zip(rows[:k], cols[:k], strict=True):
        out.append([int(r), int(c), *(float(a[r, c]) for a in arrays)])
    return out


def compare_dtm(package: np.ndarray, download: np.ndarray, download_nodata: float | None = None,
                valid_range: tuple[float, float] = PACKAGE_VALID_RANGE) -> dict[str, Any]:
    """Compare a package raster with its unmodified re-download (both float32, same shape).

    - Where the package is not NaN, the values must be bitwise equal. Any difference is counted, with magnitudes.
    - Every package NaN must be NoData (NaN or the nodata tag) or outside `valid_range` in the download.
    - Every NoData or out-of-range download cell must be NaN in the package (the manifest's stated change).
    `ok` is True only if all three hold.
    """
    if package.shape != download.shape:
        return {"ok": False, "shape_equal": False, "package_shape": list(package.shape),
                "download_shape": list(download.shape)}
    pkg_nan = np.isnan(package)
    valid = ~pkg_nan
    eq = _bits(package) == _bits(download)
    diff = valid & ~eq
    nod, oor = invalid_masks(download, download_nodata, valid_range)
    lo, hi = valid_range
    with np.errstate(invalid="ignore"):
        below, above = oor & (download < lo), oor & (download > hi)
    unexplained = pkg_nan & ~(nod | oor)
    invalid_kept = valid & (nod | oor)

    n_diff = int(diff.sum())
    diffs: dict[str, Any] = {"n": n_diff}
    if n_diff:
        d = download[diff].astype(np.float64) - package[diff].astype(np.float64)
        fin = np.isfinite(d)
        ad = np.abs(d[fin])
        diffs.update({
            "n_nonfinite": int((~fin).sum()),
            "max_abs": float(ad.max()) if ad.size else None,
            "mean_abs": float(ad.mean()) if ad.size else None,
            "median_abs": float(np.median(ad)) if ad.size else None,
            "histogram_abs": {f"[{a:g}, {b:g})": int(((ad >= a) & (ad < b)).sum())
                              for a, b in pairwise(_DIFF_BINS)},
            "examples_row_col_package_download": _examples(diff, package, download),
        })
    dl_fin = download[np.isfinite(download)]
    out = {
        "shape_equal": True,
        "n_pixels": int(package.size),
        "n_package_nan": int(pkg_nan.sum()),
        "n_compared": int(valid.sum()),
        "n_bitwise_equal": int((valid & eq).sum()),
        "diff": diffs,
        "package_nan_explained": {"nodata": int((pkg_nan & nod).sum()),
                                  "below_range": int((pkg_nan & below).sum()),
                                  "above_range": int((pkg_nan & above).sum())},
        "n_package_nan_unexplained": int(unexplained.sum()),
        "unexplained_examples_row_col_download": _examples(unexplained, download),
        "n_download_invalid_but_package_valued": int(invalid_kept.sum()),
        "download": {"nodata_tag": download_nodata, "n_nodata": int(nod.sum()), "n_below_range": int(below.sum()),
                     "n_above_range": int(above.sum()), "n_zero": int((download == 0).sum()),
                     "min": float(dl_fin.min()) if dl_fin.size else None,
                     "max": float(dl_fin.max()) if dl_fin.size else None},
        "package_n_zero": int((package == 0).sum()),
    }
    out["ok"] = bool(n_diff == 0 and out["n_package_nan_unexplained"] == 0
                     and out["n_download_invalid_but_package_valued"] == 0)
    return out


def package_stats(z: np.ndarray) -> dict[str, Any]:
    """NaN count, zero count and NaN-aware min/max of a package raster (float32)."""
    nan = np.isnan(z)
    fin = z[~nan]
    return {"n_pixels": int(z.size), "n_nan": int(nan.sum()), "n_zero": int((fin == 0).sum()),
            "n_negative": int((fin < 0).sum()),
            "zmin": float(fin.min()) if fin.size else None, "zmax": float(fin.max()) if fin.size else None}


def compare_fetch_log_stats(stats: Mapping[str, Any], rec: Mapping[str, Any]) -> dict[str, bool]:
    """A raster's `package_stats` against its fetch-log record (`invalid_frac`, `zmin`, `zmax`).

    `invalid_frac` is read as the NaN fraction; `zmin`/`zmax` are float32 values written as doubles, so they are
    compared after rounding to float32.
    """
    def f32(v: Any) -> float | None:
        return None if v is None else float(np.float32(v))

    return {"invalid_frac": math.isclose(stats["n_nan"] / stats["n_pixels"], float(rec["invalid_frac"]),
                                         rel_tol=0, abs_tol=0.5 / stats["n_pixels"]),
            "zmin": f32(rec.get("zmin")) == stats["zmin"], "zmax": f32(rec.get("zmax")) == stats["zmax"]}


def to_package_array(download: np.ndarray, nodata: float | None,
                     valid_range: tuple[float, float] = PACKAGE_VALID_RANGE) -> np.ndarray:
    """The manifest's stated change: NoData and values outside the valid range set to NaN (float32)."""
    z = np.array(download, dtype=np.float32, copy=True)
    nod, oor = invalid_masks(z, nodata, valid_range)
    z[nod | oor] = np.nan
    return z


def write_package_raster(download_path: Path, out_path: Path, epsg: int) -> None:
    """Rebuild a package-format raster from a download: NaN for invalid cells, deflate-tiled GeoTIFF, same grid."""
    import rasterio

    with rasterio.open(download_path) as src:
        z = to_package_array(src.read(1), src.nodata)
        profile = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": src.width, "height": src.height,
                   "transform": src.transform, "crs": f"EPSG:{epsg}", "nodata": float("nan"), "tiled": True,
                   "blockxsize": 256, "blockysize": 256, "compress": "deflate"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(z, 1)


# ======================================================================================================================
# Manifest
# ======================================================================================================================


def sha256_file(path: Path | str, chunk_bytes: int = 8 << 20,
                on_chunk: Callable[[int], None] | None = None) -> str:
    """sha256 of a file, streamed in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
            if on_chunk is not None:
                on_chunk(len(chunk))
    return h.hexdigest()


def read_manifest(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_manifest(raw_dir: Path, rows: Sequence[Mapping[str, str]],
                   on_chunk: Callable[[int], None] | None = None) -> dict[str, Any]:
    """sha256 of every file listed in the manifest, plus every file in `raw_dir` the manifest does not list.

    The manifest itself is reported as unlisted too (it cannot list its own hash).
    """
    listed = {}
    for r in rows:
        rel = r["file"].replace("\\", "/")
        if rel in listed:
            raise ValueError(f"manifest lists {rel} twice")
        listed[rel] = r["sha256"].strip().lower()
    mismatches, missing = [], []
    n_ok = n_bytes = 0
    for rel, expected in listed.items():
        p = raw_dir / rel
        if not p.is_file():
            missing.append(rel)
            continue
        got = sha256_file(p, on_chunk=on_chunk)
        n_bytes += p.stat().st_size
        if got == expected:
            n_ok += 1
        else:
            mismatches.append({"file": rel, "manifest": expected, "computed": got})
    unlisted = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(raw_dir).as_posix()
            if rel not in listed:
                unlisted.append({"file": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return {"n_listed": len(listed), "n_ok": n_ok, "n_mismatch": len(mismatches), "n_missing": len(missing),
            "bytes_hashed": n_bytes, "mismatches": mismatches, "missing": missing, "unlisted": unlisted,
            "ok": not mismatches and not missing}


# ======================================================================================================================
# ArcGIS layers (rows 2-4)
# ======================================================================================================================

LAYER_META_KEYS = ("name", "type", "geometryType", "maxRecordCount", "sourceSpatialReference", "spatialReference",
                   "extent", "currentVersion", "supportedQueryFormats", "capabilities", "objectIdField",
                   "displayField", "copyrightText")
IMAGE_META_KEYS = ("name", "currentVersion", "spatialReference", "extent", "pixelSizeX", "pixelSizeY", "pixelType",
                   "bandCount", "noDataValue", "noDataValues", "maxImageWidth", "maxImageHeight", "maxRecordCount",
                   "heightModelInfo", "defaultResamplingMethod", "defaultMosaicMethod", "capabilities",
                   "copyrightText")


def layer_metadata(client: Any, url: str, keys: Sequence[str] = LAYER_META_KEYS) -> dict[str, Any]:
    """The layer (or image service) description with `f=json`: the listed keys plus the field names and types."""
    m = client.get_json(url, params={"f": "json"})
    out = {k: m[k] for k in keys if k in m}
    if m.get("fields"):
        out["fields"] = [[f["name"], f["type"]] for f in m["fields"]]
    return out


def native_wkid(meta: Mapping[str, Any]) -> int | None:
    """The `latestWkid` (or `wkid`) of a layer's source spatial reference."""
    sr = meta.get("sourceSpatialReference") or meta.get("spatialReference") or {}
    return sr.get("latestWkid", sr.get("wkid"))


def oid_field(meta: Mapping[str, Any]) -> str:
    if meta.get("objectIdField"):
        return str(meta["objectIdField"])
    for name, typ in meta.get("fields", []):
        if typ == "esriFieldTypeOID":
            return str(name)
    raise ValueError("layer has no object-id field")


def sql_in(field: str, values: Iterable[str]) -> str:
    """`field IN ('a', 'b')` with single quotes doubled."""
    quoted = ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)
    return f"{field} IN ({quoted})"


def count_features(client: Any, url: str, where: str = "1=1") -> int:
    """`returnCountOnly=true` on a layer."""
    body = client.get_json(url + "/query", params={"where": where, "returnCountOnly": "true", "f": "json"})
    return int(body["count"])


def object_ids(client: Any, url: str, where: str = "1=1") -> list[int]:
    body = client.get_json(url + "/query", params={"where": where, "returnIdsOnly": "true", "f": "json"})
    return sorted(int(i) for i in (body.get("objectIds") or []))


def chunks(ids: Sequence[int], size: int) -> list[list[int]]:
    if size < 1:
        raise ValueError("page size must be >= 1")
    return [list(ids[k:k + size]) for k in range(0, len(ids), size)]


def query_pages(client: Any, url: str, where: str, out_fields: str, fmt: str, page_size: int, oid: str,
                out_dir: Path) -> dict[str, Any]:
    """Download all features matching `where`, paginated by object id, saving each response page unmodified.

    Pages are contiguous runs of the sorted object ids, queried as `(<where>) AND oid >= lo AND oid <= hi`, so each
    page returns exactly its ids (checked). `fmt` is `json` (Esri JSON, native SR) or `geojson`. No `outSR`.
    """
    ids = object_ids(client, url, where)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for k, ch in enumerate(chunks(ids, page_size)):
        w = f"({where}) AND {oid} >= {ch[0]} AND {oid} <= {ch[-1]}"
        params = {"where": w, "outFields": out_fields, "returnGeometry": "true", "orderByFields": oid, "f": fmt}
        resp = client.get(url + "/query", params=params)
        body = resp.content
        data = json.loads(body)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"ArcGIS error on page {k}: {data['error']}")
        feats = data.get("features", [])
        got = sorted(int((f.get("attributes") or f.get("properties"))[oid]) for f in feats)
        if got != ch:
            raise RuntimeError(f"page {k}: expected {len(ch)} ids {ch[0]}..{ch[-1]}, got {len(got)}")
        if data.get("exceededTransferLimit") or (data.get("properties") or {}).get("exceededTransferLimit"):
            raise RuntimeError(f"page {k}: exceededTransferLimit")
        p = out_dir / f"page_{k:04d}.{fmt}"
        p.write_bytes(body)
        pages.append({"file": p.name, "n": len(feats), "first_oid": ch[0], "last_oid": ch[-1],
                      "sha256": sha256_bytes(body)})
    return {"n_ids": len(ids), "n_pages": len(pages), "pages": pages}


def esri_pages_to_frame(page_paths: Sequence[Path], epsg: int) -> Any:
    """Concatenate Esri JSON pages (read with GDAL's ESRIJSON driver) into one GeoDataFrame in `epsg`.

    The pages declare the layer's native SR (ESRI wkid 102067, latestWkid 5514); it is checked, then labelled
    EPSG:`epsg` without any coordinate change.
    """
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    frames = []
    for p in page_paths:
        sr = json.loads(Path(p).read_text(encoding="utf-8")).get("spatialReference", {})
        if epsg not in (sr.get("latestWkid"), sr.get("wkid")):
            raise ValueError(f"{p}: spatialReference {sr} is not EPSG:{epsg}")
        df = pyogrio.read_dataframe(f"ESRIJSON:{Path(p)}")
        frames.append(df.set_crs(f"EPSG:{epsg}", allow_override=True))
    if not frames:
        raise ValueError("no pages")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=f"EPSG:{epsg}")


def geojson_pages_to_collection(page_paths: Sequence[Path]) -> dict[str, Any]:
    """Concatenate GeoJSON pages into one FeatureCollection, features unchanged."""
    feats: list[Any] = []
    for p in page_paths:
        feats.extend(json.loads(Path(p).read_text(encoding="utf-8"))["features"])
    return {"type": "FeatureCollection", "features": feats}


def to_jsonable(obj: Any) -> Any:
    """Dataclasses, numpy scalars and paths to plain JSON types; a float NaN becomes the string "NaN"."""
    if hasattr(obj, "__dataclass_fields__"):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, float) and math.isnan(obj):
        return "NaN"
    return obj
