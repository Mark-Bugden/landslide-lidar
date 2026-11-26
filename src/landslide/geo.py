"""Geo preparation for the splits (task P1-08): register polygons, the flysch outline, sheet attributes, DTM validity.

- **Polygons** (row 2): `in_mapped` (D04), `eligible` (D05) and `kraj1` (the first-listed `gen_kraj`).
- **Flysch outline** (row 4, D07) and each sheet's flysch fraction.
- **DTM validity** per sheet (row 1), one raster at a time, with the grid read from the raster itself.
- **Polygon -> sheet** (the sheet holding the largest share of the polygon's area) and the **sheet kraj** (D06).

Everything works on shapely geometries in EPSG:5514 and returns plain arrays or frames; `scripts/build_geo.py` writes
the files. Sheets come from `landslide.sheets.load_sheets` (D03), never reprojected here.

**Invalid register polygons.** 68 of the 21,356 register polygons are invalid (mostly self-touching rings). Their
area (D05) and representative point (D04) use the geometry as delivered, the same as for every other polygon.
Overlays (intersection areas, the `intersects` test of the sheet kraj) use a repaired copy,
`shapely.make_valid(method="structure", keep_collapsed=False)`, because GEOS overlays can fail on invalid input.

Two method values of Appendix A §3 have no config key: interior holes are filled and parts under 1 km² are dropped.
They are the module constants below (`FLYSCH_FILL_HOLES`, `FLYSCH_MIN_PART_M2`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon

from landslide import sheets as S

# ---- method values that are not config keys (Appendix A §3) ----

#: Appendix A §3: "interior holes (Quaternary valley fills) are filled".
FLYSCH_FILL_HOLES = True
#: Appendix A §3: "parts under 1 km² are dropped".
FLYSCH_MIN_PART_M2 = 1_000_000.0
#: Buffer resolution of the closing: segments per quarter circle (shapely's default). Round joins and caps.
BUFFER_QUAD_SEGS = 8

#: `gen_kraj` lists several kraje separated by commas ("Jihomoravský, Zlínský"); the first one counts (D06).
KRAJ_SEP = ","
#: Source column of the register's own area field (ArcGIS `st_area(shape)`), reported for D05 only.
SRC_AREA_FIELD = "st_area(shape)"


# ======================================================================================================================
# Small geometry helpers
# ======================================================================================================================


def parts(geom: Any) -> list[Polygon]:
    """The polygons of a Polygon or MultiPolygon (empty list for an empty geometry)."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def repair(geoms: np.ndarray) -> np.ndarray:
    """A copy with every invalid geometry repaired (`make_valid`, structure method, polygonal output only)."""
    geoms = np.asarray(geoms, dtype=object)
    out = geoms.copy()
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        out[bad] = shapely.make_valid(geoms[bad], method="structure", keep_collapsed=False)
    return out


# ======================================================================================================================
# Register polygons (D04, D05, kraj1)
# ======================================================================================================================


def first_kraj(gen_kraj: Any) -> str | None:
    """The first-listed kraj of a `gen_kraj` value: `"Jihomoravský, Zlínský"` -> `"Jihomoravský"`."""
    if gen_kraj is None or (isinstance(gen_kraj, float) and np.isnan(gen_kraj)):
        return None
    tok = str(gen_kraj).split(KRAJ_SEP)[0].strip()
    return tok or None


def load_register(path: Path | str) -> Any:
    """Row 2 as a GeoDataFrame in EPSG:5514: `objectid`, `skupina`, `gen_kraj`, `st_area`, `geometry`, by objectid."""
    import geopandas as gpd

    reg = gpd.read_file(path)
    if reg.crs is None or reg.crs.to_epsg() != 5514:
        raise ValueError(f"{path}: expected EPSG:5514, got {reg.crs}")
    if reg["objectid"].duplicated().any():
        raise ValueError(f"{path}: duplicate objectids")
    out = gpd.GeoDataFrame(
        {"objectid": reg["objectid"].astype("int64").values, "skupina": reg["skupina"].astype("int64").values,
         "gen_kraj": reg["gen_kraj"].values,
         "st_area": reg[SRC_AREA_FIELD].astype("float64").values if SRC_AREA_FIELD in reg else np.nan},
        geometry=reg.geometry.values, crs=reg.crs)
    return out.sort_values("objectid").reset_index(drop=True)


def representative_points(geoms: np.ndarray) -> np.ndarray:
    """`representative_point()` of each geometry (GEOS PointOnSurface, as geopandas does)."""
    return shapely.point_on_surface(np.asarray(geoms, dtype=object))


def in_mapped(geoms: np.ndarray, mapped_area: Any) -> np.ndarray:
    """D04 / BRIEF.md note 2: True where the polygon's representative point lies within `mapped_area`."""
    shapely.prepare(mapped_area)
    return shapely.within(representative_points(geoms), mapped_area)


def eligible_mask(skupina: np.ndarray, in_mapped_: np.ndarray, area_m2: np.ndarray,
                  cfg: Mapping[str, Any]) -> np.ndarray:
    """D05: `skupina == labels.skupina`, `in_mapped == labels.in_mapped`, `area >= labels.min_area_m2`."""
    lab = cfg["labels"]
    return ((np.asarray(skupina) == int(lab["skupina"])) & (np.asarray(in_mapped_) == bool(lab["in_mapped"]))
            & (np.asarray(area_m2, dtype=float) >= float(lab["min_area_m2"])))


# ======================================================================================================================
# Flysch outline (D07)
# ======================================================================================================================


def close_fill_drop(geom: Any, buffer_m: float, fill_holes: bool = FLYSCH_FILL_HOLES,
                    min_part_m2: float = FLYSCH_MIN_PART_M2,
                    quad_segs: int = BUFFER_QUAD_SEGS) -> tuple[Any, dict[str, Any]]:
    """Morphological closing `buffer(+d).buffer(-d)`, then fill interior holes, then drop parts under `min_part_m2`.

    Returns the outline (a MultiPolygon) and the area (km²) and part count after each step.
    """
    closed = shapely.buffer(shapely.buffer(geom, buffer_m, quad_segs=quad_segs), -buffer_m, quad_segs=quad_segs)
    filled = shapely.union_all([Polygon(p.exterior) for p in parts(closed)]) if fill_holes else closed
    kept = [p for p in parts(filled) if p.area >= min_part_m2]
    kept.sort(key=lambda p: (p.bounds, p.area))
    out = MultiPolygon(kept)

    def st(g: Any) -> dict[str, Any]:
        ps = parts(g)
        return {"area_km2": g.area / 1e6, "n_parts": len(ps), "n_holes": sum(len(p.interiors) for p in ps)}

    stages = {"input": st(geom), "closed": st(closed), "filled": st(filled), "final": st(out),
              "buffer_m": float(buffer_m), "fill_holes": bool(fill_holes), "min_part_m2": float(min_part_m2),
              "quad_segs": int(quad_segs)}
    return out, stages


def flysch_union(geoms: np.ndarray, subregion: Sequence[Any], subregions: Sequence[str]) -> tuple[Any, int]:
    """The union of the geology polygons whose `subregion` is in `subregions` (Appendix A §3), and how many there are."""
    sel = np.isin(np.asarray(subregion, dtype=object), list(subregions))
    return shapely.union_all(repair(np.asarray(geoms, dtype=object)[sel])), int(sel.sum())


def flysch_outline(geoms: np.ndarray, subregion: Sequence[Any], cfg: Mapping[str, Any],
                   buffer_m: float | None = None) -> tuple[Any, dict[str, Any]]:
    """D07: the closed flysch outline. `buffer_m` defaults to the config's `flysch.closing_m` (the buffer distance)."""
    u, n = flysch_union(geoms, subregion, cfg["flysch"]["subregions"])
    d = float(cfg["flysch"]["closing_m"]) if buffer_m is None else float(buffer_m)
    out, stages = close_fill_drop(u, d)
    stages["n_geology_polygons"] = n
    return out, stages


def area_fraction_inside(geoms: np.ndarray, region: Any) -> np.ndarray:
    """area(g ∩ region) / area(g) for each geometry."""
    geoms = np.asarray(geoms, dtype=object)
    shapely.prepare(region)
    return shapely.area(shapely.intersection(geoms, region)) / shapely.area(geoms)


# ======================================================================================================================
# Polygon -> sheet, sheet kraj (D06)
# ======================================================================================================================


def overlap_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs (i, j) with a[i] intersecting b[j] and the area of a[i] ∩ b[j]. Both inputs are repaired first."""
    a, b = repair(a), repair(b)
    tree = shapely.STRtree(b)
    i, j = tree.query(a, predicate="intersects")
    area = shapely.area(shapely.intersection(a[i], b[j])) if len(i) else np.zeros(0)
    return i.astype(np.int64), j.astype(np.int64), np.asarray(area, dtype=float)


def assign_to_sheets(poly_geoms: np.ndarray, sheet_geoms: np.ndarray,
                     sheet_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Appendix A §4: each polygon goes to the sheet that holds the largest part of its area.

    Exact ties go to the lowest sheet id. A polygon that overlaps no sheet with positive area gets None.
    Returns (sheet id or None, share of the polygon's area inside that sheet).
    """
    poly_geoms = np.asarray(poly_geoms, dtype=object)
    ids = np.asarray(list(sheet_ids), dtype=object)
    n = len(poly_geoms)
    out = np.full(n, None, dtype=object)
    share = np.zeros(n, dtype=float)
    i, j, area = overlap_pairs(poly_geoms, np.asarray(sheet_geoms, dtype=object))
    keep = area > 0
    i, j, area = i[keep], j[keep], area[keep]
    if len(i):
        # Sort by polygon, then area descending, then sheet id ascending; the first row per polygon wins.
        rank_id = pd.Series(ids[j]).rank(method="dense").to_numpy()
        order = np.lexsort((rank_id, -area, i))
        i, j, area = i[order], j[order], area[order]
        first = np.r_[True, i[1:] != i[:-1]]
        out[i[first]] = ids[j[first]]
        poly_area = shapely.area(repair(poly_geoms))
        share[i[first]] = area[first] / poly_area[i[first]]
    return out, share


def majority_kraj(votes: Mapping[str, int], areas: Mapping[str, float]) -> tuple[str | None, str]:
    """D06 for one sheet: the most frequent kraj; ties to the larger summed intersection area, then alphabetical.

    Returns (kraj, how it was decided: "majority", "area" or "alphabetical"; "none" without votes).
    "Alphabetical" is Python string order (Unicode code points).
    """
    if not votes:
        return None, "none"
    top = max(votes.values())
    tied = sorted(k for k, v in votes.items() if v == top)
    if len(tied) == 1:
        return tied[0], "majority"
    best_area = max(areas.get(k, 0.0) for k in tied)
    tied_area = sorted(k for k in tied if areas.get(k, 0.0) == best_area)
    return tied_area[0], ("area" if len(tied_area) == 1 else "alphabetical")


def sheet_kraj(sheet_geoms: np.ndarray, poly_geoms: np.ndarray, poly_kraj: Sequence[Any]) -> pd.DataFrame:
    """D06: each sheet's kraj from all register polygons that intersect it (one row per sheet, in input order).

    Columns: `kraj`, `kraj_rule` (majority | area | alphabetical | none), `kraj_n_polygons`, `kraj_n_votes` (votes
    of the winner), `kraj_votes` (JSON object, kraj -> count).
    """
    import json

    pk = np.asarray([first_kraj(k) for k in poly_kraj], dtype=object)
    si, pi, area = overlap_pairs(np.asarray(sheet_geoms, dtype=object), np.asarray(poly_geoms, dtype=object))
    df = pd.DataFrame({"s": si, "kraj": pk[pi], "area": area}).dropna(subset=["kraj"])
    grp = df.groupby(["s", "kraj"], sort=True).agg(n=("area", "size"), area=("area", "sum")).reset_index()
    rows = []
    by_sheet = {s: g for s, g in grp.groupby("s", sort=True)}
    for s in range(len(sheet_geoms)):
        g = by_sheet.get(s)
        votes = {} if g is None else dict(zip(g["kraj"], g["n"].astype(int)))
        areas = {} if g is None else dict(zip(g["kraj"], g["area"].astype(float)))
        k, rule = majority_kraj(votes, areas)
        rows.append({"kraj": k, "kraj_rule": rule, "kraj_n_polygons": int(sum(votes.values())),
                     "kraj_n_votes": int(votes.get(k, 0)) if k else 0,
                     "kraj_votes": json.dumps(dict(sorted(votes.items())), ensure_ascii=False)})
    out = pd.DataFrame(rows)
    out["kraj"] = pd.Series([r["kraj"] for r in rows], index=out.index, dtype=object)  # keep None, not NaN
    return out


# ======================================================================================================================
# DTM validity (row 1)
# ======================================================================================================================

_RULE_LE = re.compile(r"\bz\s*<=\s*(-?\d+(?:\.\d+)?)")
_RULE_GT = re.compile(r"\bz\s*>\s*(-?\d+(?:\.\d+)?)")


def invalid_bounds(cfg: Mapping[str, Any]) -> tuple[float, float]:
    """(le, gt) from the config's `dtm.invalid_rule`: z <= le or z > gt is invalid (as are NaN and nodata).

    The same parse as `landslide.dtm.invalid_bounds` (P1-04); the rule must contain exactly one `z <= a` and one
    `z > b`, so a reworded rule can't be silently misread.
    """
    rule = cfg["dtm"]["invalid_rule"]
    le, gt = _RULE_LE.findall(rule), _RULE_GT.findall(rule)
    if len(le) != 1 or len(gt) != 1:
        raise ValueError(f"cannot parse dtm.invalid_rule {rule!r}: need one 'z <= a' and one 'z > b'")
    lo, hi = float(le[0]), float(gt[0])
    if not lo < hi:
        raise ValueError(f"dtm.invalid_rule bounds out of order: {lo} >= {hi}")
    return lo, hi


def dtm_validity(z: np.ndarray, cfg: Mapping[str, Any], nodata: float | None = None) -> dict[str, Any]:
    """Cell counts of one raster against the invalid rule, and the two flags of PLAN.md §4.3.

    - `has_zero_cells`: some cell is exactly 0 m (the ImageServer's value outside the border).
    - `dtm_all_valid`: config `splits.stage2.require_valid_dtm` ("no 0-m cells or NaN"), read with the full
      `dtm.invalid_rule`: no NaN or non-finite cell, no nodata cell, no z <= le, no z > gt.
    """
    lo, hi = invalid_bounds(cfg)
    z = np.asarray(z)
    fin = np.isfinite(z)
    with np.errstate(invalid="ignore"):
        le = fin & (z <= lo)
        gt = fin & (z > hi)
        nod = fin & (z == nodata) if nodata is not None and np.isfinite(nodata) else np.zeros_like(fin)
    bad = ~fin | le | gt | nod
    zf = z[fin]
    return {"n_cells": int(z.size), "n_nonfinite": int((~fin).sum()), "n_zero": int((fin & (z == 0)).sum()),
            "n_le_min": int(le.sum()), "n_gt_max": int(gt.sum()), "n_nodata": int(nod.sum()),
            "n_invalid": int(bad.sum()),
            "zmin": float(zf.min()) if zf.size else None, "zmax": float(zf.max()) if zf.size else None,
            "has_zero_cells": bool((fin & (z == 0)).any()), "dtm_all_valid": not bool(bad.any())}


def scan_dtm(path: Path | str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Read one package raster (read-only, whole band 1) and return its lattice grid and `dtm_validity`."""
    import rasterio

    cell = float(cfg["dtm"]["cell_m"])
    with rasterio.open(path, mode="r") as ds:
        if ds.count != 1:
            raise ValueError(f"{path}: expected 1 band, found {ds.count}")
        grid = S.grid_from_transform(ds.transform, ds.width, ds.height, cell=cell)
        z = ds.read(1)
        nodata = ds.nodata
    out = {"grid": grid, "grid_xmin": grid.xmin, "grid_ymax": grid.ymax, "grid_w": grid.w, "grid_h": grid.h}
    out.update(dtm_validity(z, cfg, nodata=nodata))
    return out
