"""The provenance check of BRIEF.md (rows 1-4), gate G1 of PLAN.md. Writes `results/provenance.json`.

1. **Sources.** Metadata of the four layers (spatialReference, maxRecordCount, geometry type, fields), and
   `returnCountOnly=true` on rows 2 and 3 (and row 4 with the Carpathian `subregion IN (...)` filter), compared with
   the brief's numbers and with the hand-over files.
2. **Grids.** The recipe grid (note 1) of all 860 sheets against `dmr5g/fetch_log.jsonl` and each raster's own
   transform and size. Sheets are reprojected with pyproj (D03); the PROJ operation used is recorded. Every package
   raster is also scanned (NaN cells, 0 m cells, min/max) and compared with the fetch log's invalid_frac, zmin, zmax.
3. **DTM.** The provenance sample (config `replication.provenance.n_sheets`, stream `provenance_sample`) is
   re-downloaded with the recipe into `data/cache/provenance/` (cached files are reused unless `--refetch`) and
   compared with the package rasters: georeferencing tags equal, values bitwise equal where the package is not NaN,
   every package NaN explained by NoData or a value outside [-100, 2000] m in the download.
4. **Manifest.** sha256 of every `data/raw_data/` file against `MANIFEST.csv`.

G1 passes only if every check agrees. The script exits 0 on a pass and 2 on a fail. The committed
`results/provenance.json` has the sheet-grid coordinates removed (they derive from row 3, whose licence is not
stated); the full record goes to the gitignored `data/cache/provenance/provenance_full.json`.

    uv run python scripts/provenance_check.py
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio

from landslide import sheets as S
from landslide.config import config_hash, load_config
from landslide.logs import get_logger, progress, setup_logging
from landslide.net import Client
from landslide.paths import Paths, sheet_to_stem, stem_to_sheet

#: BRIEF.md, "Provenance check": the feature counts on 2026-11-25.
EXPECTED_COUNTS = {"register": 21356, "sheets": 860}
#: BRIEF.md, "Provenance check": re-download at least 10 randomly chosen sheets.
MIN_DTM_SHEETS = 10

log = get_logger("provenance")


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=root).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_sources(cfg: dict, paths: Paths, client: Client) -> dict[str, Any]:
    """Layer metadata and live counts for rows 1-4, against the brief and the hand-over files."""
    out: dict[str, Any] = {}
    dtm_url = S.dtm_service_url(cfg)
    out["dtm"] = {"url": dtm_url, "metadata": S.layer_metadata(client, dtm_url, keys=S.IMAGE_META_KEYS)}
    geology_where = S.sql_in("subregion", cfg["flysch"]["subregions"])
    rows = (("register", S.REGISTER_URL, "1=1", paths.register_gpkg),
            ("sheets", S.SHEETS_URL, "1=1", paths.surveyed_sheets),
            ("geology", S.GEOLOGY_URL, geology_where, paths.geology_gpkg))
    for name, url, where, pkg in rows:
        meta = S.layer_metadata(client, url)
        live = S.count_features(client, url, where)
        info = pyogrio.read_info(pkg)
        rec = {"url": url, "where": where, "metadata": meta, "count_live": live,
               "count_package": int(info["features"]), "package_crs": info["crs"],
               "package_fields": list(info["fields"])}
        if name in EXPECTED_COUNTS:
            rec["count_expected_brief"] = EXPECTED_COUNTS[name]
        rec["counts_agree"] = live == rec["count_package"] and live == rec.get("count_expected_brief", live)
        live_fields = [f for f, t in meta.get("fields", []) if t != "esriFieldTypeGeometry"]
        rec["package_fields_in_layer"] = all(f in live_fields for f in rec["package_fields"])
        log.info("%s: live %d, package %d, brief %s -> %s", name, live, rec["count_package"],
                 rec.get("count_expected_brief"), "agree" if rec["counts_agree"] else "DISAGREE")
        out[name] = rec
    return out


def check_grids(cfg: dict, paths: Paths, sheets_gdf, service_meta: dict) -> dict[str, Any]:
    """Recipe grid vs fetch log vs raster header, all sheets."""
    cell = float(cfg["dtm"]["cell_m"])
    epsg = S.epsg_code(cfg)
    recipe = {r.sheet: S.dtm_grid(r.geometry.bounds, cfg) for r in sheets_gdf.itertuples()}
    clearance = {r.sheet: S.edge_clearance(r.geometry.bounds, cfg) for r in sheets_gdf.itertuples()}
    flog = S.read_fetch_log(paths.dtm_fetch_log)
    rasters: dict[str, dict[str, Any]] = {}
    header_problems = []
    for s in progress(sorted(set(recipe) | set(flog)), desc="raster headers", unit="sheet"):
        p = paths.dtm(s)
        if not p.exists():
            rasters[s] = {"error": "missing"}
            continue
        with rasterio.open(p) as ds:
            rasters[s] = {"transform": list(ds.transform)[:6], "width": ds.width, "height": ds.height}
            if ds.count != 1 or ds.dtypes[0] != "float32" or (ds.crs.to_epsg() if ds.crs else None) != epsg:
                header_problems.append({"sheet": s, "count": ds.count, "dtype": ds.dtypes[0],
                                        "epsg": ds.crs.to_epsg() if ds.crs else None})
    extra = sorted({stem_to_sheet(p.stem) for p in paths.dmr5g_dir.glob("*.tif")} - set(recipe))
    res = S.compare_grids(recipe, flog, rasters, cell)
    worst = min(clearance, key=clearance.get)
    max_w, max_h = max(g.w for g in recipe.values()), max(g.h for g in recipe.values())
    res.update({
        "raster_header_problems": header_problems,
        "rasters_not_in_sheet_layer": extra,
        "min_edge_clearance_m": clearance[worst], "min_edge_clearance_sheet": worst,
        "n_sheets_clearance_below_1mm": sum(v < 1e-3 for v in clearance.values()),
        "max_request_w_h": [max_w, max_h],
        "service_max_image_w_h": [service_meta.get("maxImageWidth"), service_meta.get("maxImageHeight")],
    })
    res["within_service_limits"] = (max_w <= (service_meta.get("maxImageWidth") or 0)
                                    and max_h <= (service_meta.get("maxImageHeight") or 0))
    res["agree"] = (res["n_mismatched_sheets"] == 0 and not header_problems and not extra
                    and all(v == res["n_sheets"] for v in res["equal"].values()) and res["n_sheets"] == 860)
    log.info("grids: %s equal of %d; %d mismatched sheets; min edge clearance %.3g m (%s)", res["equal"],
             res["n_sheets"], res["n_mismatched_sheets"], clearance[worst], worst)
    return res


def check_package_rasters(paths: Paths) -> dict[str, Any]:
    """Every package raster: NaN count, zero cells and min/max, against the fetch log's invalid_frac, zmin, zmax."""
    flog = S.read_fetch_log(paths.dtm_fetch_log)
    disagree, n_nan, n_zero_sheets, n_neg = [], 0, 0, 0
    for s in progress(sorted(flog), desc="package scan", unit="sheet"):
        with rasterio.open(paths.dtm(s)) as ds:
            st = S.package_stats(ds.read(1))
        n_nan += st["n_nan"]
        n_zero_sheets += st["n_zero"] > 0
        n_neg += st["n_negative"]
        cmp = S.compare_fetch_log_stats(st, flog[s])
        if not all(cmp.values()):
            disagree.append({"sheet": s, **cmp, "raster": st,
                             "fetch_log": {k: flog[s].get(k) for k in ("invalid_frac", "zmin", "zmax")}})
    res = {"n_sheets": len(flog), "n_package_nan_total": n_nan, "n_sheets_with_zero_cells": n_zero_sheets,
           "n_negative_cells_total": n_neg, "n_fetch_log_stats_disagree": len(disagree),
           "fetch_log_stats_disagree": disagree[:50], "agree": not disagree}
    log.info("package scan: %d sheets, %d NaN cells, %d sheets with 0 m cells, %d fetch-log stat disagreements",
             len(flog), n_nan, n_zero_sheets, len(disagree))
    return res


def check_dtm(cfg: dict, paths: Paths, client: Client, sheets_gdf, refetch: bool) -> dict[str, Any]:
    """Re-download the provenance sample and compare each sheet with the package raster."""
    recipe = {r.sheet: S.dtm_grid(r.geometry.bounds, cfg) for r in sheets_gdf.itertuples()}
    drawn = S.draw_provenance_sample(cfg, recipe)
    log.info("provenance sample (draw order): %s", " ".join(drawn))
    out_dir = paths.provenance_dir
    per_sheet = []
    for s in progress(drawn, desc="dtm sample", unit="sheet"):
        g = recipe[s]
        tif, side = out_dir / f"{sheet_to_stem(s)}.tif", out_dir / f"{sheet_to_stem(s)}.json"
        if refetch or not (tif.exists() and side.exists()):
            fetch = S.fetch_dtm(client, cfg, s, g, out_dir)
        else:
            fetch = json.loads(side.read_text(encoding="utf-8"))
            fetch["reused_cache"] = True
        file_sha = S.sha256_file(tif)
        geo_pkg, geo_dl = S.geotiff_georef(paths.dtm(s)), S.geotiff_georef(tif)
        georef = S.compare_georef(geo_pkg, geo_dl)
        georef["download_transform_is_recipe"] = tuple(geo_dl["transform"]) == g.transform and \
            (geo_dl["width"], geo_dl["height"]) == (g.w, g.h)
        with rasterio.open(paths.dtm(s)) as a, rasterio.open(tif) as b:
            pkg, dl = a.read(1), b.read(1)
            dl_nodata = b.nodata
        values = S.compare_dtm(pkg, dl, dl_nodata)
        ok = bool(values["ok"] and georef["equal"] and georef["download_transform_is_recipe"]
                  and file_sha == fetch["sha256"])
        per_sheet.append({"sheet": s, "ok": ok, "grid": g.as_dict(),
                          "request": {"url": fetch["url"], "params": fetch["params"]},
                          "response": {k: fetch.get(k) for k in ("retrieved_utc", "status", "content_type", "etag",
                                                                 "bytes", "sha256", "reused_cache")},
                          "cached_file_sha256_matches_response": file_sha == fetch["sha256"],
                          "georef": georef,
                          "georef_package": {k: geo_pkg[k] for k in ("geokeys", "gdal_crs_epsg", "nodata",
                                                                     "gdal_nodata_tag")},
                          "georef_download": {k: geo_dl[k] for k in ("geokeys", "gdal_crs_epsg", "nodata",
                                                                     "gdal_nodata_tag", "gdal_crs_wkt")},
                          "values": values})
        log.info("%s: %s; compared %d px, %d bitwise equal, %d differ; package NaN %d (unexplained %d)", s,
                 "OK" if ok else "DISAGREE", values.get("n_compared", 0), values.get("n_bitwise_equal", 0),
                 values.get("diff", {}).get("n", 0), values.get("n_package_nan", 0),
                 values.get("n_package_nan_unexplained", 0))
    def tot(k: str) -> int:
        return int(sum(p["values"].get(k, 0) for p in per_sheet))

    summary = {
        "n_sheets": len(per_sheet), "n_ok": sum(p["ok"] for p in per_sheet),
        "n_pixels": tot("n_pixels"), "n_compared": tot("n_compared"), "n_bitwise_equal": tot("n_bitwise_equal"),
        "n_diff": int(sum(p["values"].get("diff", {}).get("n", 0) for p in per_sheet)),
        "n_package_nan": tot("n_package_nan"), "n_package_nan_unexplained": tot("n_package_nan_unexplained"),
        "n_download_invalid_but_package_valued": tot("n_download_invalid_but_package_valued"),
        "n_georef_equal": sum(p["georef"]["equal"] for p in per_sheet),
    }
    return {"sample": {**S.provenance_sample_spec(cfg), "drawn": drawn, "drawn_sorted": sorted(drawn)},
            "summary": summary, "per_sheet": per_sheet, "agree": summary["n_ok"] == len(per_sheet) > 0}


def check_manifest(paths: Paths) -> dict[str, Any]:
    rows = S.read_manifest(paths.manifest)
    total = sum((paths.raw / r["file"]).stat().st_size for r in rows if (paths.raw / r["file"]).is_file())
    t0 = time.monotonic()
    with progress(total=total, desc="sha256", unit="B", unit_scale=True) as bar:
        res = S.check_manifest(paths.raw, rows, on_chunk=bar.update)
    res["seconds"] = round(time.monotonic() - t0, 1)
    log.info("manifest: %d listed, %d ok, %d mismatched, %d missing, unlisted %s (%.0f s)", res["n_listed"],
             res["n_ok"], res["n_mismatch"], res["n_missing"], [u["file"] for u in res["unlisted"]],
             res["seconds"])
    return res


def redact_coordinates(result: dict[str, Any]) -> dict[str, Any]:
    """A copy without the EPSG:5514 coordinates of sheet grids (they derive from row 3, whose licence is not stated).

    Sizes, pixel offsets, counts and sheet numbers stay; the full record is kept in the gitignored cache.
    """
    out = copy.deepcopy(result)
    for p in out["dtm"]["per_sheet"]:
        p["grid"] = {"w": p["grid"]["w"], "h": p["grid"]["h"]}
        p["request"]["params"]["bbox"] = "redacted: see data/cache/provenance/<sheet>.json"
    for m in out["grids"]["mismatches"]:
        for k in ("grid_recipe", "grid_fetch_log", "grid_raster"):
            m.pop(k, None)
        for k in ("fetch_log", "raster"):
            if isinstance(m.get(k), str) and "corner" in m[k]:
                m[k] = "off the lattice (coordinates in the full record)"
    for name in ("sheets", "geology"):  # rows 3 and 4: no stated or verified licence
        out["sources"][name]["metadata"].pop("extent", None)
    out["redacted"] = ("EPSG:5514 sheet-grid coordinates and the layer extents of rows 3 and 4 removed (licence not "
                       "stated or verified); the full record is data/cache/provenance/provenance_full.json")
    return out


def main(argv: list[str] | None = None) -> int:
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=paths.results / "provenance.json")
    ap.add_argument("--refetch", action="store_true", help="re-download the DTM sample even if cached")
    ap.add_argument("--skip-manifest", action="store_true", help="skip the sha256 pass (the gate then fails)")
    args = ap.parse_args(argv)

    setup_logging(paths.logs / "provenance_check.log")
    cfg = load_config()
    started = datetime.now(timezone.utc)
    sheets_gdf = S.load_sheets(paths.surveyed_sheets)
    lonlat = np.unique(np.concatenate([np.asarray(g.exterior.coords)
                                       for g in gpd.read_file(paths.surveyed_sheets).geometry]), axis=0)

    result: dict[str, Any] = {
        "task": "P1-03",
        "started_utc": started.isoformat(timespec="seconds"),
        "git_commit": _git_commit(paths.root),
        "config_hash": config_hash(cfg),
        "environment": {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
                        "numpy": np.__version__, "pyogrio": pyogrio.__version__},
        "reprojection_D03": S.reprojection_report(lonlat),
        "sheets": {"n": len(sheets_gdf), "n_full": int(sheets_gdf["full"].sum()),
                   "n_partial": int((~sheets_gdf["full"]).sum())},
    }
    with Client() as client:
        result["sources"] = check_sources(cfg, paths, client)
        result["grids"] = check_grids(cfg, paths, sheets_gdf, result["sources"]["dtm"]["metadata"])
        result["dtm"] = check_dtm(cfg, paths, client, sheets_gdf, args.refetch)
        result["n_requests"] = client.n_requests
    result["package_rasters"] = check_package_rasters(paths)
    result["manifest"] = {"skipped": True, "ok": False} if args.skip_manifest else check_manifest(paths)

    failures = []
    for name in ("register", "sheets", "geology"):
        src = result["sources"][name]
        if not src["counts_agree"]:
            failures.append(f"count {name}: live {src['count_live']}, package {src['count_package']}, "
                            f"brief {src.get('count_expected_brief')}")
        if not src["package_fields_in_layer"]:
            failures.append(f"fields {name}: package fields not all in the live layer")
    if not result["grids"]["agree"]:
        failures.append(f"grids: {result['grids']['n_mismatched_sheets']} sheets disagree "
                        f"(equal counts {result['grids']['equal']})")
    if not result["package_rasters"]["agree"]:
        failures.append(f"package rasters: {result['package_rasters']['n_fetch_log_stats_disagree']} sheets whose "
                        "NaN fraction, zmin or zmax disagree with fetch_log.jsonl")
    if not result["grids"]["within_service_limits"]:
        failures.append("grids: a request exceeds the service's maxImageWidth/Height")
    if result["dtm"]["summary"]["n_sheets"] < MIN_DTM_SHEETS:
        failures.append(f"dtm: only {result['dtm']['summary']['n_sheets']} sheets compared (brief: at least "
                        f"{MIN_DTM_SHEETS})")
    for p in result["dtm"]["per_sheet"]:
        if not p["ok"]:
            failures.append(f"dtm {p['sheet']}: georef {p['georef']}, diff {p['values'].get('diff')}, "
                            f"unexplained NaN {p['values'].get('n_package_nan_unexplained')}, "
                            f"invalid kept {p['values'].get('n_download_invalid_but_package_valued')}")
    if not result["manifest"]["ok"]:
        failures.append("manifest: skipped" if result["manifest"].get("skipped") else
                        f"manifest: {result['manifest']['n_mismatch']} mismatched, "
                        f"{result['manifest']['n_missing']} missing")
    result["gate_G1"] = {"pass": not failures, "failures": failures}
    result["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    full = paths.provenance_dir / "provenance_full.json"
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(S.to_jsonable(result), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(S.to_jsonable(redact_coordinates(result)), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    log.info("G1 %s; wrote %s (full record: %s)", "PASS" if not failures else "FAIL", args.out, full)
    for f in failures:
        log.error("G1: %s", f)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
