"""Geo preparation and splits (task P1-08, gate G2). Reads rows 1-4 in place; writes the derived files and two
tracked summaries.

Steps (PLAN.md P1-08; Appendix A §3-§4; decisions D04-D10 in docs/decisions/P1-08.md):
1. Polygons: `in_mapped` (representative point within the union of all 860 sheets), `eligible`, `kraj1`.
2. Flysch outline (closing, fill holes, drop parts < 1 km²), each sheet's flysch fraction and flag. The other reading
   of "2 km closing" (buffer 1 km) is computed alongside and reported, never used.
3. DTM validity: the 860 package rasters, one at a time (grid from the raster, 0 m cells, invalid cells).
4. Eligible polygon -> sheet (largest share of area); `n_eligible` per sheet.
5. Sheet kraj (D06), transfer group, stratum.
6. Stage-2 sheets: the 20 of Appendix A §4, checked (full, >= 5 eligible, valid DTM, flysch flag). Never swapped.
7. Stage-1 sheets, 15 km blocks (D08), (a) folds (D09), (b) groups and inner-val blocks (D10).

Outputs (PLAN.md §4.3):
- `data/derived/sheets.parquet`, `polygons.parquet`, `flysch_outline.gpkg`, `splits.json` (gitignored);
- `results/splits_summary.json` and `results/sheet_splits.csv` (tracked): sheet numbers, kraj, flysch flags, opaque
  block ids, folds, groups, stage flags and aggregate counts only. No geometry, coordinates or flysch fractions.

`--out DIR` writes every output under DIR instead (DIR/derived, DIR/results), e.g. to check that a re-run is
identical. Exit code 0; 2 if a stage-2 sheet fails its checks (reported, not fixed).

    uv run python scripts/build_geo.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from landslide import geo as G
from landslide import sheets as S
from landslide import splits as SP
from landslide.config import config_hash, load_config
from landslide.logs import get_logger, progress, setup_logging
from landslide.paths import RUN_NAMES, Paths

log = get_logger("build_geo")


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=root).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment() -> dict[str, str]:
    """Library versions that can change the geometry (PROJ picks the S-JTSK operation per area; see P1-03 D03)."""
    import pyogrio
    import pyproj
    import rasterio

    return {"pyproj": pyproj.__version__, "proj": pyproj.proj_version_str, "shapely": shapely.__version__,
            "geos": shapely.geos_version_string, "geopandas": gpd.__version__, "pyogrio": pyogrio.__version__,
            "gdal": rasterio.__gdal_version__, "numpy": np.__version__, "pandas": pd.__version__}


def _counts(values: Any) -> dict[str, int]:
    s = pd.Series(values).dropna()
    return {str(k): int(v) for k, v in sorted(s.value_counts().items(), key=lambda kv: str(kv[0]))}


# ======================================================================================================================
# Steps
# ======================================================================================================================


def polygons_step(cfg: dict, reg: gpd.GeoDataFrame, mapped: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """In_mapped (D04), eligible (D05) and kraj1 for every register polygon."""
    geoms = reg.geometry.values
    inm = G.in_mapped(geoms, mapped)
    area = shapely.area(geoms)
    elig = G.eligible_mask(reg["skupina"].values, inm, area, cfg)
    elig_src = G.eligible_mask(reg["skupina"].values, inm, reg["st_area"].values, cfg)
    valid = shapely.is_valid(geoms)
    # Checks of the reading, reported only: boundary-inclusive test, repaired geometries.
    inm_cov = shapely.covered_by(G.representative_points(geoms), mapped)
    rep = G.repair(geoms)
    inm_rep = shapely.within(G.representative_points(rep), mapped)
    elig_rep = G.eligible_mask(reg["skupina"].values, inm_rep, shapely.area(rep), cfg)
    df = pd.DataFrame({"objectid": reg["objectid"].values, "skupina": reg["skupina"].values, "area_m2": area,
                       "in_mapped": inm, "eligible": elig,
                       "kraj1": [G.first_kraj(k) for k in reg["gen_kraj"].values], "geom_valid": valid})
    diff_src = reg["objectid"].values[elig != elig_src]
    summary = {
        "n_total": int(len(df)), "n_in_mapped": int(inm.sum()), "n_eligible": int(elig.sum()),
        "n_skupina": _counts(reg["skupina"].values),
        "n_invalid_geometries": int((~valid).sum()),
        "eligible_by_kraj1": _counts(df.loc[elig, "kraj1"]),
        "checks": {
            "n_in_mapped_covered_by": int(inm_cov.sum()),
            "n_in_mapped_repaired_geometry": int(inm_rep.sum()),
            "n_eligible_repaired_geometry": int(elig_rep.sum()),
            "n_eligible_source_area_field": int(elig_src.sum()),
            "objectids_eligible_differs_source_area_field": [int(v) for v in diff_src],
            "source_area_field": G.SRC_AREA_FIELD,
            "geometry_minus_source_area_m2": {
                "median": float(np.median(area - reg["st_area"].values)),
                "max_abs": float(np.max(np.abs(area - reg["st_area"].values)))},
        },
    }
    return df, summary


def flysch_step(cfg: dict, paths: Paths, sheet_geoms: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    """The outline under the default reading (buffer = closing_m) and the other one (buffer = closing_m / 2)."""
    geo = gpd.read_file(paths.geology_gpkg)
    if geo.crs is None or geo.crs.to_epsg() != 5514:
        raise ValueError(f"{paths.geology_gpkg}: expected EPSG:5514, got {geo.crs}")
    d = float(cfg["flysch"]["closing_m"])
    thr = float(cfg["flysch"]["min_sheet_frac"])
    out: dict[str, Any] = {}
    summ: dict[str, Any] = {"n_geology_polygons_file": int(len(geo)), "min_sheet_frac": thr}
    for name, buf in (("default", d), ("alt", d / 2)):
        outline, stages = G.flysch_outline(geo.geometry.values, geo["subregion"].values, cfg, buffer_m=buf)
        frac = G.area_fraction_inside(sheet_geoms, outline)
        out[name] = {"outline": outline, "frac": frac, "flag": frac >= thr, "stages": stages}
        summ[name] = {"reading": ("closing = buffer(+d).buffer(-d) with d = closing_m" if name == "default" else
                                  "closing with a 2 km structuring element (diameter): d = closing_m / 2"),
                      "buffer_m": buf, "area_km2": stages["final"]["area_km2"], "stages": stages,
                      "n_flysch_sheets": int((frac >= thr).sum()),
                      "n_sheets_frac_within_0.05_of_threshold": int((np.abs(frac - thr) < 0.05).sum())}
    summ["n_sheets_flag_differs_between_readings"] = int((out["default"]["flag"] != out["alt"]["flag"]).sum())
    summ["used"] = "default"
    return out, summ


def dtm_step(cfg: dict, paths: Paths, sheets: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream the 860 rasters one at a time: grid from the raster, validity flags and cell counts."""
    rows = []
    for s, geom in progress(list(zip(sheets["sheet"], sheets.geometry.values)), desc="DTM validity", unit="sheet"):
        p = paths.dtm(s)
        r = G.scan_dtm(p, cfg)
        recipe = S.dtm_grid(geom.bounds, cfg)
        grid = r.pop("grid")
        r.update({"sheet": s, "dtm_path": p.relative_to(paths.raw).as_posix(),
                  "grid_matches_recipe": grid.key() == recipe.key()})
        rows.append(r)
    df = pd.DataFrame(rows)
    summ = {"n_sheets": int(len(df)), "n_has_zero_cells": int(df["has_zero_cells"].sum()),
            "n_dtm_all_valid": int(df["dtm_all_valid"].sum()),
            "n_with_nonfinite": int((df["n_nonfinite"] > 0).sum()),
            "n_with_le_min": int((df["n_le_min"] > 0).sum()), "n_with_gt_max": int((df["n_gt_max"] > 0).sum()),
            "n_with_negative": int((df["zmin"] < 0).sum()),
            "n_grid_matches_recipe": int(df["grid_matches_recipe"].sum()),
            "invalid_rule": cfg["dtm"]["invalid_rule"], "invalid_bounds": list(G.invalid_bounds(cfg)),
            "zmax_all_sheets": float(df["zmax"].max()),
            "n_zero_cells_total": int(df["n_zero"].sum())}
    return df, summ


# ======================================================================================================================
# Build
# ======================================================================================================================


def build(cfg: dict, paths: Paths) -> dict[str, Any]:
    t0 = time.time()
    sheets = S.load_sheets(paths.surveyed_sheets)  # sorted by sheet, EPSG:5514 (D03)
    sg = sheets.geometry.values
    sheets["area_m2"] = shapely.area(sg)
    cen = shapely.centroid(sg)
    sheets["centroid_x"], sheets["centroid_y"] = shapely.get_x(cen), shapely.get_y(cen)
    mapped = shapely.union_all(sg)  # D04: all 860 sheets, partial ones included
    log.info("sheets: %d (%d full); union %.1f km²", len(sheets), int(sheets["full"].sum()), mapped.area / 1e6)

    # 1. polygons
    reg = G.load_register(paths.register_gpkg)
    poly, poly_summ = polygons_step(cfg, reg, mapped)
    log.info("polygons: %d, in_mapped %d, eligible %d (source area field: %d)", poly_summ["n_total"],
             poly_summ["n_in_mapped"], poly_summ["n_eligible"],
             poly_summ["checks"]["n_eligible_source_area_field"])

    # 2. flysch
    fl, fl_summ = flysch_step(cfg, paths, sg)
    sheets["flysch_frac"] = fl["default"]["frac"]
    sheets["flysch"] = fl["default"]["flag"]
    sheets["flysch_frac_alt"] = fl["alt"]["frac"]
    log.info("flysch outline %.1f km² (alt %.1f km²); flysch sheets %d (alt %d)", fl_summ["default"]["area_km2"],
             fl_summ["alt"]["area_km2"], fl_summ["default"]["n_flysch_sheets"], fl_summ["alt"]["n_flysch_sheets"])

    # 3. DTM validity
    dtm, dtm_summ = dtm_step(cfg, paths, sheets)
    sheets = sheets.merge(dtm, on="sheet", how="left", validate="one_to_one")
    log.info("DTM: %d sheets with 0 m cells, %d fully valid, grids equal to the recipe %d",
             dtm_summ["n_has_zero_cells"], dtm_summ["n_dtm_all_valid"], dtm_summ["n_grid_matches_recipe"])

    # 4. eligible polygon -> sheet
    el = np.flatnonzero(poly["eligible"].values)
    asg, share = G.assign_to_sheets(reg.geometry.values[el], sg, sheets["sheet"].tolist())
    poly["sheet"] = pd.Series([None] * len(poly), dtype=object)
    poly.loc[el, "sheet"] = asg
    poly["sheet_share"] = np.nan
    poly.loc[el, "sheet_share"] = share
    n_el = poly.loc[el, "sheet"].value_counts()
    sheets["n_eligible"] = sheets["sheet"].map(n_el).fillna(0).astype("int64")
    asg_summ = {"n_eligible_assigned": int(pd.notna(asg).sum()), "n_eligible_unassigned": int(pd.isna(asg).sum()),
                "n_share_below_0.5": int((share < 0.5).sum()), "n_share_below_1": int((share < 1 - 1e-9).sum()),
                "min_share": float(share.min()) if len(share) else None,
                "n_sheets_with_eligible": int((sheets["n_eligible"] > 0).sum())}

    # 5. kraj, transfer group, stratum
    kr = G.sheet_kraj(sg, reg.geometry.values, reg["gen_kraj"].values)
    for c in kr.columns:
        sheets[c] = pd.Series(kr[c].tolist(), index=sheets.index, dtype=object)
    sheets["kraj"] = pd.Series([None if SP._missing(k) else k for k in kr["kraj"]], index=sheets.index, dtype=object)
    train_kraje = cfg["splits"]["transfer_train_kraje"]
    sheets["group_b"] = pd.Series([SP.transfer_group(k, train_kraje) for k in sheets["kraj"]], index=sheets.index,
                                  dtype=object)
    sheets["stratum"] = pd.Series([SP.stratum(f, g) for f, g in zip(sheets["flysch"], sheets["group_b"])],
                                  index=sheets.index, dtype=object)

    # 6. stage-2 sheets
    expected = SP.STAGE2_SHEETS
    s2_all = set(expected["flysch"]) | set(expected["nonflysch"])
    sheets["is_stage2"] = sheets["sheet"].isin(s2_all)
    s2_checks = SP.check_stage2(sheets, expected, cfg)
    s2_sizes = SP.stage2_list_sizes_ok(expected, cfg)
    s2_fail = [c for c in s2_checks if not c["pass"]]
    for c in s2_fail:
        log.error("stage-2 sheet %s (%s list) fails: %s", c["sheet"], c["list"], ", ".join(c["failed"]))

    # 7. stage-1 sheets and blocks
    sheets["is_stage1"] = ~sheets["is_stage2"] & (sheets["n_eligible"] >= 1)
    if sheets.loc[sheets["is_stage1"], "stratum"].isna().any():
        raise RuntimeError("a stage-1 sheet has no kraj")
    block_m = float(cfg["splits"]["block_km"]) * 1000.0
    cells = [SP.block_cell(x, y, block_m) for x, y in zip(sheets["centroid_x"], sheets["centroid_y"])]
    sheets["block_cx"] = [c[0] for c in cells]
    sheets["block_cy"] = [c[1] for c in cells]
    st1 = sheets["is_stage1"].values
    ids = SP.block_ids(sheets.loc[st1, "sheet"].tolist(), sheets.loc[st1, "stratum"].tolist(),
                       sheets.loc[st1, "block_cx"].tolist(), sheets.loc[st1, "block_cy"].tolist())
    sheets["block"] = pd.Series([None] * len(sheets), dtype=object)
    sheets.loc[st1, "block"] = ids

    s1 = sheets[sheets["is_stage1"]]
    bt = (s1.groupby("block", sort=True)
          .agg(stratum=("stratum", "first"), group_b=("group_b", "first"), n_stratum=("stratum", "nunique"),
               n_sheets=("sheet", "size"), n_eligible=("n_eligible", "sum"), sheets=("sheet", lambda v: sorted(v)))
          .reset_index())
    if (bt["n_stratum"] != 1).any():
        raise RuntimeError("a block spans two strata")

    # (a) folds over the flysch blocks (D09)
    n_folds = int(cfg["splits"]["n_folds_a"])
    fly = bt[bt["stratum"].isin(["F_ZLMS", "F_other"])]
    folds = SP.assign_folds(dict(zip(fly["block"], fly["n_eligible"])), n_folds, SP.FOLD_SEED)
    bt["fold_a"] = bt["block"].map(folds).astype("Int64")
    sheets["fold_a"] = sheets["block"].map(folds).astype("Int64")

    # inner validation (D10)
    frac = float(cfg["splits"]["inner_val_frac"])
    bn = dict(zip(bt["block"], bt["n_eligible"]))
    inner: dict[str, list[str]] = {}
    inner_summ: dict[str, Any] = {}
    for run in RUN_NAMES:
        tb = SP.training_blocks(run, bt)
        iv = SP.inner_val_blocks({b: bn[b] for b in tb}, frac, SP.inner_val_seed(run))
        inner[run] = iv
        n_train = int(sum(bn[b] for b in tb))
        n_iv = int(sum(bn[b] for b in iv))
        inner_summ[run] = {"seed": SP.inner_val_seed(run), "n_training_blocks": len(tb),
                           "n_training_eligible": n_train, "n_inner_val_blocks": len(iv),
                           "n_inner_val_eligible": n_iv, "inner_val_frac_eligible": n_iv / n_train if n_train else None,
                           "n_inner_val_sheets": int(sum(len(bt.set_index("block").at[b, "sheets"]) for b in iv))}

    # ---- splits.json ----
    chash = config_hash(cfg)
    folds_a = {}
    for k in range(n_folds):
        fb = sorted(b for b, f in folds.items() if f == k)
        folds_a[str(k)] = {"blocks": fb, "sheets": sorted(s1.loc[s1["block"].isin(fb), "sheet"]),
                           "n_eligible": int(sum(bn[b] for b in fb))}
    splits = {
        "config_hash": chash,
        "stage2": {"flysch": list(expected["flysch"]), "nonflysch": list(expected["nonflysch"])},
        "folds_a": folds_a,
        "group_b": {"train_blocks": sorted(bt.loc[bt["group_b"] == SP.GROUP_TRAIN, "block"]),
                    "test_blocks": sorted(bt.loc[bt["group_b"] == SP.GROUP_TEST, "block"])},
        "inner_val": {run: sorted(v) for run, v in inner.items()},
        "stage1_sheets": sorted(s1["sheet"]),
        "blocks": {r.block: {"stratum": r.stratum, "group_b": r.group_b,
                             "fold_a": None if pd.isna(r.fold_a) else int(r.fold_a),
                             "n_eligible": int(r.n_eligible), "sheets": list(r.sheets)} for r in bt.itertuples()},
        "seeds": {"folds_a": SP.FOLD_SEED, "inner_val": {run: SP.inner_val_seed(run) for run in RUN_NAMES}},
    }

    # ---- summary (tracked; aggregate only) ----
    def strat_table(df: pd.DataFrame, blocks: pd.DataFrame) -> dict[str, Any]:
        return {s: {"n_sheets": int((df["stratum"] == s).sum()), "n_blocks": int((blocks["stratum"] == s).sum()),
                    "n_eligible": int(df.loc[df["stratum"] == s, "n_eligible"].sum())} for s in SP.STRATA}

    s2df = sheets[sheets["is_stage2"]]
    full = sheets["full"].values
    pool = full & (sheets["n_eligible"].values >= int(cfg["splits"]["stage2"]["min_eligible_polygons"])) \
        & sheets["dtm_all_valid"].values
    summary: dict[str, Any] = {
        "task": "P1-08",
        "config_hash": chash,
        "code_commit": _git_commit(paths.root),
        "environment": _environment(),
        "polygons": poly_summ,
        "assignment": asg_summ,
        "sheets": {"n": int(len(sheets)), "n_full": int(full.sum()), "n_partial": int((~full).sum()),
                   "n_with_kraj": int(sheets["kraj"].notna().sum()), "kraj": _counts(sheets["kraj"]),
                   "kraj_rule": _counts(sheets["kraj_rule"]),
                   "n_flysch": int(sheets["flysch"].sum()), "n_with_eligible": asg_summ["n_sheets_with_eligible"]},
        "dtm": dtm_summ,
        "flysch": fl_summ,
        "stage2": {"lists": s2_sizes, "checks": s2_checks, "all_pass": not s2_fail,
                   "n_targets": int(s2df["n_eligible"].sum()),
                   "n_targets_flysch_list": int(s2df.loc[s2df["sheet"].isin(expected["flysch"]), "n_eligible"].sum()),
                   "n_targets_nonflysch_list": int(
                       s2df.loc[s2df["sheet"].isin(expected["nonflysch"]), "n_eligible"].sum()),
                   "candidate_pool_info": {
                       "note": "full sheets with >= min_eligible_polygons and a fully valid DTM (not redrawn)",
                       "n_flysch": int((pool & sheets["flysch"].values).sum()),
                       "n_nonflysch": int((pool & ~sheets["flysch"].values).sum())}},
        "stage1": {"n_sheets": int(len(s1)), "n_eligible": int(s1["n_eligible"].sum()),
                   "n_blocks": int(len(bt)), "strata": strat_table(s1, bt),
                   "n_sheets_with_zero_cells": int(s1["has_zero_cells"].sum()),
                   "block_km": cfg["splits"]["block_km"],
                   "n_sheets_per_block": {"min": int(bt["n_sheets"].min()), "max": int(bt["n_sheets"].max())}},
        "folds_a": {"seed": SP.FOLD_SEED, "n_folds": n_folds, "per_fold": {
            k: {"n_blocks": len(v["blocks"]), "n_sheets": len(v["sheets"]), "n_eligible": v["n_eligible"],
                "n_blocks_F_ZLMS": int(bt.loc[bt["block"].isin(v["blocks"]), "stratum"].eq("F_ZLMS").sum()),
                "n_blocks_F_other": int(bt.loc[bt["block"].isin(v["blocks"]), "stratum"].eq("F_other").sum())}
            for k, v in folds_a.items()}},
        "group_b": {g: {"n_blocks": int((bt["group_b"] == g).sum()), "n_sheets": int((s1["group_b"] == g).sum()),
                        "n_eligible": int(s1.loc[s1["group_b"] == g, "n_eligible"].sum())}
                    for g in (SP.GROUP_TRAIN, SP.GROUP_TEST)},
        "inner_val": {"frac": frac, "rule": "random training blocks until >= frac of the run's training eligible "
                                            "polygons (D10)", "runs": inner_summ},
    }
    log.info("build finished in %.1f s", time.time() - t0)  # not in the summary, so re-runs are byte-identical
    return {"sheets": sheets, "poly": poly, "reg": reg, "flysch": fl, "splits": splits, "summary": summary,
            "s2_fail": s2_fail}


# ======================================================================================================================
# Writing
# ======================================================================================================================


SHEET_COLS = ["sheet", "objectid", "poznamka", "full", "area_m2", "centroid_x", "centroid_y",
              "grid_xmin", "grid_ymax", "grid_w", "grid_h", "dtm_path", "grid_matches_recipe",
              "has_zero_cells", "dtm_all_valid", "n_cells", "n_zero", "n_nonfinite", "n_le_min", "n_gt_max",
              "n_nodata", "n_invalid", "zmin", "zmax",
              "flysch_frac", "flysch", "flysch_frac_alt",
              "kraj", "kraj_rule", "kraj_n_polygons", "kraj_n_votes", "kraj_votes",
              "group_b", "stratum", "block_cx", "block_cy", "block", "fold_a", "is_stage1", "is_stage2", "n_eligible",
              "geometry"]
POLY_COLS = ["objectid", "skupina", "area_m2", "in_mapped", "eligible", "kraj1", "sheet", "sheet_share",
             "geom_valid", "geometry"]
CSV_COLS = ["sheet", "kraj", "flysch", "stratum", "block", "fold_a", "group_b", "is_stage1", "is_stage2"]


def write_outputs(res: dict[str, Any], derived: Path, results: Path) -> list[Path]:
    derived.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    sheets = gpd.GeoDataFrame(res["sheets"][SHEET_COLS], geometry="geometry", crs="EPSG:5514")
    poly = gpd.GeoDataFrame(res["poly"].assign(geometry=res["reg"].geometry.values)[POLY_COLS],
                            geometry="geometry", crs="EPSG:5514")
    out = []
    p = derived / "sheets.parquet"
    sheets.to_parquet(p, index=False)
    out.append(p)
    p = derived / "polygons.parquet"
    poly.to_parquet(p, index=False)
    out.append(p)

    p = derived / "flysch_outline.gpkg"
    if p.exists():
        p.unlink()
    for layer, name in (("flysch_outline", "default"), ("flysch_outline_alt", "alt")):
        f = res["flysch"][name]
        gdf = gpd.GeoDataFrame({"reading": [name], "buffer_m": [f["stages"]["buffer_m"]],
                                "area_km2": [f["stages"]["final"]["area_km2"]]},
                               geometry=[f["outline"]], crs="EPSG:5514")
        gdf.to_file(p, layer=layer, driver="GPKG")
    out.append(p)

    p = derived / "splits.json"
    p.write_text(json.dumps(S.to_jsonable(res["splits"]), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    out.append(p)

    p = results / "splits_summary.json"
    p.write_text(json.dumps(S.to_jsonable(res["summary"]), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    out.append(p)
    p = results / "sheet_splits.csv"
    res["sheets"][CSV_COLS].to_csv(p, index=False, lineterminator="\n", encoding="utf-8")
    out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="write everything under this directory (OUT/derived, OUT/results) instead")
    args = ap.parse_args(argv)
    derived = args.out / "derived" if args.out else paths.derived
    results = args.out / "results" if args.out else paths.results

    setup_logging(paths.logs / "build_geo.log")
    cfg = load_config()
    log.info("build_geo: config %s, derived -> %s, results -> %s", config_hash(cfg)[:12], derived, results)
    res = build(cfg, paths)
    for p in write_outputs(res, derived, results):
        log.info("wrote %s", p)
    if res["s2_fail"]:
        log.error("%d stage-2 sheet(s) fail their checks; reported, not swapped", len(res["s2_fail"]))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
