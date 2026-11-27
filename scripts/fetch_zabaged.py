"""Row 6: ZABAGED polohopis (ČÚZK, CC BY 4.0) layers 118, 119, 130, 129, 79 and 75 around every surveyed sheet.

Each layer's metadata (and its national feature count) is read first. Then, per layer and per sheet, an envelope
query on the sheet's DTM grid (bbox of the package DTM raster, `inSR=5514`, `esriSpatialRelIntersects`, no `outSR`)
returns the object ids, and the features are downloaded as Esri JSON in the native SR, paginated by object id with
`landslide.sheets.query_pages` (each page must return exactly its ids). Every response page is saved unmodified under
`data/cache/zabaged/pages/<layer>/<sheet>/`, and each finished (layer, sheet) is appended to `fetch_log.jsonl`, so a
re-run skips it. The run stops at the first error (after the client's retries).

The build step reads the pages locally (GDAL ESRIJSON), drops duplicate object ids (a feature near a sheet edge is
returned for several sheets; duplicates must be identical, which is checked) and writes one GeoPackage per layer,
`data/cache/zabaged/<layer>.gpkg` (EPSG:5514), with the service's attributes unchanged.

    uv run python scripts/fetch_zabaged.py --sheets 25-14-17 04-33-10   # try two sheets
    uv run python scripts/fetch_zabaged.py --all                        # all 860 sheets (resumes), then build
    uv run python scripts/fetch_zabaged.py --build-only                 # rebuild the GeoPackages; no network
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from landslide import sheets as S
from landslide.config import config_hash, load_config
from landslide.logs import get_logger, progress, setup_logging
from landslide.net import Client
from landslide.paths import Paths, sheet_to_stem

SECTION = "zabaged"
#: The brief's name of each role (BRIEF.md row 6), written as `name_brief` in `results/sources.json`. Labels only:
#: the service and the layer ids come from the config's `sources.zabaged` block (`service`, `layers`: role -> id).
BRIEF_NAMES: dict[str, str] = {"quarry": "quarry", "spoil": "spoil heap", "rock": "rock formation",
                               "landslide": "landslide/scree", "road": "road", "railway": "railway"}
SERVICE_META_KEYS = ("currentVersion", "mapName", "serviceDescription", "copyrightText", "spatialReference",
                     "maxRecordCount", "capabilities", "supportedQueryFormats")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def service_url(cfg: Mapping[str, Any]) -> str:
    """The ZABAGED polohopis MapServer (`sources.zabaged.service`)."""
    return str(cfg["sources"]["zabaged"]["service"]).rstrip("/")


def layers(cfg: Mapping[str, Any]) -> dict[int, str]:
    """Layer id -> the brief's name, in the brief's order (BRIEF.md row 6; `sources.zabaged.layers`). The GeoPackage
    is named by the layer id."""
    return {int(i): BRIEF_NAMES[role] for role, i in cfg["sources"]["zabaged"]["layers"].items()}


def layer_key(layer: int) -> str:
    """The `Paths.zabaged` key of a layer: its id as text (`data/cache/zabaged/118.gpkg`)."""
    return str(int(layer))


# ======================================================================================================================
# Envelope queries
# ======================================================================================================================


def envelope_params(bbox: tuple[float, float, float, float], epsg: int) -> dict[str, Any]:
    """The spatial filter of one sheet: its DTM grid's envelope in the native SR. No outSR."""
    return {"geometry": ",".join(repr(float(round(v, 6))) for v in bbox), "geometryType": "esriGeometryEnvelope",
            "inSR": epsg, "spatialRel": "esriSpatialRelIntersects"}


class EnvelopeClient:
    """Forwards to a `landslide.net.Client` and adds one envelope filter to every request.

    `landslide.sheets.query_pages` (P1-03) then does the ids-only query and the object-id pages inside the envelope:
    the page `(1=1) AND OBJECTID >= lo AND OBJECTID <= hi` plus the envelope returns exactly the envelope's ids in
    [lo, hi], which `query_pages` checks.
    """

    def __init__(self, client: Any, bbox: tuple[float, float, float, float], epsg: int) -> None:
        self.client = client
        self.filter = envelope_params(bbox, epsg)

    def _params(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        p = dict(params or {})
        clash = set(p) & set(self.filter)
        if clash:
            raise ValueError(f"spatial parameters already set: {sorted(clash)}")
        return {**p, **self.filter}

    def get(self, url: str, params: Mapping[str, Any] | None = None, **kw: Any) -> Any:
        return self.client.get(url, params=self._params(params), **kw)

    def get_json(self, url: str, params: Mapping[str, Any] | None = None, **kw: Any) -> Any:
        return self.client.get_json(url, params=self._params(params), **kw)


def fetch_pair(client: Any, layer: int, meta: Mapping[str, Any], sheet: str, bbox: tuple[float, float, float, float],
               epsg: int, page_dir: Path, *, service: str) -> dict[str, Any]:
    """All features of `layer` that intersect one sheet's envelope, saved as unmodified pages in `page_dir`."""
    t0 = now_utc()
    if page_dir.exists():
        shutil.rmtree(page_dir)  # a page left by an interrupted attempt
    res = S.query_pages(EnvelopeClient(client, bbox, epsg), f"{service}/{layer}", "1=1", "*", "json",
                        int(meta["maxRecordCount"]), S.oid_field(meta), page_dir)
    if res["n_pages"] == 0 and page_dir.exists():
        page_dir.rmdir()
    return {"layer": int(layer), "sheet": sheet, "retrieved_utc": t0, "bbox": list(bbox), **res}


# ======================================================================================================================
# Build: pages -> one deduplicated GeoPackage per layer
# ======================================================================================================================


def dedup(gdf: Any, oid: str) -> tuple[Any, dict[str, Any]]:
    """Keep one row per object id. Rows with the same id must be identical (attributes and geometry WKB).

    Returns the unique rows sorted by id and the counts: raw rows, unique ids, duplicate rows, and the ids that came
    back with more than one version (with up to 20 examples).
    """
    import pandas as pd
    import shapely

    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    df["_wkb"] = shapely.to_wkb(gdf.geometry.values)
    versions = df.drop_duplicates()
    conflicting = sorted(int(v) for v in versions.loc[versions[oid].duplicated(), oid].unique())
    keep = ~df.duplicated(oid, keep="first")
    out = gdf.loc[keep.values].sort_values(oid).reset_index(drop=True)
    stats = {"n_raw": int(len(df)), "n_unique": int(keep.sum()), "n_duplicate_rows": int((~keep).sum()),
             "n_conflicting_ids": len(conflicting), "conflicting_ids_examples": conflicting[:20]}
    return out, stats


def geometry_summary(gdf: Any) -> dict[str, Any]:
    geoms = gdf.geometry
    types = Counter(str(t) for t in geoms.geom_type.fillna("None"))
    return {"types": dict(types.most_common()), "n_null": int(geoms.isna().sum()),
            "n_empty": int(geoms.is_empty.fillna(False).sum()), "n_invalid": int((~geoms.is_valid).sum())}


def build_layer(layer: int, records: list[dict[str, Any]], pages_root: Path, out_path: Path, epsg: int,
                oid: str) -> dict[str, Any]:
    """Read every page of the layer, deduplicate by object id and write the GeoPackage (atomically)."""
    import geopandas as gpd
    import shapely

    pages = []
    declared: Counter[str] = Counter()
    page_bytes = 0
    for rec in records:
        for p in rec["pages"]:
            path = pages_root / layer_key(layer) / sheet_to_stem(rec["sheet"]) / p["file"]
            body = path.read_bytes()
            if S.sha256_bytes(body) != p["sha256"]:
                raise ValueError(f"{path}: sha256 differs from the fetch log")
            declared[json.loads(body).get("geometryType", "?")] += 1
            page_bytes += len(body)
            pages.append(path)
    if not pages:
        raise ValueError(f"layer {layer}: no features in any sheet envelope")
    raw = S.esri_pages_to_frame(pages, epsg)
    if oid not in raw.columns:
        raise ValueError(f"layer {layer}: object-id field {oid} not read from the pages (columns {list(raw.columns)})")
    gdf, st = dedup(raw, oid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".part.gpkg")
    if tmp.exists():
        tmp.unlink()
    # promote_to_multi=False: keep each geometry as served (pyogrio would otherwise turn a mix of Polygon and
    # MultiPolygon into MultiPolygon for GPKG); a mixed layer gets the generic geometry type.
    gdf.to_file(tmp, layer=f"zabaged_{layer_key(layer)}", driver="GPKG", promote_to_multi=False)
    back = gpd.read_file(tmp)
    same = (len(back) == len(gdf) and back[oid].tolist() == gdf[oid].tolist()
            and bool((shapely.to_wkb(back.geometry.values) == shapely.to_wkb(gdf.geometry.values)).all()))
    if not same:
        raise ValueError(f"layer {layer}: the GeoPackage does not read back equal to the deduplicated features")
    tmp.replace(out_path)
    per_sheet = [r["n_ids"] for r in records]
    extra = {}
    if "fid_zbg" in gdf.columns:
        extra["n_unique_fid_zbg"] = int(gdf["fid_zbg"].nunique(dropna=True))
    return {"file": out_path.name, "bytes": out_path.stat().st_size, "gpkg_layer": f"zabaged_{layer_key(layer)}",
            "n_sheets_queried": len(records), "n_sheets_nonempty": sum(1 for n in per_sheet if n),
            "max_features_per_sheet": max(per_sheet) if per_sheet else 0,
            "n_pages": len(pages), "page_bytes": page_bytes, "declared_geometry_types": dict(declared),
            **st, "geometry": geometry_summary(gdf), "roundtrip_equal": same, **extra}


# ======================================================================================================================
# CLI
# ======================================================================================================================


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records of a jsonl log. A truncated last line (a killed run) is ignored; any other bad line raises."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if n == len(lines):
                break
            raise
    return out


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(S.to_jsonable(rec), ensure_ascii=False) + "\n")


def update_sources(path: Path, key: str, section: dict[str, Any]) -> None:
    """Replace one source's section of `results/sources.json`, keeping the others."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = S.to_jsonable(section)
    data["updated_utc"] = now_utc()
    data = {"task": "P1-07", **{k: v for k, v in data.items() if k != "task"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_grids(paths: Paths, sheets: list[str], cell: float, origin: tuple[float, float]) -> dict[str, S.Grid]:
    """Each sheet's DTM grid, read from its package raster (transform and size), never recomputed."""
    import rasterio

    out = {}
    for s in progress(sheets, desc="grids", unit="sheet"):
        with rasterio.open(paths.dtm(s)) as ds:
            out[s] = S.grid_from_transform(list(ds.transform)[:6], ds.width, ds.height, cell=cell, origin=origin)
    return out


def _git_commit(root: Path) -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    cfg = load_config()
    paths = Paths()
    url_service = service_url(cfg)
    all_layers = layers(cfg)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--sheets", nargs="+", metavar="SHEET", help="sheet ids (fetch only; no GeoPackages)")
    sel.add_argument("--all", action="store_true", help="all 860 surveyed sheets, then build the GeoPackages")
    sel.add_argument("--build-only", action="store_true", help="build the GeoPackages from the saved pages")
    ap.add_argument("--layers", nargs="+", type=int, default=list(all_layers), help="layer ids (default: all six)")
    args = ap.parse_args(argv)
    bad = sorted(set(args.layers) - set(all_layers))
    if bad:
        ap.error(f"not a P1-07 layer: {bad}")

    setup_logging(paths.logs / "fetch_zabaged.log")
    log = get_logger("fetch_zabaged")
    epsg = S.epsg_code(cfg)
    cell = float(cfg["dtm"]["cell_m"])
    d = paths.zabaged_dir
    pages_root = d / "pages"
    log_path = d / "fetch_log.jsonl"
    meta_path = d / "service_metadata.json"
    all_sheets = sorted(S.load_sheets(paths.surveyed_sheets)["sheet"])
    status, n_pairs = "ok", 0
    run = {"started_utc": now_utc(), "argv": sys.argv[1:] if argv is None else argv,
           "git_commit": _git_commit(paths.root), "config_hash": config_hash(cfg)}

    if not args.build_only:
        chosen = all_sheets if args.all else list(dict.fromkeys(args.sheets))
        unknown = sorted(set(chosen) - set(all_sheets))
        if unknown:
            ap.error(f"not surveyed sheets: {unknown}")
        grids = read_grids(paths, chosen, cell, S.grid_origin(cfg))
        done = {(r["layer"], r["sheet"]) for r in read_jsonl(log_path)}
        with Client() as client:
            try:
                svc = client.get_json(url_service, params={"f": "json"})
                meta: dict[str, Any] = {"retrieved_utc": now_utc(), "url": url_service,
                                        "service": {k: svc[k] for k in SERVICE_META_KEYS if k in svc}, "layers": {}}
                for lyr in args.layers:
                    url = f"{url_service}/{lyr}"
                    m = S.layer_metadata(client, url, keys=(*S.LAYER_META_KEYS, "advancedQueryCapabilities"))
                    m["count_national"] = S.count_features(client, url)
                    meta["layers"][layer_key(lyr)] = m
                    log.info("layer %d %r: %s, maxRecordCount %s, SR %s, oid %s, %d features nationally", lyr,
                             m.get("name"), m.get("geometryType"), m.get("maxRecordCount"), S.native_wkid(m),
                             S.oid_field(m), m["count_national"])
                    if S.native_wkid(m) != epsg:
                        raise RuntimeError(f"layer {lyr}: native SR {S.native_wkid(m)} is not EPSG:{epsg}")
                    if m["count_national"] == 0:
                        raise RuntimeError(f"layer {lyr} is empty")
                old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"layers": {}}
                meta["layers"] = {**old.get("layers", {}), **meta["layers"]}
                d.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
                todo = [(lyr, s) for lyr in args.layers for s in chosen if (lyr, s) not in done]
                log.info("%d (layer, sheet) pairs selected, %d to fetch", len(args.layers) * len(chosen), len(todo))
                for lyr, s in progress(todo, desc="zabaged", unit="query"):
                    t0 = time.monotonic()
                    rec = fetch_pair(client, lyr, meta["layers"][layer_key(lyr)], s, grids[s].bbox, epsg,
                                     pages_root / layer_key(lyr) / sheet_to_stem(s), service=url_service)
                    rec["seconds"] = round(time.monotonic() - t0, 2)
                    append_jsonl(log_path, rec)
                    n_pairs += 1
                    if rec["n_ids"]:
                        log.info("layer %d %s: %d features, %d pages, %.1f s", lyr, s, rec["n_ids"], rec["n_pages"],
                                 rec["seconds"])
            except Exception as e:  # stop and report: no work-arounds
                status = f"error: {type(e).__name__}: {e}"
                log.error("stopped: %s\n%s", status, traceback.format_exc())
            finally:
                run.update({"finished_utc": now_utc(), "status": status, "n_requests": client.n_requests,
                            "n_pairs": n_pairs})
                append_jsonl(d / "runs.jsonl", run)
        log.info("fetch %s: %d pairs, %d requests", status, n_pairs, run["n_requests"])
        if status != "ok" or args.sheets:
            return 0 if status == "ok" else 1

    # ---- build ----
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    recs: dict[tuple[int, str], dict[str, Any]] = {(r["layer"], r["sheet"]): r for r in read_jsonl(log_path)}
    layers_out: dict[str, Any] = {}
    try:
        for lyr in args.layers:
            missing = [s for s in all_sheets if (lyr, s) not in recs]
            if missing:
                raise RuntimeError(f"layer {lyr}: {len(missing)} sheets not fetched yet, e.g. {missing[:5]}")
            m = meta["layers"][layer_key(lyr)]
            st = build_layer(lyr, [recs[(lyr, s)] for s in all_sheets], pages_root, paths.zabaged(layer_key(lyr)),
                             epsg, S.oid_field(m))
            if st["n_conflicting_ids"]:
                raise RuntimeError(f"layer {lyr}: {st['n_conflicting_ids']} object ids came back with different "
                                   f"content, e.g. {st['conflicting_ids_examples']}")
            layers_out[layer_key(lyr)] = st
            log.info("layer %d: %d unique of %d rows, geometry %s, %d bytes", lyr, st["n_unique"], st["n_raw"],
                     st["geometry"]["types"], st["bytes"])
    except Exception as e:
        log.error("build stopped: %s\n%s", e, traceback.format_exc())
        return 1
    runs = read_jsonl(d / "runs.jsonl")
    dates = sorted(r["retrieved_utc"] for r in recs.values())
    section = {
        "row": 6,
        "source": "ČÚZK ZABAGED polohopis, CC BY 4.0 (© ČÚZK); data stay in data/, never committed",
        "service": url_service,
        "service_metadata": meta["service"],
        "metadata_retrieved_utc": meta["retrieved_utc"],
        "request": {"per_sheet_and_layer": "returnIdsOnly with the envelope, then pages of <= maxRecordCount ids "
                                           "(OBJECTID range + the same envelope), Esri JSON (f=json) in the native SR",
                    "envelope": "bbox of the sheet's package DTM raster (sheet + 250 m on the native lattice)",
                    "params": {k: v for k, v in envelope_params((0, 0, 1, 1), epsg).items() if k != "geometry"},
                    "outFields": "*", "no_outSR": True},
        "retrieved_utc": {"first": dates[0], "last": dates[-1]},
        "runs": [{k: r.get(k) for k in ("started_utc", "finished_utc", "status", "n_requests", "n_pairs", "argv",
                                        "git_commit")} for r in runs],
        "n_requests_total": sum(int(r.get("n_requests", 0)) for r in runs),
        "n_sheets": len(all_sheets),
        "layers": {k: {"name_brief": all_layers[int(k)], "metadata": meta["layers"][k], **v}
                   for k, v in layers_out.items()},
        "total_bytes": {"gpkg": sum(v["bytes"] for v in layers_out.values()),
                        "pages": sum(v["page_bytes"] for v in layers_out.values())},
    }
    update_sources(paths.results / "sources.json", SECTION, section)
    log.info("wrote %d GeoPackages and the %s section of sources.json", len(layers_out), SECTION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
