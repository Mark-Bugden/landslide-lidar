"""Row 5: render the ČGS landslide-susceptibility map per class on every sheet's DTM grid (BRIEF.md note 5).

Per sheet, two `MapServer/export` requests (mid, high) with the bbox and size of the sheet's package DTM raster,
`bboxSR = imageSR = 5514`, `format=png32`, `transparent=true`, `layers=show:0`, `dpi=96` and the class's
`layerDefs`. Each PNG must have the DTM's size. alpha >= 128 -> class. The codes (0 low/no data, 1 mid, 2 high; a
pixel in both renders gets 2) are written as a uint8 deflate GeoTIFF on the DTM grid to
`data/cache/susceptibility/<sheet>.tif`; the unmodified PNGs are kept in `png/`. Every finished sheet is appended to
`fetch_log.jsonl`, so a re-run skips it. The run stops at the first error (after the client's retries).

    uv run python scripts/fetch_susceptibility.py --sheets 25-14-17 04-33-10 --inspect   # render and check two
    uv run python scripts/fetch_susceptibility.py --all                                  # the rest (resumes)
    uv run python scripts/fetch_susceptibility.py --summary-only                         # rewrite sources.json

`--inspect` also asks the service to describe each render (`f=json`: extent, size, scale) and checks a few pixels per
category against point queries of the vector layer. Its per-point record stays in `data/` (row 5 has no stated
licence); `results/sources.json` gets aggregate numbers only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from landslide import sheets as S
from landslide import susceptibility as SU
from landslide.config import config_hash, load_config
from landslide.logs import get_logger, progress, setup_logging
from landslide.net import Client
from landslide.paths import Paths, sheet_to_stem

SECTION = "susceptibility"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def latest_by_sheet(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The last record of each sheet (a forced re-render replaces the earlier one)."""
    return {r["sheet"]: r for r in records}


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


def _quantiles(v: list[float]) -> dict[str, float] | None:
    if not v:
        return None
    q = np.quantile(np.asarray(v, dtype=float), [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    return {k: round(float(x), 6) for k, x in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), q,
                                                  strict=True)}


def summarise(paths: Paths, cfg: dict, all_sheets: list[str]) -> dict[str, Any]:
    """Aggregate numbers only (row 5 has no stated licence): no per-sheet values, no coordinates."""
    d = paths.susceptibility_dir
    recs = latest_by_sheet(read_jsonl(d / "fetch_log.jsonl"))
    runs = read_jsonl(d / "runs.jsonl")
    meta_p = d / "service_metadata.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else None
    done = [s for s in all_sheets if s in recs and paths.susceptibility(s).exists()]
    missing = sorted(set(all_sheets) - set(done))
    tot = {"n_px": 0, "n_mid_render": 0, "n_high_render": 0, "n_overlap": 0, "0": 0, "1": 0, "2": 0}
    pairs: dict[str, int] = {}
    alpha = {k: {} for k in SU.CLASSES}
    other_colour = 0
    frac_mid, frac_high, frac_any = [], [], []
    n_no_class = n_overlap_sheets = 0
    png_bytes = 0
    retrieved = []
    for s in done:
        r = recs[s]
        for k in ("n_px", "n_mid_render", "n_high_render", "n_overlap"):
            tot[k] += r[k]
        for c in ("0", "1", "2"):
            tot[c] += r["n_code"][c]
        for p, n in r["overlap_alpha_pairs"].items():
            pairs[p] = pairs.get(p, 0) + n
        for cls in SU.CLASSES:
            q = r["requests"][cls]
            png_bytes += q["bytes"]
            retrieved.append(q["retrieved_utc"])
            other_colour += q["colour_check"]["n_other_colour"]
            for a, n in q["alpha_histogram"].items():
                alpha[cls][a] = alpha[cls].get(a, 0) + n
        frac_mid.append(r["n_code"]["1"] / r["n_px"])
        frac_high.append(r["n_code"]["2"] / r["n_px"])
        frac_any.append((r["n_code"]["1"] + r["n_code"]["2"]) / r["n_px"])
        n_no_class += r["n_code"]["0"] == r["n_px"]
        n_overlap_sheets += r["n_overlap"] > 0
    tif_bytes = sum(paths.susceptibility(s).stat().st_size for s in done)
    n_png = len(list((d / "png").glob("*.png"))) if (d / "png").exists() else 0
    insp_p = d / "inspection.json"
    insp = json.loads(insp_p.read_text(encoding="utf-8"))["summary"] if insp_p.exists() else None
    meta_c = None
    if meta:
        meta_c = {k: v for k, v in meta.items() if k != "retrieved_utc"}
    npx = max(tot["n_px"], 1)
    return {
        "row": 5,
        "source": "ČGS landslide susceptibility map (no stated licence: rasters stay in data/, aggregates only here)",
        "service": SU.SERVICE_URL,
        "request": {"url": SU.EXPORT_URL, "per_sheet": "one request per class (mid, high) on the sheet's DTM grid",
                    "params": {k: v for k, v in SU.export_params(S.Grid(0, 0, 1, 1), S.epsg_code(cfg), "mid").items()
                               if k not in ("bbox", "size", "layerDefs")},
                    "bbox_size": "bbox and size of the sheet's package DTM raster (transform and size read from it)",
                    "layerDefs": {k: SU.layer_def(p) for k, (_, p) in SU.CLASSES.items()},
                    "class_rule": f"alpha >= {SU.ALPHA_MIN}", "codes": SU.CLASS_LABELS,
                    "overlap_rule": "a pixel in both the mid and the high render gets code 2 (high)"},
        "metadata": meta_c,
        "metadata_retrieved_utc": meta.get("retrieved_utc") if meta else None,
        "retrieved_utc": {"first": min(retrieved) if retrieved else None, "last": max(retrieved) if retrieved else None},
        "runs": [{k: r.get(k) for k in ("started_utc", "finished_utc", "status", "n_requests", "n_rendered",
                                        "argv", "git_commit")} for r in runs],
        "n_requests_total": sum(int(r.get("n_requests", 0)) for r in runs),
        "n_export_requests_in_log": 2 * len(done),
        "n_sheets_expected": len(all_sheets),
        "n_sheets_done": len(done),
        "missing_sheets": missing,
        "size_mismatches": 0 if done else None,
        "size_check": "every PNG's size equals its DTM raster's (asserted per request; a mismatch stops the run)",
        "files": {"class_rasters": {"n": len(done), "bytes": tif_bytes, "format": "uint8 deflate GeoTIFF"},
                  "pngs": {"n": n_png, "bytes_in_log": png_bytes}},
        "class_fractions": {
            "pixel_weighted": {"low_or_no_data": round(tot["0"] / npx, 6), "mid": round(tot["1"] / npx, 6),
                               "high": round(tot["2"] / npx, 6)},
            "note": "over all pixels of all sheet rasters (sheet + 250 m margin); overlapping margins count once per "
                    "raster",
            "n_px": tot["n_px"], "n_code": {k: tot[k] for k in ("0", "1", "2")},
            "per_sheet_quantiles": {"mid": _quantiles(frac_mid), "high": _quantiles(frac_high),
                                    "mid_or_high": _quantiles(frac_any)},
            "n_sheets_all_low_or_no_data": int(n_no_class),
        },
        "overlap_mid_high": {"n_px": tot["n_overlap"], "fraction_of_px": round(tot["n_overlap"] / npx, 8),
                             "n_sheets_with_overlap": int(n_overlap_sheets),
                             "alpha_pairs_mid_high": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
                             "resolution": "code 2 (high wins)"},
        "renders": {"n_mid_px": tot["n_mid_render"], "n_high_px": tot["n_high_render"],
                    "alpha_histogram": {k: dict(sorted(v.items(), key=lambda kv: int(kv[0])))
                                        for k, v in alpha.items()},
                    "class_pixels_with_unexpected_colour": other_colour},
        "inspection": insp,
    }


def inspect_sheet(client: Client, sheet: str, paths: Paths, epsg: int, cell: float, n_points: int) -> dict[str, Any]:
    """`f=json` description of both renders, then point queries of the vector layer at chosen pixels."""
    from PIL import Image

    dtm = SU.read_dtm_grid(paths.dtm(sheet), cell)
    g = dtm.grid
    out: dict[str, Any] = {"sheet": sheet, "grid": g.as_dict(), "export_json": {}}
    for cls in SU.CLASSES:
        body = client.get_json(SU.EXPORT_URL, params=SU.export_params(g, epsg, cls, f="json"))
        out["export_json"][cls] = SU.check_export_json(body, g, epsg)
    alphas = {}
    for cls in SU.CLASSES:
        with Image.open(paths.susceptibility_dir / "png" / f"{sheet_to_stem(sheet)}_{cls}.png") as im:
            alphas[cls] = np.asarray(im)[..., 3].copy()
    import rasterio

    with rasterio.open(paths.susceptibility(sheet)) as ds:
        codes = ds.read(1)
    pts = SU.inspection_points(alphas, n_points)
    rows = []
    for cat, cells in pts.items():
        for r, c in cells:
            x, y = SU.pixel_centre(g, r, c)
            vals = SU.point_query(client, x, y, epsg)
            sc = SU.code_from_values(vals)
            rows.append({"category": cat, "row": r, "col": c, "x": x, "y": y, "raster_code": int(codes[r, c]),
                         "service_values": [v.split(" –")[0] for v in vals], "service_code": sc,
                         "agree": sc == int(codes[r, c])})
    out["points"] = rows
    return out


def summarise_inspection(sheets_out: list[dict[str, Any]]) -> dict[str, Any]:
    cats: dict[str, dict[str, int]] = {}
    for s in sheets_out:
        for p in s["points"]:
            c = cats.setdefault(p["category"], {"n": 0, "agree": 0})
            c["n"] += 1
            c["agree"] += int(p["agree"])
    return {"n_sheets": len(sheets_out),
            "export_json_ok": sum(int(v["ok"]) for s in sheets_out for v in s["export_json"].values()),
            "export_json_n": sum(len(s["export_json"]) for s in sheets_out),
            "export_scale": sorted({v["scale"] for s in sheets_out for v in s["export_json"].values()}),
            "points": cats,
            "points_agree": sum(c["agree"] for c in cats.values()), "points_n": sum(c["n"] for c in cats.values())}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    cfg = load_config()
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--sheets", nargs="+", metavar="SHEET", help="sheet ids, e.g. 25-14-17")
    sel.add_argument("--all", action="store_true", help="all 860 surveyed sheets (finished ones are skipped)")
    sel.add_argument("--summary-only", action="store_true", help="rewrite the sources.json section; no network")
    ap.add_argument("--force", action="store_true", help="re-render sheets that are already done")
    ap.add_argument("--inspect", action="store_true", help="f=json and point-query checks of the selected sheets")
    ap.add_argument("--inspect-points", type=int, default=10, help="points per category and sheet for --inspect")
    args = ap.parse_args(argv)

    setup_logging(paths.logs / "fetch_susceptibility.log")
    log = get_logger("fetch_susceptibility")
    epsg = S.epsg_code(cfg)
    cell = float(cfg["dtm"]["cell_m"])
    d = paths.susceptibility_dir
    sources_json = paths.results / "sources.json"

    all_sheets = sorted(S.load_sheets(paths.surveyed_sheets)["sheet"])
    if args.summary_only:
        update_sources(sources_json, SECTION, summarise(paths, cfg, all_sheets))
        log.info("wrote the %s section of %s", SECTION, sources_json)
        return 0
    if args.sheets:
        unknown = sorted(set(args.sheets) - set(all_sheets))
        if unknown:
            ap.error(f"not surveyed sheets: {unknown}")
        chosen = list(dict.fromkeys(args.sheets))
    else:
        chosen = all_sheets
    missing_dtm = [s for s in chosen if not paths.dtm(s).exists()]
    if missing_dtm:
        log.error("no DTM raster for %s", missing_dtm)
        return 1
    grids = {s: SU.read_dtm_grid(paths.dtm(s), cell).grid for s in progress(all_sheets, desc="grids", unit="sheet")}
    max_w, max_h = max(g.w for g in grids.values()), max(g.h for g in grids.values())

    run = {"started_utc": now_utc(), "argv": sys.argv[1:] if argv is None else argv,
           "git_commit": _git_commit(paths.root), "config_hash": config_hash(cfg)}
    log_path = d / "fetch_log.jsonl"
    done = latest_by_sheet(read_jsonl(log_path))
    todo = [s for s in chosen if args.force or s not in done or not paths.susceptibility(s).exists()]
    log.info("%d sheets selected, %d to render, %d already done", len(chosen), len(todo), len(chosen) - len(todo))
    status, n_rendered = "ok", 0
    with Client() as client:
        try:
            meta = SU.service_metadata(client)
            meta["retrieved_utc"] = now_utc()
            d.mkdir(parents=True, exist_ok=True)
            (d / "service_metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
            problems = SU.check_service(meta, max_w, max_h, epsg)
            if problems:
                for p in problems:
                    log.error("service check: %s", p)
                raise SU.UnexpectedResponse("; ".join(problems))
            colours = {k: tuple(v) for k, v in meta["class_colours"].items()}
            log.info("service %s v%s, layer %r; colours %s; max grid %dx%d (limit %sx%s)", SU.SERVICE_URL,
                     meta["service"].get("currentVersion"), meta["layer"].get("name"), colours, max_w, max_h,
                     meta["service"].get("maxImageWidth"), meta["service"].get("maxImageHeight"))
            for s in progress(todo, desc="susceptibility", unit="sheet"):
                t0 = time.monotonic()
                rec = SU.render_sheet(client, s, paths.dtm(s), paths.susceptibility(s), d / "png", epsg, cell,
                                      colours)
                rec["seconds"] = round(time.monotonic() - t0, 2)
                append_jsonl(log_path, rec)
                n_rendered += 1
                n = rec["n_px"]
                log.info("%s: %dx%d, mid %.4f, high %.4f, overlap %d px, %.1f s", s, rec["grid"]["w"],
                         rec["grid"]["h"], rec["n_code"]["1"] / n, rec["n_code"]["2"] / n, rec["n_overlap"],
                         rec["seconds"])
            if args.inspect:
                out = [inspect_sheet(client, s, paths, epsg, cell, args.inspect_points) for s in chosen]
                summ = summarise_inspection(out)
                prev = json.loads((d / "inspection.json").read_text(encoding="utf-8")) \
                    if (d / "inspection.json").exists() else {"sheets": []}
                keep = [x for x in prev["sheets"] if x["sheet"] not in set(chosen)] + out
                (d / "inspection.json").write_text(json.dumps(S.to_jsonable(
                    {"written_utc": now_utc(), "summary": summarise_inspection(keep), "sheets": keep}),
                    indent=1, ensure_ascii=False), encoding="utf-8")
                log.info("inspection: %s", json.dumps(summ, ensure_ascii=False))
        except Exception as e:  # stop and report: no work-arounds
            status = f"error: {type(e).__name__}: {e}"
            log.error("stopped: %s\n%s", status, traceback.format_exc())
        finally:
            run.update({"finished_utc": now_utc(), "status": status, "n_requests": client.n_requests,
                        "n_rendered": n_rendered})
            append_jsonl(d / "runs.jsonl", run)
    update_sources(sources_json, SECTION, summarise(paths, cfg, all_sheets))
    log.info("%s: %d rendered, %d requests; sources.json updated", status, n_rendered, run["n_requests"])
    return 0 if status == "ok" else 1


def _git_commit(root: Path) -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
