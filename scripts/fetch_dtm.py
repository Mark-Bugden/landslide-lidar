"""Row 1: download DMR 5G sheet rasters from the ČÚZK ImageServer with the brief's recipe (note 1).

One `exportImage` request per sheet: bbox = sheet bounds in EPSG:5514 + 250 m, snapped outwards to the native lattice
(x = 0.4, y = -0.12 mod 2 m); bboxSR = imageSR = 5514; size = bbox / 2 m; format=tiff, pixelType=F32,
interpolation=RSP_NearestNeighbor. The unmodified response is saved as `<out>/<sheet>.tif` with a `<sheet>.json`
sidecar. `--package-out DIR` also writes the package format (NoData and values outside [-100, 2000] m set to NaN,
deflate GeoTIFF on the same grid), which rebuilds `data/raw_data/dmr5g/`.

In phase 1 only the provenance sample is fetched (a full fetch is about 11 GB):

    uv run python scripts/fetch_dtm.py --sample             # the config's provenance sample (12 sheets)
    uv run python scripts/fetch_dtm.py --sheets 14-34-07 25-23-01
    uv run python scripts/fetch_dtm.py --all --count-only   # what a full fetch would request; no download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from landslide import sheets as S
from landslide.config import load_config
from landslide.logs import get_logger, progress, setup_logging
from landslide.net import Client
from landslide.paths import Paths, sheet_to_stem


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--sheets", nargs="+", metavar="SHEET", help="sheet ids, e.g. 14-34-07")
    sel.add_argument("--sample", nargs="?", type=int, const=-1, metavar="N",
                     help="draw N sheets (default: replication.provenance.n_sheets) with the provenance stream")
    sel.add_argument("--all", action="store_true", help="all surveyed sheets")
    ap.add_argument("--seed", type=int, default=None, help="seed for --sample (default: replication.seed)")
    ap.add_argument("--out", type=Path, default=paths.provenance_dir, help="where the unmodified TIFFs go")
    ap.add_argument("--package-out", type=Path, default=None, help="also write package-format rasters here")
    ap.add_argument("--force", action="store_true", help="re-download sheets that are already cached")
    ap.add_argument("--count-only", action="store_true",
                    help="read the service metadata and report the requests; download nothing")
    args = ap.parse_args(argv)

    setup_logging(paths.logs / "fetch_dtm.log")
    log = get_logger("fetch_dtm")

    sheets = S.load_sheets(paths.surveyed_sheets)
    grids = {r.sheet: S.dtm_grid(r.geometry.bounds, cfg) for r in sheets.itertuples()}
    if args.all:
        chosen = sorted(grids)
    elif args.sheets:
        unknown = sorted(set(args.sheets) - set(grids))
        if unknown:
            ap.error(f"not surveyed sheets: {unknown}")
        chosen = list(args.sheets)
    else:
        chosen = S.draw_provenance_sample(cfg, grids, n=None if args.sample == -1 else args.sample, seed=args.seed)
        log.info("provenance sample (draw order): %s", " ".join(chosen))

    url = S.dtm_service_url(cfg)
    with Client() as client:
        meta = S.layer_metadata(client, url, keys=S.IMAGE_META_KEYS)
        log.info("service %s: pixel %sx%s %s, SR %s, max image %sx%s", url, meta.get("pixelSizeX"),
                 meta.get("pixelSizeY"), meta.get("pixelType"), meta.get("spatialReference"),
                 meta.get("maxImageWidth"), meta.get("maxImageHeight"))
        too_big = [s for s in chosen
                   if grids[s].w > meta.get("maxImageWidth", 0) or grids[s].h > meta.get("maxImageHeight", 0)]
        summary = {"service": url, "metadata": meta, "n_requests": len(chosen),
                   "n_pixels": sum(grids[s].w * grids[s].h for s in chosen),
                   "max_w": max(grids[s].w for s in chosen), "max_h": max(grids[s].h for s in chosen),
                   "over_service_limit": too_big, "sheets": chosen}
        summary["uncompressed_f32_bytes"] = 4 * summary["n_pixels"]
        if args.count_only:
            print(json.dumps(S.to_jsonable(summary), indent=1, ensure_ascii=False))
            return 0
        if too_big:
            log.error("requests larger than the service limit: %s", too_big)
            return 1
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "service_metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
        for s in progress(chosen, desc="dtm", unit="sheet"):
            tif = args.out / f"{sheet_to_stem(s)}.tif"
            if tif.exists() and (args.out / f"{sheet_to_stem(s)}.json").exists() and not args.force:
                log.info("%s: cached, skipped", s)
            else:
                rec = S.fetch_dtm(client, cfg, s, grids[s], args.out)
                log.info("%s: %d bytes, sha256 %s", s, rec["bytes"], rec["sha256"])
            if args.package_out is not None:
                S.write_package_raster(tif, args.package_out / tif.name, S.epsg_code(cfg))
    log.info("done: %d sheets in %s (%d requests)", len(chosen), args.out, client.n_requests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
