"""Row 2: the ČGS slope-deformation register (labels), layer 1, all features, all fields, native SR (EPSG:5514).

Reads the layer metadata first, then either counts the features (`--count-only`, `returnCountOnly=true`) or downloads
them paginated by object id (page = the layer's maxRecordCount) as Esri JSON in the native SR. Each page is saved
unmodified; the pages are then combined into a GeoPackage. No `outSR`.

In phase 1 only the count is run (the hand-over file `data/raw_data/cgs_register_polygons.gpkg` is used):

    uv run python scripts/fetch_register.py --count-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from landslide import sheets as S
from landslide.config import load_config
from landslide.logs import get_logger, setup_logging
from landslide.net import Client
from landslide.paths import Paths

LAYER = "cgs_register_polygons"


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count-only", action="store_true", help="read the metadata and count; download nothing")
    ap.add_argument("--out", type=Path, default=paths.cache / "rebuild" / f"{LAYER}.gpkg")
    ap.add_argument("--pages", type=Path, default=paths.cache / "rebuild" / "register_pages",
                    help="where the unmodified response pages go")
    args = ap.parse_args(argv)

    setup_logging(paths.logs / "fetch_register.log")
    log = get_logger("fetch_register")
    epsg = S.epsg_code(cfg)
    with Client() as client:
        meta = S.layer_metadata(client, S.REGISTER_URL)
        log.info("%s: %s, %s, maxRecordCount %s, SR %s", S.REGISTER_URL, meta.get("name"),
                 meta.get("geometryType"), meta.get("maxRecordCount"), meta.get("sourceSpatialReference"))
        if S.native_wkid(meta) != epsg:
            log.error("native SR is %s, not EPSG:%d", S.native_wkid(meta), epsg)
            return 1
        n = S.count_features(client, S.REGISTER_URL)
        log.info("count: %d", n)
        if args.count_only:
            print(json.dumps({"url": S.REGISTER_URL, "where": "1=1", "count": n, "metadata": meta},
                             indent=1, ensure_ascii=False))
            return 0
        res = S.query_pages(client, S.REGISTER_URL, "1=1", "*", "json", int(meta["maxRecordCount"]),
                            S.oid_field(meta), args.pages)
    gdf = S.esri_pages_to_frame([args.pages / p["file"] for p in res["pages"]], epsg)
    if len(gdf) != n:
        log.error("downloaded %d features, count says %d", len(gdf), n)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.out, layer=LAYER, driver="GPKG")
    (args.pages / "fetch.json").write_text(json.dumps({"metadata": meta, "count": n, **res}, indent=1,
                                                      ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d features to %s", len(gdf), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
