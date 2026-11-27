"""Row 4: the ČGS 1:500,000 geological map, layer 7, Carpathian subregions only, native SR (EPSG:5514).

The query is `subregion IN (...)` with the subregions of Appendix A §3 (config `flysch.subregions`) and the fields
`objectid, region, subregion, hor_karto` (BRIEF.md note 4). Reads the layer metadata first, then either counts the
features (`--count-only`) or downloads them paginated by object id as Esri JSON; pages are saved unmodified and
combined into a GeoPackage. No `outSR`.

In phase 1 only the count is run (the hand-over file `data/raw_data/cgs_geol500_carpathian_units.gpkg` is used):

    uv run python scripts/fetch_geology.py --count-only

The licence of this layer is not verified: the output stays in the gitignored `data/`.
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

LAYER = "geol500_carpathian_units"


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count-only", action="store_true", help="read the metadata and count; download nothing")
    ap.add_argument("--out", type=Path, default=paths.cache / "rebuild" / "cgs_geol500_carpathian_units.gpkg")
    ap.add_argument("--pages", type=Path, default=paths.cache / "rebuild" / "geology_pages",
                    help="where the unmodified response pages go")
    args = ap.parse_args(argv)

    setup_logging(paths.logs / "fetch_geology.log")
    log = get_logger("fetch_geology")
    epsg = S.epsg_code(cfg)
    where = S.sql_in("subregion", cfg["flysch"]["subregions"])
    url = S.source_url(cfg, "geology")
    with Client() as client:
        meta = S.layer_metadata(client, url)
        log.info("%s: %s, %s, maxRecordCount %s, SR %s", url, meta.get("name"),
                 meta.get("geometryType"), meta.get("maxRecordCount"), meta.get("sourceSpatialReference"))
        if S.native_wkid(meta) != epsg:
            log.error("native SR is %s, not EPSG:%d", S.native_wkid(meta), epsg)
            return 1
        n = S.count_features(client, url, where)
        log.info("count (%s): %d", where, n)
        if args.count_only:
            print(json.dumps({"url": url, "where": where, "count": n, "metadata": meta},
                             indent=1, ensure_ascii=False))
            return 0
        res = S.query_pages(client, url, where, ",".join(S.geology_fields(cfg)), "json",
                            int(meta["maxRecordCount"]), S.oid_field(meta), args.pages)
    gdf = S.esri_pages_to_frame([args.pages / p["file"] for p in res["pages"]], epsg)
    if len(gdf) != n:
        log.error("downloaded %d features, count says %d", len(gdf), n)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.out, layer=LAYER, driver="GPKG")
    (args.pages / "fetch.json").write_text(json.dumps({"metadata": meta, "count": n, "where": where, **res},
                                                      indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d features to %s", len(gdf), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
