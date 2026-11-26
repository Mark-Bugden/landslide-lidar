"""Row 3: the ČGS surveyed map sheets, layer 0, all features, in EPSG:4326.

Reads the layer metadata first, then either counts the features (`--count-only`, `returnCountOnly=true`) or downloads
them with `f=geojson`, which the server delivers in EPSG:4326 without `outSR` (the layer's native SR is EPSG:5514;
the sheets are reprojected locally with pyproj when used, decision D03). Pages (by object id) are saved unmodified,
then combined into one GeoJSON FeatureCollection.

In phase 1 only the count is run (the hand-over file `data/raw_data/cgs_surveyed_sheets.geojson` is used):

    uv run python scripts/fetch_sheets.py --count-only

The licence of this layer is not stated: the output stays in the gitignored `data/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from landslide import sheets as S
from landslide.logs import get_logger, setup_logging
from landslide.net import Client
from landslide.paths import Paths


def _positions(coords) -> list:
    """All [x, y] positions of a GeoJSON coordinates array, at any nesting depth."""
    if coords and isinstance(coords[0], (int, float)):
        return [coords[:2]]
    return [p for c in coords for p in _positions(c)]


def _lonlat_range(fc: dict) -> tuple[float, float, float, float]:
    xy = np.array([p for f in fc["features"] for p in _positions(f["geometry"]["coordinates"])], dtype=float)
    return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())


def main(argv: list[str] | None = None) -> int:
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count-only", action="store_true", help="read the metadata and count; download nothing")
    ap.add_argument("--out", type=Path, default=paths.cache / "rebuild" / "cgs_surveyed_sheets.geojson")
    ap.add_argument("--pages", type=Path, default=paths.cache / "rebuild" / "sheets_pages",
                    help="where the unmodified response pages go")
    args = ap.parse_args(argv)

    setup_logging(paths.logs / "fetch_sheets.log")
    log = get_logger("fetch_sheets")
    with Client() as client:
        meta = S.layer_metadata(client, S.SHEETS_URL)
        log.info("%s: %s, %s, maxRecordCount %s, SR %s", S.SHEETS_URL, meta.get("name"), meta.get("geometryType"),
                 meta.get("maxRecordCount"), meta.get("sourceSpatialReference"))
        n = S.count_features(client, S.SHEETS_URL)
        log.info("count: %d", n)
        if args.count_only:
            print(json.dumps({"url": S.SHEETS_URL, "where": "1=1", "count": n, "metadata": meta},
                             indent=1, ensure_ascii=False))
            return 0
        res = S.query_pages(client, S.SHEETS_URL, "1=1", "*", "geojson", int(meta["maxRecordCount"]),
                            S.oid_field(meta), args.pages)
    fc = S.geojson_pages_to_collection([args.pages / p["file"] for p in res["pages"]])
    xmin, ymin, xmax, ymax = _lonlat_range(fc)
    if len(fc["features"]) != n or not (-180 <= xmin <= xmax <= 180 and -90 <= ymin <= ymax <= 90):
        log.error("got %d features (count %d), coordinate range %s", len(fc["features"]), n,
                  (xmin, ymin, xmax, ymax))
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    (args.pages / "fetch.json").write_text(json.dumps({"metadata": meta, "count": n, **res}, indent=1,
                                                      ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d features to %s", n, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
