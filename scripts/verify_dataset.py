"""Check the P1-10 dataset (`patches.npy`, `meta.csv`); exit non-zero on any failure.

Checks (`landslide.patches.verify_dataset`):
- the files exist; the `.npy` header is float16, C order, (len(meta.csv), 7, 150, 150); the file size is exactly
  header + data (and as recorded in `build_info.json`);
- every stored value is finite; no float16 overflow or saturation (max |value| < 65504); slope in [0, 90] deg and
  svf in [0, 1];
- `meta.csv` has the contract's columns with finite statistics; ids unique and equal to the row number;
- label, source, pair_id and match_level are consistent; each pair_id has exactly one positive and one negative, in
  the same stratum, obeying the slope-bin rule of its match level;
- every sample column equals `samples.parquet` row by row (when it exists);
- the patch rule, rechecked from the coordinates: x, y are the pixel corners (row, col) of the sheet grid; the patch
  plus its context lies inside the raster; the 300 m square lies inside its own stage-1 sheet and more than 150 m from
  every stage-2 sheet (exact shapely geometry);
- meta <-> patches alignment: a seeded sample of rows (plus the first and last) is re-extracted from the rasters,
  each with its own `read_dtm` window (patch plus 28 px context); the float16 patches must be bit-identical and the
  statistics equal;
- `build_info.json` exists (the build writes it last) and records the sha256 of `meta.csv` and `patches.npy`
  (`--no-hash` skips the 3.5 GB hash);
- `results/dataset_summary.json` agrees on the row count and config hash;
- a warning (not a failure) if the current config hash differs from the build's.

    uv run python scripts/verify_dataset.py
    uv run python scripts/verify_dataset.py --n-check 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from landslide import patches as P
from landslide.config import config_hash, load_config
from landslide.logs import get_logger, setup_logging
from landslide.paths import Paths
from landslide.sampling import grid_of_sheet

log = get_logger("verify_dataset")


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=None, help="dataset directory (default: data/dataset)")
    ap.add_argument("--summary", type=Path, default=None,
                    help="dataset_summary.json (default: results/dataset_summary.json)")
    ap.add_argument("--n-check", type=int, default=200, help="seeded rows to re-extract (default 200)")
    ap.add_argument("--no-hash", action="store_true", help="skip the sha256 of patches.npy")
    ap.add_argument("--log", type=Path, default=None, help="log file (default: logs/verify_dataset.log)")
    args = ap.parse_args(argv)
    dataset = args.dataset or paths.dataset
    summary_path = args.summary or (paths.results / P.SUMMARY_NAME)

    setup_logging(args.log or (paths.logs / "verify_dataset.log"))
    cfg = load_config()
    t0 = time.perf_counter()
    sheets = gpd.read_parquet(paths.sheets_parquet)
    cell = float(cfg["dtm"]["cell_m"])
    sources = {s: P.SheetSource(sheet=s, dtm_path=paths.raw / str(row.dtm_path), susc_path=paths.susceptibility(s),
                                grid=grid_of_sheet(row, cell))
               for s, row in zip(sheets["sheet"], sheets.itertuples(index=False))}
    samples_path = dataset / "samples.parquet"
    samples = pd.read_parquet(samples_path) if samples_path.is_file() else None
    checks = P.verify_dataset(cfg, dataset / "patches.npy", dataset / "meta.csv", sheets, sources,
                              samples=samples, build_info=_json(dataset / P.BUILD_INFO_NAME),
                              summary=_json(summary_path), config_hash=config_hash(cfg), n_check=args.n_check,
                              full_hash=not args.no_hash, require_build_info=True)
    if samples is None:
        log.warning("no samples.parquet next to the dataset: the meta_vs_samples check was skipped")
    for c in checks:
        (log.info if c.ok else log.error if c.level == "fail" else log.warning)(c.line())
    bad = P.failures(checks)
    log.info("%d checks, %d failed, %d warnings, %.1f s", len(checks), len(bad),
             sum(1 for c in checks if not c.ok and c.level == "warn"), time.perf_counter() - t0)
    if bad:
        log.error("DATASET CHECK FAILED: %s", ", ".join(c.name for c in bad))
        return 1
    log.info("DATASET OK: no patch is missing or contains NaN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
