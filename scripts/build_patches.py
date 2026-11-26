"""Patch extraction and statistics (task P1-10): build `patches.npy`, `meta.csv`, the dataset summary and the slope
figure from P1-09's `samples.parquet`.

Reads (never rebuilds) `data/dataset/samples.parquet`, `data/derived/sheets.parquet` (sheet grids and DTM paths only;
no geometry), the DMR 5G rasters in place and P1-07's susceptibility class rasters. Streams sheet by sheet: each
sheet's DTM and class raster are opened once, over the window that holds all its patches plus their 28 px context.
Method: `landslide.patches` and `landslide.features`; decisions in `docs/decisions/P1-10.md`.

Outputs:
- `data/dataset/patches.npy` (gitignored): float16, (N, 7, 150, 150), channel order `config.channels`, physical units;
- `data/dataset/meta.csv` (gitignored): row i describes patches[i] (PLAN.md §4.3 columns, P1-09's `source_layer`,
  `source_objectid`, and the D19 statistics, susceptibility included);
- `data/dataset/build_info.json` (gitignored, written last): config hash, git commit, file sizes and sha256, channel
  ranges, float16 error, the slope-mean check;
- `results/dataset_summary.json` (tracked): aggregate counts only;
- `figures/slope_distributions.png` (tracked): slope histograms, positives vs negatives per test set (no maps).

`--out DIR` writes DIR/dataset, DIR/results, DIR/figures and DIR/logs instead. Then check the result with
`scripts/verify_dataset.py`.

    uv run python scripts/build_patches.py
    uv run python scripts/verify_dataset.py
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from landslide import patches as P
from landslide.channels import DEFAULT_WORKERS
from landslide.config import config_hash, load_config
from landslide.logs import get_logger, setup_logging
from landslide.paths import Paths
from landslide.sampling import grid_of_sheet

log = get_logger("build_patches")

#: Columns read from sheets.parquet (no geometry: the sheet outlines are not needed for extraction).
SHEET_COLUMNS = ["sheet", "grid_xmin", "grid_ymax", "grid_w", "grid_h", "dtm_path", "is_stage1"]
#: Free disk required beyond the patch file itself (meta.csv, build_info, the temporary files).
DISK_MARGIN_BYTES = 1 << 30


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=root).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _release_frames(exc: BaseException | None) -> None:
    """Clear the local variables of the frames in an exception chain's tracebacks (the message and stack stay)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__


def sheet_sources(cfg: dict[str, Any], paths: Paths, sheet_ids: set[str]) -> dict[str, P.SheetSource]:
    """Each sheet's DTM (`paths.raw / dtm_path`, as P1-09 reads it), class raster and expected grid."""
    sheets = pd.read_parquet(paths.sheets_parquet, columns=SHEET_COLUMNS)
    sh = sheets.set_index("sheet")
    missing = sorted(sheet_ids - set(sh.index))
    if missing:
        raise ValueError(f"samples on sheets missing from sheets.parquet: {missing[:5]}")
    not_stage1 = sorted(s for s in sheet_ids if not bool(sh.loc[s, "is_stage1"]))
    if not_stage1:
        raise ValueError(f"samples on sheets that are not stage-1 sheets: {not_stage1[:5]}")
    cell = float(cfg["dtm"]["cell_m"])
    out = {}
    for s in sorted(sheet_ids):
        row = sh.loc[s]
        out[s] = P.SheetSource(sheet=s, dtm_path=paths.raw / str(row["dtm_path"]), susc_path=paths.susceptibility(s),
                               grid=grid_of_sheet(row, cell))
    absent = [str(p) for src in out.values() for p in (src.dtm_path, src.susc_path) if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} rasters missing, e.g. {absent[:3]}")
    return out


def main(argv: list[str] | None = None) -> int:
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="write under this directory (OUT/dataset, OUT/results, OUT/figures, OUT/logs) instead")
    ap.add_argument("--samples", type=Path, default=None, help="samples.parquet to read (default: the dataset's)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"threads per sheet (default {DEFAULT_WORKERS}; the output does not depend on it)")
    args = ap.parse_args(argv)
    dataset = args.out / "dataset" if args.out else paths.dataset
    results = args.out / "results" if args.out else paths.results
    figures = args.out / "figures" if args.out else paths.figures
    logs = args.out / "logs" if args.out else paths.logs
    samples_path = args.samples or paths.samples_parquet

    setup_logging(logs / "build_patches.log")
    cfg = load_config()
    chash = config_hash(cfg)
    log.info("build_patches: config %s, samples %s, dataset -> %s", chash[:12], samples_path, dataset)
    t0 = time.perf_counter()

    samples = pd.read_parquet(samples_path)
    P.check_samples_table(samples)
    samp_summary = results / "sampling_summary.json"
    if samp_summary.is_file():
        h = json.loads(samp_summary.read_text(encoding="utf-8")).get("config_hash")
        if h != chash:
            log.warning("samples were built with config %s, the current config is %s", str(h)[:12], chash[:12])
    pp = P.patch_params(cfg)
    N = len(samples)
    shape = (N, *pp.shape)
    data_bytes = int(np.prod(shape)) * P.PATCH_DTYPE.itemsize
    dataset.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dataset).free
    if free < data_bytes + DISK_MARGIN_BYTES:
        log.error("not enough free disk: %.2f GB free, %.2f GB needed", free / 1e9,
                  (data_bytes + DISK_MARGIN_BYTES) / 1e9)
        return 2
    sources = sheet_sources(cfg, paths, set(samples["sheet"].astype(str)))
    log.info("%d samples on %d sheets -> patches %s float16 (%.2f GB)", N, len(sources), shape, data_bytes / 1e9)

    def open_window(sheet: str, rows: np.ndarray, cols: np.ndarray) -> P.SheetWindow:
        return P.read_sheet_window(sources[sheet], rows, cols, cfg)

    npy = dataset / "patches.npy"
    tmp = dataset / "patches.npy.part"
    # build_info.json is written last: without it (or with other hashes) the dataset is not a finished build
    (dataset / P.BUILD_INFO_NAME).unlink(missing_ok=True)
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=P.PATCH_DTYPE, shape=shape)
    try:
        ext = P.extract_dataset(cfg, samples, open_window, mm, workers=args.workers, show_progress=True)
        mm.flush()
    except BaseException as e:
        # Windows keeps a mapped file locked: drop every reference (the traceback's frames hold the array) and delete
        # the partial file, then re-raise.
        _release_frames(e)
        del mm
        gc.collect()
        tmp.unlink(missing_ok=True)
        raise
    del mm
    gc.collect()
    tmp.replace(npy)
    t_extract = time.perf_counter() - t0
    log.info("extracted %d patches in %.1f s; max |slope_mean recomputed - stored| = %.3g deg", N, t_extract,
             ext.info["max_slope_mean_diff_deg"])

    meta = P.build_meta(samples, ext.stats)
    meta_path = dataset / "meta.csv"
    P.write_meta(meta, meta_path)
    back = P.read_meta(meta_path)
    pd.testing.assert_frame_equal(back, meta, check_dtype=False, check_exact=True)

    summary = {"config_hash": chash, "git_commit": _git_commit(paths.root), **P.summarise(meta, cfg)}
    P.dump_json(summary, results / P.SUMMARY_NAME)
    P.plot_slope_distributions(meta, cfg, figures / P.FIGURE_NAME)

    _, _, _, offset = P.npy_header(npy)
    info = {
        "config_hash": chash,
        "git_commit": summary["git_commit"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": {"file": samples_path.name, "rows": N, "sha256": P.sha256_file(samples_path)},
        "patches": {"file": npy.name, "shape": list(shape), "dtype": str(P.PATCH_DTYPE), "header_bytes": offset,
                    "bytes": npy.stat().st_size, "sha256": P.sha256_file(npy)},
        "meta": {"file": meta_path.name, "rows": len(meta), "columns": list(P.META_COLUMNS),
                 "bytes": meta_path.stat().st_size, "sha256": P.sha256_file(meta_path)},
        "extraction": ext.info,
        "workers": args.workers,
        "seconds": {"extraction": round(t_extract, 1), "total": round(time.perf_counter() - t0, 1)},
    }
    P.dump_json(info, dataset / P.BUILD_INFO_NAME)
    log.info("wrote %s (%.2f GB), %s, %s, %s and %s in %.1f s", npy, npy.stat().st_size / 1e9, meta_path,
             dataset / P.BUILD_INFO_NAME, results / P.SUMMARY_NAME, figures / P.FIGURE_NAME,
             time.perf_counter() - t0)
    log.info("counts by kind %s; channel ranges %s", summary["by_kind"],
             {k: (round(ext.info["channel_min"][k], 3), round(ext.info["channel_max"][k], 3))
              for k in pp.names})
    return 0


if __name__ == "__main__":
    sys.exit(main())
