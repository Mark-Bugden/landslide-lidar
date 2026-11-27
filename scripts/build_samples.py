"""Sampling and slope matching (task P1-09): build `data/dataset/samples.parquet` and `results/sampling_summary.json`.

Reads (never rebuilds) P1-08's `data/derived/{sheets,polygons}.parquet` and `splits.json`, the DMR 5G rasters in place
and P1-07's ZABAGED cache (`landslide.sampling.load_zabaged`). Method: `landslide.sampling` (Appendix A §5, D11-D17);
decisions in `docs/decisions/P1-09.md`.

After the build, every rule is re-checked on the final table (`landslide.sampling.check_samples`): strata, the
slope-bin rule per match level, >= 150 m spacing of the negatives and background, >= 150 m from every register polygon
and ZABAGED landslide/scree feature (exact distances), the patch rule with exact shapely squares, positives inside
their polygons, and the validity and slope mean of every patch recomputed from its raster window. A failure stops the
script before anything is written. `--check-determinism` builds a second time and requires identical output.

Outputs:
- `data/dataset/samples.parquet` (gitignored): one row per sample, PLAN.md §4.3 columns (plus `source_layer`,
  `source_objectid` for look-alikes);
- `results/sampling_summary.json` (tracked): aggregate counts only (no coordinates), the checks' margins and the
  comparison with PLAN.md §5. Build time and peak memory go to the log only.

`--out DIR` writes DIR/dataset/samples.parquet and DIR/results/sampling_summary.json instead.

    uv run python scripts/build_samples.py --check-determinism
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from landslide import sampling as S
from landslide.config import config_hash, load_config
from landslide.dtm import read_dtm
from landslide.logs import get_logger, setup_logging
from landslide.paths import Paths

log = get_logger("build_samples")

#: PLAN.md §5 reference numbers for this task (checks, not targets) and the Appendix A §4 count they rest on.
SPIKE = [
    ("eligible stage-1 polygons", ("positives", "eligible_stage1_polygons"), 14617),
    ("eligible stage-1 polygons with no feasible patch", ("positives", "infeasible_polygons"), 870),
    ("water-filling cap, flysch (points per sheet)", ("positives", "pools", "flysch", "cap"), 8),
    ("water-filling cap, non-flysch (points per sheet)", ("positives", "pools", "nonflysch", "cap"), 3),
    ("positives selected", ("positives", "selected_points"), 3613),
    ("positives dropped by validity", ("positives", "validity_dropped"), 29),
    ("positives kept (before matching)", ("positives", "kept"), 3584),
]


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=root).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def peak_memory_gb() -> float | None:
    """Peak working set of this process (Windows), in GB."""
    if sys.platform != "win32":
        return None

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    proc = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(proc, ctypes.byref(pmc), pmc.cb):
        return None
    return pmc.PeakWorkingSetSize / 2**30


def load_inputs(cfg: dict[str, Any], paths: Paths) -> tuple[S.Inputs, dict[str, Any]]:
    """P1-08's derived files, the ZABAGED cache, and a DTM reader. Returns the inputs and a provenance record."""
    sheets = gpd.read_parquet(paths.sheets_parquet)
    polygons = gpd.read_parquet(paths.polygons_parquet)
    splits = json.loads(paths.splits_json.read_text(encoding="utf-8"))
    for df, name in ((sheets, "sheets"), (polygons, "polygons")):
        if df.crs is None or df.crs.to_epsg() != 5514:
            raise ValueError(f"{name}.parquet: expected EPSG:5514, got {df.crs}")
    if len(polygons) != 21356 or len(sheets) != 860:
        log.warning("unexpected counts: %d polygons, %d sheets", len(polygons), len(sheets))
    s2 = set(splits["stage2"]["flysch"]) | set(splits["stage2"]["nonflysch"])
    if s2 != set(sheets.loc[sheets["is_stage2"], "sheet"]):
        raise ValueError("splits.json and sheets.parquet disagree on the stage-2 sheets")
    if set(splits["stage1_sheets"]) != set(sheets.loc[sheets["is_stage1"], "sheet"]):
        raise ValueError("splits.json and sheets.parquet disagree on the stage-1 sheets")
    zab = S.load_zabaged(cfg, paths)
    dtm_path = dict(zip(sheets["sheet"], sheets["dtm_path"]))
    inputs = S.Inputs(sheets=sheets, polygons=polygons, zabaged=zab,
                      read_dtm=lambda s: read_dtm(paths.raw / dtm_path[s], cfg))
    prov = {"splits_config_hash": splits.get("config_hash"),
            "zabaged_features": {role: int(len(df)) for role, df in zab.items()},
            "zabaged_layers": S.sampling_params(cfg).zabaged_layers}
    return inputs, prov


def _get(d: dict[str, Any], path: tuple[str, ...]) -> Any:
    for k in path:
        d = d[k]
    return d


def spike_comparison(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"quantity": q, "spike": v, "ours": _get(summary, path), "equal": _get(summary, path) == v}
            for q, path, v in SPIKE]


def main(argv: list[str] | None = None) -> int:
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="write under this directory (OUT/dataset, OUT/results) instead")
    ap.add_argument("--check-determinism", action="store_true", help="build twice and require identical output")
    ap.add_argument("--no-recheck", action="store_true",
                    help="skip re-reading every patch window from its raster in the final checks")
    args = ap.parse_args(argv)
    dataset = args.out / "dataset" if args.out else paths.dataset
    results = args.out / "results" if args.out else paths.results

    setup_logging(paths.logs / "build_samples.log")
    cfg = load_config()
    chash = config_hash(cfg)
    log.info("build_samples: config %s, dataset -> %s, results -> %s", chash[:12], dataset, results)
    t0 = time.perf_counter()
    inputs, prov = load_inputs(cfg, paths)
    if prov["splits_config_hash"] != chash:
        log.warning("splits.json was built with config %s, the current config is %s",
                    str(prov["splits_config_hash"])[:12], chash[:12])
    res = S.build_samples(cfg, inputs, show_progress=True)
    t_build = time.perf_counter() - t0
    log.info("build done in %.1f s; peak memory %.2f GB", t_build, peak_memory_gb() or float("nan"))

    checks = S.check_samples(cfg, res.samples, inputs, recheck_dtm=not args.no_recheck)
    log.info("checks passed: %s", checks)

    determinism = None
    if args.check_determinism:
        t1 = time.perf_counter()
        res2 = S.build_samples(cfg, inputs, show_progress=True)
        pd.testing.assert_frame_equal(res.samples, res2.samples, check_exact=True)
        same = json.dumps(res.summary, sort_keys=True) == json.dumps(res2.summary, sort_keys=True)
        assert same, "the summaries of two builds differ"
        determinism = {"rebuilt": True, "samples_identical": True, "summary_identical": True}
        log.info("determinism: second build identical (%.1f s)", time.perf_counter() - t1)
        del res2

    summary: dict[str, Any] = {
        "config_hash": chash,
        "git_commit": _git_commit(paths.root),
        "inputs": prov,
        **res.summary,
        "checks": checks,
        "determinism": determinism,
        "spike_comparison": spike_comparison(res.summary),
    }
    dataset.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    out_parquet = dataset / "samples.parquet"
    res.samples.to_parquet(out_parquet, index=False)
    out_json = results / "sampling_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("wrote %s (%d rows) and %s", out_parquet, len(res.samples), out_json)
    for row in summary["spike_comparison"]:
        log.info("§5 %-52s spike %6s  ours %6s", row["quantity"], row["spike"], row["ours"])
    log.info("total %.1f s; peak memory %.2f GB", time.perf_counter() - t0, peak_memory_gb() or float("nan"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
