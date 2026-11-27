"""Record the environment and confirm CUDA and the hand-over data are usable.

Writes `results/environment.json`. Exits non-zero if CUDA is unavailable or `data/raw_data/` can't be read.

    uv run python scripts/check_env.py
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pyproj
import rasterio
import torch

from landslide.paths import Paths

PACKAGES = ("torch", "torchvision", "numpy", "scipy", "pandas", "pyarrow", "geopandas", "shapely", "pyproj",
            "rasterio", "pyogrio", "scikit-learn", "matplotlib", "pillow", "requests", "tqdm", "pytest")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Paths().root).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _driver_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return out.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _raw_data_check(paths: Paths) -> dict:
    """Open the manifest and one raster read-only; count the hand-over files."""
    with paths.manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    tifs = sorted(paths.dmr5g_dir.glob("*.tif"))
    with rasterio.open(tifs[0]) as src:
        block = src.read(1, window=((0, 16), (0, 16)))
        crs = src.crs.to_epsg()
        res = src.res
    return {
        "raw_dir": (paths.raw.relative_to(paths.root) if paths.raw.is_relative_to(paths.root)
                    else paths.raw).as_posix(),
        "manifest_rows": len(rows),
        "dtm_rasters": len(tifs),
        "fetch_log_lines": sum(1 for _ in paths.dtm_fetch_log.open(encoding="utf-8")),
        "vector_files_present": all(p.exists() for p in (paths.register_gpkg, paths.surveyed_sheets,
                                                          paths.geology_gpkg)),
        "sample_raster": {"file": tifs[0].name, "epsg": crs, "res_m": list(res),
                          "dtype": str(block.dtype), "readable": bool(np.isfinite(block).any())},
    }


def main() -> int:
    paths = Paths()
    cuda = torch.cuda.is_available()
    gpu: dict = {"cuda": cuda}
    if cuda:
        props = torch.cuda.get_device_properties(0)
        gpu.update({
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 2**30, 2),
            "capability": f"{props.major}.{props.minor}",
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "driver": _driver_version(),
        })
    try:
        raw = _raw_data_check(paths)
        raw_ok = raw["dtm_rasters"] > 0 and raw["vector_files_present"]
    except Exception as e:  # report, don't hide
        raw, raw_ok = {"error": f"{type(e).__name__}: {e}"}, False

    env = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: version(p) for p in PACKAGES},
        "proj": pyproj.proj_version_str,
        "gdal": rasterio.__gdal_version__,
        "gpu": gpu,
        "raw_data": raw,
    }
    out = paths.results / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(env, indent=2, ensure_ascii=False))
    print(f"\nwrote {out.relative_to(paths.root)}  cuda: {str(cuda).lower()}  raw data readable: {raw_ok}")
    return 0 if cuda and raw_ok else 1


if __name__ == "__main__":
    sys.exit(main())
