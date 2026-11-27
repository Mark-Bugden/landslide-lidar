"""Every file location in the pipeline (PLAN.md §4.2).

Inputs (raw data, caches, derived files, the dataset, baseline predictions) always live in the real `data/`.
Outputs of runs (run directories, stage 2, results, figures, logs) are redirected under `data/smoke/` in smoke
mode, so a smoke test can never write to the tracked `results/` or `figures/`.

`data/raw_data/` is the read-only hand-over (env override `LANDSLIDE_RAW_DIR`). `data/` itself can be moved with
`LANDSLIDE_DATA_DIR`, so git worktrees (which don't contain the gitignored `data/`) share the main checkout's data.
Tracked outputs (`results/`, `figures/`) and `logs/` stay in the checkout. Sheet ids are canonical as
`14-34-07`; raster file names use underscores (`14_34_07.tif`). Convert only here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "landslide.json"
CONFIG_SHA_PATH = REPO_ROOT / "configs" / "landslide.sha256"

RUN_NAMES = ("a_f0", "a_f1", "a_f2", "a_f3", "a_f4", "b", "s2")


def sheet_to_stem(sheet: str) -> str:
    """`14-34-07` -> `14_34_07`."""
    return sheet.replace("-", "_")


def stem_to_sheet(stem: str) -> str:
    """`14_34_07` -> `14-34-07`."""
    return stem.replace("_", "-")


@dataclass(frozen=True)
class Paths:
    """Resolved locations. `Paths(smoke=True)` redirects run outputs to `data/smoke/`."""

    smoke: bool = False
    root: Path = REPO_ROOT

    def _env(self, name: str) -> str | None:
        """An env override, honoured only for the checkout's own root. A `Paths` built on another root (a test's
        tmp dir) ignores the overrides, so it can never read or write the real data tree."""
        if Path(self.root).resolve() != REPO_ROOT:
            return None
        return os.environ.get(name) or None

    # ---- inputs (never redirected) ----
    @property
    def data(self) -> Path:
        env = self._env("LANDSLIDE_DATA_DIR")
        return Path(env) if env else self.root / "data"

    @property
    def raw(self) -> Path:
        env = self._env("LANDSLIDE_RAW_DIR")
        return Path(env) if env else self.data / "raw_data"

    @property
    def manifest(self) -> Path:
        return self.raw / "MANIFEST.csv"

    @property
    def register_gpkg(self) -> Path:
        return self.raw / "cgs_register_polygons.gpkg"

    @property
    def surveyed_sheets(self) -> Path:
        return self.raw / "cgs_surveyed_sheets.geojson"

    @property
    def geology_gpkg(self) -> Path:
        return self.raw / "cgs_geol500_carpathian_units.gpkg"

    @property
    def dmr5g_dir(self) -> Path:
        return self.raw / "dmr5g"

    @property
    def dtm_fetch_log(self) -> Path:
        return self.dmr5g_dir / "fetch_log.jsonl"

    def dtm(self, sheet: str) -> Path:
        return self.dmr5g_dir / f"{sheet_to_stem(sheet)}.tif"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def provenance_dir(self) -> Path:
        return self.cache / "provenance"

    @property
    def susceptibility_dir(self) -> Path:
        return self.cache / "susceptibility"

    def susceptibility(self, sheet: str) -> Path:
        return self.susceptibility_dir / f"{sheet_to_stem(sheet)}.tif"

    @property
    def zabaged_dir(self) -> Path:
        return self.cache / "zabaged"

    def zabaged(self, layer: str) -> Path:
        return self.zabaged_dir / f"{layer}.gpkg"

    @property
    def torch_home(self) -> Path:
        return self.cache / "torch"

    @property
    def derived(self) -> Path:
        return self.data / "derived"

    @property
    def sheets_parquet(self) -> Path:
        return self.derived / "sheets.parquet"

    @property
    def polygons_parquet(self) -> Path:
        return self.derived / "polygons.parquet"

    @property
    def flysch_outline(self) -> Path:
        return self.derived / "flysch_outline.gpkg"

    @property
    def splits_json(self) -> Path:
        return self.derived / "splits.json"

    @property
    def dataset(self) -> Path:
        return self.data / "dataset"

    @property
    def samples_parquet(self) -> Path:
        return self.dataset / "samples.parquet"

    @property
    def patches_npy(self) -> Path:
        return self.dataset / "patches.npy"

    @property
    def meta_csv(self) -> Path:
        return self.dataset / "meta.csv"

    @property
    def baselines_dir(self) -> Path:
        return self.data / "baselines"

    # ---- outputs (redirected in smoke mode) ----
    @property
    def out_root(self) -> Path:
        """Where run outputs go: `data/` normally, `data/smoke/` in smoke mode."""
        return self.data / "smoke" if self.smoke else self.data

    @property
    def runs(self) -> Path:
        return self.out_root / "runs"

    def run_dir(self, run: str, seed: int) -> Path:
        return self.runs / f"{run}_s{seed}"

    @property
    def stage2(self) -> Path:
        return self.out_root / "stage2"

    @property
    def results(self) -> Path:
        return self.out_root / "results" if self.smoke else self.root / "results"

    @property
    def figures(self) -> Path:
        return self.out_root / "figures" if self.smoke else self.root / "figures"

    @property
    def logs(self) -> Path:
        return self.out_root / "logs" if self.smoke else self.root / "logs"
