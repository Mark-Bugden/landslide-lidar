"""Synthetic inputs for the runner tests (P1-14), all under a temporary root. Nothing here reads real data.

- `write_tree` writes every file the preflight requires, consistent with a given meta table (the dataset, splits,
  baselines, rasters, stage-2 baseline outputs, weights) plus the frozen config (`configs/landslide.json` and
  `configs/landslide.sha256`).
- `FakeBackend` replaces training and the stage-2 CNN: it writes synthetic run directories (`eval_synth.write_unit`)
  and CNN detections, so the real evaluation and verdict run on them.
- `toy_stage2_world` builds two small reserved sheets with DTM and susceptibility rasters, so the real stage-2 code
  can run in the end-to-end smoke test.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval_synth import (cnn_probs, synth_detections, synth_meta, synth_targets, unit_test_ids, write_baselines,
                        write_unit)
from landslide import preflight as PF
from landslide.config import config_hash, load_config
from landslide.paths import Paths

GIB = 2 ** 30
WEIGHTS_NAME = "resnet18-f37072fd.pth"
BASELINES = PF.STAGE2_BASELINES


def make_cfg(px: int = 8, n_boot: int = 200, stage2_sheets: dict[str, list[str]] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg["status"] = "FROZEN (synthetic test config)"  # the hash excludes it
    cfg["patch"]["px"] = px
    cfg["eval"]["bootstrap"]["n"] = n_boot
    if stage2_sheets is not None:
        cfg["replication"]["splits"]["stage2_sheets"] = stage2_sheets
    return cfg


def full_meta(seed: int = 0) -> pd.DataFrame:
    """eval_synth's meta with the statistics the runner's preflight and the LR comparator need."""
    meta = synth_meta(seed)
    rng = np.random.default_rng(seed + 1)
    y = meta["label"].to_numpy()
    meta["slope_std"] = rng.uniform(1, 5, len(meta))
    meta["susc_mean"] = np.clip(rng.normal(0.5 + 0.4 * y, 0.4), 0, 2)
    meta["susc_high_frac"] = np.clip(rng.normal(0.1 + 0.2 * y, 0.1), 0, 1)
    return meta


def synth_splits(meta: pd.DataFrame, chash: str) -> dict[str, Any]:
    blocks = meta.drop_duplicates("block")[["block", "fold", "group_b"]]
    folds = {str(k): {"blocks": sorted(blocks.loc[blocks["fold"] == k, "block"]), "sheets": [], "n_eligible": 0}
             for k in range(5)}
    return {"config_hash": chash,
            "folds_a": folds,
            "group_b": {"train_blocks": sorted(blocks.loc[blocks["group_b"] == "train", "block"]),
                        "test_blocks": sorted(blocks.loc[blocks["group_b"] == "test", "block"])},
            "inner_val": {"a_f0": ["FB1"], "a_f1": ["FB2"], "a_f2": ["FB3"], "a_f3": ["FB4"], "a_f4": ["FB0"],
                          "b": ["FB2"], "s2": ["FB3", "NB1"]}}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze(root: Path, cfg: dict[str, Any]) -> str:
    """Write `configs/landslide.json` and the frozen hash, as P1-16 will."""
    d = root / "configs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "landslide.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    h = config_hash(cfg)
    (d / "landslide.sha256").write_text(h + "\n", encoding="ascii")
    return h


def _write_tif(path: Path, arr: np.ndarray, x0: float = -600_000.4, y0: float = -1_150_000.12) -> None:
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype=arr.dtype.name, crs="EPSG:5514", transform=Affine(2.0, 0, x0, 0, -2.0, y0)) as dst:
        dst.write(arr, 1)


def write_dataset(paths: Paths, cfg: dict[str, Any], meta: pd.DataFrame, patches: np.ndarray | None = None,
                  seed: int = 0) -> np.ndarray:
    px, c = int(cfg["patch"]["px"]), len(cfg["channels"])
    if patches is None:
        rng = np.random.default_rng(seed)
        patches = (rng.normal(size=(len(meta), c, px, px)) * 3 + 10).astype(np.float16)
        patches[meta["label"].to_numpy() == 1, 0] += 0.5
    paths.dataset.mkdir(parents=True, exist_ok=True)
    np.save(paths.patches_npy, patches)
    meta.to_csv(paths.meta_csv, index=False)
    shape, _, _, _ = PF.npy_header(paths.patches_npy)
    info = {"config_hash": config_hash(cfg),
            "patches": {"shape": list(shape), "bytes": paths.patches_npy.stat().st_size},
            "meta": {"rows": len(meta), "bytes": paths.meta_csv.stat().st_size, "sha256": _sha256(paths.meta_csv)}}
    (paths.dataset / "build_info.json").write_text(json.dumps(info), encoding="utf-8")
    return patches


def write_tree(root: Path, cfg: dict[str, Any], meta: pd.DataFrame | None = None, *, patches: np.ndarray | None = None,
               stage2_outputs: bool = True, rasters: bool = True, sheets_parquet: bool = True,
               polygons: bool = True, weights: bytes | None = None) -> Paths:
    """Everything the preflight checks, consistent with `meta` and `cfg`, frozen. Returns the full-mode Paths."""
    meta = full_meta() if meta is None else meta
    paths = Paths(root=root)
    h = freeze(root, cfg)
    sheets = PF.stage2_sheet_list(cfg)
    paths.derived.mkdir(parents=True, exist_ok=True)
    paths.splits_json.write_text(json.dumps(synth_splits(meta, h)), encoding="utf-8")
    if sheets_parquet:
        pd.DataFrame({"sheet": sheets, "grid_w": 5, "grid_h": 4, "is_stage2": True}).to_parquet(
            paths.sheets_parquet, index=False)
    if polygons:
        synth_targets(sheets).to_parquet(paths.polygons_parquet, index=False)
    write_dataset(paths, cfg, meta, patches)
    write_baselines(paths, meta)
    (paths.baselines_dir / "build_info.json").write_text(json.dumps(
        {"config_hash": h, "inputs": {"meta_csv_sha256": _sha256(paths.meta_csv)}}), encoding="utf-8")
    if rasters:
        for s in sheets:
            _write_tif(paths.dtm(s), np.full((4, 5), 400.0, dtype=np.float32))
            _write_tif(paths.susceptibility(s), np.zeros((4, 5), dtype=np.uint8))
    if stage2_outputs:
        write_stage2_baseline_outputs(paths, cfg, meta)
    wp = paths.torch_home / "hub" / "checkpoints" / WEIGHTS_NAME
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_bytes(weights if weights is not None else b"fake weights")
    return paths


def write_stage2_baseline_outputs(paths: Paths, cfg: dict[str, Any], meta: pd.DataFrame | None = None) -> None:
    """Dummy phase-1 stage-2 baseline outputs (P1-12's files) carrying the config hash; the detections are
    eval_synth's, so the evaluation can read them."""
    h = config_hash(cfg)
    sheets = PF.stage2_sheet_list(cfg)
    targets = synth_targets(sheets)
    frames = [synth_detections(b, sheets, targets, tp_frac=0.7, n_fp=20, quality=0.2, seed=10 + i)
              for i, b in enumerate(BASELINES)]
    paths.stage2.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(paths.stage2 / "detections.parquet", index=False)
    (paths.stage2 / "summary.json").write_text(json.dumps({"config_hash": h, "scorers": {b: {} for b in BASELINES},
                                                           "lattice": {"n_windows": 146_622, "per_sheet": {
                                                               s: {} for s in sheets}}}), encoding="utf-8")
    for s in sheets:
        (paths.stage2 / "lattice").mkdir(parents=True, exist_ok=True)
        np.savez(paths.stage2 / "lattice" / f"{s}.npz", row=np.arange(3), config_hash=np.str_(h))
        for b in BASELINES:
            (paths.stage2 / b).mkdir(parents=True, exist_ok=True)
            np.savez(paths.stage2 / b / f"{s}.npz", score=np.arange(3.0))
    paths.results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"scorer": list(BASELINES), "config_hash": h}).to_csv(paths.results / "stage2_baselines.csv",
                                                                       index=False)


# ======================================================================================================================
# Machine probes
# ======================================================================================================================


def gpu(free_gib: float = 11.0, available: bool = True):
    def probe() -> PF.GpuInfo:
        if not available:
            return PF.GpuInfo(available=False, torch_version="x", cuda_version=None,
                              error="torch.cuda.is_available() is False")
        return PF.GpuInfo(True, "Fake RTX", int(free_gib * GIB), int(12 * GIB), "x", "13.0")
    return probe


def disk(free_gib: float = 100.0):
    class Usage:
        free = int(free_gib * GIB)
        total = int(500 * GIB)
        used = total - free

    def usage(_p: Path) -> Usage:
        return Usage()
    return usage


# ======================================================================================================================
# Fake backend: synthetic training results and CNN detections; the real evaluation and verdict
# ======================================================================================================================


class FakeBackend:
    """Mimics `train_unit`'s contract (done.json / failed.json / interrupted.json, should_stop between epochs) and
    the stage-2 calls, recording every call. The evaluation is the real one (`Backend.evaluate`)."""

    def __init__(self, meta: pd.DataFrame, *, strength_a: float = 2.0, strength_b: float = 1.5,
                 fail: set[str] | frozenset[str] = frozenset(), epochs: int = 3, stage2: str = "pass",
                 tick: Any = None):
        from landslide.runner import Backend

        self._real = Backend()
        self.meta = meta
        self.strength = {"a": strength_a, "b": strength_b, "s2": 1.0}
        self.fail = set(fail)
        self.epochs = epochs
        self.stage2 = stage2
        self.tick = tick  # called after every simulated epoch (advances a fake clock)
        self.calls: list[tuple[str, Any]] = []
        self.network_blocked: list[bool] = []
        self.keeps: dict[str, np.ndarray | None] = {}

    # training
    def train(self, run, seed, cfg, paths, *, keep, should_stop, show_progress):
        name = f"{run}_s{seed}"
        self.calls.append(("train", name))
        self.network_blocked.append(PF.network_blocked())
        self.keeps[name] = keep
        d = paths.run_dir(run, seed)
        if (d / "done.json").exists():
            raise FileExistsError(f"{d / 'done.json'} exists")
        d.mkdir(parents=True, exist_ok=True)
        for f in ("done.json", "failed.json", "interrupted.json", "history.csv", "preds_test.csv", "preds_val.csv"):
            (d / f).unlink(missing_ok=True)
        chash = config_hash(cfg)
        if name in self.fail:
            (d / "failed.json").write_text(json.dumps({"unit": name, "error": "RuntimeError: boom",
                                                       "config_hash": chash}), encoding="utf-8")
            raise RuntimeError(f"boom in {name}")
        for epoch in range(1, self.epochs + 1):
            if epoch == 1 and should_stop():
                return self._interrupted(d, name, chash, 0)
            if self.tick is not None:
                self.tick()
            if epoch < self.epochs and should_stop():
                return self._interrupted(d, name, chash, epoch)
        ids = unit_test_ids(self.meta, run)
        if keep is not None:
            kept = set(self.meta["id"].to_numpy()[np.asarray(keep)].tolist())
            ids = np.array([i for i in ids if i in kept], dtype=np.int64)
        kind = "a" if run.startswith("a_f") else run
        p = cnn_probs(self.meta, ids, seed, self.strength[kind]) if len(ids) else np.empty(0)
        write_unit(paths, run, seed, ids, p, chash, epochs=self.epochs, best_epoch=min(2, self.epochs))
        done = json.loads((d / "done.json").read_text(encoding="utf-8"))
        return {"status": "done", **done}

    @staticmethod
    def _interrupted(d: Path, name: str, chash: str, epochs: int) -> dict[str, Any]:
        info = {"unit": name, "config_hash": chash, "epochs_completed": epochs, "reason": "should_stop callback"}
        (d / "interrupted.json").write_text(json.dumps(info), encoding="utf-8")
        return {"status": "interrupted", **info}

    # evaluation (real)
    def evaluate(self, cfg, paths):
        self.calls.append(("evaluate", None))
        return self._real.evaluate(cfg, paths)

    def stage2_allowed(self, metrics):
        return self._real.stage2_allowed(metrics)

    # stage 2
    def stage2_inputs(self, cfg, paths, sheets):
        self.calls.append(("stage2_inputs", None if sheets is None else list(sheets)))
        return list(sheets) if sheets is not None else PF.stage2_sheet_list(cfg)

    def stage2_sheet_ids(self, inputs):
        return list(inputs)

    def stage2_baselines(self, cfg, paths, inputs):
        self.calls.append(("stage2_baselines", list(inputs)))
        targets = synth_targets(list(inputs))
        frames = [synth_detections(b, list(inputs), targets, tp_frac=0.7, n_fp=20, quality=0.2, seed=20 + i)
                  for i, b in enumerate(BASELINES)]
        _upsert(paths, pd.concat(frames, ignore_index=True))
        return {b: {"status": "reached"} for b in BASELINES}

    def stage2_models(self, cfg, paths, seeds):
        self.calls.append(("stage2_models", list(seeds)))
        return "models"

    def stage2_cnn_sheet(self, cfg, paths, inputs, sheet, models, show_progress):
        self.calls.append(("stage2_cnn_sheet", sheet))
        self.network_blocked.append(PF.network_blocked())
        if self.tick is not None:
            self.tick()
        return {"status": "done", "n_windows": 100, "seconds": 2.0, "windows_per_s": 50.0}

    def stage2_evaluate_cnn(self, cfg, paths, inputs):
        self.calls.append(("stage2_evaluate_cnn", list(inputs)))
        sheets = list(inputs)
        targets = synth_targets(sheets)
        frac = {"pass": 0.85, "fail": 0.85}[self.stage2]
        quality = {"pass": 3.0, "fail": -0.5}[self.stage2]
        n_fp = {"pass": 4, "fail": 40}[self.stage2]
        _upsert(paths, synth_detections("cnn", sheets, targets, tp_frac=frac, n_fp=n_fp, quality=quality, seed=1))
        return {"status": "reached"}

    def gpu_peak_reset(self):
        pass

    def gpu_peak_gib(self):
        return None

    def release(self, models):
        pass

    def full_unit_sizes(self, cfg, paths):
        return self._real.full_unit_sizes(cfg, paths)

    # helpers for assertions
    def trained(self) -> list[str]:
        return [n for k, n in self.calls if k == "train"]


def _upsert(paths: Paths, new: pd.DataFrame) -> None:
    p = paths.stage2 / "detections.parquet"
    if p.exists():
        old = pd.read_parquet(p)
        new = pd.concat([old.loc[~old["scorer"].isin(set(new["scorer"]))], new], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(p, index=False)


class FakeClock:
    """A monotonic clock that the test advances explicitly."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def file_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime_ns) of every file under `root`."""
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


# ======================================================================================================================
# Toy reserved sheets for the real stage-2 code (end-to-end smoke test)
# ======================================================================================================================


def rotated_square(cx: float, cy: float, side: float, deg: float = 3.0):
    from shapely.geometry import Polygon

    t = math.radians(deg)
    h = side / 2.0
    pts = [(-h, -h), (h, -h), (h, h), (-h, h)]
    return Polygon([(cx + u * math.cos(t) - v * math.sin(t), cy + u * math.sin(t) + v * math.cos(t)) for u, v in pts])


def toy_stage2_world(paths: Paths, cfg: dict[str, Any], side_m: float = 900.0) -> None:
    """sheets.parquet, polygons.parquet and the DTM / susceptibility rasters of the cfg's reserved sheets (which
    the caller sets to toy sheet numbers), on the DMR 5G lattice."""
    import geopandas as gpd
    from shapely.geometry import box

    from landslide.sheets import dtm_grid
    from terrain_synth import synthetic_dtm

    lists = cfg["replication"]["splits"]["stage2_sheets"]
    x0, y0 = -600_000.4, -1_150_000.12
    rows, polys, oid = [], [], 100
    rng = np.random.default_rng(7)
    sheets = [(s, k == "flysch") for k in ("flysch", "nonflysch") for s in lists[k]]
    for i, (sheet, flysch) in enumerate(sorted(sheets)):
        geom = rotated_square(x0 + 1500 + 8000 * i, y0 - 1200, side_m, deg=3.0 + i)
        g = dtm_grid(geom.bounds, cfg)
        row = {"sheet": sheet, "flysch": flysch, "grid_xmin": g.xmin, "grid_ymax": g.ymax, "grid_w": g.w,
               "grid_h": g.h, "is_stage2": True, "geometry": geom}
        rows.append(row)
        z = synthetic_dtm((g.h, g.w), seed=10 + i, border=False, holes=False, lake=False).astype(np.float32)
        code = np.digitize(z, np.quantile(z, [0.5, 0.8])).astype(np.uint8)
        _write_tif(paths.dtm(sheet), z, g.xmin, g.ymax)
        _write_tif(paths.susceptibility(sheet), code, g.xmin, g.ymax)
        cx, cy = geom.centroid.x, geom.centroid.y
        for _ in range(6):
            px_, py_ = cx + rng.uniform(-300, 300), cy + rng.uniform(-300, 300)
            elig = bool(rng.random() < 0.8)
            polys.append({"objectid": oid, "eligible": elig, "sheet": sheet if elig else None,
                          "geometry": box(px_ - 40, py_ - 40, px_ + 40, py_ + 40)})
            oid += 1
    paths.derived.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:5514").to_parquet(paths.sheets_parquet)
    gpd.GeoDataFrame(polys, geometry="geometry", crs="EPSG:5514").to_parquet(paths.polygons_parquet)
