"""Training of one unit (run x seed): BRIEF.md Appendix A §7, Appendix B `train`, PLAN.md P1-06 and D20.

`train_unit(run, seed, cfg, paths)` trains one model and writes the run directory of PLAN.md §4.3
(`data/runs/<run>_s<seed>/`):

- `norm.json`: per-channel float64 mean and std over the training rows (inner-val rows excluded);
- `history.csv`: epoch, lr, train_loss, train_auc, val_loss, val_auc, seconds (rewritten after every epoch);
- `best.pt`: the weights of the best epoch, with metadata;
- `preds_val.csv`, `preds_test.csv`: `id`, `prob` (8-view TTA) from the restored best weights;
- `done.json` on success; `failed.json` (with the traceback) when an exception escapes; `interrupted.json` when the
  `should_stop` callback ends the unit between epochs. Only a finished unit has `done.json`.

The patches of the unit are copied once to the GPU as float16; there is no DataLoader and there are no worker
processes. Each epoch draws a permutation and one D4 transform per sample from `numpy.random.default_rng(seed)`, keeps
only whole batches (`drop_last`), and sets the learning rate before every optimiser step (linear warm-up, then cosine
to 0 at `max_epochs`, both per step). Early stopping follows the inner-val ROC AUC of the 8-view TTA probabilities;
ties go to the earliest epoch, and the best weights are restored before predicting.

Determinism: python, numpy and torch are seeded with the unit's seed; cuDNN is deterministic with benchmarking off;
`torch.use_deterministic_algorithms(True)`; `CUBLAS_WORKSPACE_CONFIG` must be `:4096:8` before CUDA starts (the entry
point sets it; `ensure_cublas_workspace` sets it if CUDA has not started yet and refuses otherwise).

Nothing here reads real data unless the caller passes no data source: `data`, `meta` and `splits` accept in-memory
objects (tests use synthetic ones) or paths, and default to `paths.patches_npy`, `paths.meta_csv` and
`paths.splits_json`.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

from landslide import d4
from landslide.config import config_hash
from landslide.logs import get_logger, progress
from landslide.model import build_model, imagenet_weights_path
from landslide.paths import REPO_ROOT, Paths
from landslide.predict import Normaliser, probs_from_logits, tta_logits, tta_predict

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_DETERMINISTIC_CUBLAS = (":4096:8", ":16:8")

BACKGROUND = "background"
HISTORY_COLUMNS = ("epoch", "lr", "train_loss", "train_auc", "val_loss", "val_auc", "seconds")
OUTPUT_FILES = ("history.csv", "best.pt", "norm.json", "preds_val.csv", "preds_test.csv", "done.json",
                "failed.json", "interrupted.json")
META_COLUMNS = ("id", "label", "source", "fold", "group_b", "block")

# Rows copied to the GPU (and reduced for the statistics) per chunk: 128 x 7 x 150 x 150 float16 = 40 MB.
_CHUNK = 128

log = get_logger("train")


# ---------------------------------------------------------------------------------------------------------------
# Unit selection
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitRows:
    """Positional row indices into `meta` (= into the patch array), sorted ascending."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _folds_a(splits: dict[str, Any]) -> dict[int, set[str]]:
    fa = splits["folds_a"]
    items = fa.items() if isinstance(fa, dict) else enumerate(fa)
    return {int(k): {str(b) for b in v["blocks"]} for k, v in items}


def _keep_mask(keep: np.ndarray | None, n: int) -> np.ndarray:
    if keep is None:
        return np.ones(n, dtype=bool)
    keep = np.asarray(keep)
    if keep.dtype == bool:
        if keep.shape != (n,):
            raise ValueError(f"boolean keep mask must have length {n}")
        return keep
    mask = np.zeros(n, dtype=bool)
    mask[keep.astype(np.int64)] = True
    return mask


def select_unit_rows(run: str, meta: pd.DataFrame, splits: dict[str, Any], *,
                     keep: np.ndarray | None = None) -> UnitRows:
    """Train, inner-val and test rows of a run, from `meta.csv` and `splits.json` (PLAN.md §4.3).

    - `a_f<k>`: the pool is the flysch stage-1 patches (`fold >= 0`) of the other folds, the test set is fold k;
    - `b`: the pool is `group_b == "train"`, the test set is `group_b == "test"`;
    - `s2`: the pool is every stage-1 patch plus the background sample; there is no test set.

    Background patches (`source == "background"`) belong to `s2` only. Inner-val = the pool rows whose block is in
    `splits["inner_val"][run]`; train = the rest of the pool. `keep` (boolean mask or row indices, e.g. a smoke subset)
    restricts all three sets. The fold and group columns of `meta` are checked against `splits`, and no block may be
    in both the test set and the pool.
    """
    missing = [c for c in META_COLUMNS if c not in meta.columns]
    if missing:
        raise KeyError(f"meta lacks columns {missing}")
    n = len(meta)
    label = meta["label"].to_numpy()
    if not np.isin(label, (0, 1)).all():
        raise ValueError("meta.label must be 0 or 1")
    is_bg = meta["source"].astype(str).to_numpy() == BACKGROUND
    block = meta["block"].astype(str).to_numpy()
    fold = meta["fold"].to_numpy().astype(np.int64)
    group = meta["group_b"].astype(str).to_numpy()
    keep_m = _keep_mask(keep, n)

    if run not in splits.get("inner_val", {}):
        raise KeyError(f"splits.inner_val has no entry for run {run!r}")
    inner = np.isin(block, [str(b) for b in splits["inner_val"][run]])

    m = re.fullmatch(r"a_f(\d+)", run)
    if m:
        k = int(m.group(1))
        folds = _folds_a(splits)
        if k not in folds:
            raise KeyError(f"splits.folds_a has no fold {k}")
        all_fold_blocks = set().union(*folds.values())
        for j, blocks in folds.items():
            stray = set(block[fold == j]) - blocks
            if stray:
                raise ValueError(f"meta rows with fold {j} lie in blocks not listed for fold {j}: {sorted(stray)[:5]}")
        if set(block[fold < 0]) & all_fold_blocks:
            raise ValueError("meta rows with fold -1 lie in blocks of the (a) folds")
        test = ~is_bg & (fold == k)
        pool = ~is_bg & (fold >= 0) & (fold != k)
    elif run == "b":
        gb = splits["group_b"]
        for g, key in (("train", "train_blocks"), ("test", "test_blocks")):
            stray = set(block[group == g]) - {str(b) for b in gb[key]}
            if stray:
                raise ValueError(f"meta rows with group_b {g!r} lie in blocks not in splits.group_b.{key}: "
                                 f"{sorted(stray)[:5]}")
        test = ~is_bg & (group == "test")
        pool = ~is_bg & (group == "train")
    elif run == "s2":
        test = np.zeros(n, dtype=bool)
        pool = np.ones(n, dtype=bool)
    else:
        raise ValueError(f"unknown run {run!r} (expected a_f<k>, b or s2)")

    test &= keep_m
    pool &= keep_m
    val = pool & inner
    train = pool & ~inner
    if (test & inner).any():
        raise ValueError(f"inner-val blocks of {run} contain test rows")
    shared = set(block[test]) & set(block[pool])
    if shared:
        raise ValueError(f"blocks in both the test set and the training pool of {run}: {sorted(shared)[:5]}")
    sets = [("train", train), ("inner-val", val)] + ([("test", test)] if run != "s2" else [])
    for name, mask in sets:
        if len(np.unique(label[mask])) < 2:
            raise ValueError(f"{run}: the {name} set ({int(mask.sum())} rows) does not contain both classes")
    return UnitRows(np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test))


# ---------------------------------------------------------------------------------------------------------------
# Schedule and early stopping
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LRSchedule:
    """Per-step learning rate (D20): linear warm-up, then cosine to exactly 0 at the last step.

    Step t (0-based) uses lr(t) with tau = t + 1 optimiser steps counted:
    tau <= W: base * tau / W;   tau > W: base * (1 + cos(pi * (tau - W) / (T - W))) / 2,
    with W = warmup_epochs x steps/epoch and T = max_epochs x steps/epoch. So the first step uses base / W, the last
    warm-up step uses base, and step T - 1 (the last of epoch max_epochs) uses 0.
    """

    base_lr: float
    warmup_steps: int
    total_steps: int

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_steps <= self.total_steps or self.total_steps < 1:
            raise ValueError(f"need 0 <= warmup_steps <= total_steps and total_steps >= 1, got {self}")

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], steps_per_epoch: int) -> LRSchedule:
        tr = cfg["train"]
        return cls(float(tr["lr"]), int(tr["warmup_epochs"]) * steps_per_epoch,
                   int(tr["max_epochs"]) * steps_per_epoch)

    def __call__(self, step: int) -> float:
        if not 0 <= step < self.total_steps:
            raise ValueError(f"step {step} outside 0..{self.total_steps - 1}")
        tau = step + 1
        if tau <= self.warmup_steps:
            return self.base_lr * tau / self.warmup_steps
        progress_ = (tau - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress_))


class EarlyStopping:
    """Maximise a metric. A strictly higher value is an improvement, so ties keep the earliest epoch. Stop once
    `patience` epochs have passed without improvement. Non-finite values never improve."""

    def __init__(self, patience: int):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = int(patience)
        self.best_value = -math.inf
        self.best_epoch = 0
        self.last_epoch = 0

    def update(self, epoch: int, value: float) -> bool:
        """Record `value` for `epoch` (1-based, increasing); True if it is the new best."""
        if epoch <= self.last_epoch:
            raise ValueError("epochs must increase")
        self.last_epoch = epoch
        if math.isfinite(value) and value > self.best_value:
            self.best_value, self.best_epoch = float(value), epoch
            return True
        return False

    @property
    def should_stop(self) -> bool:
        return self.last_epoch - self.best_epoch >= self.patience


# ---------------------------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------------------------

def ensure_cublas_workspace() -> None:
    """Make sure `CUBLAS_WORKSPACE_CONFIG` has a deterministic value; set it if CUDA has not started yet."""
    val = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if val in _DETERMINISTIC_CUBLAS:
        return
    if torch.cuda.is_initialized():
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG is {val!r} but CUDA has already started; the entry point must "
                           f"set it to {CUBLAS_WORKSPACE_CONFIG!r} before CUDA is first used")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


@contextmanager
def deterministic_mode() -> Iterator[None]:
    """cuDNN deterministic, no benchmarking, `use_deterministic_algorithms(True)`; the previous state is restored."""
    ensure_cublas_workspace()
    prev = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev[2], prev[3]


def seed_everything(seed: int) -> None:
    """Seed python's `random` (unused by the pipeline, D02), numpy's legacy global generator and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------

def _open_patches(data: np.ndarray | Path | str | None, paths: Paths) -> np.ndarray:
    if data is None:
        data = paths.patches_npy
    if isinstance(data, (str, Path)):
        return np.load(Path(data), mmap_mode="r")
    if isinstance(data, np.ndarray):
        return data
    raise TypeError(f"data must be an array or a .npy path, got {type(data).__name__}")


def _load_meta(meta: pd.DataFrame | Path | str | None, paths: Paths) -> pd.DataFrame:
    if meta is None:
        meta = paths.meta_csv
    if isinstance(meta, (str, Path)):
        return pd.read_csv(Path(meta))
    return meta.reset_index(drop=True)


def _load_splits(splits: dict[str, Any] | Path | str | None, paths: Paths) -> dict[str, Any]:
    if splits is None:
        splits = paths.splits_json
    if isinstance(splits, (str, Path)):
        with Path(splits).open(encoding="utf-8") as f:
            return json.load(f)
    return splits


def _check_patch_array(arr: np.ndarray, meta: pd.DataFrame, cfg: dict[str, Any]) -> None:
    px, c = int(cfg["patch"]["px"]), len(cfg["channels"])
    if arr.ndim != 4 or arr.shape[1:] != (c, px, px):
        raise ValueError(f"patch array shape {arr.shape}, expected (N, {c}, {px}, {px})")
    if arr.shape[0] != len(meta):
        raise ValueError(f"patch array has {arr.shape[0]} rows but meta has {len(meta)}")


def rows_to_device(arr: np.ndarray, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy `arr[rows]` to `device` as float16, chunk by chunk (memmap friendly). Non-finite values raise."""
    out = torch.empty((len(rows), *arr.shape[1:]), dtype=torch.float16, device=device)
    for i in range(0, len(rows), _CHUNK):
        r = rows[i:i + _CHUNK]
        out[i:i + len(r)] = torch.from_numpy(np.ascontiguousarray(arr[r], dtype=np.float16)).to(device)
        if not torch.isfinite(out[i:i + len(r)]).all():
            raise ValueError(f"non-finite values (or float16 overflow) in patch rows {r[0]}..{r[-1]}")
    return out


def norm_stats(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std (ddof 0) over every pixel of x (N, C, H, W), in float64 with two passes."""
    n, c, h, w = x.shape
    count = n * h * w
    s = torch.zeros(c, dtype=torch.float64, device=x.device)
    for i in range(0, n, _CHUNK):
        s += x[i:i + _CHUNK].double().sum(dim=(0, 2, 3))
    mean = s / count
    ss = torch.zeros(c, dtype=torch.float64, device=x.device)
    for i in range(0, n, _CHUNK):
        ss += (x[i:i + _CHUNK].double() - mean.reshape(1, c, 1, 1)).square().sum(dim=(0, 2, 3))
    return mean.cpu().numpy(), torch.sqrt(ss / count).cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------

def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC (ties count one half); NaN when only one class is present."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float64)))


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_state() -> tuple[str | None, bool | None]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return commit, (None if status is None else bool(status))


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    _atomic_write_text(path, df.to_csv(index=False, lineterminator="\n"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_device(device: str | torch.device | None) -> torch.device:
    dev = torch.device(device) if device is not None else torch.device("cuda")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training needs CUDA, but torch.cuda.is_available() is False")
    return dev


def _check_train_cfg(cfg: dict[str, Any]) -> None:
    """The descriptive config entries must name what this module implements."""
    tr = cfg["train"]
    expected = {"loss": "BCEWithLogits", "optimizer": "AdamW", "amp": "bf16 autocast",
                "schedule": "linear warmup then cosine to 0 over max_epochs"}
    for key, val in expected.items():
        if tr[key] != val:
            raise ValueError(f"train.{key} is {tr[key]!r}; this code implements {val!r}")
    if tr["deterministic"] is not True:
        raise ValueError("train.deterministic must be true")
    if tr["early_stopping"]["metric"] != "inner-val ROC AUC":
        raise ValueError("train.early_stopping.metric: only 'inner-val ROC AUC' is implemented")


def _evaluate_val(model: nn.Module, x_val: torch.Tensor, y_val: np.ndarray, norm: Normaliser, batch_size: int,
                  show_progress: bool) -> tuple[float, float]:
    """Inner-val loss (mean BCE over the 8 views) and ROC AUC of the 8-view TTA probabilities."""
    logits = tta_logits(model, x_val, batch_size=batch_size, norm=norm,
                        progress_desc="val" if show_progress else None)
    y = torch.from_numpy(np.asarray(y_val, dtype=np.float64)).expand(logits.shape[0], -1)
    loss = float(F.binary_cross_entropy_with_logits(logits.double(), y))
    return loss, _auc(y_val, probs_from_logits(logits).numpy())


# ---------------------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------------------

def train_unit(run: str, seed: int, cfg: dict[str, Any], paths: Paths, *,
               data: np.ndarray | Path | str | None = None,
               meta: pd.DataFrame | Path | str | None = None,
               splits: dict[str, Any] | Path | str | None = None,
               weights_path: Path | str | None = None,
               keep: np.ndarray | None = None,
               should_stop: Callable[[], bool] | None = None,
               epoch_limit: int | None = None,
               device: str | torch.device | None = None,
               overwrite: bool = False,
               show_progress: bool = True) -> dict[str, Any]:
    """Train run `run` (a_f0..a_f4, b, s2) with seed `seed` and write `paths.run_dir(run, seed)`.

    Data sources (`data`, `meta`, `splits`) default to the real dataset files; pass arrays/frames/dicts or paths to
    use others. `weights_path` defaults to torchvision's cached ImageNet file under `paths.torch_home`. `keep`
    restricts the rows (smoke subset). `should_stop()` is asked before the first epoch and after every epoch that
    does not end the unit; when it returns True the unit stops, writes `interrupted.json` (no `done.json`) and
    returns with status "interrupted". `epoch_limit` caps the epochs of this call without changing the LR schedule;
    in smoke mode (`paths.smoke`) it defaults to `cfg["smoke"]["max_epochs"]`.

    Returns the `done.json` content plus `status` ("done" or "interrupted"). An exception writes `failed.json` and
    is re-raised. A unit that already has `done.json` raises FileExistsError unless `overwrite`.
    """
    run_dir = paths.run_dir(run, seed)
    if (run_dir / "done.json").exists() and not overwrite:
        raise FileExistsError(f"{run_dir / 'done.json'} exists; pass overwrite=True to retrain")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (run_dir / name).unlink(missing_ok=True)
    t0 = time.perf_counter()
    try:
        return _train(run, seed, cfg, paths, run_dir, t0, data=data, meta=meta, splits=splits,
                      weights_path=weights_path, keep=keep, should_stop=should_stop, epoch_limit=epoch_limit,
                      device=device, show_progress=show_progress)
    except Exception as e:
        commit, dirty = _git_state()
        _write_json(run_dir / "failed.json", {
            "unit": f"{run}_s{seed}", "run": run, "seed": seed, "config_hash": config_hash(cfg),
            "git_commit": commit, "git_dirty": dirty, "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(), "seconds": round(time.perf_counter() - t0, 2), "time": _now(),
        })
        log.error("%s_s%d failed: %s: %s", run, seed, type(e).__name__, e)
        raise
    finally:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def _train(run: str, seed: int, cfg: dict[str, Any], paths: Paths, run_dir: Path, t0: float, *,
           data, meta, splits, weights_path, keep, should_stop, epoch_limit, device,
           show_progress: bool) -> dict[str, Any]:
    unit = f"{run}_s{seed}"
    chash = config_hash(cfg)
    _check_train_cfg(cfg)
    tr = cfg["train"]
    batch = int(tr["batch_size"])
    max_epochs = int(tr["max_epochs"])
    if epoch_limit is None and paths.smoke:
        epoch_limit = int(cfg["smoke"]["max_epochs"])
    limit = max_epochs if epoch_limit is None else min(int(epoch_limit), max_epochs)
    if limit < 1:
        raise ValueError("epoch_limit must be >= 1")

    dev = _resolve_device(device)
    meta_df = _load_meta(meta, paths)
    splits_d = _load_splits(splits, paths)
    rows = select_unit_rows(run, meta_df, splits_d, keep=keep)
    arr = _open_patches(data, paths)
    _check_patch_array(arr, meta_df, cfg)
    steps_per_epoch = len(rows.train) // batch  # drop_last (D20)
    if steps_per_epoch < 1:
        raise ValueError(f"{unit}: {len(rows.train)} training rows is less than one batch of {batch}")
    schedule = LRSchedule.from_cfg(cfg, steps_per_epoch)
    weights = Path(weights_path) if weights_path is not None else imagenet_weights_path(cfg, paths)
    labels = meta_df["label"].to_numpy().astype(np.float32)
    ids = meta_df["id"].to_numpy()
    log.info("%s: %d train, %d inner-val, %d test rows; %d steps/epoch; device %s", unit, len(rows.train),
             len(rows.val), len(rows.test), steps_per_epoch, dev)

    with deterministic_mode():  # sets CUBLAS_WORKSPACE_CONFIG if CUDA has not started yet, so it comes first
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        seed_everything(seed)
        rng = np.random.default_rng(seed)

        x_train = rows_to_device(arr, rows.train, dev)
        x_val = rows_to_device(arr, rows.val, dev)
        y_train = torch.from_numpy(labels[rows.train]).to(dev)
        y_train_np = labels[rows.train]
        y_val = labels[rows.val]

        mean, std = norm_stats(x_train)
        norm = Normaliser(mean, std)
        _write_json(run_dir / "norm.json", {
            "channels": list(cfg["channels"]), **norm.to_dict(), "dtype": "float64", "ddof": 0,
            "rows": "training rows of the unit (inner-val excluded)", "n_patches": len(rows.train),
            "unit": unit, "config_hash": chash,
        })

        model = build_model(cfg, weights).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=schedule(0), weight_decay=float(tr["weight_decay"]))
        stopper = EarlyStopping(int(tr["early_stopping"]["patience"]))
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, float]] = []
        step = 0
        stop_reason: str | None = None

        def interrupted(epochs_done: int) -> dict[str, Any]:
            info = {"unit": unit, "run": run, "seed": seed, "config_hash": chash, "epochs_completed": epochs_done,
                    "reason": "should_stop callback", "seconds": round(time.perf_counter() - t0, 2),
                    "time": _now()}
            _write_json(run_dir / "interrupted.json", info)
            log.warning("%s interrupted after %d epoch(s)", unit, epochs_done)
            return {"status": "interrupted", **info}

        if should_stop is not None and should_stop():
            return interrupted(0)

        epoch_bar = progress(total=limit, desc=unit, unit="epoch", disable=not show_progress)
        for epoch in range(1, limit + 1):
            te = time.perf_counter()
            model.train()
            perm = rng.permutation(len(rows.train))
            ks = rng.integers(0, d4.N_TRANSFORMS, size=len(rows.train))
            perm_t = torch.from_numpy(perm).to(dev)
            ks_t = torch.from_numpy(ks).to(dev)
            n_used = steps_per_epoch * batch
            batch_losses = torch.empty(steps_per_epoch, dtype=torch.float32, device=dev)
            train_logits = torch.empty(n_used, dtype=torch.float32, device=dev)
            lr = math.nan
            bar = progress(total=steps_per_epoch, desc=f"epoch {epoch}", unit="batch", leave=False,
                           disable=not show_progress)
            for s in range(steps_per_epoch):
                lr = schedule(step)
                for g in opt.param_groups:
                    g["lr"] = lr
                sl = slice(s * batch, (s + 1) * batch)
                idx = perm_t[sl]
                xb = norm.apply(d4.t_apply_per_sample(x_train[idx], ks_t[sl]))
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    logits = model(xb)[:, 0]
                loss = F.binary_cross_entropy_with_logits(logits.float(), y_train[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                batch_losses[s] = loss.detach()
                train_logits[sl] = logits.detach().float()
                step += 1
                bar.update(1)
            bar.close()
            train_loss = float(batch_losses.double().mean())
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"{unit}: non-finite training loss in epoch {epoch}")
            train_auc = _auc(y_train_np[perm[:n_used]], train_logits.cpu().numpy())
            val_loss, val_auc = _evaluate_val(model, x_val, y_val, norm, batch, show_progress)
            if stopper.update(epoch, val_auc):
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_auc": train_auc,
                            "val_loss": val_loss, "val_auc": val_auc,
                            "seconds": round(time.perf_counter() - te, 3)})
            _write_csv(run_dir / "history.csv", pd.DataFrame(history, columns=list(HISTORY_COLUMNS)))
            log.info("%s epoch %d/%d lr %.3g train_loss %.4f train_auc %.4f val_loss %.4f val_auc %.4f %.1fs%s",
                     unit, epoch, max_epochs, lr, train_loss, train_auc, val_loss, val_auc,
                     history[-1]["seconds"], " (best)" if stopper.best_epoch == epoch else "")
            epoch_bar.update(1)
            if stopper.should_stop:
                stop_reason = "early_stopping"
            elif epoch == max_epochs:
                stop_reason = "max_epochs"
            elif epoch == limit:
                stop_reason = "epoch_limit"
            if stop_reason:
                break
            if should_stop is not None and should_stop():
                epoch_bar.close()
                return interrupted(epoch)
        epoch_bar.close()

        if best_state is None:
            raise RuntimeError(f"{unit}: no epoch gave a finite inner-val AUC")
        if tr["early_stopping"]["restore_best"]:
            model.load_state_dict(best_state)
            saved_epoch, saved_auc = stopper.best_epoch, stopper.best_value
        else:
            saved_epoch, saved_auc = history[-1]["epoch"], history[-1]["val_auc"]
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "unit": unit, "run": run, "seed": seed, "epoch": saved_epoch,
                    "val_auc": saved_auc, "config_hash": chash,
                    "arch": cfg["model"]["arch"], "in_channels": int(cfg["model"]["in_channels"])},
                   run_dir / "best.pt")

        p_val = tta_predict(model, x_val, batch_size=batch, norm=norm).numpy()
        _write_csv(run_dir / "preds_val.csv", pd.DataFrame({"id": ids[rows.val], "prob": p_val}))
        val_auc_check = _auc(y_val, p_val)
        if tr["early_stopping"]["restore_best"] and val_auc_check != stopper.best_value:
            log.warning("%s: inner-val AUC of the restored weights %.6f != best epoch's %.6f", unit,
                        val_auc_check, stopper.best_value)

        del x_train, y_train, opt, x_val
        if len(rows.test):
            x_test = rows_to_device(arr, rows.test, dev)
            p_test = tta_predict(model, x_test, batch_size=batch, norm=norm,
                                 progress_desc="test" if show_progress else None).numpy()
            del x_test
        else:
            p_test = np.empty(0, dtype=np.float64)
        _write_csv(run_dir / "preds_test.csv", pd.DataFrame({"id": ids[rows.test], "prob": p_test}))

    commit, dirty = _git_state()
    peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else None
    peak_reserved = torch.cuda.max_memory_reserved(dev) / 2**30 if dev.type == "cuda" else None
    done = {
        "unit": unit, "run": run, "seed": seed, "config_hash": chash, "git_commit": commit, "git_dirty": dirty,
        "best_epoch": stopper.best_epoch, "best_val_auc": stopper.best_value,
        "n_train": len(rows.train), "n_val": len(rows.val), "n_test": len(rows.test),
        "seconds": round(time.perf_counter() - t0, 2),
        "peak_gpu_mem_gb": None if peak is None else round(peak, 3),
        "peak_gpu_reserved_gb": None if peak_reserved is None else round(peak_reserved, 3),
        "epochs_run": len(history), "stop_reason": stop_reason, "saved_epoch": saved_epoch,
        "val_auc_recomputed": val_auc_check, "steps_per_epoch": steps_per_epoch, "batch_size": batch,
        "smoke": paths.smoke, "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev),
        "torch": torch.__version__, "finished": _now(),
    }
    _write_json(run_dir / "done.json", done)
    log.info("%s done: best epoch %d, inner-val AUC %.4f, %.1f s, peak GPU %s GiB", unit, stopper.best_epoch,
             stopper.best_value, done["seconds"], done["peak_gpu_mem_gb"])
    return {"status": "done", **done}
