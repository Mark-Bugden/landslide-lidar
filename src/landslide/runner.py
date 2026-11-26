"""The overnight runner (task P1-14): training -> evaluation -> stage 2 (only if (a) and (b) pass) -> verdict.

BRIEF.md Phase 1 item 6 and Appendix A §11; PLAN.md P1-14 and D24. Every interpretation is recorded in
`docs/decisions/P1-14.md`. The entry point is `scripts/run_overnight.py`; this module never imports torch at import
time, so `main()` can set `CUBLAS_WORKSPACE_CONFIG` and `TORCH_HOME` before torch loads.

**Order.** Preflight (`landslide.preflight`; any problem exits with code 2 within 60 s) -> sockets disabled and
Windows kept awake -> a_f0 ... a_f4 x seeds, then b x seeds -> evaluation -> if (a) and (b) both PASS
(`evaluate.stage2_allowed`): s2 x seeds, stage-2 CNN inference sheet by sheet, the CNN's stage-2 evaluation
(next to the phase-1 baselines), and the final evaluation with the verdict. Otherwise the first evaluation is final.

**Smoke mode** (`--smoke`, Appendix B `smoke`, D24): the units a_f0, b and s2 with seed 0, `smoke.max_epochs`
epochs, on a seeded subset of 5 % of the pair_ids plus 5 % of the background; then stage 2 on `smoke.stage2_sheets`
sheet(s) whatever (a) and (b) give, with the stage-2 baselines recomputed on that sheet. Everything is written under
`data/smoke/` (`Paths(smoke=True)`).

**Resume.** A unit with `done.json` is skipped and never overwritten. A unit left with `failed.json` or
`interrupted.json` (or with no status file, after Ctrl-C) is trained again from scratch, once per invocation; its
previous status is logged and kept in the run summary. Stage-2 inference skips finished sheets (`stage2.run_cnn`).

**Robustness.** An exception in a unit, a stage-2 sheet or an evaluation is logged with its traceback and the runner
moves on. Units that are not done at the end are listed (`missing_units`) and never pass.

**Wall-clock cap.** `runner.wall_clock_h` (or `--max-hours`) from the start of the runner, with 15 min kept for the
evaluation: after `start + cap - 15 min` no unit or stage-2 sheet starts, and a training unit stops at the next
epoch boundary (`train_unit(should_stop=...)`, which writes `interrupted.json`). The evaluation then runs on what has
finished, so the verdict is INCOMPLETE (never REPLICATED).

**Logging.** `<logs>/overnight_<timestamp>.log` (progress bars go to the console only) and the run summary
`<results>/run_summary.json` (tracked in full mode), rewritten after every step.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from landslide import preflight as PF
from landslide.config import config_hash, load_config, rng_for
from landslide.logs import LOGGER_NAME, get_logger, setup_logging
from landslide.paths import REPO_ROOT, Paths
from landslide.winawake import keep_awake

log = get_logger("runner")

SCHEMA_VERSION = 1
#: `landslide.train.CUBLAS_WORKSPACE_CONFIG` (tested equal); repeated here because importing train loads torch.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_DETERMINISTIC_CUBLAS = (":4096:8", ":16:8")
#: Appendix A §11 "15 min reserved for evaluation". No config key exists (requested for P1-16's batch).
EVAL_RESERVE_MIN = 15.0
SUMMARY_NAME = "run_summary.json"
MARKERS = ("done.json", "failed.json", "interrupted.json")

EXIT_OK, EXIT_ERROR, EXIT_PREFLIGHT, EXIT_ABORTED = 0, 1, 2, 130

# Unit statuses in run_summary.json
DONE = "done"                # trained to the end in this invocation
SKIPPED = "skipped"          # had done.json already (finished earlier)
FAILED = "failed"            # raised in this invocation (failed.json written by train_unit)
INTERRUPTED = "interrupted"  # stopped at an epoch boundary by the wall-clock cap (interrupted.json)
NOT_STARTED = "not_started"  # never started: the wall-clock cap was reached before it
NOT_RUN = "not_run"          # not part of this run: stage 2 is not allowed because (a) or (b) did not pass
ABORTED = "aborted"          # the run was stopped by the user (Ctrl-C) during this unit
FINISHED_UNIT = (DONE, SKIPPED)


# ======================================================================================================================
# Environment, time, small helpers
# ======================================================================================================================


def prepare_environment(paths: Paths) -> None:
    """Set `CUBLAS_WORKSPACE_CONFIG` (deterministic cuBLAS, P1-06) and `TORCH_HOME` (the cached ImageNet weights)
    before torch is imported. Refuses if CUDA has already started with a non-deterministic cuBLAS setting."""
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in _DETERMINISTIC_CUBLAS:
            raise RuntimeError("CUDA started before the runner could set CUBLAS_WORKSPACE_CONFIG; run the runner "
                               "as `uv run python scripts/run_overnight.py`")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    os.environ["TORCH_HOME"] = str(paths.torch_home)


@dataclass
class Deadline:
    """The wall-clock cap: work may start (and an epoch may follow another) until `start + cap - reserve`."""

    start: float
    cap_s: float
    reserve_s: float
    clock: Callable[[], float] = time.monotonic
    reached: bool = False

    @property
    def work_until(self) -> float:
        return self.start + max(0.0, self.cap_s - self.reserve_s)

    def over(self) -> bool:
        if self.clock() >= self.work_until:
            if not self.reached:
                log.warning("wall-clock cap: %.2f h minus %.0f min for the evaluation reached; no further work "
                            "starts", self.cap_s / 3600, self.reserve_s / 60)
            self.reached = True
        return self.reached

    def remaining_s(self) -> float:
        return self.work_until - self.clock()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _json_default(o: Any) -> Any:
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        x = float(o)
        return x if math.isfinite(x) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"cannot serialise {type(o).__name__}")


def _clean(o: Any) -> Any:
    """Non-finite floats -> None, recursively (JSON has no NaN)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(obj), indent=2, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


# ======================================================================================================================
# Plan and smoke subset (D24)
# ======================================================================================================================


@dataclass(frozen=True)
class Plan:
    a: tuple[str, ...]
    b: tuple[str, ...]
    s2: tuple[str, ...]
    seeds: tuple[int, ...]

    def units(self, runs: Sequence[str]) -> list[tuple[str, int]]:
        """Run-major order: a_f0_s0, a_f0_s1, ..., a_f1_s0, ..."""
        return [(r, s) for r in runs for s in self.seeds]


def make_plan(cfg: Mapping[str, Any], smoke: bool) -> Plan:
    """The units of the run, as the evaluation expects them (`evaluate.expected_runs` / `expected_seeds`)."""
    if smoke:
        folds = [int(k) for k in cfg["smoke"]["folds"]]
        runs = [str(r) for r in cfg["smoke"]["runs"]]
        seeds = [int(s) for s in cfg["smoke"]["seeds"]]
    else:
        folds = list(range(int(cfg["splits"]["n_folds_a"])))
        runs = [f"a_f{k}" for k in folds] + ["b", "s2"]
        seeds = [int(s) for s in cfg["train"]["seeds"]]
    a_runs = [f"a_f{k}" for k in folds]
    unknown = [r for r in runs if r not in a_runs and r not in ("b", "s2")]
    if unknown:
        raise ValueError(f"unexpected runs {unknown} (folds {folds})")
    return Plan(a=tuple(r for r in a_runs if r in runs), b=tuple(r for r in runs if r == "b"),
                s2=tuple(r for r in runs if r == "s2"), seeds=tuple(seeds))


def smoke_subset(cfg: Mapping[str, Any], meta: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """D24: a seeded subset of `smoke.patch_frac` of the pair_ids (both members of each pair) plus the same share
    of the background rows. Returns a boolean mask over the rows of `meta` and a description.

    One generator, `rng_for(cfg, replication.seed, "smoke_subset")`, draws first the pairs (from the sorted unique
    pair_ids >= 0) and then the background rows (from their sorted ids), both without replacement; the counts are
    round-half-up of frac x n.
    """
    frac = float(cfg["smoke"]["patch_frac"])
    if not 0 < frac <= 1:
        raise ValueError(f"smoke.patch_frac {frac} must be in (0, 1]")
    seed = int(cfg["replication"]["seed"])
    rng = rng_for(dict(cfg), seed, "smoke_subset")
    pid = meta["pair_id"].to_numpy(dtype=np.int64)
    ids = meta["id"].to_numpy(dtype=np.int64)
    is_bg = meta["source"].astype(str).to_numpy() == "background"
    pairs = np.unique(pid[(pid >= 0) & ~is_bg])
    k = min(pairs.size, _round_half_up(frac * pairs.size))
    chosen_pairs = np.sort(rng.choice(pairs, size=k, replace=False)) if k else np.empty(0, np.int64)
    bg = np.sort(ids[is_bg])
    kb = min(bg.size, _round_half_up(frac * bg.size))
    chosen_bg = np.sort(rng.choice(bg, size=kb, replace=False)) if kb else np.empty(0, np.int64)
    keep = (np.isin(pid, chosen_pairs) & ~is_bg) | (is_bg & np.isin(ids, chosen_bg))
    kept_ids = np.sort(ids[keep])
    info = {
        "rule": "D24: smoke.patch_frac of the pair_ids (both members) plus the same share of the background rows, "
                "drawn without replacement with rng_for(replication.seed, 'smoke_subset'): pairs first, then "
                "background; counts round half up",
        "patch_frac": frac, "seed": seed, "stream": "smoke_subset",
        "n_pairs": int(pairs.size), "n_pairs_chosen": int(k),
        "n_background": int(bg.size), "n_background_chosen": int(kb),
        "n_rows": int(keep.sum()), "n_rows_total": int(len(meta)),
        "n_rows_outside_rule": int(((pid < 0) & ~is_bg).sum()),
        "ids_sha256": hashlib.sha256(",".join(map(str, kept_ids.tolist())).encode("ascii")).hexdigest(),
    }
    return keep, info


def smoke_stage2_sheets(cfg: Mapping[str, Any]) -> list[str]:
    """The first `smoke.stage2_sheets` of the 20 reserved sheets in sorted order (the rule of
    `scripts/stage2_baselines.py --smoke`)."""
    return PF.stage2_sheet_list(cfg)[:int(cfg["smoke"]["stage2_sheets"])]


# ======================================================================================================================
# Backend: the work the runner orchestrates (tests substitute a fake)
# ======================================================================================================================


class Backend:
    """Calls into training (P1-06), evaluation (P1-13) and stage 2 (P1-12). `device` and `weights_path` override
    the defaults (CUDA; the cached ImageNet file) for tests on synthetic data."""

    def __init__(self, device: Any = None, weights_path: Path | str | None = None):
        self.device = device
        self.weights_path = weights_path

    # ---- training and evaluation ----
    def train(self, run: str, seed: int, cfg: Mapping[str, Any], paths: Paths, *, keep: np.ndarray | None,
              should_stop: Callable[[], bool], show_progress: bool) -> dict[str, Any]:
        from landslide.train import train_unit

        return train_unit(run, seed, dict(cfg), paths, keep=keep, should_stop=should_stop, device=self.device,
                          weights_path=self.weights_path, show_progress=show_progress)

    def evaluate(self, cfg: Mapping[str, Any], paths: Paths) -> dict[str, Any]:
        from landslide.evaluate import run_evaluation

        return run_evaluation(cfg, paths)

    def stage2_allowed(self, metrics: Mapping[str, Any]) -> bool:
        from landslide.evaluate import stage2_allowed

        return stage2_allowed(metrics)

    # ---- stage 2 ----
    def stage2_inputs(self, cfg: Mapping[str, Any], paths: Paths, sheets: Sequence[str] | None) -> Any:
        from landslide import stage2 as S2

        return S2.load_inputs(cfg, paths, sheets)

    def stage2_sheet_ids(self, inputs: Any) -> list[str]:
        return list(inputs.sheet_ids)

    def stage2_baselines(self, cfg: Mapping[str, Any], paths: Paths, inputs: Any) -> dict[str, Any]:
        from landslide import stage2 as S2

        res = S2.run_baselines(cfg, paths, list(S2.BASELINE_KINDS), inputs, write_results=False)
        return {k: {"status": r.pooled.status, "p_at_r50": r.pooled.precision_at_recall,
                    "n_detections": int(len(r.detections))} for k, r in res.items()}

    def stage2_models(self, cfg: Mapping[str, Any], paths: Paths, seeds: Sequence[int]) -> Any:
        from landslide import stage2 as S2

        return S2.load_s2_models(cfg, paths, device=self.device, seeds=list(seeds))

    def stage2_cnn_sheet(self, cfg: Mapping[str, Any], paths: Paths, inputs: Any, sheet: str, models: Any,
                         show_progress: bool) -> dict[str, Any]:
        from landslide import stage2 as S2

        one = dataclasses.replace(inputs, sheets=inputs.sheets.loc[inputs.sheets["sheet"] == sheet]
                                  .reset_index(drop=True))
        return dict(S2.run_cnn(cfg, paths, one, models=models, show_progress=show_progress)["sheets"][sheet])

    def stage2_evaluate_cnn(self, cfg: Mapping[str, Any], paths: Paths, inputs: Any) -> dict[str, Any]:
        from landslide import stage2 as S2

        r = S2.evaluate_scorers(cfg, paths, [S2.CNN], inputs, write_results=False)[S2.CNN]
        return {"status": r.pooled.status, "p_at_r50": r.pooled.precision_at_recall,
                "n_detections": int(len(r.detections)), "n_windows": r.n_windows, "n_sheets": r.n_sheets}

    # ---- GPU memory (measurements only) ----
    def gpu_peak_reset(self) -> None:
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.reset_peak_memory_stats()

    def gpu_peak_gib(self) -> float | None:
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                return round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        return None

    def release(self, models: Any) -> None:
        del models
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    # ---- sizes for the runtime estimate (smoke only) ----
    def full_unit_sizes(self, cfg: Mapping[str, Any], paths: Paths) -> dict[str, Any]:
        from landslide.train import select_unit_rows

        meta = pd.read_csv(paths.meta_csv)
        splits = json.loads(paths.splits_json.read_text(encoding="utf-8"))
        plan = make_plan(cfg, smoke=False)
        out = {}
        for run in plan.a + plan.b + plan.s2:
            r = select_unit_rows(run, meta, splits)
            out[run] = {"n_train": int(r.train.size), "n_val": int(r.val.size), "n_test": int(r.test.size)}
        return out


# ======================================================================================================================
# The run
# ======================================================================================================================


@dataclass
class RunOptions:
    smoke: bool = False
    max_hours: float | None = None
    show_progress: bool = True


def unit_measurements(run_dir: Path) -> dict[str, Any] | None:
    """Seconds per epoch and memory of a finished unit, from its `history.csv` and `done.json` (for P1-17)."""
    done = _read_json(run_dir / "done.json")
    hp = run_dir / "history.csv"
    if done is None or not hp.is_file():
        return None
    try:
        sec = pd.read_csv(hp)["seconds"].to_numpy(dtype=np.float64)
    except (OSError, ValueError, KeyError):
        return None
    out = {"epochs": int(sec.size),
           "seconds_per_epoch_mean": float(sec.mean()) if sec.size else None,
           "seconds_first_epoch": float(sec[0]) if sec.size else None,
           "seconds_per_epoch_after_first": float(sec[1:].mean()) if sec.size > 1 else None}
    for k in ("seconds", "peak_gpu_mem_gb", "peak_gpu_reserved_gb", "n_train", "n_val", "n_test", "steps_per_epoch",
              "batch_size", "stop_reason", "best_epoch"):
        out[k] = done.get(k)
    return out


class Runner:
    """One invocation of the overnight run after a passed preflight."""

    def __init__(self, cfg: Mapping[str, Any], paths: Paths, opts: RunOptions, backend: Backend, *,
                 deadline: Deadline, preflight: PF.PreflightReport | None = None, log_file: Path | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.paths = paths
        self.opts = opts
        self.backend = backend
        self.deadline = deadline
        self.clock = clock
        self.t0 = deadline.start
        self.plan = make_plan(cfg, opts.smoke)
        self.chash = config_hash(dict(cfg))
        self.units: dict[str, dict[str, Any]] = {}
        self.keep: np.ndarray | None = None
        cap_src = "--max-hours" if opts.max_hours is not None else "runner.wall_clock_h"
        pf = preflight.to_dict() if preflight is not None else None
        info = preflight.info if preflight is not None else {}
        commit = _git(paths.root, "rev-parse", "HEAD")
        dirty = _git(paths.root, "status", "--porcelain", "--untracked-files=no")
        # A resumed run keeps a short record of the invocations before it (the summary itself is rewritten).
        prev = _read_json(self.summary_path)
        earlier: list[dict[str, Any]] = []
        if prev and prev.get("schema_version") == SCHEMA_VERSION and prev.get("smoke") == opts.smoke:
            earlier = list(prev.get("previous_invocations") or []) + [
                {k: prev.get(k) for k in ("started_utc", "finished_utc", "status", "verdict", "git_commit")}]
        self.summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "smoke": opts.smoke,
            "status": "running",
            "started_utc": _utc(),
            "finished_utc": None,
            "wall_seconds": None,
            "config_hash": self.chash,
            "frozen_hash": info.get("frozen_hash"),
            "freeze_commit": info.get("freeze_commit"),
            "git_commit": commit,
            "git_dirty": None if dirty is None else bool(dirty),
            "log_file": None if log_file is None else self._rel(log_file),
            "cap": {"wall_clock_h": round(deadline.cap_s / 3600, 4), "source": cap_src,
                    "eval_reserve_min": round(deadline.reserve_s / 60, 2), "reached": False},
            "preflight": pf,
            "plan": {"a": list(self.plan.a), "b": list(self.plan.b), "s2": list(self.plan.s2),
                     "seeds": list(self.plan.seeds), "order": "a folds x seeds, b x seeds, evaluation; s2 x seeds "
                                                              "and stage 2 only if (a) and (b) pass"
                                                              + ("" if not opts.smoke else
                                                                 " (smoke: stage 2 whatever (a) and (b) give)")},
            "smoke_subset": None,
            "units": self.units,
            "evaluations": [],
            "stage2": {"allowed": None, "reason": None},
            "verdict": None,
            "verdict_detail": None,
            "missing_units": None,
            "measurements": {"units": {}},
            "previous_invocations": earlier,
        }

    # ---- bookkeeping ----
    def _rel(self, p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(self.paths.root.resolve()).as_posix()
        except ValueError:
            return Path(p).name

    @property
    def summary_path(self) -> Path:
        return self.paths.results / SUMMARY_NAME

    def save(self) -> None:
        self.summary["cap"]["reached"] = self.deadline.reached
        self.summary["wall_seconds"] = round(self.clock() - self.t0, 1)
        try:
            write_json_atomic(self.summary_path, self.summary)
        except Exception:  # noqa: BLE001 - the summary must never stop the run
            log.exception("could not write %s", self.summary_path)

    # ---- the work ----
    def run(self) -> int:
        self.save()
        log.info("plan: %s; seeds %s; cap %.2f h (%s), work until %.2f h", self.summary["plan"]["order"],
                 list(self.plan.seeds), self.deadline.cap_s / 3600, self.summary["cap"]["source"],
                 (self.deadline.work_until - self.t0) / 3600)
        if self.opts.smoke:
            self._prepare_smoke()
        n_stage1 = len(self.plan.units(self.plan.a + self.plan.b))
        for i, (run, seed) in enumerate(self.plan.units(self.plan.a + self.plan.b)):
            self.unit(run, seed, i + 1, n_stage1)
        first = self.evaluate("after_stage1")

        if self.opts.smoke:
            allowed, reason = True, "smoke: stage 2 runs whatever (a) and (b) give (D24)"
        elif first is None:
            allowed, reason = False, "the evaluation after (a) and (b) failed"
        else:
            try:
                allowed = bool(self.backend.stage2_allowed(first))
                st = {t: first["tests"][t]["status"] for t in ("a", "b")}
                reason = f"(a) {st['a']}, (b) {st['b']}: " + (
                    "both pass" if allowed else "stage 2 runs only if (a) and (b) both pass (Appendix A, section 1)")
            except Exception as e:  # noqa: BLE001 - a malformed metrics dict never allows stage 2
                log.exception("could not read the (a) and (b) statuses")
                allowed, reason = False, f"the (a) and (b) statuses could not be read: {type(e).__name__}: {e}"
        self.summary["stage2"].update(allowed=allowed, reason=reason)
        log.info("stage 2 %s: %s", "allowed" if allowed else "skipped", reason)
        self.save()

        if allowed and self.plan.s2:
            units = self.plan.units(self.plan.s2)
            for i, (run, seed) in enumerate(units):
                self.unit(run, seed, i + 1, len(units))
            self.stage2()
            final = self.evaluate("final")
        else:
            for run, seed in self.plan.units(self.plan.s2):
                self.units[f"{run}_s{seed}"] = {"run": run, "seed": seed, "status": NOT_RUN, "reason": reason}
            final = first  # nothing changed since: the first evaluation is the final one
        if self.summary["evaluations"]:
            self.summary["evaluations"][-1]["final"] = True
        return self.finish(final)

    def _prepare_smoke(self) -> None:
        meta = pd.read_csv(self.paths.meta_csv, usecols=["id", "source", "pair_id"])
        self.keep, info = smoke_subset(self.cfg, meta)
        self.summary["smoke_subset"] = info
        log.info("smoke subset: %d of %d pairs and %d of %d background rows -> %d of %d patches",
                 info["n_pairs_chosen"], info["n_pairs"], info["n_background_chosen"], info["n_background"],
                 info["n_rows"], info["n_rows_total"])
        try:
            self.summary["measurements"]["full_unit_sizes"] = self.backend.full_unit_sizes(self.cfg, self.paths)
        except Exception as e:  # noqa: BLE001 - measurements never stop the run
            log.warning("full-size unit counts not recorded: %s: %s", type(e).__name__, e)
        full_s2 = _read_json(dataclasses.replace(self.paths, smoke=False).stage2 / "summary.json") or {}
        lat = full_s2.get("lattice") or {}
        self.summary["measurements"]["full_stage2"] = {"n_windows": lat.get("n_windows"),
                                                       "n_sheets": len(lat.get("per_sheet") or {}) or None,
                                                       "n_seeds": len(self.cfg["train"]["seeds"])}
        self.save()

    def unit(self, run: str, seed: int, i: int, n: int) -> None:
        name = f"{run}_s{seed}"
        d = self.paths.run_dir(run, seed)
        rec: dict[str, Any] = {"run": run, "seed": seed}
        present = [m for m in MARKERS if (d / m).exists()]
        if "done.json" in present:
            done = _read_json(d / "done.json") or {}
            rec.update(status=SKIPPED, seconds=done.get("seconds"), best_epoch=done.get("best_epoch"),
                       epochs_run=done.get("epochs_run"), stop_reason=done.get("stop_reason"),
                       peak_gpu_mem_gb=done.get("peak_gpu_mem_gb"))
            problems = []
            if done.get("config_hash") != self.chash:
                problems.append(f"done.json carries config {str(done.get('config_hash'))[:12]}, not the frozen "
                                f"{self.chash[:12]}; it is never overwritten and counts as missing")
            if len(present) > 1:
                problems.append(f"several status files {present}; the evaluation counts the unit as invalid")
            if problems:
                rec["problem"] = "; ".join(problems)
                log.error("%s: %s", name, rec["problem"])
            log.info("[%d/%d] %s: done.json exists, skipped", i, n, name)
            self._record(name, rec, d)
            return
        if present:
            prev = _read_json(d / present[0]) or {}
            kind = present[0].removesuffix(".json")
            rec["previous_attempt"] = {"status": kind, "error": prev.get("error"),
                                       "epochs_completed": prev.get("epochs_completed"), "time": prev.get("time")}
            log.warning("[%d/%d] %s: the previous attempt ended %s%s; training it again from scratch", i, n, name,
                        kind, f" ({prev.get('error')})" if prev.get("error") else "")
            if prev.get("traceback"):
                log.info("%s: traceback of the previous attempt:\n%s", name, prev["traceback"].rstrip())
        elif (d / "history.csv").exists():
            rec["previous_attempt"] = {"status": "no status file (stopped during training)"}
            log.warning("[%d/%d] %s: a previous attempt left no status file; training it again from scratch", i, n,
                        name)
        if self.deadline.over():
            rec.update(status=NOT_STARTED, reason="wall-clock cap reached before the unit started")
            log.warning("[%d/%d] %s: not started (wall-clock cap)", i, n, name)
            self._record(name, rec, d)
            return
        log.info("[%d/%d] %s: training (%.2f h left before the evaluation reserve)", i, n, name,
                 self.deadline.remaining_s() / 3600)
        t = self.clock()
        rec["status"] = ABORTED  # until the call returns; a Ctrl-C leaves it so
        self.units[name] = rec
        try:
            res = self.backend.train(run, seed, self.cfg, self.paths, keep=self.keep, should_stop=self.deadline.over,
                                     show_progress=self.opts.show_progress)
        except Exception as e:  # noqa: BLE001 - one unit never stops the run
            log.exception("[%d/%d] %s failed; moving on to the next unit", i, n, name)
            rec.update(status=FAILED, error=f"{type(e).__name__}: {e}", seconds=round(self.clock() - t, 1))
        else:
            status = res.get("status")
            rec["seconds"] = round(self.clock() - t, 1)
            if status == "done":
                rec.update(status=DONE, best_epoch=res.get("best_epoch"), best_val_auc=res.get("best_val_auc"),
                           epochs_run=res.get("epochs_run"), stop_reason=res.get("stop_reason"),
                           peak_gpu_mem_gb=res.get("peak_gpu_mem_gb"))
                log.info("[%d/%d] %s done in %.0f s", i, n, name, rec["seconds"])
            elif status == "interrupted":
                rec.update(status=INTERRUPTED, epochs_completed=res.get("epochs_completed"),
                           reason="wall-clock cap (stopped between epochs)")
                log.warning("[%d/%d] %s interrupted by the wall-clock cap after %s epoch(s)", i, n, name,
                            res.get("epochs_completed"))
            else:
                rec.update(status=FAILED, error=f"train_unit returned status {status!r}")
                log.error("[%d/%d] %s: %s", i, n, name, rec["error"])
        self._record(name, rec, d)

    def _record(self, name: str, rec: dict[str, Any], run_dir: Path) -> None:
        self.units[name] = rec
        if rec.get("status") in FINISHED_UNIT:
            m = unit_measurements(run_dir)
            if m is not None:
                self.summary["measurements"]["units"][name] = m
        self.save()

    def evaluate(self, when: str) -> dict[str, Any] | None:
        t = self.clock()
        log.info("evaluation (%s)", when)
        rec: dict[str, Any] = {"when": when}
        try:
            m = self.backend.evaluate(self.cfg, self.paths)
        except Exception as e:  # noqa: BLE001
            log.exception("the evaluation (%s) failed", when)
            rec.update(error=f"{type(e).__name__}: {e}", seconds=round(self.clock() - t, 1))
            self.summary["evaluations"].append(rec)
            self.save()
            return None
        tests = m.get("tests") or {}
        rec.update(verdict=m.get("verdict"),
                   statuses={k: (tests.get(k) or {}).get("status") for k in ("a", "b", "stage2")},
                   estimates={k: (tests.get(k) or {}).get("estimate") for k in ("a", "b", "stage2")},
                   seconds=round(self.clock() - t, 1))
        log.info("evaluation (%s): %s; (a) %s, (b) %s, stage 2 %s", when, rec["verdict"], *rec["statuses"].values())
        self.summary["evaluations"].append(rec)
        self.save()
        return m

    def stage2(self) -> None:
        s = self.summary["stage2"]
        seeds = list(self.plan.seeds)
        unfinished = [f"s2_s{x}" for x in seeds if self.units.get(f"s2_s{x}", {}).get("status") not in FINISHED_UNIT]
        if unfinished:
            s["cnn"] = {"status": "skipped", "reason": f"s2 units not finished: {unfinished}"}
            log.warning("stage-2 inference skipped: %s (stage 2 is INCOMPLETE)", s["cnn"]["reason"])
            self.save()
            return
        sheets = smoke_stage2_sheets(self.cfg) if self.opts.smoke else None
        try:
            inputs = self.backend.stage2_inputs(self.cfg, self.paths, sheets)
        except Exception as e:  # noqa: BLE001
            log.exception("stage-2 inputs could not be loaded")
            s["cnn"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            self.save()
            return
        if self.opts.smoke:
            t = self.clock()
            try:
                s["baselines"] = {"sheets": sheets, "scorers": self.backend.stage2_baselines(self.cfg, self.paths,
                                                                                             inputs),
                                  "seconds": round(self.clock() - t, 1)}
            except Exception as e:  # noqa: BLE001
                log.exception("smoke stage-2 baselines failed")
                s["baselines"] = {"sheets": sheets, "error": f"{type(e).__name__}: {e}"}
            self.save()
        try:
            models = self.backend.stage2_models(self.cfg, self.paths, seeds)
        except Exception as e:  # noqa: BLE001
            log.exception("the s2 models could not be loaded")
            s["cnn"] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            self.save()
            return
        sheet_ids = self.backend.stage2_sheet_ids(inputs)
        per: dict[str, Any] = {}
        s["cnn"] = {"status": "running", "seeds": seeds, "sheets": per}
        self.backend.gpu_peak_reset()
        t_all = self.clock()
        for i, sheet in enumerate(sheet_ids):
            if self.deadline.over():
                per[sheet] = {"status": NOT_STARTED, "reason": "wall-clock cap"}
                continue
            log.info("[%d/%d] stage-2 CNN on %s", i + 1, len(sheet_ids), sheet)
            try:
                per[sheet] = self.backend.stage2_cnn_sheet(self.cfg, self.paths, inputs, sheet, models,
                                                           self.opts.show_progress)
            except Exception as e:  # noqa: BLE001
                log.exception("stage-2 CNN on %s failed; moving on", sheet)
                per[sheet] = {"status": FAILED, "error": f"{type(e).__name__}: {e}"}
            self.save()
        self.backend.release(models)
        del models
        scored = [v for v in per.values() if v.get("status") == "done"]
        n_win = sum(int(v.get("n_windows") or 0) for v in scored)
        sec = sum(float(v.get("seconds") or 0.0) for v in scored)
        complete = bool(per) and all(v.get("status") in ("done", "skipped") for v in per.values())
        s["cnn"].update(status="done" if complete else "incomplete", seconds=round(self.clock() - t_all, 1),
                        n_windows_scored=n_win, windows_per_s=round(n_win / sec, 2) if sec > 0 else None,
                        n_seeds=len(seeds), peak_gpu_mem_gb=self.backend.gpu_peak_gib())
        self.summary["measurements"]["stage2"] = {k: s["cnn"][k] for k in ("n_windows_scored", "windows_per_s",
                                                                          "n_seeds", "peak_gpu_mem_gb")}
        self.summary["measurements"]["stage2"]["seconds_scoring"] = round(sec, 2)
        if complete:
            try:
                s["cnn_evaluation"] = self.backend.stage2_evaluate_cnn(self.cfg, self.paths, inputs)
                log.info("stage-2 CNN detections: %s", s["cnn_evaluation"])
            except Exception as e:  # noqa: BLE001
                log.exception("the stage-2 CNN evaluation failed")
                s["cnn_evaluation"] = {"error": f"{type(e).__name__}: {e}"}
        else:
            bad = {k: v.get("status") for k, v in per.items() if v.get("status") not in ("done", "skipped")}
            log.warning("stage-2 CNN scores are incomplete (%s): stage 2 is INCOMPLETE", bad)
        self.save()

    def finish(self, final: dict[str, Any] | None) -> int:
        cap = self.deadline.reached
        if final is not None:
            integ = final.get("integrity", {})
            missing = sorted(set().union(*(integ.get(k, []) for k in ("missing", "failed", "interrupted",
                                                                       "hash_mismatch", "invalid"))))
            self.summary.update(verdict=final.get("verdict"), verdict_detail=final.get("verdict_detail"),
                                missing_units=missing)
        else:
            self.summary["missing_units"] = sorted(k for k, v in self.units.items()
                                                   if v.get("status") not in FINISHED_UNIT)
        self.summary["status"] = "stopped_at_cap" if cap else "finished"
        self.summary["finished_utc"] = _utc()
        self.save()
        v = self.summary["verdict"] or "none (the evaluation failed)"
        detail = self.summary["verdict_detail"] or {}
        log.info("RUN %s: verdict %s ((a) %s, (b) %s, stage 2 %s)", "STOPPED AT THE WALL-CLOCK CAP" if cap
                 else "FINISHED", v, detail.get("a"), detail.get("b"), detail.get("stage2"))
        if self.summary["missing_units"]:
            log.warning("missing or unusable units (never a pass): %s", ", ".join(self.summary["missing_units"]))
        log.info("summary: %s", self._rel(self.summary_path))
        return EXIT_OK if final is not None else EXIT_ERROR

    def abort(self, why: str, status: str) -> None:
        self.summary["status"] = status
        self.summary["abort_reason"] = why
        self.summary["finished_utc"] = _utc()
        self.save()


# ======================================================================================================================
# Command line
# ======================================================================================================================


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_overnight.py",
        description="Overnight run: training -> evaluation -> stage 2 (only if (a) and (b) pass) -> verdict. "
                    "Re-run the same command to resume (finished units are skipped).")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test (Appendix B `smoke`): a_f0, b, s2 x seed 0, 2 epochs, 5%% of the pairs and of "
                         "the background, stage 2 on 1 sheet; writes only under data/smoke/")
    ap.add_argument("--dry-run", action="store_true", help="run the preflight checks only, then exit")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock cap in hours (default: runner.wall_clock_h in the config); 15 min of it are "
                         "kept for the evaluation")
    ap.add_argument("--no-progress", action="store_true", help="no progress bars on the console")
    ap.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)  # tests: another repository root
    args = ap.parse_args(argv)
    if args.max_hours is not None and not (args.max_hours > 0 and math.isfinite(args.max_hours)):
        ap.error("--max-hours must be a positive number")
    return args


def _attach_warnings(handlers: Sequence[logging.Handler]) -> logging.Logger:
    """Route Python warnings (e.g. sklearn's) to the run log as well."""
    logging.captureWarnings(True)
    wl = logging.getLogger("py.warnings")
    wl.propagate = False
    for h in handlers:
        wl.addHandler(h)
    return wl


def _close_logging(wl: logging.Logger | None) -> None:
    if wl is not None:
        for h in list(wl.handlers):
            wl.removeHandler(h)
        logging.captureWarnings(False)
    pkg = logging.getLogger(LOGGER_NAME)
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
        h.close()


def main(argv: Sequence[str] | None = None, *, backend: Backend | None = None,
         gpu_probe: Callable[[], PF.GpuInfo] | None = None, disk_usage: Callable[[Path], Any] | None = None,
         clock: Callable[[], float] = time.monotonic, cfg: Mapping[str, Any] | None = None) -> int:
    """The overnight runner. Returns the exit code: 0 run finished (any verdict) or dry run passed; 2 preflight
    failed; 1 the runner itself failed; 130 stopped with Ctrl-C.

    `backend`, `gpu_probe`, `disk_usage`, `clock` and `cfg` replace the real ones (tests).
    """
    args = parse_args(argv)
    root = args.root.resolve() if args.root is not None else REPO_ROOT
    paths = Paths(smoke=args.smoke, root=root)
    prepare_environment(paths)  # before anything imports torch
    t_start = clock()
    log_file = paths.logs / f"overnight_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = setup_logging(log_file)
    wl = _attach_warnings(logger.handlers)
    try:
        log.info("overnight runner (%s%s) in %s; log %s", "smoke" if args.smoke else "full run",
                 ", dry run" if args.dry_run else "", root, log_file.name)
        if cfg is None:
            try:
                cfg = load_config(root / "configs" / "landslide.json")
            except Exception as e:  # noqa: BLE001
                log.error("PREFLIGHT FAILED: the config %s cannot be read: %s: %s",
                          root / "configs" / "landslide.json", type(e).__name__, e)
                return EXIT_PREFLIGHT
        report = PF.run_preflight(cfg, paths, gpu_probe=gpu_probe, disk_usage=disk_usage, clock=clock,
                                  t_start=t_start)
        for chk in report.checks:
            (log.info if chk.ok else (log.error if chk.fatal else log.warning))(
                "preflight %-14s %s %s", chk.name, "ok  " if chk.ok else ("FAIL" if chk.fatal else "warn"),
                chk.detail)
        if not report.ok:
            log.error("%s", report.message())
            return EXIT_PREFLIGHT
        log.info("%s", report.message())
        if args.dry_run:
            log.info("dry run: preflight only, nothing trained")
            return EXIT_OK

        cap_h = float(args.max_hours) if args.max_hours is not None else float(cfg["runner"]["wall_clock_h"])
        deadline = Deadline(start=t_start, cap_s=cap_h * 3600.0, reserve_s=EVAL_RESERVE_MIN * 60.0, clock=clock)
        opts = RunOptions(smoke=args.smoke, max_hours=args.max_hours, show_progress=not args.no_progress)
        runner = Runner(cfg, paths, opts, backend or Backend(), deadline=deadline, preflight=report,
                        log_file=log_file, clock=clock)
        with PF.offline(), keep_awake():
            try:
                return runner.run()
            except KeyboardInterrupt:
                log.error("stopped by the user (Ctrl-C); run the same command again to resume")
                runner.abort("KeyboardInterrupt", "aborted")
                return EXIT_ABORTED
            except Exception as e:  # noqa: BLE001
                log.exception("the runner failed")
                runner.abort(f"{type(e).__name__}: {e}", "error")
                return EXIT_ERROR
    finally:
        _close_logging(wl)
