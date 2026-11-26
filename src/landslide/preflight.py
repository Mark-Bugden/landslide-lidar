"""Preflight checks of the overnight runner, and the network block for the rest of the run.

BRIEF.md Phase 1 item 6 ("Offline", "Fail-fast"), Appendix A §11 and PLAN.md P1-14 item 2. Decisions are in
`docs/decisions/P1-14.md`.

`run_preflight(cfg, paths)` runs every check (it does not stop at the first problem, so one run lists them all) and
returns a `PreflightReport`. Nothing is hashed in full and nothing touches the network; with a warm disk cache it
takes a few seconds, most of it importing torch and starting CUDA. The checks:

1. **Freeze gate (G3).** `configs/landslide.sha256` exists and equals the hash of the config on disk; in a git
   checkout, the config and the sha256 file are committed (the commit order proves the freeze came first). This
   applies to `--smoke` too.
2. **Built artefacts carry the frozen hash:** `splits.json`, `data/dataset/build_info.json`,
   `data/baselines/build_info.json` (which must also name the dataset's `meta.csv`), and the stage-2 baseline outputs
   (`results/stage2_baselines.csv`, `data/stage2/summary.json`, every lattice file).
3. **Every cached input is present and non-empty** (a required-files list); sizes are compared where they are
   recorded (the dataset's `build_info.json`, the weights' `results/sources.json`).
4. **Dataset shape.** The `patches.npy` header is float16, C order, (N, channels, px, px); the file size equals the
   header plus the data; N equals the rows of `meta.csv`, which holds every column the runner needs.
5. **Stage-2 rasters.** The DTM and susceptibility rasters of the 20 reserved sheets have the size `sheets.parquet`
   records (header only), and `is_stage2` marks exactly the config's 20 sheets.
6. **Machine.** CUDA is available with >= 7 GiB free GPU memory, and the data drive has >= `runner.min_free_disk_gb`
   GiB free. The GPU probe runs with a timeout so that a hung driver still ends the preflight within 60 s.
7. **Config.** `runner.num_workers` is 0 (there is no DataLoader).

After a passed preflight the runner calls `block_network()`: every socket connection or name lookup to a
non-loopback address raises `NetworkBlocked` until `unblock_network()`.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import math
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from landslide.config import config_hash, frozen_hash
from landslide.logs import get_logger
from landslide.paths import Paths

log = get_logger("preflight")

GIB = 2 ** 30
#: Appendix A §11 "≥ 7 GB free GPU memory". There is no config key for it (requested for P1-16's batch).
MIN_FREE_GPU_GB = 7.0
#: BRIEF.md Phase 1 item 6: on any problem the runner exits within 60 s.
TIME_BUDGET_S = 60.0
#: Time kept after the GPU probe for the remaining checks and the exit, and the shortest probe timeout.
_PROBE_MARGIN_S = 8.0
_MIN_PROBE_TIMEOUT_S = 5.0
_GIT_TIMEOUT_S = 15.0

#: The stage-2 baseline scorers whose phase-1 outputs the full run evaluates next to the CNN (P1-12; equal to
#: `landslide.stage2.BASELINE_KINDS`, tested; written here so the preflight does not import the stage-2 stack).
STAGE2_BASELINES = ("slope_only", "susceptibility", "slope_susc_lr", "random")
#: meta.csv columns that training (`train.META_COLUMNS`), the evaluation (`evaluate.META_COLUMNS`), the smoke subset
#: and the stage-2 LR comparator (`stage2.LR_FEATURES`) read (tested against those modules).
META_REQUIRED = ("id", "label", "sheet", "kraj", "flysch", "source", "fold", "group_b", "block", "pair_id",
                 "match_level", "slope_mean", "slope_std", "susc_mean", "susc_high_frac")
#: Where each group of files comes from, for the error messages.
HINTS = {
    "splits": "built by `uv run python scripts/build_geo.py` (P1-08)",
    "dataset": "built by `uv run python scripts/build_patches.py` (P1-10)",
    "baselines": "built by `uv run python scripts/run_baselines.py` (P1-11)",
    "weights": "fetched by `uv run python scripts/fetch_weights.py` (P1-07)",
    "dtm": "the hand-over DTM rasters in data/raw_data/dmr5g/",
    "susceptibility": "fetched by `uv run python scripts/fetch_susceptibility.py` (P1-07)",
    "stage2_baselines": "built by `uv run python scripts/stage2_baselines.py` (P1-12)",
}


# ======================================================================================================================
# Report
# ======================================================================================================================


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "fatal": self.fatal, "detail": self.detail}


@dataclass
class PreflightReport:
    checks: list[Check]
    seconds: float
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def problems(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.fatal]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.fatal]

    @property
    def ok(self) -> bool:
        return not self.problems

    def message(self) -> str:
        if self.ok:
            head = f"PREFLIGHT PASSED in {self.seconds:.1f} s"
        else:
            head = (f"PREFLIGHT FAILED in {self.seconds:.1f} s with {len(self.problems)} problem(s); nothing was "
                    f"trained. Fix them and run the same command again:")
        lines = [head]
        lines += [f"  - [{c.name}] {c.detail}" for c in self.problems]
        lines += [f"  - warning [{c.name}] {c.detail}" for c in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "seconds": round(self.seconds, 2), "problems": [c.detail for c in self.problems],
                "warnings": [c.detail for c in self.warnings], "info": self.info}


class _Collector:
    def __init__(self, root: Path):
        self.root = root
        self.checks: list[Check] = []

    def rel(self, p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return Path(p).as_posix()

    def ok(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, True, detail))

    def fail(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, False, detail))

    def warn(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, False, detail, fatal=False))

    @contextmanager
    def guard(self, name: str) -> Iterator[None]:
        """Turn an unexpected exception inside a check into a problem instead of a crash."""
        try:
            yield
        except Exception as e:  # noqa: BLE001 - reported, never fatal to the preflight itself
            self.fail(name, f"could not be checked: {type(e).__name__}: {e}")


# ======================================================================================================================
# Machine probes (injectable for tests)
# ======================================================================================================================


@dataclass
class GpuInfo:
    available: bool
    name: str | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    error: str | None = None


def probe_gpu() -> GpuInfo:
    """CUDA availability and free memory of device 0 (imports torch and starts CUDA)."""
    import torch

    info = GpuInfo(available=False, torch_version=torch.__version__, cuda_version=torch.version.cuda)
    if not torch.cuda.is_available():
        info.error = "torch.cuda.is_available() is False"
        return info
    free, total = torch.cuda.mem_get_info(0)
    return dataclasses.replace(info, available=True, name=torch.cuda.get_device_name(0), free_bytes=int(free),
                               total_bytes=int(total))


def call_with_timeout(fn: Callable[[], Any], timeout_s: float, what: str) -> Any:
    """Run `fn` in a daemon thread; raise TimeoutError if it has not returned after `timeout_s`."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = e

    t = threading.Thread(target=target, name=f"preflight-{what}", daemon=True)
    t.start()
    t.join(max(0.0, timeout_s))
    if t.is_alive():
        raise TimeoutError(f"{what} did not finish within {timeout_s:.1f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _git(root: Path, *args: str) -> tuple[int, str]:
    """(returncode, stdout) of a git command in `root`; (-1, "") if git cannot run."""
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return r.returncode, r.stdout.strip()


def git_checkout(root: Path) -> bool:
    """True if `root` is the top level of a git work tree."""
    rc, top = _git(root, "rev-parse", "--show-toplevel")
    if rc != 0 or not top:
        return False
    try:
        return Path(top).resolve() == Path(root).resolve()
    except OSError:
        return False


# ======================================================================================================================
# Helpers
# ======================================================================================================================


def stage2_sheet_list(cfg: Mapping[str, Any]) -> list[str]:
    s2 = cfg["replication"]["splits"]["stage2_sheets"]
    return sorted(str(s) for k in ("flysch", "nonflysch") for s in s2[k])


def sha_path(paths: Paths) -> Path:
    return paths.root / "configs" / "landslide.sha256"


def npy_header(path: Path) -> tuple[tuple[int, ...], bool, np.dtype, int]:
    """(shape, fortran_order, dtype, data offset in bytes) of a `.npy` file, from its header only."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"unsupported .npy version {version}")
        return tuple(int(s) for s in shape), bool(fortran), np.dtype(dtype), f.tell()


def _read_json(p: Path) -> dict[str, Any]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{p.name} does not hold a JSON object")
    return obj


def _short(h: Any) -> str:
    return f"{str(h)[:12]}..." if h else repr(h)


@dataclass(frozen=True)
class Required:
    path: Path
    group: str


def required_files(cfg: Mapping[str, Any], paths: Paths) -> list[Required]:
    """The cached inputs a full run reads (the smoke run reads a subset of them). Stage-2 baseline outputs are
    always those of the full run (`data/stage2/`, `results/`), also in smoke mode."""
    full = dataclasses.replace(paths, smoke=False)
    sheets = stage2_sheet_list(cfg)
    out = [Required(paths.splits_json, "splits"), Required(paths.sheets_parquet, "splits"),
           Required(paths.polygons_parquet, "splits"),
           Required(paths.patches_npy, "dataset"), Required(paths.meta_csv, "dataset"),
           Required(paths.dataset / "build_info.json", "dataset"),
           Required(paths.baselines_dir / "preds_a.csv", "baselines"),
           Required(paths.baselines_dir / "preds_b.csv", "baselines"),
           Required(paths.baselines_dir / "build_info.json", "baselines")]
    out += [Required(paths.dtm(s), "dtm") for s in sheets]
    out += [Required(paths.susceptibility(s), "susceptibility") for s in sheets]
    out += [Required(full.results / "stage2_baselines.csv", "stage2_baselines"),
            Required(full.stage2 / "detections.parquet", "stage2_baselines"),
            Required(full.stage2 / "summary.json", "stage2_baselines")]
    out += [Required(full.stage2 / "lattice" / f"{s}.npz", "stage2_baselines") for s in sheets]
    out += [Required(full.stage2 / b / f"{s}.npz", "stage2_baselines") for b in STAGE2_BASELINES for s in sheets]
    return out


# ======================================================================================================================
# The checks
# ======================================================================================================================


def _check_freeze(c: _Collector, cfg: Mapping[str, Any], paths: Paths, info: dict[str, Any]) -> str | None:
    """The freeze gate. Returns the frozen hash (None before the freeze)."""
    current = config_hash(dict(cfg))
    info["config_hash"] = current
    sp = sha_path(paths)
    frozen = frozen_hash(sp) if sp.is_file() else None
    info["frozen_hash"] = frozen
    if frozen is None:
        c.fail("freeze", f"{c.rel(sp)} does not exist: the pre-registration is not frozen. No CNN may train before "
                         f"the freeze (gate G3, PLAN.md P1-16); this applies to --smoke too")
        return None
    if frozen != current:
        c.fail("freeze", f"the config on disk (hash {_short(current)}) differs from the frozen hash in {c.rel(sp)} "
                         f"({_short(frozen)}): the run must use the frozen config")
    else:
        c.ok("freeze", f"config hash {_short(current)} equals {c.rel(sp)}")
    status = str(cfg.get("status", ""))
    if not status.upper().startswith("FROZEN"):
        c.warn("freeze", f"the config's status is {status[:40]!r}, not FROZEN (the hash excludes it)")
    if git_checkout(paths.root):
        rc, out = _git(paths.root, "status", "--porcelain", "--", "configs/landslide.json",
                       "configs/landslide.sha256")
        if rc != 0:
            c.warn("freeze", "git status of the config failed; the freeze commit was not checked")
        elif out:
            c.fail("freeze", f"the frozen config is not committed ({'; '.join(out.splitlines())}): commit the "
                             f"freeze before any training (gate G3)")
        else:
            rc, sha = _git(paths.root, "log", "-1", "--format=%H", "--", "configs/landslide.sha256")
            info["freeze_commit"] = sha if rc == 0 and sha else None
    else:
        c.warn("freeze", "the repository root is not a git checkout; the freeze commit was not checked")
    return frozen


def _check_files(c: _Collector, req: Sequence[Required], info: dict[str, Any]) -> set[Path]:
    """Present, regular, non-empty files. Returns the set of usable paths."""
    usable: set[Path] = set()
    by_group: dict[str, list[str]] = {}
    n_bytes = 0
    for r in req:
        p = r.path
        if not p.is_file():
            by_group.setdefault(r.group, []).append(f"{c.rel(p)} is missing")
            continue
        size = p.stat().st_size
        if size == 0:
            by_group.setdefault(r.group, []).append(f"{c.rel(p)} is empty")
            continue
        usable.add(p)
        n_bytes += size
    for group, msgs in by_group.items():
        shown = "; ".join(msgs[:3]) + (f"; and {len(msgs) - 3} more" if len(msgs) > 3 else "")
        c.fail("files", f"{group}: {shown} ({HINTS.get(group, '')})")
    if not by_group:
        c.ok("files", f"{len(req)} required files present ({n_bytes / GIB:.2f} GiB)")
    info["required_files"] = {"n": len(req), "n_usable": len(usable), "gib": round(n_bytes / GIB, 3)}
    return usable


def _check_dataset(c: _Collector, cfg: Mapping[str, Any], paths: Paths, expected: str | None,
                   usable: set[Path], info: dict[str, Any]) -> dict[str, Any] | None:
    import pandas as pd

    npy, meta, bi_path = paths.patches_npy, paths.meta_csv, paths.dataset / "build_info.json"
    if not {npy, meta} <= usable:
        return None
    shape, fortran, dtype, offset = npy_header(npy)
    n_ch, px = len(cfg["channels"]), int(cfg["patch"]["px"])
    size = npy.stat().st_size
    bad = []
    if dtype != np.dtype(np.float16):
        bad.append(f"dtype {dtype}, expected float16")
    if fortran:
        bad.append("Fortran order, expected C order")
    if len(shape) != 4 or tuple(shape[1:]) != (n_ch, px, px):
        bad.append(f"shape {shape}, expected (N, {n_ch}, {px}, {px})")
    elif size != offset + math.prod(shape) * dtype.itemsize:
        bad.append(f"{size:,} bytes, but the header promises {offset + math.prod(shape) * dtype.itemsize:,} "
                   f"(truncated or padded)")
    cols = list(pd.read_csv(meta, nrows=0).columns)
    missing = [k for k in META_REQUIRED if k not in cols]
    n_rows = len(pd.read_csv(meta, usecols=["id"]))
    if missing:
        bad.append(f"meta.csv lacks columns {missing}")
    if shape and shape[0] != n_rows:
        bad.append(f"patches.npy holds {shape[0]:,} patches but meta.csv has {n_rows:,} rows")
    build = None
    if bi_path in usable:
        build = _read_json(bi_path)
        bp, bm = build.get("patches", {}), build.get("meta", {})
        if build.get("config_hash") != expected:
            bad.append(f"{c.rel(bi_path)} records config {_short(build.get('config_hash'))}, not the frozen "
                       f"{_short(expected)}")
        if list(bp.get("shape", [])) != list(shape) or bp.get("bytes") != size:
            bad.append(f"{c.rel(bi_path)} records patches {bp.get('shape')} / {bp.get('bytes')} bytes, the file is "
                       f"{list(shape)} / {size} bytes")
        if bm.get("rows") != n_rows or bm.get("bytes") != meta.stat().st_size:
            bad.append(f"{c.rel(bi_path)} records meta.csv {bm.get('rows')} rows / {bm.get('bytes')} bytes, the file "
                       f"has {n_rows} rows / {meta.stat().st_size} bytes")
    for b in bad:
        c.fail("dataset", f"{b} ({HINTS['dataset']})")
    if not bad:
        c.ok("dataset", f"patches.npy {shape} float16 matches meta.csv ({n_rows:,} rows) and build_info.json")
    info["dataset"] = {"shape": list(shape), "meta_rows": n_rows, "gib": round(size / GIB, 3)}
    return build


def _check_hash(c: _Collector, name: str, p: Path, got: Any, expected: str | None, hint: str) -> None:
    if got != expected:
        c.fail(name, f"{c.rel(p)} records config {_short(got)}, not the frozen {_short(expected)} ({hint})")


def _check_artefacts(c: _Collector, cfg: Mapping[str, Any], paths: Paths, expected: str | None,
                     usable: set[Path], dataset_build: dict[str, Any] | None) -> None:
    import pandas as pd

    n0 = len(c.checks)
    if paths.splits_json in usable:
        with c.guard("artefacts"):
            _check_hash(c, "artefacts", paths.splits_json, _read_json(paths.splits_json).get("config_hash"),
                        expected, HINTS["splits"])
    bb = paths.baselines_dir / "build_info.json"
    if bb in usable:
        with c.guard("artefacts"):
            b = _read_json(bb)
            _check_hash(c, "artefacts", bb, b.get("config_hash"), expected, HINTS["baselines"])
            want = (dataset_build or {}).get("meta", {}).get("sha256")
            got = b.get("inputs", {}).get("meta_csv_sha256")
            if want is not None and got != want:
                c.fail("artefacts", f"{c.rel(bb)} was built from a meta.csv with sha256 {_short(got)}, the dataset's "
                                    f"build_info.json records {_short(want)} ({HINTS['baselines']})")
    full = dataclasses.replace(paths, smoke=False)
    hint = HINTS["stage2_baselines"]
    csv = full.results / "stage2_baselines.csv"
    if csv in usable:
        with c.guard("artefacts"):
            df = pd.read_csv(csv, usecols=["scorer", "config_hash"])
            lacking = [s for s in STAGE2_BASELINES if s not in set(df["scorer"])]
            if lacking:
                c.fail("artefacts", f"{c.rel(csv)} lacks the scorers {lacking} ({hint})")
            other = sorted({str(h) for h in df["config_hash"]} - {str(expected)})
            if other:
                c.fail("artefacts", f"{c.rel(csv)} has rows built with config {[_short(h) for h in other]}, not the "
                                    f"frozen {_short(expected)} ({hint})")
    summ = full.stage2 / "summary.json"
    if summ in usable:
        with c.guard("artefacts"):
            s = _read_json(summ)
            _check_hash(c, "artefacts", summ, s.get("config_hash"), expected, hint)
            lacking = [b for b in STAGE2_BASELINES if b not in s.get("scorers", {})]
            if lacking:
                c.fail("artefacts", f"{c.rel(summ)} lacks the scorers {lacking} ({hint})")
    det = full.stage2 / "detections.parquet"
    if det in usable:
        with c.guard("artefacts"):
            scorers = set(pd.read_parquet(det, columns=["scorer"])["scorer"].astype(str))
            lacking = [b for b in STAGE2_BASELINES if b not in scorers]
            if lacking:
                c.fail("artefacts", f"{c.rel(det)} has no detections of {lacking} ({hint})")
    bad_lat = []
    for sheet in stage2_sheet_list(cfg):
        lp = full.stage2 / "lattice" / f"{sheet}.npz"
        if lp in usable:
            with c.guard("artefacts"), np.load(lp, allow_pickle=False) as z:
                h = str(z["config_hash"]) if "config_hash" in z.files else None
                if h != expected:
                    bad_lat.append(sheet)
    if bad_lat:
        c.fail("artefacts", f"the stage-2 lattices of {len(bad_lat)} sheet(s) (e.g. {bad_lat[0]}) were not built "
                            f"with the frozen config ({hint})")
    needed = [paths.splits_json, bb, csv, summ, det] + [full.stage2 / "lattice" / f"{s}.npz"
                                                        for s in stage2_sheet_list(cfg)]
    if len(c.checks) == n0 and all(p in usable for p in needed):
        c.ok("artefacts", f"splits, baselines and stage-2 baseline outputs carry config {_short(expected)}")


def _check_stage2_rasters(c: _Collector, cfg: Mapping[str, Any], paths: Paths, usable: set[Path]) -> None:
    import pandas as pd

    if paths.sheets_parquet not in usable:
        return
    sh = pd.read_parquet(paths.sheets_parquet, columns=["sheet", "grid_w", "grid_h", "is_stage2"])
    want = stage2_sheet_list(cfg)
    marked = sorted(sh.loc[sh["is_stage2"].astype(bool), "sheet"].astype(str))
    if marked != want:
        c.fail("stage2_rasters", f"sheets.parquet marks {len(marked)} stage-2 sheets that differ from the config's "
                                 f"20 ({HINTS['splits']})")
    grid = {str(r.sheet): (int(r.grid_h), int(r.grid_w)) for r in sh.itertuples()}
    import rasterio

    bad = []
    for sheet in want:
        for kind, p in (("dtm", paths.dtm(sheet)), ("susceptibility", paths.susceptibility(sheet))):
            if p not in usable or sheet not in grid:
                continue
            try:
                with rasterio.open(p) as src:
                    hw = (src.height, src.width)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{c.rel(p)} cannot be opened ({type(e).__name__})")
                continue
            if hw != grid[sheet]:
                bad.append(f"{c.rel(p)} is {hw[0]} x {hw[1]} px, sheets.parquet says {grid[sheet][0]} x "
                           f"{grid[sheet][1]}")
    for b in bad[:5]:
        c.fail("stage2_rasters", b)
    if len(bad) > 5:
        c.fail("stage2_rasters", f"and {len(bad) - 5} more raster problems")
    if not bad and marked == want:
        c.ok("stage2_rasters", f"{len(want)} reserved sheets: DTM and susceptibility rasters match sheets.parquet")


def _check_weights(c: _Collector, cfg: Mapping[str, Any], paths: Paths, info: dict[str, Any]) -> None:
    from landslide.model import imagenet_weights_path  # torch is loaded by now

    p = imagenet_weights_path(dict(cfg), paths)
    info["weights"] = c.rel(p)
    if not p.is_file() or p.stat().st_size == 0:
        c.fail("weights", f"the ImageNet weights {c.rel(p)} are missing or empty ({HINTS['weights']})")
        return
    size = p.stat().st_size
    src = paths.root / "results" / "sources.json"
    want = None
    if src.is_file():
        want = (_read_json(src).get("weights") or {}).get("bytes")
    if want is not None and int(want) != size:
        c.fail("weights", f"{c.rel(p)} has {size:,} bytes, results/sources.json records {int(want):,}")
    else:
        c.ok("weights", f"{c.rel(p)} ({size:,} bytes{', as recorded' if want is not None else ''})")


def _check_disk(c: _Collector, cfg: Mapping[str, Any], paths: Paths, disk_usage: Callable[[Path], Any],
                info: dict[str, Any]) -> None:
    need = float(cfg["runner"]["min_free_disk_gb"])
    if not paths.data.is_dir():
        c.fail("disk", f"the data directory {c.rel(paths.data)}/ does not exist")
        return
    free = float(disk_usage(paths.data).free) / GIB
    info["free_disk_gib"] = round(free, 2)
    if free < need:
        c.fail("disk", f"only {free:.1f} GiB free on the drive of {c.rel(paths.data)}; the run needs >= {need:g} GiB "
                       f"(runner.min_free_disk_gb)")
    else:
        c.ok("disk", f"{free:.1f} GiB free (>= {need:g} GiB)")


def _check_gpu(c: _Collector, gpu_probe: Callable[[], GpuInfo], timeout_s: float, info: dict[str, Any]) -> bool:
    """Returns True if torch loaded (so later torch-based checks may run)."""
    try:
        g = call_with_timeout(gpu_probe, timeout_s, "CUDA probe")
    except TimeoutError as e:
        c.fail("gpu", f"{e}: torch or the CUDA driver hangs; restart the machine and try again")
        return False
    except Exception as e:  # noqa: BLE001
        c.fail("gpu", f"torch / CUDA could not start: {type(e).__name__}: {e}")
        return False
    info["gpu"] = {"available": g.available, "name": g.name, "torch": g.torch_version, "cuda": g.cuda_version,
                   "free_gib": None if g.free_bytes is None else round(g.free_bytes / GIB, 2),
                   "total_gib": None if g.total_bytes is None else round(g.total_bytes / GIB, 2)}
    if not g.available:
        c.fail("gpu", f"CUDA is not available ({g.error or 'no CUDA device'}; torch {g.torch_version}, CUDA build "
                      f"{g.cuda_version}): training needs the GPU")
        return True
    free = (g.free_bytes or 0) / GIB
    if free < MIN_FREE_GPU_GB:
        c.fail("gpu", f"only {free:.1f} GiB of GPU memory free on {g.name}; the run needs >= {MIN_FREE_GPU_GB:g} GiB "
                      f"(close other GPU programs)")
    else:
        c.ok("gpu", f"{g.name}: {free:.1f} GiB free (>= {MIN_FREE_GPU_GB:g} GiB)")
    return True


def run_preflight(cfg: Mapping[str, Any], paths: Paths, *, gpu_probe: Callable[[], GpuInfo] | None = None,
                  disk_usage: Callable[[Path], Any] | None = None, clock: Callable[[], float] | None = None,
                  t_start: float | None = None) -> PreflightReport:
    """Run every check (see the module docstring) and return the report.

    `gpu_probe`, `disk_usage` and `clock` replace the machine probes (tests). `t_start` is when the runner started
    (default: now); the GPU probe gets what is left of the 60 s budget, less a margin for the exit.
    """
    clock = clock or time.monotonic
    t0 = clock() if t_start is None else t_start
    c = _Collector(paths.root)
    info: dict[str, Any] = {}
    with c.guard("freeze"):
        frozen = _check_freeze(c, cfg, paths, info)
    expected = frozen or info.get("config_hash")
    with c.guard("config"):
        nw = cfg["runner"]["num_workers"]
        if int(nw) != 0:
            c.fail("config", f"runner.num_workers is {nw}; the runner has no DataLoader and implements 0 only")
    usable: set[Path] = set()
    with c.guard("files"):
        usable = _check_files(c, required_files(cfg, paths), info)
    build = None
    with c.guard("dataset"):
        build = _check_dataset(c, cfg, paths, expected, usable, info)
    with c.guard("artefacts"):
        _check_artefacts(c, cfg, paths, expected, usable, build)
    with c.guard("stage2_rasters"):
        _check_stage2_rasters(c, cfg, paths, usable)
    with c.guard("disk"):
        _check_disk(c, cfg, paths, disk_usage or shutil.disk_usage, info)
    timeout = max(_MIN_PROBE_TIMEOUT_S, TIME_BUDGET_S - _PROBE_MARGIN_S - (clock() - t0))
    torch_ok = _check_gpu(c, gpu_probe or probe_gpu, timeout, info)
    if torch_ok:
        with c.guard("weights"):
            _check_weights(c, cfg, paths, info)
    else:
        c.fail("weights", "not checked: torch did not load")
    seconds = clock() - t0
    if seconds > TIME_BUDGET_S:
        c.warn("time", f"the preflight took {seconds:.0f} s, more than the {TIME_BUDGET_S:.0f} s budget")
    return PreflightReport(checks=c.checks, seconds=seconds, info=info)


# ======================================================================================================================
# Network block (offline for the rest of the run)
# ======================================================================================================================


class NetworkBlocked(RuntimeError):
    """A network connection or name lookup was attempted after the preflight (the run is offline)."""


_saved: dict[str, Any] | None = None
_lock = threading.Lock()


def _is_local(host: Any) -> bool:
    """Loopback addresses (127.0.0.0/8, ::1, "localhost") and the passive None host never leave the machine."""
    if host is None:
        return True
    h = host.decode("ascii", "replace") if isinstance(host, (bytes, bytearray)) else str(host)
    h = h.strip().strip("[]").split("%", 1)[0].lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _address_host(address: Any) -> Any:
    return address[0] if isinstance(address, tuple) and address else address


def _refuse(what: str, target: Any) -> None:
    raise NetworkBlocked(f"network access is disabled for the overnight run (offline after the preflight): "
                         f"{what} {target!r}")


def block_network() -> None:
    """Make every socket connection, datagram and name lookup to a non-loopback address raise `NetworkBlocked`.

    Patched: `socket.socket.connect`, `connect_ex`, `sendto`, and `socket.getaddrinfo`, `gethostbyname`,
    `gethostbyname_ex` (so `socket.create_connection`, urllib3 and requests fail before any packet is sent).
    Loopback stays allowed: it never leaves the machine, and Python's own `socket.socketpair` on Windows needs it.
    Idempotent; undone by `unblock_network()`.
    """
    global _saved
    with _lock:
        if _saved is not None:
            return
        S = socket.socket
        saved = {"connect": S.connect, "connect_ex": S.connect_ex, "sendto": S.sendto,
                 "getaddrinfo": socket.getaddrinfo, "gethostbyname": socket.gethostbyname,
                 "gethostbyname_ex": socket.gethostbyname_ex}

        def connect(self, address):  # type: ignore[no-untyped-def]
            if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(_address_host(address)):
                _refuse("connect to", address)
            return saved["connect"](self, address)

        def connect_ex(self, address):  # type: ignore[no-untyped-def]
            if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(_address_host(address)):
                _refuse("connect to", address)
            return saved["connect_ex"](self, address)

        def sendto(self, data, *args):  # type: ignore[no-untyped-def]
            address = args[-1] if args else None
            if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(_address_host(address)):
                _refuse("send to", address)
            return saved["sendto"](self, data, *args)

        def getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
            if not _is_local(host):
                _refuse("name lookup of", host)
            return saved["getaddrinfo"](host, *args, **kwargs)

        def gethostbyname(name):  # type: ignore[no-untyped-def]
            if not _is_local(name):
                _refuse("name lookup of", name)
            return saved["gethostbyname"](name)

        def gethostbyname_ex(name):  # type: ignore[no-untyped-def]
            if not _is_local(name):
                _refuse("name lookup of", name)
            return saved["gethostbyname_ex"](name)

        S.connect, S.connect_ex, S.sendto = connect, connect_ex, sendto  # type: ignore[method-assign]
        socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
        socket.gethostbyname = gethostbyname  # type: ignore[assignment]
        socket.gethostbyname_ex = gethostbyname_ex  # type: ignore[assignment]
        _saved = saved
    log.info("network disabled for the rest of the run (loopback only)")


def unblock_network() -> None:
    """Restore the original socket functions (tests; the runner calls it on exit)."""
    global _saved
    with _lock:
        if _saved is None:
            return
        S = socket.socket
        S.connect, S.connect_ex, S.sendto = _saved["connect"], _saved["connect_ex"], _saved["sendto"]  # type: ignore
        socket.getaddrinfo = _saved["getaddrinfo"]  # type: ignore[assignment]
        socket.gethostbyname = _saved["gethostbyname"]  # type: ignore[assignment]
        socket.gethostbyname_ex = _saved["gethostbyname_ex"]  # type: ignore[assignment]
        _saved = None


def network_blocked() -> bool:
    return _saved is not None


@contextmanager
def offline() -> Iterator[None]:
    """`block_network()` for the duration of the block."""
    already = network_blocked()
    block_network()
    try:
        yield
    finally:
        if not already:
            unblock_network()
