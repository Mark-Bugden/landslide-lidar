"""Stage-1 baselines and classical-ML comparators on the CPU (task P1-11; BRIEF.md Appendix A §8, §10).

Reads (never rebuilds) `data/dataset/meta.csv` (P1-10) and `data/derived/splits.json` (P1-08). The patches are not
read: every scorer works on the per-patch statistics in `meta.csv`. Method: `landslide.baselines`; decisions in
`docs/decisions/P1-11.md`.

- Test (a): for each fold, the fitted scorers train on that fold's train + inner-val rows (`select_unit_rows`, as
  the CNN) and score its test rows; the out-of-fold predictions are pooled. Test (b): one fit.
- Metrics: ROC AUC with sheet- and block-level cluster-bootstrap CIs (`eval.bootstrap`: 2,000 replicates, seed
  123), per fold (a), by kraj and flysch flag (b), positives against each negative source, slope-only as the
  matching check.

Outputs:
- `data/baselines/preds_a.csv` (`id`, `fold`, one column per scorer) and `preds_b.csv` (`id`, one column per
  scorer): one row per test patch, higher = more landslide-like (the interface for P1-13's paired differences);
- `data/baselines/build_info.json`: config hash, git commit, input hashes, library versions, file hashes, fits;
- `results/baselines.csv` (one row per test x scorer x breakdown group, `kind` = baseline | classical_ml) and
  `results/baselines.json` (headline AUCs, best scorers, matching check, settings, fits, the spike's reference).

`--out DIR` writes DIR/baselines/... and DIR/results/... instead; `--meta` and `--splits` read other inputs.
`--check-determinism` computes everything twice and requires identical outputs before writing.

    uv run python scripts/run_baselines.py --check-determinism
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from landslide import baselines as B
from landslide.config import config_hash, load_config
from landslide.logs import get_logger, setup_logging
from landslide.paths import Paths

log = get_logger("run_baselines")


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=root,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _rel(path: Path, root: Path) -> str:
    """`path` relative to the repo root (posix) when it lies inside it, else as it is."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, lineterminator="\n")


def read_inputs(meta_path: Path, splits_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    for p in (meta_path, splits_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist (meta.csv comes from P1-10, splits.json from P1-08)")
    meta = pd.read_csv(meta_path)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    return meta, splits


def outputs_text(res: B.BaselineResults) -> dict[str, str]:
    """The text of every deterministic output (the build-info file is added by `main`)."""
    return {"baselines/preds_a.csv": _csv_text(res.preds["a"]), "baselines/preds_b.csv": _csv_text(res.preds["b"]),
            "results/baselines.csv": _csv_text(res.table)}


def main(argv: list[str] | None = None) -> int:
    paths = Paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, default=None, help=f"meta.csv (default {paths.meta_csv})")
    ap.add_argument("--splits", type=Path, default=None, help=f"splits.json (default {paths.splits_json})")
    ap.add_argument("--out", type=Path, default=None,
                    help="write under this directory (OUT/baselines, OUT/results, OUT/logs) instead")
    ap.add_argument("--check-determinism", action="store_true",
                    help="compute everything twice and require identical outputs")
    args = ap.parse_args(argv)
    meta_path = args.meta or paths.meta_csv
    splits_path = args.splits or paths.splits_json
    base_dir = args.out / "baselines" if args.out else paths.baselines_dir
    results_dir = args.out / "results" if args.out else paths.results
    log_dir = args.out / "logs" if args.out else paths.logs

    setup_logging(log_dir / "run_baselines.log")
    cfg = load_config()
    chash = config_hash(cfg)
    t0 = time.perf_counter()
    log.info("run_baselines: config %s; meta %s; splits %s; -> %s, %s", chash[:12], meta_path, splits_path,
             base_dir, results_dir)
    meta, splits = read_inputs(meta_path, splits_path)
    if splits.get("config_hash") != chash:
        log.warning("splits.json was built with config %s, the current config is %s",
                    str(splits.get("config_hash"))[:12], chash[:12])

    res = B.run_baselines(meta, splits, cfg)
    log.info("computed in %.1f s", time.perf_counter() - t0)
    texts = outputs_text(res)
    summary_text = _json_text(res.summary)

    determinism = None
    if args.check_determinism:
        t1 = time.perf_counter()
        res2 = B.run_baselines(meta, splits, cfg)
        if outputs_text(res2) != texts or _json_text(res2.summary) != summary_text:
            log.error("determinism check failed: two computations differ; nothing written")
            return 1
        determinism = {"recomputed": True, "identical": True}
        log.info("determinism: second computation identical (%.1f s)", time.perf_counter() - t1)

    commit = _git(paths.root, "rev-parse", "HEAD")
    dirty = _git(paths.root, "status", "--porcelain", "--untracked-files=no")
    inputs = {"meta_csv": str(meta_path), "meta_csv_sha256": _sha256(meta_path), "meta_rows": int(len(meta)),
              "splits_json": str(splits_path), "splits_json_sha256": _sha256(splits_path),
              "splits_config_hash": splits.get("config_hash")}
    results_json = {"task": "P1-11", "config_hash": chash, "git_commit": commit,
                    "inputs": {k: inputs[k] for k in ("meta_csv_sha256", "meta_rows", "splits_json_sha256",
                                                      "splits_config_hash")},
                    "outputs": {"preds_a": _rel(base_dir / "preds_a.csv", paths.root),
                                "preds_b": _rel(base_dir / "preds_b.csv", paths.root),
                                "build_info": _rel(base_dir / "build_info.json", paths.root),
                                "table": _rel(results_dir / "baselines.csv", paths.root)},
                    **res.summary}

    for rel, text in texts.items():
        root = base_dir if rel.startswith("baselines/") else results_dir
        _atomic_write(root / rel.split("/", 1)[1], text)
    _atomic_write(results_dir / "baselines.json", _json_text(results_json))
    files = {name: {"rows": int(len(res.preds[t])), "sha256": _sha256(base_dir / name)}
             for t, name in (("a", "preds_a.csv"), ("b", "preds_b.csv"))}
    build_info = {
        "task": "P1-11", "config_hash": chash, "git_commit": commit,
        "git_dirty": None if dirty is None else bool(dirty),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs, "files": files,
        "columns": {"preds_a.csv": list(res.preds["a"].columns), "preds_b.csv": list(res.preds["b"].columns)},
        "score_direction": "higher = more landslide-like",
        "random_state": res.summary["random_state"], "scorers": res.summary["scorers"],
        "fits": res.summary["fits"], "all_lr_fits_converged": res.summary["all_lr_fits_converged"],
        "determinism": determinism,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "seconds": round(time.perf_counter() - t0, 2),
    }
    _atomic_write(base_dir / "build_info.json", _json_text(build_info))

    if not res.summary["all_lr_fits_converged"]:
        bad = [f"{f['run']}/{f['baseline']}" for f in res.summary["fits"] if f["converged"] is False]
        log.warning("logistic regression did not converge for %s (kept as fitted; see build_info.json)", bad)
    for t, entry in res.summary["tests"].items():
        for name, r in entry["results"].items():
            u = res.summary["bootstrap"]["primary_unit"]
            log.info("(%s) %-18s %-12s AUC %s  %s CI %s", t, name, r["kind"],
                     "n/a" if r["auc"] is None else f"{r['auc']:.3f}", u,
                     [None if v is None else round(v, 3) for v in r[f"{u}_ci"]])
        ref = entry.get("spike_reference")
        if ref:
            log.info("(%s) spike's best non-CNN (reference only): %s %.3f [%.3f, %.3f]; ours %.3f", t,
                     ref["baseline"], ref["auc"], ref["ci_low"], ref["ci_high"], ref["ours"])
    log.info("wrote %s, %s, %s and %s (%.1f s)", base_dir / "preds_a.csv", base_dir / "preds_b.csv",
             results_dir / "baselines.csv", results_dir / "baselines.json", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
