"""Evaluation and replication verdict (task P1-13), re-runnable from saved predictions only.

Reads the run directories, `data/dataset/meta.csv`, the baseline predictions in `data/baselines/` and the stage-2
detections, and writes:
- `results/metrics.json` (the single source for REPLICATION.md and the model card; schema in
  `docs/decisions/P1-13.md`);
- `results/metrics_table.md` (our numbers next to the spike's claims, with discrepancy flags);
- `figures/learning_curves.png` (Appendix A §7).

With `--smoke` it reads the smoke run outputs and writes only under `data/smoke/` (`Paths(smoke=True)`), never to
the tracked `results/` or `figures/`. Re-running on the same inputs gives byte-identical files.

    uv run python scripts/evaluate.py [--smoke]
"""

from __future__ import annotations

import argparse
import sys

from landslide.config import load_config
from landslide.evaluate import run_evaluation
from landslide.logs import get_logger, setup_logging
from landslide.paths import Paths

log = get_logger("evaluate")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="evaluate the smoke test (reads/writes data/smoke/ only)")
    args = ap.parse_args(argv)
    paths = Paths(smoke=args.smoke)
    setup_logging(paths.logs / "evaluate.log")
    metrics = run_evaluation(load_config(), paths)
    t = metrics["tests"]
    print(f"verdict: {metrics['verdict']}  (a) {t['a']['status']}  (b) {t['b']['status']}  "
          f"stage 2 {t['stage2']['status']}")
    print(f"written to {paths.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
