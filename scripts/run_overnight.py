"""The overnight run (task P1-14): training -> evaluation -> stage 2 (only if (a) and (b) pass) -> verdict.

    uv run python scripts/run_overnight.py               # the full run (phase 2, Mark)
    uv run python scripts/run_overnight.py --smoke       # the smoke test (P1-17); writes only under data/smoke/
    uv run python scripts/run_overnight.py --dry-run     # preflight checks only
    uv run python scripts/run_overnight.py --max-hours 2 # another wall-clock cap (default runner.wall_clock_h)

Resume: run the same command again; finished units are skipped. Progress: the newest `logs/overnight_*.log`
(`data/smoke/logs/` for the smoke test); summary: `results/run_summary.json` (`data/smoke/results/`).

The preflight refuses to start before the freeze (`configs/landslide.sha256`, gate G3) and exits within 60 s on any
problem. Exit codes: 0 finished (whatever the verdict) or dry run passed, 2 preflight failed, 1 runner error,
130 Ctrl-C. Method and decisions: `src/landslide/runner.py`, `src/landslide/preflight.py`,
`docs/decisions/P1-14.md`.
"""

import sys


def _main() -> int:
    # landslide.runner does not import torch; runner.main() sets CUBLAS_WORKSPACE_CONFIG and TORCH_HOME first thing,
    # before anything loads torch (P1-06's determinism requirement).
    from landslide.runner import main

    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main())
