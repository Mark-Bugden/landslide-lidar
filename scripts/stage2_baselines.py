"""Stage-2 baselines (task P1-12): window lattices of the 20 reserved sheets, baseline scores, NMS, matching, P@R50.

    uv run python scripts/stage2_baselines.py                       # slope_only, susceptibility, random (+ LR if
                                                                    # data/dataset/meta.csv exists)
    uv run python scripts/stage2_baselines.py --scorers slope_susc_lr   # add the LR comparator row later
    uv run python scripts/stage2_baselines.py --evaluate-only       # re-evaluate from the saved lattices
    uv run python scripts/stage2_baselines.py --smoke               # one sheet, everything under data/smoke/

CPU only; one sheet raster in memory at a time. Writes `data/stage2/{lattice,<scorer>}/<sheet>.npz`,
`data/stage2/detections.parquet`, `data/stage2/summary.json` and the tracked `results/stage2_baselines.csv`
(one row per scorer; rows of scorers not run are kept). The CNN scorer is not run here: the overnight runner calls
`landslide.stage2.run_cnn` after the freeze (gate G3). Method and decisions: `src/landslide/stage2.py`,
`docs/decisions/P1-12.md`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from landslide import stage2
from landslide.config import load_config
from landslide.logs import get_logger, setup_logging
from landslide.paths import Paths


def _fmt(v) -> str:
    return "-" if v is None or v != v else f"{v:.4f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorers", default=None,
                    help=f"comma-separated, from {list(stage2.BASELINE_KINDS)}; default: "
                         f"{','.join(stage2.UNFITTED_BASELINES)}, plus {stage2.LR_SCORER} when meta.csv exists")
    ap.add_argument("--smoke", action="store_true", help="write under data/smoke/ (smoke.stage2_sheets sheets)")
    ap.add_argument("--sheets", default=None, help="comma-separated subset of the stage-2 sheets (smoke only)")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="skip the lattices and re-evaluate from data/stage2/<scorer>/<sheet>.npz")
    ap.add_argument("--quiet", action="store_true", help="no progress bars")
    args = ap.parse_args(argv)

    cfg = load_config()
    paths = Paths(smoke=args.smoke)
    setup_logging(paths.logs / f"stage2_baselines_{datetime.now():%Y%m%d_%H%M%S}.log")
    log = get_logger("stage2_baselines")

    sheets = None
    if args.sheets:
        if not args.smoke:
            ap.error("--sheets is allowed only with --smoke (the tracked results cover all 20 sheets)")
        sheets = [s.strip() for s in args.sheets.split(",") if s.strip()]
    elif args.smoke:
        sheets = sorted(s for v in stage2.stage2_sheets(cfg).values() for s in v)[:int(cfg["smoke"]["stage2_sheets"])]
    inputs = stage2.load_inputs(cfg, paths, sheets)

    if args.scorers:
        scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    else:
        scorers = list(stage2.UNFITTED_BASELINES)
        if paths.meta_csv.is_file():
            scorers.append(stage2.LR_SCORER)
        else:
            log.info("%s skipped: %s does not exist yet (it runs once P1-10 has built the dataset)",
                     stage2.LR_SCORER, paths.meta_csv)
    bad = [s for s in scorers if s not in stage2.BASELINE_KINDS]
    if bad:
        ap.error(f"unknown or non-baseline scorers {bad}; the CNN is scored only by the overnight runner")

    log.info("stage 2 baselines %s on %d sheets (%s)", scorers, len(inputs.sheet_ids),
             "smoke" if args.smoke else "full")
    if args.evaluate_only:
        results = stage2.evaluate_scorers(cfg, paths, scorers, inputs)
    else:
        results = stage2.run_baselines(cfg, paths, scorers, inputs, show_progress=not args.quiet)

    print(f"\n{'scorer':<16}{'status':<13}{'P@R50':>8}{'interp':>8}{'maxrec':>8}{'ringFP':>8}"
          f"{'F P@R50':>9}{'N P@R50':>9}  detections")
    for r in results.values():
        s, fl, nf = r.pooled, r.strata.get("flysch"), r.strata.get("nonflysch")
        print(f"{r.scorer:<16}{s.status:<13}{_fmt(s.precision_at_recall):>8}{_fmt(s.interpolated_precision):>8}"
              f"{_fmt(s.max_recall):>8}{_fmt(r.ring_fp.precision_at_recall):>8}"
              f"{_fmt(fl.precision_at_recall if fl else None):>9}{_fmt(nf.precision_at_recall if nf else None):>9}"
              f"  {len(r.detections)}")
    print(f"\nwrote {paths.stage2} and {paths.results / 'stage2_baselines.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
