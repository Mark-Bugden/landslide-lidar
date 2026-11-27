# Orchestrator decisions and incidents (phase 1)

Decisions taken while reviewing task branches, before any training. Each also sits in the owning task's file where
noted. P1-16 folds them into `PREREGISTRATION.md`.

**Mark's review (2026-11-27, before any training).** Mark approved O2, O3, O5 and O6 as recommended, and approved the
orchestrator running P1-09's sampling build (I2). After the investigation of the sampling counts, he approved
O8 the same day.

## O1 · Config hash leaves out `status` (P1-01, D02a)

Setting `status` to FROZEN at the freeze must not change the hash recorded by files built before it. Any other
config change still changes the hash.

## O2 · Kraj ties go to the lowest objectid (P1-08, D06)

- **Decision.** When two kraje tie in a sheet's majority vote, the kraj whose tied polygons include the lowest
  objectid wins (config `replication.splits.kraj_tie_rule`).
- **Why.** Appendix A §4 gives no tie rule. PLAN §5 allows a spike count to choose between readings of genuinely
  ambiguous text if it is said. The §5 strata were used this way: the lowest-objectid rule reproduces all four, and
  the area rule (PLAN D06's default) moves sheets 25-12-13 and 25-13-15 across the ZLMS/other line. Lowest objectid
  is also what a plain first-occurrence count in source order gives. Only two sheets (2 eligible polygons each)
  are affected.

## O3 · TRI is Riley's (P1-10, D19)

Appendix A §10 says only "TRI". PLAN D19 labelled its formula "Riley" but wrote the mean absolute difference
(GDAL's Wilson variant). We use Riley et al. (1999): per pixel √Σ(z_n − z)² over the 8 neighbours, averaged over the
patch. Decided before any data was built.

## O4 · Logistic regression keeps sklearn defaults (P1-11, P1-12)

`StandardScaler` + `LogisticRegression` with sklearn's defaults (lbfgs, `max_iter` 100) and `random_state` =
`replication.seed`. If a fit hits the iteration cap, it is kept and the warning is reported. The cap isn't raised
after the fact.

## O5 · Stage-2 windows and stage-1 patches near reserved sheets (P1-09, P1-12)

- **Finding (P1-12).** Appendix A §4 keeps stage-1 patches from touching any stage-2 sheet buffered by 150 m and
  says that therefore stage-2 windows never overlap a stage-1 patch. A 300 m axis-aligned window centred inside a
  sheet reaches up to 150√2 m from the sheet near its corners (and slightly more than 150 m along edges that aren't
  axis-aligned in EPSG:5514), so a small overlap (up to about 44 × 44 m at a corner) is geometrically possible.
- **Decision.** Keep Appendix A's literal rule (it reproduces the spike's dataset construction). Once
  `samples.parquet` exists, count the stage-1 patches that overlap any stage-2 window and the overlap areas, and
  report them as a known limitation. Not changed after seeing results.
- **Count [E] (final sampling build, 2026-11-27).** Of 11,132 samples, one (a random negative) overlaps one of the
  146,622 stage-2 windows, by 4 m × 50 m = 200 m², 0.2 % of the window. No positive overlaps any window.

## O6 · Float32 head under bf16 autocast (P1-06), approved by Mark

Only the final 512 → 1 layer runs in float32, so scores aren't rounded to bf16. The literal reading of "bf16
autocast" would give bf16 logits. Recorded as a decision, not a deviation; Mark approved it on 2026-11-27.

## O7 · Config keys are added in one batch at the freeze (P1-16)

Several tasks asked for keys for values that now sit in module constants (source URLs and layer ids, the
susceptibility alpha threshold, the stage-2 buffer, the verification stream, `eval.bootstrap.secondary_unit`, the
stage-2 bootstrap rule). P1-16 adds them in one batch, then re-runs every deterministic build (geo, samples,
patches, baselines, stage-2 baselines) under the final config and confirms the content is unchanged.

## O8 · Water-filling fills each sheet until it reaches the cap (P1-09, D13), approved by Mark

- **Finding.** The first sampling build (polygons taken only while the sheet stays ≤ the cap) gave caps 8 / 4 and
  3,779 positives against the spike's 8 / 3 and 3,613; at equal caps it selected about 6 % fewer points.
- **Investigation.** Read-only, with four readings of the cap and four of feasibility fixed before computing. Only
  "take polygons until the sheet reaches the cap" gives caps 8 / 3 (in all 21 order seeds tried), overshoots both
  budgets at cap + 1 as the spike says, and puts 3,613 inside its seed range (3,592–3,652).
- **Decision.** Adopt it, as PLAN §5 allows for genuinely ambiguous text. Mark approved it on 2026-11-27, before any
  training.

## O9 · 793 vs 870 infeasible polygons: unexplained, kept as a reported difference

Our exact patch rule leaves 793 of 14,617 eligible stage-1 polygons with no feasible patch; the spike reports 870.
None of the alternatives fixed in advance gives 870: the square plus its 56 m context inside the sheet (1,379), the
pixel-centre test (780), the stage-2 rule on the raster extent (971). Feasibility is seed-free, so randomness can't
explain it. The literal rule is kept, and the 77-polygon difference (0.5 %) is reported as a discrepancy rather than
searched further.

## I1 · Incident: a test run wrote synthetic files into the real data tree (2026-11-26, fixed)

- **What happened.** `LANDSLIDE_DATA_DIR` overrode every `Paths`, including those tests build on a tmp root. A full
  test run with the variable set (23:04) wrote a synthetic `data/dataset/meta.csv` and `data/baselines/preds_{a,b}.csv`
  and overwrote `data/derived/polygons.parquet` and `data/stage2/detections.parquet`.
- **Fix.** The overrides now apply only to the checkout's own root (commit "Keep the data-dir overrides away from
  test roots"), with a test.
- **Recovery and checks.** The three synthetic files were deleted (no real dataset existed yet). `build_geo.py` and
  `stage2_baselines.py` were re-run: `sheets.parquet` and `splits.json` came out byte-identical, the tracked
  `results/sheet_splits.csv`, `splits_summary.json` (apart from its `code_commit`) and `stage2_baselines.csv` were
  unchanged, and the real-data tests pass. A full test run after the fix changed no file under `data/`.

## I2 · P1-09's real build was blocked by the permission system

The P1-09 agent was denied reading the ZABAGED cache, a real-data dry run and a `git merge`. The orchestrator did not
run the build in its place and asked Mark how to proceed. The likely cause was the orchestrator's own earlier
instruction to that agent not to read the cache while it was still downloading. On 2026-11-27 Mark approved the
orchestrator running the build (`scripts/build_samples.py --check-determinism`) from the main checkout.
