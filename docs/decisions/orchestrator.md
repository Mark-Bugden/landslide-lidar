# Orchestrator decisions and incidents (phase 1)

Decisions taken while reviewing task branches, before any training. Each also sits in the owning task's file where
noted. P1-16 folds them into `PREREGISTRATION.md`.

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

## O6 · Float32 head under bf16 autocast (P1-06), flagged to Mark

Only the final 512 → 1 layer runs in float32, so scores aren't rounded to bf16. The literal reading of "bf16
autocast" would give bf16 logits. Recorded as a decision, not a deviation; Mark may veto before the freeze.

## O7 · Config keys are added in one batch at the freeze (P1-16)

Several tasks asked for keys for values that now sit in module constants (source URLs and layer ids, the
susceptibility alpha threshold, the stage-2 buffer, the verification stream, `eval.bootstrap.secondary_unit`, the
stage-2 bootstrap rule). P1-16 adds them in one batch, then re-runs every deterministic build (geo, samples,
patches, baselines, stage-2 baselines) under the final config and confirms the content is unchanged.

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
run the build in its place and asked Mark how to proceed.
