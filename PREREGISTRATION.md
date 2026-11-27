# Pre-registration: independent replication of the landslide kill test

**Status: FROZEN on 2026-11-27**, in the commit "Freeze pre-registration" on `phase-1`, before any CNN training
(the smoke test included). From this commit on, nothing in this file, the method or the config changes; the config
hash is recorded in `configs/landslide.sha256`, and the overnight runner refuses to train unless the committed config
matches it.

## 1. Purpose

A research spike in a separate private repo reported that a ResNet-18 trained on DMR 5G LiDAR terrain recognises the
landslides mapped in the ČGS slope-deformation register, on map sheets and regions it never saw, once slope is
controlled. This repo tests that claim independently: its own code, inputs rebuilt from public sources, and a newly
trained model. The verdict is one of REPLICATED, NOT REPLICATED or INCOMPLETE, under the criteria in §4, fixed here
before any training.

## 2. Independence

- **Code.** Every step is written in this repo from the method text in `docs/BRIEF.md` Appendix A: the dataset
  build, baselines, training, evaluation (bootstrap included), and the stage-2 detection and matching. No spike code
  was provided or read.
- **Data.** Only the eight public inputs listed in `docs/BRIEF.md` ("Data sources"), checked against their sources
  before use (provenance check, P1-03). Everything the spike derived (patches, `in_mapped`, the flysch outline,
  susceptibility rasters, ZABAGED subsets, splits, baseline scores) is derived again here.
- **Results.** The spike's numbers below are claims to verify. They are used only as sanity checks while building
  the pipeline (`docs/PLAN.md` §5), never as targets, and never to tune sampling, splits or the model. No spike
  result produces or appears in any result here.
- **Splits.** The 20 reserved stage-2 sheets are the spike's (Appendix A §4). Everything else (kraj and flysch flags,
  blocks, folds, groups, inner-validation blocks) is rebuilt with the rules and seeds of Appendix A §4, so the folds
  will not match the spike's exactly.

## 3. The spike's claims (to verify)

Copied verbatim from `docs/BRIEF.md` ("The spike's claims (to verify)"):

- **Labels.** The ČGS slope-deformation register.
- **Inputs.** 300 m × 300 m patches of DMR 5G at 2 m: 7 channels, invariant under rotations and flips.
- **Model.** ResNet-18, 3 seeds.
- **Negatives.** Matched on slope, from the same map sheets, at least 150 m from any mapped polygon.

| Pre-registered test | Threshold | Spike CNN | Spike best non-CNN |
| --- | --- | --- | --- |
| (a) held-out flysch sheets, ROC AUC (5 folds, pooled) | ≥ 0.75 | 0.811 [0.796, 0.826] | 0.695 [0.678, 0.713] (terrain-stats LR) |
| (b) train Zlínský + Moravskoslezský, test other kraje, ROC AUC | ≥ 0.65 | 0.749 [0.727, 0.770] | 0.624 [0.603, 0.647] (terrain-stats GBM) |
| Stage 2: 20 reserved sheets, precision at 50 % object recall | ≥ 0.30 | 0.559 [0.407, 0.812]† | 0.136 (slope + susceptibility LR); random 0.085 |

† Post hoc sheet-level bootstrap, not pre-registered.

**Post hoc checks from the spike:**
- same-sheet pair accuracy 0.824 (a) and 0.762 (b);
- stage 2 is 0.418 if detections in the 150 m ring count as false positives;
- non-flysch Jihomoravský AUC 0.622.

**Nothing from the spike may produce your results or appear in the demo.** You have its claims (above), its method
and config (appendices) and raw downloads from public sources. Every number you publish comes from your own run.

## 4. Replication criteria

Copied verbatim from `docs/BRIEF.md` ("Replication criteria (fixed now; never change them after training)"):

- **REPLICATED** requires all three of your own tests to pass, under the same definitions:
  - (a) AUC ≥ 0.75;
  - (b) AUC ≥ 0.65;
  - stage-2 P@R50 ≥ 0.30.

  The point estimate decides. A test with any missing run is INCOMPLETE, which is never a pass.
- **Report every number next to the spike's claim** (the table above). If one of your point estimates falls outside
  the spike's 95 % CI, report it as a discrepancy and investigate it in phase 3. A discrepancy does not change the
  verdict.
- **Recompute on your own data.** This covers the baselines and comparators, and any post hoc check you want to cite
  (e.g. same-sheet pair accuracy, stage 2 with the ring counted as false positives).
- **If NOT REPLICATED or INCOMPLETE:** stop and report to Mark. The demo must not present the spike's claims, and
  Mark decides what happens next.

## 5. Method

The method is the spike's pre-registration §1–§12, reproduced verbatim as **Appendix A of
[`docs/BRIEF.md`](docs/BRIEF.md)**, which is part of this repo. The values are **Appendix B** of the same file.

- **Frozen configuration:** [`configs/landslide.json`](configs/landslide.json). It carries every Appendix B key and
  value unchanged; only `name`, `status` and `paths` (the spike's own repo paths) differ.
  `tests/test_config_appendix_b.py` embeds Appendix B verbatim and fails on any changed or missing value.
- **Our own fixed choices** where Appendix A/B is silent are in the config's `replication` block: base seed, named
  random streams, provenance sample size, negative jitter, NMS tie jitter, verdict words, channel constants, split
  values and tie rule, flysch settings, stage-2 buffer, baseline scorers, secondary bootstrap unit, stage-2
  bootstrap rule and runner limits. §7 below lists every one with its reason.
- **Source facts** that the brief states outside Appendix B are in the config's `sources` block: service URLs and
  layer ids, the DMR 5G lattice origin, the package's valid range, the expected feature counts, the susceptibility
  render recipe and the ImageNet weights' file and sha256. They are under the config hash like every other value.
- **Fixed settings.** No hyperparameter search. Fixed seeds and thresholds. The configuration is never changed after
  seeing any CNN result.
- **Verdict words.** The spike's PASS / FAIL become REPLICATED / NOT REPLICATED, with the same precedence
  (Appendix A §1): a completed test below its threshold means NOT REPLICATED even when other runs are missing; a
  test with any missing run is INCOMPLETE, which is never a pass.

## 6. Data sources and licences

| # | Input | Source | Licence | How we get it |
| --- | --- | --- | --- | --- |
| 1 | DMR 5G terrain, 2 m, one raster per surveyed sheet | ČÚZK `dmr5g` ImageServer | CC BY 4.0 | Hand-over files, verified by re-download (P1-03) |
| 2 | Slope-deformation register (labels) | ČGS `Geohazardy/svahove_nestability` MapServer layer 1 | CC BY 4.0 | Hand-over file, verified by feature count |
| 3 | Surveyed map sheets | ČGS `Prozkoumanost/prehled_mapovani_svahove_nestability` layer 0 | not stated | Hand-over file, verified by feature count |
| 4 | Geological map 1:500k (flysch outline) | ČGS `Geologie/geologicka_mapa500_CR` layer 7 | not verified | Hand-over file |
| 5 | Landslide susceptibility map (baselines) | ČGS `Geohazardy/sesuvna_nachylnost` MapServer | not stated | Fetched (rendered per class) |
| 6 | ZABAGED look-alikes and landslide/scree exclusion | ČÚZK `ZABAGED_POLOHOPIS` layers 118, 119, 130, 129, 79, 75 | CC BY 4.0 | Fetched |
| 7 | Orthophoto (demo only) | ČÚZK `ORTOFOTO` MapServer | CC BY 4.0 | Fetched in phase 4 only |
| 8 | ImageNet weights for ResNet-18 | torchvision `ResNet18_Weights.IMAGENET1K_V1` | torchvision's terms | Downloaded through torchvision |

Every input lives in the gitignored `data/`, and a fetch script rebuilds each one from its source. Rows 3, 4 and 5
have no stated or verified licence: they and anything derived from them (sheet geometry, the flysch outline,
susceptibility rasters) are never committed, drawn in a figure or shown in the demo; only aggregate results (AUCs,
per-sheet flysch flags, map-sheet numbers) are published.

## 7. Decisions

Appendix A leaves many details open, and a few of its sentences can be read in more than one way. Every choice made
to implement it is listed here, grouped by stage. All of them were made before any CNN saw a real patch. Each row
gives the decision, the reason, the owning task, and the evidence. Full details, including the alternatives
considered, are in `docs/decisions/<task>.md` (the "Detail" column names the entry, e.g. `P1-08 D06`).
`docs/decisions/orchestrator.md` holds the orchestrator's decisions (O1–O9) and the incidents (I1, I2). The decision
ids D01–D25 refer to the decision register in `docs/PLAN.md` §6.

Tags: [V] verified against a source, [E] measured on our data, [I] inference. "§5-guided" marks the two places where
a spike number was used to choose between readings of ambiguous text (§7.2).

### 7.1 What was established before training

**Sources (provenance, gate G1: passed).** [E] The checks ran on 2026-11-26 (`results/provenance.json`,
`docs/provenance.md`). The live feature counts equal the brief's 2026-11-25 counts: 21,356 register polygons and 860
surveyed sheets; the geology query returns 597 features, as many as the hand-over file. Twelve DTM sheets drawn at
random (seed 2611) were downloaded again with the brief's recipe. Their georeferencing is equal, and 81,849,089 of
81,849,089 compared pixels are bitwise equal, including 5.7 million 0 m border cells. The recipe grid equals the fetch
log and each raster's own grid for all 860 sheets. All 863 hand-over files match `MANIFEST.csv` (sha256). **So the
checks found no change in the sources of rows 1–4 since 2026-11-25.** Rows 5 and 6 were fetched on 2026-11-26. The
brief gives no earlier reference for them, so their service versions, layer metadata and counts are recorded in
`results/sources.json` (susceptibility: 475 of 475 point queries on 39 sheets agree with the rendered rasters). The
ImageNet weights are `resnet18-f37072fd.pth`, 46,830,571 bytes, sha256 `f37072fd…c560e07ec` (the full hash is in
the config's `sources.weights`).

**Our counts against the spike's (`docs/PLAN.md` §5).** These numbers were used to catch bugs, and twice to choose
between readings of ambiguous text (§7.2). They were never used to tune sampling or splits.

| Quantity | Spike | Ours [E] | Agree | Detail |
| --- | --- | --- | --- | --- |
| Register / surveyed-sheet features (`returnCountOnly`) | 21,356 / 860 | 21,356 / 860 | yes | P1-03 D-new-6 |
| Per-sheet DTM grids, recipe vs `fetch_log.jsonl` | 860 identical | 860 of 860 | yes | P1-03 D03 |
| `in_mapped` polygons | 18,649 | 18,649 | yes | P1-08 D04 |
| Eligible positives (skupina 1, in_mapped, ≥ 3,000 m²) | 15,082 | 15,082 | yes | P1-08 D05 (the source's `st_area` field would give 15,083) |
| Sheet rasters with 0 m cells | 96 | 96 | yes | P1-08 |
| Flysch outline area | 6,586 km² | 6,585.6 km² | yes | P1-08 D07 (the other reading of the closing gives 6,525.3) |
| Flysch sheets (≥ 50 % inside) | 300 | 300 | yes | P1-08 D07 (other reading: 299) |
| Stage-1 sheets | 751 | 751 | yes | P1-08 |
| Strata, sheets / blocks / eligible: F_ZLMS · F_other · N_ZLMS · N_other | 242/34/8,000 · 42/18/575 · 39/18/372 · 428/96/5,670 | the same | yes | P1-08 D06, **§5-guided** (O2) |
| Eligible stage-1 polygons | 14,617 | 14,617 | yes | P1-08 |
| (a) folds | 10–11 blocks, 1,714–1,716 eligible each | 10, 11, 10, 11, 10 blocks; 1,715, 1,716, 1,715, 1,715, 1,714 | yes | P1-08 D09 |
| Stage-2 targets | 465 (278 flysch, 187 non-flysch) | 465 (278, 187) | yes | P1-08, P1-12 |
| Eligible stage-1 polygons with no feasible patch | 870 | **793** | **no** | P1-09 D12a, O9: unexplained (below) |
| Water-filling caps, flysch / non-flysch | 8 / 3 | 8 / 3 | yes | P1-09 D13, **§5-guided** (O8) |
| Positives selected | 3,613 | **3,592** | no (−21) | P1-09 D13: the spike's value lies in our seed range, 3,592–3,652 |
| Positives dropped by validity (patch context touches 0 m cells) | 29 | **26** | no (−3) | P1-09 D13a |
| Positives kept | 3,584 | **3,566** | no (−18, −0.5 %) | all 3,566 matched |

- **Positives, 3,566 vs 3,584.** Water-filling takes each sheet's polygons in a seeded random order. Our streams are
  not the spike's, so the selection differs. Under our reading the total over 21 order seeds ranges from 3,592 to
  3,652, and the spike's 3,613 lies inside that range [E]. The validity drop (26 vs 29) depends on which polygons
  were selected. We read the gap as sampling variation, not a different rule [I].
- **Infeasible polygons, 793 vs 870 (O9).** Our exact patch rule (the 300 m square inside its own sheet, and more
  than 150 m from every stage-2 sheet) leaves 793 of 14,617 polygons without a feasible patch. Three alternative
  rules were fixed in advance and computed: square plus its 56 m channel context inside the sheet, 1,379; a
  pixel-centre test, 780; the stage-2 rule on the raster extent, 971. None gives 870. Feasibility uses no random
  numbers, so chance cannot explain the gap. The literal rule is kept, and the 77-polygon difference (0.5 %) is
  reported as an unexplained discrepancy rather than searched further.
- **The dataset** [E] (`results/sampling_summary.json`, `results/dataset_summary.json`): 11,132 patches. That is
  3,566 positives, 3,566 slope-matched negatives and 4,000 background patches. The negatives are 3,138 random and
  428 look-alikes (cut 187, rock 188, quarry 41, spoil 12). Match levels: 3,473 at level 1, 83 at level 2, 3 at
  level 3 and 7 at level 5; no positive is unmatched. Test sets: (a) 2,205 pairs on 284 sheets; (b) 1,545 pairs on
  465 sheets ((b) training: 2,021 pairs on 281 sheets). `scripts/verify_dataset.py` passes: no patch is missing or
  contains NaN.
- **Stage-2 windows near stage-1 patches (O5).** Appendix A §4 says stage-2 windows never overlap a stage-1 patch.
  For axis-aligned squares near a stage-2 sheet's corner that is not guaranteed (overlaps of up to about 44 × 44 m
  are possible). Count [E]: of 11,132 samples, one (a random negative) overlaps one of the 146,622 stage-2 windows,
  by 4 m × 50 m = 200 m², 0.2 % of the window. No positive overlaps any window. This is reported as a known
  limitation. The rule is not changed.

**Stage-1 baselines before training** [E] (`results/baselines.csv`, `results/baselines.json`). Sheet-level cluster
bootstrap, 2,000 replicates, seed 123, 95 % percentile CI; the block-level CIs are in the CSV. Every logistic
regression fit converged. These numbers are part of the record and changed nothing.

| Scorer | Kind | (a) AUC [95 % CI] | (b) AUC [95 % CI] |
| --- | --- | --- | --- |
| `slope_only` | non-ML baseline | 0.507 [0.504, 0.511] | 0.510 [0.504, 0.517] |
| `susceptibility` | non-ML baseline | 0.540 [0.525, 0.555] | 0.525 [0.511, 0.541] |
| `slope_susc_lr` | classical ML | 0.577 [0.559, 0.595] | 0.516 [0.496, 0.536] |
| `terrain_stats_lr` | classical ML | **0.685 [0.666, 0.704]** | **0.624 [0.604, 0.642]** |
| `terrain_stats_gbm` | classical ML (secondary) | 0.669 [0.652, 0.687] | 0.618 [0.597, 0.639] |
| Spike's best non-CNN | | 0.695 [0.678, 0.713] (terrain-stats LR) | 0.624 [0.603, 0.647] (terrain-stats GBM) |

- Our terrain-stats LR for (a) (0.685) and terrain-stats GBM for (b) (0.618) both lie inside the spike's CIs for the
  same scorers. Our best (b) scorer is the LR (0.624).
- The slope-only AUC is the check that matching worked; about 0.5 is expected. Ours is 0.507 (a) and 0.510 (b), and
  neither sheet CI quite contains 0.5. A likely reason [I] is that matching within 2° bins leaves a small
  within-bin slope difference: the median |Δ slope| of a pair is 0.59° and the 90th percentile 1.41° [E]. This is
  reported as it is and has not been acted on.

**Stage-2 baselines before training** [E] (`results/stage2_baselines.csv`; 20 reserved sheets, 146,622 windows, 465
targets; no window was dropped as invalid). None reaches the 0.30 threshold.

| Scorer | Kind | P@R50 | Max recall | Flysch P@R50 (278 targets) | Non-flysch P@R50 (187) | Ring as FP (post hoc) |
| --- | --- | --- | --- | --- | --- | --- |
| `slope_only` | non-ML baseline | 0.120 | 0.920 | 0.220 | 0.068 | 0.091 |
| `susceptibility` | non-ML baseline | 0.101 | 0.892 | 0.121 | 0.056 | 0.078 |
| `slope_susc_lr` | classical ML | 0.141 | 0.918 | 0.215 | 0.083 | 0.103 |
| `random` | chance | 0.094 | 0.886 | 0.122 | 0.072 | 0.080 |

- The spike reports 0.136 for its best non-CNN (slope + susceptibility LR) and 0.085 for random. It also says every
  baseline stayed ≤ 0.14 and random ≤ 0.085 under all four rule variants. Ours: LR 0.141, 0.005 above the spike's
  value and just above its 0.14 bound; random 0.094, 0.009 above. The spike gives no CI for these numbers, so they
  are reported, not flagged as discrepancies. Random is one seeded draw at chance level, so its precision is close to
  the base rate of TP among TP + FP detections [I].

### 7.2 Where a spike number chose between readings, and Mark's approvals

`docs/PLAN.md` §5 allows a spike number to choose between readings of **genuinely ambiguous** method text, provided
the choice is stated. It was used exactly twice:

1. **Kraj ties (O2, P1-08 D06).** Appendix A §4 defines a sheet's kraj as a majority vote and gives no tie rule. Four
   sheets tie. Our first rule (PLAN D06's default: the larger summed intersection area wins) moved sheets 25-12-13 and
   25-13-15 across the ZLMS/other line and missed all four §5 strata by one sheet. The rule "the kraj whose tied
   polygons include the lowest objectid wins" reproduces all four strata exactly. It is also what a plain
   first-occurrence count over the register in source order gives. It affects 2 sheets with 2 eligible polygons
   each. Config: `replication.splits.kraj_tie_rule` = `lowest_objectid`.
2. **Water-filling cap (O8, P1-09 D13).** "Water-filled across sheets in random order up to a common per-sheet cap"
   (Appendix A §5) fits two readings:
   - take a polygon only while the sheet stays at or under the cap (PLAN D13's default; our first build);
   - take polygons until the sheet reaches the cap, so the last one may overshoot it.

   The first reading gave caps 8 / 4 and 3,779 positives against the spike's 8 / 3 and 3,613. A read-only
   investigation fixed four readings of the cap and four of feasibility before computing. Only "until the sheet
   reaches the cap" gives caps 8 / 3 (under all 21 order seeds tried), overshoots both pool budgets at cap + 1 (as
   the spike's §12 item 11 says), and has 3,613 inside its seed range.

In both cases the text has no rule for the case, and both readings are faithful to it. The spike's numbers decided
which one the spike used. Neither changes a threshold, a seed or a model setting.

**Mark's approvals (2026-11-27, before any training).** Mark approved O2 (kraj ties), O3 (Riley TRI), O5 (keep the
literal stage-2 buffer rule and report the overlap) and O6 (float32 head under bf16 autocast) as recommended. He also
approved the orchestrator running P1-09's sampling build (I2), and, after the investigation of the sampling counts,
O8 (water-filling) the same day.

### 7.3 Incidents

- **I1 · A test run wrote synthetic files into the real data tree (2026-11-26, fixed).**
  - *What happened.* The `LANDSLIDE_DATA_DIR` override also redirected the paths that tests build on a temporary
    root. A full test run with the variable set wrote a synthetic `meta.csv` and `preds_{a,b}.csv` and overwrote
    `polygons.parquet` and `detections.parquet`.
  - *Fix.* The overrides now apply only to the checkout's own root, with a test.
  - *Recovery.* The synthetic files were deleted; no real dataset existed yet. `build_geo.py` and
    `stage2_baselines.py` were re-run: `sheets.parquet` and `splits.json` came out byte-identical, and the tracked
    results were unchanged.
  - *Found by P1-16 on 2026-11-27.* The same test run also left 21 synthetic run directories,
    `data/runs/{a_f0…a_f4,b,s2}_s{0,1,2}/`, dated 2026-11-26 23:00:41. They were written by `tests/eval_synth.py`
    and hold `done.json`, `history.csv` and `preds_*.csv` fixtures: git commit `c0ffee`, config hash `93367585…`, no
    checkpoint and no `norm.json`. They are not training output; no CNN has been trained. But the runner skips any
    unit that has `done.json`, and the evaluation would count these as hash mismatches, so a phase-2 run would be
    INCOMPLETE without training anything. They must be removed before the smoke test; §9 records it.
- **I2 · P1-09's real build was blocked by the permission system.** The P1-09 agent was denied reading the ZABAGED
  cache, a real-data dry run and a `git merge`. The orchestrator did not run the build in its place; it asked Mark.
  On 2026-11-27 Mark approved the orchestrator running `scripts/build_samples.py --check-determinism` from the main
  checkout. Two builds in that run were identical.

### 7.4 Decision register

Unless a row says otherwise, the decision is an implementation detail or a reading that Appendix A does not settle,
and it is not a deviation (§8).

**A. Data and provenance**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| Python 3.13.15, `torch==2.14.1+cu130`, `torchvision==0.29.1+cu130`; every dependency pinned in `uv.lock` | Brief Phase 1 item 1 (the spike's build) | [E] `results/environment.json`: CUDA on the RTX 3060, bf16 supported | P1-01 |
| Hand-over files are read in place from `data/raw_data/` (read-only) and their sha256 checked against `MANIFEST.csv` | The brief allows reading in place; avoids 11 GB of copies | [E] 863 of 863 match; 2 unlisted files (the manifest, the fetch log) checked separately | P1-01 D01, P1-03 D01 |
| Sheets reprojected EPSG:4326 → 5514 with pyproj's default operation, vertex by vertex, no densifying; a sheet's bounds are those of the reprojected polygon | Brief note 1 names no method; PLAN D03 default | [E] PROJ 9.8.1 picks Helmert variants per area of use (824 / 493 / 3 of 1,320 vertices) and reproduces 860 of 860 grids; any single fixed variant reproduces only 47–600 | P1-03 D03 |
| Grid snapping on integer lattice indices (origin 0.4, −0.12, tolerance 10⁻⁹ cell); grids compared as integers | Exact comparison without float tolerances | [E] closest buffered edge 0.08 mm from a lattice line | P1-03 D-new-1 |
| DTM request: note 1 exactly, plus `f=image` (returns the TIFF itself); nothing else sent | The brief names no `f` | [E] 12 of 12 downloads on the recipe grid | P1-03 D-new-2 |
| Provenance sample: 12 sheets (brief: at least 10), `default_rng(SeedSequence([2611, 1]))` | PLAN recommends 12 with a config seed | [E] the 12 sheets are listed in `results/provenance.json` | P1-03 D-new-3 |
| "Georeferencing equal" is checked on the GeoTIFF tags (size, transform, tiepoint, pixel scale, GeoKeys), not on GDAL's CRS object | The download writes an ESRI CRS string that GDAL reads as `LOCAL_CS`; the grid is the same | [E] 12 of 12 equal on every tag | P1-03 D-new-4 |
| DTM values compared as float32 bit patterns; every package NaN must be NoData or out of [−100, 2000] m in the download | Brief note 1 | [E] 0 differing pixels; 0 NaN in all 860 package rasters, so the NaN clause is vacuous | P1-03 D-new-5, D-new-8 |
| Rows 2–4: metadata recorded first, then `returnCountOnly`; full-download scripts written and unit-tested but not run | The brief asks only for the counts of rows 2–4 | [E] 21,356 / 860 / 597 | P1-03 D-new-6, D-new-7 |
| No sheet-grid coordinates in committed results | They outline row-3 sheets (no stated licence) | redaction test | P1-03 D-new-9 |
| Susceptibility: one `export` per sheet and class with note 5's parameters plus `f=image`, on the grid read from the sheet's DTM raster | Brief note 5; grids are read, never recomputed | [E] 1,720 PNGs, every size equal to the DTM's; 78 of 78 `f=json` extents exact | P1-07 D-new-1 |
| Class = alpha ≥ 128; every class pixel must have the class's renderer colour | Note 5 fixes the threshold; the colour check catches a render that ignored `layerDefs` | [E] alpha is antialiased (0/64/128/191/255); 0 pixels of an unexpected colour | P1-07 D-new-2 |
| A pixel in both the mid and the high render gets code 2 | The brief does not define overlaps; higher class is the ordinal maximum | [E] 0.014 % of pixels; at 27 of 27 pixels checked, both polygons exist in the source | P1-07 D-new-3 |
| Service checks before rendering, and point-query inspection on 39 sheets | Catch a symbology or service change | [E] each pattern selects exactly one class; 475 of 475 points agree | P1-07 D-new-5, D-new-6 |
| ZABAGED: per layer and sheet, an envelope query on the sheet's DTM grid (sheet + 250 m), paginated by object id; features deduplicated by `OBJECTID` (duplicates must be identical) | Complete by construction; the envelope covers the 150 m exclusion | [E] 5,148 queries; 0 conflicting duplicates; layers 118, 119, 130, 129 are polygons (130 has no points), 79 and 75 lines | P1-07 D-new-7, D-new-8 |
| ImageNet weights fetched through torchvision with `TORCH_HOME=data/cache/torch`; loading checked with the network blocked | Brief row 8 | [E] sha256 matches the hub prefix and `sources.weights.sha256` | P1-07 D-new-9 |
| Rows 3–5 and anything derived from them stay in `data/`; only aggregates are committed | Brief "Constraints" | tests on the committed summaries | P1-07 D-new-11, P1-08 |

**B. Labels and splits**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| `in_mapped`: the polygon's `representative_point()` lies within the union of all 860 surveyed sheets (partial ones included) | Brief note 2; PLAN D04 | [E] 18,649 = spike; full sheets only would give 18,606 | P1-08 D04 |
| Area = planar area of the delivered geometry in EPSG:5514 | Brief note 2; PLAN D05 | [E] 15,082 eligible = spike (`st_area` field: 15,083) | P1-08 D05 |
| 68 invalid register polygons are repaired (`make_valid`, structure) only for overlays and distances; area and representative point use the delivered geometry | GEOS overlays can fail on invalid input | [E] the repair changes areas by ≤ 0.27 m²; counts unaffected | P1-08 D05a |
| `kraj1` = first comma-separated token of `gen_kraj` | Appendix A §4 "first-listed" | every polygon has one | P1-08 |
| Flysch closing = `buffer(+2000).buffer(−2000)` (round joins, 8 segments per quarter), holes filled, parts < 1 km² dropped; flysch if ≥ 50 % of the sheet inside | Appendix A §3; PLAN D07 default, set before computing | [E] 6,585.6 km², 300 sheets (spike 6,586, 300); the other reading gives 6,525.3 and 299 | P1-08 D07 |
| Per-sheet DTM grid read from each raster; `dtm_all_valid` = no cell that the full invalid rule rejects | Grids are read, never recomputed (P1-03) | [E] 96 sheets with 0 m cells, 764 fully valid | P1-08 D-new-1 |
| Polygon → sheet: largest intersection area; exact ties to the lowest sheet number | Appendix A §4 | [E] all 15,082 assigned; 34 have < 50 % in their sheet | P1-08 D-new-2 |
| Sheet kraj: majority of the `kraj1` of every register polygon intersecting the sheet; **ties to the kraj whose polygons include the lowest objectid** | Appendix A §4 has no tie rule; **§5-guided** (§7.2) | [E] 4 tie sheets, 2 change stratum; all four §5 strata reproduced | P1-08 D06, O2 |
| Transfer group "train" if kraj is Zlínský or Moravskoslezský; stratum = flysch × group, set for every sheet with a kraj | Appendix A §4 | 797 sheets with a kraj | P1-08 |
| Blocks: cell (⌊cx/15,000⌋, ⌊cy/15,000⌋) of the sheet centroid from the EPSG:5514 origin, nested in the stratum; opaque ids in committed files | Appendix A §4; PLAN D08 | [E] 166 stage-1 blocks; strata 34 / 18 / 18 / 96 = spike | P1-08 D08 |
| Stage-2 sheets: the brief's 20, checked (full, ≥ 5 eligible, fully valid DTM, flysch flag); never swapped | Brief "Splits" | [E] all 20 pass; 278 + 187 targets | P1-08 |
| (a) folds: flysch blocks sorted by id, permuted with `default_rng(11)`, stable-sorted by eligible count (descending), each to the fold with fewest polygons (ties to the lowest index) | Appendix A §4; PLAN D09 | [E] 1,714–1,716 per fold = spike's range | P1-08 D09 |
| Inner validation: training blocks sorted, permuted with `default_rng(1000 + k / 2000 / 3000)`, added until ≥ 15 % of the run's training eligible polygons (exact rational comparison), **for s2 too** | Appendix A §4 states this rule and names the s2 seed; Appendix B's s2 prose says "15 % of blocks" (see §8) | [E] shares 0.154–0.215 | P1-08 D10 |

**C. Sampling and patches**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| Patch centred on pixel corner (r, c), rows r−75 … r+74; patch rule: square `covered_by` its own sheet and **more than** 150 m (`replication.sampling.stage2_buffer_m`) from every stage-2 sheet | Appendix A §4; PLAN D11; "touches the buffer" read as distance ≤ 150 m | [E] every square ≥ 150.13 m from the stage-2 sheets | P1-09 D11 |
| Every geometric screen is exact (polygon erosion by the square, scanline on the corner lattice, exact distances); the selected samples are re-tested with shapely | No raster approximation needed | [E, synthetic] equal to brute force | P1-09 D11a |
| Positive points: corners strictly inside the polygon that pass the patch rule; k = clip(⌊area/40,000⌋, 1, 3), drawn greedily at random ≥ 100 m apart, fewer if there is no room | Appendix A §5; PLAN D12 | [E] 16 polygons got fewer than k | P1-09 D12 |
| A polygon is infeasible when no interior corner passes the patch rule; validity is a later filter | Appendix A §12 item 12 defines the 870 by the patch rule | [E] **793 vs 870, unexplained** (§7.1) | P1-09 D12a, O9 |
| Water-filling: budget split by eligible-polygon share (flysch 2,347, non-flysch 1,653); per sheet, polygons in seeded random order, taken whole **until the sheet reaches the cap**; the largest cap whose pool total ≤ budget | Appendix A §5; **§5-guided** (§7.2) | [E] caps 8 / 3 = spike; cap + 1 overshoots both budgets | P1-09 D13, O8 |
| After selection, a positive whose patch plus 28 px context holds an invalid cell is dropped; no top-up | Appendix A §5 "Storage", §12 item 11 | [E] 26 dropped (spike 29); 3,566 kept | P1-09 D13a |
| Negative candidates: 60 m grid anchored at the raster origin, jittered U(−15, 15) m per axis and snapped to a corner; then patch rule, validity, ≥ 150 m from every register polygon and every ZABAGED 129 feature | Appendix A §5; PLAN D14 | [E] 2,360,439 random candidates | P1-09 D14 |
| Distances are exact GEOS distances; "≥ 150 m" allows exactly 150 m | Appendix A §5 | [E] closest negative 150.01 m from the register | P1-09 D14a |
| Look-alikes: each ZABAGED area feature belongs to the sheet with the largest share of it; ≤ 2 random interior corners per feature; cut points every 100 m along each road or railway feature, kept if the p90 slope in a 21 px window is ≥ 20°; quota = round(0.12 × kept positives) | Appendix A §5; PLAN D15 | [E] 428 matched; layer 130 holds polygons, not points | P1-09 D15 |
| Matching level by level (1 → 6), seeded random order within a level; look-alikes first, round-robin over types, exact bin only (levels 1, 2, 5); a chosen negative must be ≥ 150 m from every negative chosen so far | Appendix A §5; PLAN D15, D16 | [E] 0 unmatched positives | P1-09 D16 |
| Background: unused random candidates in seeded random order, each ≥ 150 m from the selected negatives and from the background already taken, up to 4,000 | Appendix A §5; PLAN D17 | [E] 4,000 from 2,300,221 | P1-09 D17 |
| Per-sheet random streams (`SeedSequence([11, stream], spawn_key=(sheet code,))`) | A sheet's draws do not depend on the other sheets | [E, synthetic] removing a sheet changes no other sheet's draws | P1-09 D02a |
| Stage-1 patches that overlap a stage-2 window are counted, not removed | Keep Appendix A §4's literal rule (O5, approved) | [E] 1 sample, 200 m² | O5 |
| Patch statistics in float64 over the 150 × 150 patch, before the float16 cast; std with ddof 0; percentiles with numpy's linear rule | PLAN D19 | [E, synthetic] equal to written-out formulas | P1-10 D19 |
| TRI = Riley et al. (1999): per pixel √Σ(z_n − z)² over the 8 neighbours, averaged over the patch | Appendix A §10 says only "TRI"; PLAN D19's label was Riley, its formula Wilson's (O3, approved) | [E, synthetic] plane and brute-force tests | P1-10 D19a, O3 |
| Susceptibility statistics over the patch footprint: mean class code and share of code 2 | PLAN D19 | [E] class rasters on the DTM grid | P1-10 D19b |
| `meta.csv` keeps P1-09's stored `slope_mean` and `slope_bin` (recomputed mean must agree within 1e-9°) | The bin the matching used must not flip | [E] 0 bin flips | P1-10 D19c |
| Extraction asserts NaN-free channels and \|v\| ≤ 65,504 before the float16 cast; one DTM window read per sheet | Appendix A §5 "Storage"; brief Phase 1 item 3 | [E] verification passes | P1-10 |
| `meta.csv` adds `source_layer` and `source_objectid` to PLAN §4.3's columns | Trace each look-alike to its ZABAGED feature | | P1-09, P1-10 |
| `verify_dataset.py` re-extracts a seeded sample (stream `verify_sample` = 11) and requires bit-identical patches | Check meta ↔ patch alignment independently of the build | [E] passes | P1-10 |

**D. Channels**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| Float64; slope by central differences; LRM = z − `uniform_filter` (11, 51 px); curvatures from a Gaussian-smoothed DTM (σ 1, truncate 4) with Zevenbergen–Thorne derivatives and PLAN D18's formulas, 0 where p² + q² < 10⁻¹²; SVF over 16 directions, r = 1…20 px, bilinear, observer on the ground; detr = closed-form least-squares plane per window | Appendix A §6; PLAN D18 | [E, synthetic] planes and paraboloids to 1e-9 | P1-04 D18 |
| A channel is finite only where its whole footprint square is inside the array and valid | Strict, deterministic and identical for patches and full rasters | [E, synthetic] exact NaN masks | P1-04 D18a |
| Footprint radii (max 25 px) are checked against the 28 px context in code | Appendix A §6 | test | P1-04 D18b |
| One code path for patches and full rasters (tiles of 256 px with a 28 px halo) | Card: "patches and stage 2 use one code path" | [E] patch vs full ≤ 5.1e-12 (Appendix A bound 5e-9) | P1-04 D18c |
| SVF directions and the Gaussian built to be exactly closed under the 8 grid symmetries | Augmentation and TTA rely on D4 equivariance | [E] max error 1.5e-12 over the 8 transforms | P1-04 D18d |
| Method values without an Appendix B key (`context_px` 28, `curv_truncate` 4, `curv_flat_eps` 1e-12, `svf_r_min_px` 1) in `replication.channels` | PLAN §3 rule 6 | | P1-04 D18e |
| The invalid rule is parsed strictly from `dtm.invalid_rule` (z ≤ 0 or z > 2000 → NaN) | Appendix B | [E, synthetic] 8 of 8 invalid cells found | P1-04 D18f |

**E. Model and training**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| LR set per step: linear warm-up to 3e-4 over 2 epochs (first step base/W), then cosine to exactly 0 at the last step of epoch 40; early stopping truncates, never rescales | Appendix B `schedule`; PLAN D20 | [E, synthetic] boundary values | P1-06 D20 |
| `drop_last`: ⌊n/64⌋ whole batches of a fresh permutation per epoch | PLAN D20; every BatchNorm batch is 64 | | P1-06 D20 |
| Normalisation: per-channel mean and std (ddof 0) in float64 over the training rows only (inner-val excluded), from the float16 values | Appendix A §7; PLAN D20 | [E, synthetic] | P1-06 D20 |
| Early stopping on inner-val AUC (8-view TTA), patience 8; only a strictly higher AUC improves (ties keep the earliest epoch); best weights restored | Appendix A §7; PLAN D20 | [E, synthetic] scripted AUCs | P1-06 D20 |
| Rows per run: (a) fold k tests on fold k, trains on the other flysch folds; (b) trains on group "train", tests on "test"; s2 trains on everything, background included; inner-val = rows in the run's inner-val blocks (background too, for s2) | Appendix A §4, §7 | [E, synthetic] | P1-06 |
| Per unit `random`, numpy and torch seeded with the seed; data order and per-sample D4 index from `default_rng(seed)` on the CPU | Appendix A §7 seeds; CPU draws keep the order independent of the GPU | | P1-06 |
| `cudnn.deterministic`, `benchmark=False`, `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Appendix A §7 "cuDNN is deterministic" | [E, synthetic] two runs bit-identical | P1-06 |
| bf16 autocast, **except the 512 → 1 head, which runs in float32** | A bf16 logit has 8 significant bits and would quantise every score (O6, approved; see §8) | | P1-06, O6 |
| resnet18 built without weights, then the local IMAGENET1K_V1 file loaded strictly; conv1 = mean RGB filter × 3/7 over 7 channels; fc = `Linear(512, 1)` | Appendix A §7 | [E] tests of the formula | P1-06 |
| AdamW with torch defaults otherwise (betas 0.9/0.999, eps 1e-8); weight decay on every parameter | Appendix A §7 names no exclusions | | P1-06 |
| `history.csv` fields as Appendix A §7 "Monitoring"; AUCs via sklearn | Appendix A §7 | | P1-06 |
| TTA: mean of sigmoid over the 8 views, chunks of 64 samples, bf16 autocast as in training | Appendix A §7 | [E, synthetic] | P1-06 |
| Smoke mode trains the first 2 epochs of the real 40-epoch schedule | Appendix B `smoke.max_epochs`; tests the real code path | | P1-06 |

**F. Baselines**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| Five scorers (`replication.baselines.scorers`): patch-mean slope; patch-mean susceptibility class; LR on (slope mean, slope std, susc mean, susc high fraction); LR on the 12 terrain statistics; HistGradientBoosting on the same 12; all "higher = more landslide-like", never sign-flipped | Appendix A §10 | | P1-11 |
| Fitted scorers fit on train + inner-val of the CNN's run and predict exactly the CNN's test rows; (a) pooled out of fold | Appendix A §10 "same training blocks (train + inner-val)" | [E, synthetic] spy tests | P1-11 |
| sklearn defaults (`StandardScaler` + `LogisticRegression` lbfgs, `max_iter` 100; HGB defaults), `random_state` = 2611; a non-converged fit is kept and reported | Appendix A §10, no tuning (O4) | [E] every LR fit converged | P1-11, O4 |
| AUC with sheet (primary) and block (secondary, `replication.eval.secondary_bootstrap_unit`) CIs on draws shared by all scorers; breakdowns per fold, (b) by kraj and flysch flag, per negative source; the slope-only AUC as the matching check | Appendix A §8, §10 | (§7.1) | P1-11 |
| The spike's numbers are stored next to ours as reference only | PLAN P1-11: "without acting on them" | | P1-11 |

**G. Stage 2**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| Window centres at raster corners with r ≡ c ≡ 0 (mod 25), kept if inside the sheet polygon (boundary included) | Appendix A §9; PLAN D22 | [E] 146,622 windows (strictly inside gives the same) | P1-12 D22 |
| A window whose patch plus context is not fully valid is dropped from the one lattice all scorers share | The CNN needs finite channels; all scorers must score the same windows | [E] 0 dropped on the 20 sheets | P1-12 D22a |
| Window statistics from summed-area tables, with the patches' definitions | Appendix A §9; PLAN D19 | [E] equal to brute force | P1-12 D22b |
| NMS: greedy per sheet in descending score + 10⁻⁹ × U[0, 1) jitter; suppresses at distance ≤ 150 m, tested on integer pixel offsets | Appendix A §9; PLAN D23 | [E] no ties left after the jitter | P1-12 D23 |
| Jitter and random scores from per-sheet streams (seed 2611, streams 9 and 10) | The smoke sheet's draws equal the full run's | | P1-12 D23a |
| Matching pooled over sheets in descending jittered score; TP within 50 m of an unmatched target (lowest objectid if several); duplicate within 50 m of matched targets only; ignored within 150 m of any of the 21,356 polygons; else FP; exact distances | Appendix A §9; PLAN D23. "Inside an already-matched target" read as inside its 50 m buffer, so a duplicate is exactly a TP whose target was taken; this changes only the post hoc ring variant | [E] a brute-force re-simulation agrees for all 33,113 detections of the three unfitted scorers | P1-12 D23b |
| P@R50, interpolated precision and max recall via one sweep; per stratum by re-running the matching on the stratum; ring as FP = ignored → FP | Appendix A §9 | [E] strata equal the pooled outcomes | P1-12 |
| `slope_susc_lr` comparator fitted on all stage-1 positives vs the background sample | Appendix A §9 | [E] P@R50 0.141 | P1-12 |
| CNN scores: channels from the full raster, detr per window on the CPU in float64, cast to float16 as the training patches were, the s2 run's normalisation, 8-view TTA, mean over 3 seeds; a missing seed makes stage 2 INCOMPLETE | Appendix A §7, §9; the training patches' detr was made by the same function (see §8) | [E, synthetic] windows equal `patch_channels` to 5e-9 | P1-12 |

**H. Evaluation and verdict**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| ROC AUC = rank (Mann–Whitney) AUC with ties counted ½ | Appendix A §8 | [E] equal to sklearn to 1e-12 | P1-05 |
| Cluster bootstrap: labels sorted, `default_rng(123).integers(0, K, (2000, K))` once per test set and unit, shared by every scorer | Appendix A §8 (same resamples for paired differences) | [E, synthetic] | P1-05 |
| A one-class replicate is dropped and counted | PLAN D21 | counts reported | P1-05 D21 |
| Point estimate on the full test set; CI = numpy `linear` quantiles at 0.025 and 0.975 | Appendix A §8 "percentile" | | P1-05 |
| Seed ensemble = mean of probabilities, aligned on id; seeds must cover the same ids | Appendix A §7 | | P1-05, P1-13 |
| Same-sheet pair accuracy counts pairs whose members share a sheet (match levels 1 and 3); ties ½; level 1 only and all pairs also reported | Post hoc check, not defined in Appendix A | | P1-05, P1-13 |
| PR sweep: thresholds at the distinct scores; ignored and duplicate detections add to neither count; P@R50 = precision at the highest threshold with recall ≥ 0.5; not reached = FAIL | Appendix A §9 | [E, synthetic] hand-computed | P1-05 |
| Verdict: any completed test below its threshold → NOT REPLICATED; all three PASS → REPLICATED; else INCOMPLETE; equality passes | Brief criteria; Appendix A §1; PLAN D25 | [E] 36-case truth table | P1-13 D25 |
| Borderline: ci_low < threshold ≤ ci_high; never changes the verdict | Appendix A §1 | | P1-13 |
| A unit counts only if `done.json` carries the frozen config hash and its outputs are complete; evaluation refuses to run if the config differs from `configs/landslide.sha256` | Brief phase 3 | [E, synthetic] | P1-13 |
| A test is complete only if its pooled predictions cover exactly its test set | "A test with any missing run is INCOMPLETE" | | P1-13 |
| Stage-2 sheet bootstrap (post hoc): a TP counts on its target's sheet; **a replicate that never reaches 50 % recall counts as precision 0** (`replication.eval.stage2_bootstrap_not_reached`) | Dropping such replicates would bias the CI upwards and break pairing | [E, synthetic] | P1-13 |
| Discrepancy = our point estimate outside the spike's 95 % CI (bounds count as inside); claims without a CI are compared but not flagged | Brief criteria | | P1-13 |
| `metrics.json`: fixed schema, sorted keys, 6 decimals, no timestamps, training commit from `done.json`, so a re-run is byte-identical | PLAN P3-01 | [E, synthetic] | P1-13 |

**I. Runner**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| Smoke subset: 5 % of the pair ids (whole pairs) plus 5 % of the background, rounded half up, drawn from stream `smoke_subset` (8) with seed 2611 | PLAN D24 | [E, synthetic] | P1-14 D24 |
| Smoke: units a_f0, b, s2 × seed 0, 2 epochs; stage 2 on sheet 02-41-01 whatever (a) and (b) give, with its four baselines recomputed on that sheet; everything under `data/smoke/` | Appendix B `smoke` | [E, synthetic] nothing written outside `data/smoke/` | P1-14 |
| Preflight refuses to train unless `configs/landslide.sha256` exists, equals the config's hash, and both files are committed; it also checks the hash recorded by every built artefact | Gate G3 in code | [E, synthetic] | P1-14 |
| Preflight (< 60 s): required files and recorded sizes, the `patches.npy` header, the reserved sheets' rasters, ≥ 10 GiB free disk, CUDA with ≥ 7 GiB free GPU memory (`replication.runner.min_free_gpu_gb`; GB read as GiB, the stricter reading), `num_workers` = 0 | Appendix A §11; brief "Fail-fast" | [E] exits in 5.8 s on an empty root | P1-14 |
| After the preflight, every socket connection or name lookup to a non-loopback address raises | Appendix A §11 (see §8 for loopback) | [E, synthetic] | P1-14 |
| `SetThreadExecutionState(ES_CONTINUOUS \| ES_SYSTEM_REQUIRED)`, released in `finally` | Brief "Awake" | [E] | P1-14 |
| Order: a_f0…a_f4 × seeds, b × seeds, evaluate; only if (a) and (b) pass: s2 × seeds, stage-2 CNN inference, final evaluation. The stage-2 baselines are phase-1 outputs, not recomputed | Appendix A §1, §9, §11 | [E, synthetic] call sequence | P1-14 |
| Resume: a unit with `done.json` is skipped and never overwritten; a failed or interrupted unit is retrained once per invocation | Appendix A §11; brief "Resumable" | [E, synthetic] | P1-14 |
| An exception in a unit is logged and the runner moves on; missing units never pass | Appendix A §11 | [E, synthetic] | P1-14 |
| Wall-clock cap 6 h from the runner's start; from cap − 15 min (`replication.runner.eval_reserve_min`) no unit starts and a running unit stops at its next epoch boundary | Appendix A §11 | [E, synthetic] | P1-14 |
| Log under `logs/`; tracked `results/run_summary.json` with units, hashes, freeze commit, verdict, missing units and runtime measurements | Brief "Logged" | | P1-14 |

**J. Infrastructure**

| Decision | Why | Evidence | Detail |
| --- | --- | --- | --- |
| Config = every Appendix B key and value unchanged (except `name`, `status`, `paths`) plus two blocks of our own: `replication` (our fixed choices) and `sources` (source facts from the brief: URLs, layer ids, grid origin, valid range, expected counts, render parameters, the weights' hash) | Brief "Same method and values" | `tests/test_config_appendix_b.py` checks every Appendix B leaf and that no Appendix B block has an extra key | P1-02, P1-16 |
| Config hash = sha256 of the canonical JSON without `status` | Freezing must not change the hash recorded by files built before the freeze | | P1-01 D02a, O1 |
| Every config key that tasks requested was added in one batch at P1-16, with every value equal to the code's former constant | One hash change, then one rebuild of every artefact | the orchestrator rebuilds and compares contents | O7, P1-16 |
| Random numbers: `numpy.random.default_rng` only; `default_rng(seed)` where the method names the seed, otherwise `SeedSequence([seed, stream id])` with ids registered in `replication.streams`; base seed 2611 fixed before any draw | PLAN D02 | | P1-01 D02, P1-02 |
| One shared `data/` tree for all worktrees (`LANDSLIDE_DATA_DIR`); since I1 the override never applies to test roots | Worktrees have no data of their own | | P1-01 D01a, I1 |
| Network client: fixed User-Agent, ≥ 0.3 s between requests, retries with backoff, `outSR` refused, only the two service hosts | Brief "Requests" | | P1-01 |
| D4 transform k = rot90(flip_lr(x) if k ≥ 4 else x, k mod 4); numpy and torch identical | | tested | P1-01 |
| Smoke mode redirects outputs only; inputs are shared and read-only | The smoke test can never write tracked results | | P1-01 |
| Code licence MIT; data keep their own licences | | | P1-01 |

## 8. Deviations

**There are no deviations from Appendix A or Appendix B.** Every Appendix B key and value is in the frozen config
unchanged (checked by `tests/test_config_appendix_b.py`). No source changed since 2026-11-25 (§7.1). The spike's
torch build was installable, so no environment deviation was needed. Every decision in §7 either fills a gap the
method text leaves open or picks one reading of text that allows more than one.

**The two §5-guided choices are readings, not deviations.**
- *Kraj ties (O2).* Appendix A §4 defines the kraj by majority and is silent on ties. Any tie rule is an addition,
  and the lowest-objectid rule is one of them. The spike's strata chose which one; the majority rule itself is
  unchanged.
- *Water-filling (O8).* "Up to a common per-sheet cap" can mean "never above the cap" or "until the cap is reached".
  We use the second because the spike's caps and counts show it is the one the spike used. The budget, the pools,
  the random order and the cap search are as Appendix A §5 and §12 describe them.

**The closest calls.** These are the places where our implementation is furthest from a literal word of the text.
They are listed so that a reviewer can disagree:

| Text | What we do | Why it is not a deviation |
| --- | --- | --- |
| Appendix B `train.amp`: "bf16 autocast" | The final 512 → 1 layer runs in float32 inside the autocast region (O6, approved by Mark) | The network still trains and predicts under bf16 autocast, and autocast already keeps some ops in float32. Only the last dot product is not rounded to 8 significant bits, which would otherwise create ties in every AUC, ensemble and NMS ranking. |
| Appendix B `stage2.model`: "15 % of blocks as inner validation" | The s2 inner validation uses Appendix A §4's rule: blocks until ≥ 15 % of the run's training eligible polygons, seed 3000 | Appendix A §4 states this rule for "the stage-2 model" and gives its seed; Appendix B's prose summarises it. Using one rule for every run follows the more specific text. |
| Appendix A §9: "the plane is removed per window on the GPU" | The plane is removed on the CPU in float64 with the function that made the training patches' `detr` | Same computation; only the device differs. The stage-2 windows then equal a training patch at the same corner to 5e-9. |
| Appendix A §11: "Socket connections are then disabled for the rest of the run" | Connections and name lookups to any non-loopback address raise; loopback stays allowed | Loopback cannot leave the machine, and Python's own `socketpair` on Windows needs it. The run is still offline. |
| Appendix A §5: background "drawn uniformly" | Uniform random order over the unused random candidates, keeping each draw ≥ 150 m from the selected negatives and the background already taken (PLAN D17) | The ≥ 150 m spacing applies Appendix A §5's spacing of selected negatives to the background, so that no two patches nearly coincide. Without it the draw is the same uniform draw. |
| Appendix A §11: "≥ 7 GB free GPU memory" | 7 GiB (2³⁰ bytes) | The stricter of the two readings. |

**Discrepancies are not deviations.** The differences from the spike's numbers in §7.1 (793 vs 870 infeasible
polygons, 3,566 vs 3,584 positives, the stage-2 random and LR baselines) come from running the same method on our own
implementation. They are reported here and in the phase-1 report, and phase 3 relates them to any discrepancy in
the results.

## 9. Freeze record

- **Freeze date:** 2026-11-27.
- **Config hash** (`configs/landslide.sha256`, sha256 of the canonical config without `status`):
  `1ae1a9b787d8cb2567c265ae8812d28dc913c80c888edfc8103740f55d895adb`.
- **Freeze commit:** the commit on `phase-1` that adds this record, titled "Freeze pre-registration". (A commit can't
  contain its own id; the runner records it in `results/run_summary.json`, and the phase-1 report quotes it.)
- **Artefacts rebuilt under the frozen config** (commit "Rebuild every artefact under the final config", just before
  this one): the geo build, sampling (built twice in one run, identical), patches, the dataset check (16 of 16
  checks), the stage-1 baselines (computed twice, identical) and the stage-2 baselines. Compared with the build made
  before the last config keys were added, every file's content is unchanged: `sheets.parquet`, `polygons.parquet`,
  `splits.json`, `samples.parquet`, `meta.csv`, `patches.npy` (same sha256), the baseline predictions, the stage-2
  detections and all 80 stage-2 score files. Only fields that record the config hash or commit differ (including the
  20 stage-2 lattice files, which store the hash).
- **Run directories:** the 21 synthetic run directories of incident I1 were removed from `data/runs/`; no run
  directory exists at the freeze.
- **No CNN had seen a real patch before this commit.** No model checkpoint exists anywhere under `data/` (the only
  `.pth` file is the cached ImageNet weights), there is no `data/runs/` or `data/smoke/` directory, every test of the
  training, prediction and stage-2 CNN code ran on random tensors or toy rasters, and the order of commits on
  `phase-1` shows this freeze before the smoke test.
