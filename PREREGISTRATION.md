# Pre-registration: independent replication of the landslide kill test

**Status: DRAFT.** Not frozen. P1-16 fills the "Decisions" and "Deviations" sections from `docs/decisions/`, sets the
config's `status` to FROZEN and records `configs/landslide.sha256`, all in one commit before any CNN training (the
smoke test included). After that commit, nothing in this file, the method or the config changes.

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
- **Our own fixed choices** where Appendix A/B is silent are in the config's `replication` block (base seed, named
  random streams, provenance sample size, negative jitter, the NMS tie jitter, the verdict words) and in §7 below.
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

*Filled by P1-16 from `docs/decisions/*.md` at the freeze: each decision with its reason, owning task and evidence,
including the comparison of our counts with the spike's (`docs/PLAN.md` §5).*

## 8. Deviations

*Filled by P1-16 at the freeze: every departure from Appendix A or B, with its reason. Deviations are allowed only
before any training. A source that changed since 2026-11-25 is reported here.*

## 9. Freeze record

*Filled by P1-16: the freeze date, the config hash (`configs/landslide.sha256`) and a statement that no CNN had seen
a real patch before this commit.*
