# Replication report: does the landslide CNN result hold up?

**Verdict: REPLICATED.** All three pre-registered tests pass on their point estimates, none is borderline, and every
result falls inside the spike's 95 % confidence interval. [E]

This repo rebuilt the spike's method from its written description (`docs/BRIEF.md` Appendix A), rebuilt every input
from public sources, and trained a new model. The method and every threshold were frozen before any training
(`PREREGISTRATION.md`, freeze commit `1b8cdd4`, config hash `1ae1a9b7…`). Tags: [V] verified, [E] from our run,
[I] inference.

## 1. The run

- **Integrity [V].** 21 of 21 units finished (5 folds × 3 seeds for (a), 3 seeds for (b), 3 seeds for the stage-2
  model), each with the frozen config hash and training commit `dba23cf` (after the freeze, clean working tree). No
  unit failed, was interrupted or was missing, and the run needed no restart.
- **Reproducible [V].** Re-running `scripts/evaluate.py` from the saved predictions gives a byte-identical
  `results/metrics.json`, `metrics_table.md` and learning-curve figure.
- **Time [E].** 45 min on the RTX 3060 (the spike: 50 min). Every unit stopped early (best epochs 3–26 of 11–34
  run). Peak GPU memory 4.7 GB.

## 2. The three tests

| Test | Threshold | Ours | Sheet 95 % CI | Block 95 % CI | Spike | Spike 95 % CI | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (a) Held-out flysch sheets, ROC AUC (5 folds, pooled) | ≥ 0.75 | **0.800** | [0.786, 0.816] | [0.788, 0.814] | 0.811 | [0.796, 0.826] | PASS |
| (b) Train Zlínský + Moravskoslezský, test the other kraje, ROC AUC | ≥ 0.65 | **0.758** | [0.739, 0.777] | [0.732, 0.784] | 0.749 | [0.727, 0.770] | PASS |
| Stage 2: 20 reserved sheets, precision at 50 % object recall | ≥ 0.30 | **0.581** | [0.393, 0.838]† | – | 0.559 | [0.407, 0.812]† | PASS |

† Post hoc sheet-level bootstrap, not pre-registered (as in the spike). A resample that never reaches 50 % recall
counts as precision 0; none did.

Test sizes [E]: (a) 4,410 patches (2,205 slope-matched pairs) on 284 sheets in 52 blocks; (b) 3,090 patches (1,545
pairs) on 465 sheets in 114 blocks; stage 2: 146,622 windows, 465 mapped landslides on 20 sheets.

## 3. Against the baselines

The CNN beats every non-ML baseline and every classical-ML comparator by a wide margin. Differences are paired, on
the same bootstrap resamples. [E]

| | (a) AUC | (b) AUC | Stage-2 P@R50 |
| --- | --- | --- | --- |
| **CNN** | **0.800** | **0.758** | **0.581** |
| Slope only (matching check, ≈ 0.5 expected) | 0.507 | 0.510 | 0.120 |
| Susceptibility map | 0.540 | 0.525 | 0.101 |
| Slope + susceptibility LR | 0.577 | 0.516 | 0.141 (spike 0.136) |
| Terrain statistics LR | 0.685 (spike 0.695) | 0.624 | – |
| Terrain statistics GBM | 0.669 | 0.618 (spike 0.624) | – |
| Random | – | – | 0.094 (spike 0.085) |
| CNN − best comparator [95 % CI] | +0.115 [0.099, 0.132] | +0.135 [0.110, 0.159] | +0.440 [0.126, 0.745] |

The slope-only AUCs show the slope matching worked: slope alone barely separates positives from negatives (their CIs
just exclude 0.5, a tiny residual within the 2° bins).

## 4. Spread

| | Per fold (seed ensemble) | Per seed | Ensemble |
| --- | --- | --- | --- |
| (a) | 0.818, 0.784, 0.785, 0.827, 0.814 | 0.773, 0.791, 0.795 | 0.800 |
| (b) | – | 0.745, 0.749, 0.727 | 0.758 |

Every single fold and every single seed passes (a)'s threshold; every seed passes (b)'s. The three-seed ensemble adds
about 0.01–0.02 AUC over a single seed. Stage 2 uses only the ensemble. [E]

## 5. Secondary results

| Check | Ours | 95 % CI | Spike |
| --- | --- | --- | --- |
| (a) same-sheet pair accuracy | 0.814 | [0.795, 0.833] | 0.824 |
| (b) same-sheet pair accuracy | 0.758 | [0.734, 0.782] | 0.762 |
| Stage 2 with the 150 m ring counted as false positives | 0.415 | – | 0.418 |
| Stage 2, flysch / non-flysch sheets | 0.589 / 0.560 | – | – |
| Stage 2, maximum object recall | 0.920 | – | – |
| (b) flysch sheets (42) / non-flysch sheets (423) | 0.804 / 0.747 | [0.758, 0.850] / [0.726, 0.769] | – |
| (b) non-flysch Jihomoravský | 0.708 | [0.657, 0.756] | 0.622 |

- **By kraj [E]**, test (b), with positives: Ústecký 0.815 (337), Jihomoravský 0.755 (570), Liberecký 0.759 (112),
  Pardubický 0.782 (91), Středočeský 0.748 (85), Královéhradecký 0.727 (180), Olomoucký 0.698 (102), Karlovarský
  0.849 (34), Vysočina 0.577 (28); Plzeňský and Praha have 3 positives each, too few to read.
- **By negative type [E]**, all positives against each kind of negative: random terrain (a) 0.803, (b) 0.761; road
  and rail cuts (a) 0.807, (b) 0.760; rock outcrops (a) 0.695, (b) 0.715; quarries (a) 0.570 (13 negatives),
  (b) 0.898 (29). Rock outcrops are the hardest look-alike; the quarry and spoil-heap numbers rest on too few
  negatives to interpret.

## 6. Discrepancies and differences

**No discrepancy as pre-registered:** none of our point estimates falls outside the spike's 95 % CI (the three
tests, the terrain-statistics LR for (a) and the GBM for (b)). [E]

Differences worth noting, where the spike gives no CI or where the inputs differ:

1. **Non-flysch Jihomoravský, (b): 0.708 vs the spike's 0.622.** The spike's value lies below our CI
   [0.657, 0.756]. It isn't a formal discrepancy, since the spike gives no CI, and the subset is small (325
   positives on 121 sheets), so two independent samples of it can differ by this much [I]. Our dataset also differs
   slightly from the spike's (next point). Either way, the conclusion is the same as the spike's: the model transfers
   less well to non-flysch terrain (0.747) than to flysch (0.804).
2. **The dataset is close, not identical.** 793 eligible polygons have no feasible patch against the spike's 870
   (unexplained; `PREREGISTRATION.md` §7.1, O9), and 3,566 positives were kept against 3,584. The folds and draws are
   necessarily our own. These differences are small and in no direction we can see that would favour the CNN [I].
3. **Stage-2 comparators:** random 0.094 vs 0.085 and the LR 0.141 vs 0.136, both from our own sampling of the same
   20 sheets. The spike gives no CIs; ours are wide (§3). The CNN's margin over them doesn't depend on these
   differences.

## 7. What this does and doesn't show

- **Shown [E].** A CNN trained on 300 m patches of the DMR 5G terrain agrees with the ČGS register's mapped
  landslides on map sheets and regions it never saw, far better than slope, the official susceptibility map or
  simple terrain statistics, once slope is controlled. On whole reserved sheets, at the threshold where it finds half
  the mapped landslides, 58 % of its counted detections hit a mapped landslide (detections in the 150 m ring around a
  mapped polygon aren't counted, by the pre-registered rule; counting them as false alarms gives 42 %).
- **Circularity.** ČGS mapped these landslides on the same LiDAR hillshade the model sees. The model learns to agree
  with the mappers, which is not the same as finding field-verified landslides. Unmapped landslides among the
  negatives, and engineered-slope failures among the positives outside the flysch, would both lower the measured
  scores.
- **Spatial autocorrelation.** The folds are 15 km blocks, but landslide-prone terrain clusters regionally. The block
  CIs are close to the sheet CIs for (a) and somewhat wider for (b), and both stay well above the thresholds.
- **Stage-2 uncertainty.** 20 sheets give a wide CI (0.39–0.84). Its lower end is still above 0.30 and far above every
  comparator, but the point estimate is less certain than (a) and (b).
- **Calibration and overfitting.** The learning curves (`figures/learning_curves.png`) show training AUC rising past
  0.95 while validation AUC levels off at about 0.78–0.85 within 3–8 epochs and validation loss rises. Early
  stopping keeps the best epoch, so the scores above are sound, but the model's probabilities are overconfident:
  they rank well but shouldn't be read as calibrated probabilities. This also marks clear room to improve (more
  data, regularisation), which is the next phase (`docs/PLAN.md` §10a).

## 8. Files

`results/metrics.json` (every number here), `results/metrics_table.md`, `results/run_summary.json`,
`figures/learning_curves.png`; the baselines in `results/baselines.*` and `results/stage2_baselines.csv`. The run's
checkpoints, predictions and detections stay in the gitignored `data/`.
