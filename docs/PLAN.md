# Implementation plan: landslide-lidar (independent replication, then demo)

Written 2026-11-26 from `docs/BRIEF.md` and `..\site-contract-landslide-lidar.md`. Nothing has been run yet.

The brief and the raw data were moved into this repo on 2026-11-26. Where the brief says
`…\landslide-demo-handoff\raw_data\` or `raw_data/`, read `data/raw_data/` (gitignored).
If this plan and the brief disagree, the brief wins. On anything technical about the site, the site contract wins.

---

## 0. How to use this plan

- **One task card = one agent.** Give each agent:
  - §3 (rules for every agent);
  - §4 (repo layout and shared interfaces);
  - the decision-register rows it owns (§6);
  - its task card.

  Every agent also reads `docs/BRIEF.md` in full first. Demo tasks also read the site contract.
- **Gates (■) are hard stops.** No task after a gate starts until the gate is cleared.
- **Every task ends with its branch merged into the `phase-1` integration branch** (§2 item 4), or, for report tasks,
  in a message to Mark. Mark reviews phase 1 as one PR, `phase-1` → `main`.
- **Agent launch prompt (template):**
  > You are doing task `<ID>` of `PLAN.md` for the repo at `C:\Users\Mark\Code\personal_website\landslide-lidar`.
  > Read `docs/BRIEF.md` in full, then `docs/PLAN.md` §3, §4, the rows of §6 that list `<ID>` as owner, and the `<ID>` card. Do
  > only that card. Record every interpretation you make in `docs/decisions/<ID>.md`. Work on branch
  > `p1-<nn>-<name>` cut from `phase-1`, and finish with your commits on that branch plus a short summary: what you
  > built, the tests you ran, any number that differs from §5, and any open question. The orchestrator reviews the
  > branch and merges it into `phase-1`.

---

## 1. Phases, gates and waves

| Phase | Who | Tasks |
| --- | --- | --- |
| 0 | Mark | M0: decisions (done 2026-11-26, see §2); the local repo exists, and Mark adds the GitHub remote |
| 1 | Agents | P1-01 … P1-18, then **■ STOP and report** |
| 2 | Mark | Runs one command overnight (training → evaluation → stage 2 → verdict) |
| 3 (when told) | Agent | P3-01: audit the run and write `REPLICATION.md`, then **■ STOP** |
| 3b (when told, after phase 3) | Agents | Model improvement on the verified baseline, planned at that point (§10a) |
| 4 (when told, only if REPLICATED) | Agents | The demo, planned at that point (§10), with the best verified model |

**Phase-1 waves.** Tasks in the same wave can run in parallel. Use one git worktree per agent.

| Wave | Tasks | Starts when |
| --- | --- | --- |
| 1 | P1-01 scaffold and environment | M0 is done |
| 2 | P1-02 config and pre-registration draft · P1-03 provenance check · P1-04 channels · P1-05 metrics · P1-06 model and training | P1-01 is merged |
| 3 | P1-07 fetch rows 5, 6, 8 · P1-08 geo prep and splits · P1-15 demo stub (optional, on hold until 2026-11-27) | **■ G1: P1-03 passed** |
| 4 | P1-09 sampling and matching | P1-04, P1-07, P1-08 |
| 5 | P1-10 patch extraction | P1-09 |
| 6 | P1-11 stage-1 baselines · P1-12 stage 2 and its baselines · P1-13 evaluation | P1-10 (and P1-05, P1-06) |
| 7 | P1-14 overnight runner | P1-06, P1-12, P1-13 |
| 8 | P1-16 freeze the pre-registration | **■ G3**, all of P1-02 … P1-14 merged |
| 9 | P1-17 smoke test | P1-16 committed |
| 10 | P1-18 phase-1 report, then **■ STOP** | P1-17 |

```
M0 ─► P1-01 ─┬─► P1-02 (config + pre-reg draft) ─────────────────────────────────────────────────┐
             ├─► P1-03 (fetch 1–4, provenance) ■G1 ─┬─► P1-07 (fetch 5, 6, 8) ──┐                │
             │                                      ├─► P1-08 (geo + splits) ───┤                │
             │                                      └─► P1-15 (stub, optional)  │                │
             ├─► P1-04 (channels) ──────────────────────────────────────────────┴─► P1-09 ─► P1-10│
             ├─► P1-05 (metrics) ─────────────────────────────┐                          │       │
             └─► P1-06 (model + training, synthetic only) ──┐ │     P1-11 (baselines) ◄──┤       │
                                                            │ ├───► P1-12 (stage 2) ◄────┤       │
                                                            └─┴───► P1-13 (evaluation) ◄─┘       │
                                                                     └─► P1-14 (runner) ─► P1-16 FREEZE ■G3 ─► P1-17 smoke ─► P1-18 report ■
```

**Gates**
- **G0.** Mark's M0 decisions are made and the local repo exists (cleared 2026-11-26). The GitHub remote is needed
  only for the phase-1 PR and the stub release, not to start work.
- **G1 (P1-03).** The provenance check passes: the DTM re-downloads equal the package, and the feature counts are 21,356
  and 860. On **any** disagreement, stop and report to Mark. Don't switch sources.
- **G2 (P1-08, soft).** The label counts (`in_mapped` 18,649; eligible 15,082) are reproduced or the difference is
  explained. If a source or provenance problem causes it, stop per the brief. Otherwise record it and go on.
- **G3 (P1-16).** No CNN ever sees a real patch before the freeze commit, the smoke test included. Before the freeze,
  the training code runs only on synthetic tensors.
- **G4.** After P1-18, stop until Mark has run phase 2 and says go.
- **G5.** After P3-01, stop.
- **G6.** Phase 4 starts only if the verdict is REPLICATED and Mark says go.

---

## 2. Mark's decisions (M0, 2026-11-26)

1. **Names.**
   - Title: **"Locating the Landslides"**.
   - Repo: `Mark-Bugden/landslide-lidar`.
   - Slug: `landslide-lidar` (follows the repo name). Mark updates the site's project list for the new title.
2. **Demo details, "Spot the landslide" included: deferred.** The model must be verified first. Nothing in phases 1–3
   depends on the demo.
3. **Repo.**
   - Initialised locally at `C:\Users\Mark\Code\personal_website\landslide-lidar`.
   - `main` holds one root commit (`.gitignore`, README stub); the work happens on `phase-1`.
   - Mark creates the GitHub remote on 2026-11-27. Then:
     - `main`'s root commit goes up first: Mark pushes it, or explicitly allows an agent to;
     - `phase-1` is pushed;
     - the phase-1 PR is opened.

     If Mark creates the remote with its own initial commit, rebase `phase-1` onto it before pushing.
4. **Branch flow: one integration branch for phase 1.**
   - Task branches (`p1-<nn>-<name>`) are cut from `phase-1` and merged back with `--no-ff` by the orchestrator after
     review. Per-task code review happens there, because Mark sees phase 1 only as the single `phase-1` → `main` PR.
   - The order of commits on `phase-1` proves that the freeze came before any training.
   - The flow for phases 3 and 4 is decided later.
5. **Stub release (P1-15): not yet.** Mark and the agents work on it on 2026-11-27.
6. **Project goal (2026-11-27).** This is a portfolio project, ideally with real-world use. Phases 1–3 verify the
   spike's claims with its exact, fixed recipe, so the project rests on reproducible results rather than unverified
   ones. The brief's "fixed settings, no hyperparameter search" applies to that verification only. Once it's done,
   aiming for the best model is welcome (§10a).

---

## 3. Rules for every agent (digest of the brief; the brief is authoritative)

1. **Independence.**
   - Never look for or read the spike's private repo.
   - Nothing from the spike may produce a result or appear in the demo.
   - Use the spike's numbers only as claims to verify and as sanity checks (§5), never as targets.
2. **Data.**
   - Use only the 8 sources in the brief.
   - Never modify `data/raw_data/`. It is read-only input.
   - Every input and derived file lives under the gitignored `data/`.
   - Never commit data, caches, patches, rasters, geometry or checkpoints.
3. **Licences.**
   - Rows 3, 4 and 5 (surveyed sheets, 1:500k geology, susceptibility) and anything derived from them (sheet
     geometry, the flysch outline, susceptibility rasters) are never committed, never drawn in a figure and never
     shown in the demo.
   - Never draw or derive a map-sheet outline.
   - Sheet numbers, per-sheet flysch flags and aggregate metrics may be committed.
   - Credit ČÚZK and ČGS (CC BY 4.0) in the README and the demo.
4. **Network.**
   - Contact only the listed services, through `landslide.net`.
   - Never pass `outSR`.
   - Send an explicit User-Agent and pause at least 0.3 s between requests.
   - No outreach, no accounts, no licence acceptance.
5. **Training.**
   - No CNN training on real patches before the freeze (P1-16).
   - In phase 1, no GPU job runs longer than the smoke test. Baselines and dataset building run on the CPU.
6. **Config.**
   - Every value comes from `configs/landslide.json`. Never hard-code a value.
   - No hyperparameter search. Fixed seeds, fixed thresholds.
7. **Decisions.**
   - Record every interpretation of the method in `docs/decisions/<task-id>.md`: what, why, the alternatives, and the
     evidence (for example a count compared with §5).
   - P1-16 folds them into `PREREGISTRATION.md`.
   - Deviations are allowed only before any training.
8. **WebSearch.**
   - At most **10 calls across all phases and all agents**, and only for method details.
   - Before any call, read `docs/websearch_log.md` and append a row (n, date, task, query, why).
   - If the count is ≥ 8, ask the orchestrator first.
9. **Git.**
   - Work on a task branch (`p1-07-fetch-sources`, …) in your own worktree.
   - Never push to `main`. Never create tags.
   - Touch only the files your task owns (§4.4); coordinate before editing others.
   - Phase 1: branch from `phase-1`; the orchestrator merges your branch back into `phase-1` (never into `main`).
   - Your summary lists the tests you ran and the numbers you produced.
10. **Reporting.**
    - Tag claims [V] verified, [E] empirical (our run) or [I] inference.
    - Report negative or borderline results plainly.
    - Stop and report at a gate failure.
11. **Platform.**
    - Windows 10, PowerShell 5.1 or Git Bash, `uv run …`, `pathlib` everywhere.
    - RTX 3060 (12 GB), **16 GB RAM**: process one sheet at a time and memory-map large arrays.

---

## 4. Repo layout and shared interfaces

### 4.1 Layout

```
landslide-lidar/
  pyproject.toml  uv.lock  .python-version (3.13)  .gitignore  README.md  LICENSE
  PREREGISTRATION.md              # P1-02 draft, P1-16 freeze
  REPLICATION.md                  # phase 3
  configs/landslide.json          # frozen config; Appendix B values plus a "replication" block
  configs/landslide.sha256        # written at the freeze
  src/landslide/                  # library (see 4.4)
  scripts/                        # thin CLIs: fetch_*, build_*, run_*, evaluate, verify_*
  tests/                          # pytest, synthetic data unless marked @needs_data
  results/                        # tracked, small, aggregate only (list in 4.3)
  figures/                        # tracked; no maps of rows 3–5, no sheet outlines
  docs/decisions/<task>.md  docs/websearch_log.md  docs/PLAN.md (this file)  docs/BRIEF.md (the task brief)
  data/                           # gitignored (4.2)
  logs/                           # gitignored
  demo/  .github/workflows/demo.yml
```

### 4.2 Data layout (all gitignored)

```
data/raw_data/     the hand-over data, moved here unchanged: MANIFEST.csv, the 3 vector files, dmr5g/ (860 rasters
                   and fetch_log.jsonl). Read-only. Read in place, never copied or modified. The path lives in
                   landslide.paths (env override LANDSLIDE_RAW_DIR).
data/cache/        provenance/ (re-downloads) · susceptibility/<sheet>.tif · zabaged/<layer>.gpkg · torch/ (TORCH_HOME)
data/derived/      sheets.parquet · polygons.parquet · flysch_outline.gpkg · splits.json
data/dataset/      samples.parquet · patches.npy · meta.csv
data/baselines/    per-patch baseline predictions
data/runs/         <run>_s<seed>/ for run ∈ {a_f0..a_f4, b, s2}
data/stage2/       <scorer>/<sheet>.npz · detections.parquet · s2_infer/
data/smoke/        everything the smoke test writes (mirrors runs/, stage2/, results/)
```

Sheet IDs are canonical as `14-34-07`. The raster file names use underscores (`14_34_07.tif`); convert only in
`landslide.paths`.

### 4.3 File contracts

These are the interfaces between tasks. Change one only on a branch that also updates this section (`docs/PLAN.md`).

- **`data/derived/sheets.parquet`** (GeoParquet, EPSG:5514): one row per surveyed sheet (860).
  - Identity and geometry: `sheet`, `full` (poznamka == "list celý"), `geometry`, `area_m2`.
  - DTM grid: `grid_xmin`, `grid_ymax`, `grid_w`, `grid_h`, `dtm_path`.
  - DTM validity: `has_zero_cells`, `dtm_all_valid`.
  - Geology and kraj: `flysch_frac`, `flysch`, `kraj`.
  - Splits: `group_b` ("train"/"test"/null), `stratum` (F_ZLMS, F_other, N_ZLMS, N_other or null), `block`,
    `fold_a` (0–4 or null), `is_stage1`, `is_stage2`, `n_eligible`.
- **`data/derived/polygons.parquet`** (EPSG:5514): one row per register polygon (21,356).
  - Columns: `objectid`, `skupina`, `area_m2`, `in_mapped`, `eligible`, `kraj1` (first-listed `gen_kraj`), `sheet`
    (the assigned sheet; eligible polygons only), `geometry`.
- **`data/derived/splits.json`**:
  - `stage2` ({flysch, nonflysch}: sheets);
  - `folds_a` (fold → {blocks, sheets, n_eligible});
  - `group_b` ({train_blocks, test_blocks});
  - `inner_val` (run → blocks; run ∈ a_f0..a_f4, b, s2);
  - `config_hash`.
- **`data/dataset/meta.csv`**: row *i* describes `patches[i]`.
  - Identity: `id`, `label`.
  - Location: `x`, `y` (patch centre, EPSG:5514), `row`, `col` (pixel corner in the sheet raster), `sheet`.
  - Context: `kraj`, `flysch`.
  - Sample type: `source` (positive | random | quarry | spoil | rock | cut | background), `polygon_objectid`.
  - Splits: `fold` (−1 if not a flysch stage-1 sheet), `group_b`, `block`, `stratum`.
  - Matching: `slope_bin`, `match_level` (1–6 for matched negatives, 0 otherwise), `pair_id` (shared by a positive and
    its negative; −1 for background).
  - Statistics: `slope_mean`, `slope_std`, `slope_p90`, `lrm_s_std`, `lrm_l_std`, `curv_prof_std`, `curv_plan_std`,
    `svf_mean`, `svf_std`, `detr_std`, `detr_range_2_98`, `tri`, `susc_mean`, `susc_high_frac`.
- **`data/dataset/patches.npy`**: float16, shape (N, 7, 150, 150), memory-mappable.
  - Channel order follows `config.channels`.
  - Values are in physical units; normalisation happens in training.
- **Run directory `data/runs/<run>_s<seed>/`**:
  - `history.csv`: epoch, lr, train_loss, train_auc, val_loss, val_auc, seconds.
  - `best.pt`, `norm.json`.
  - `preds_val.csv`, `preds_test.csv`: columns `id`, `prob` (8-view TTA).
  - `done.json`: unit, seed, config_hash, git_commit, best_epoch, best_val_auc, n_train, n_val, n_test, seconds,
    peak_gpu_mem_gb. On failure, `failed.json` replaces it.
- **Stage 2**:
  - `data/stage2/<scorer>/<sheet>.npz`: lattice `row`, `col`, `x`, `y`, `score`.
  - `data/stage2/detections.parquet`: scorer, sheet, x, y, score, outcome (tp | duplicate | ignored | fp),
    target_objectid.
- **Tracked `results/`**:

  | File | Written by |
  | --- | --- |
  | `environment.json` | P1-01 |
  | `provenance.json` | P1-03 |
  | `sources.json` | P1-07 |
  | `splits_summary.json`, `sheet_splits.csv` (sheet number, kraj, flysch, stratum, opaque block id, fold, group, stage-2 flag) | P1-08 |
  | `sampling_summary.json` | P1-09 |
  | `dataset_summary.json` | P1-10 |
  | `baselines.csv`, `baselines.json` | P1-11 |
  | `stage2_baselines.csv` | P1-12 |
  | `metrics.json`, `metrics_table.md` | P1-13 code; filled in phase 2 |
  | `run_summary.json` | P1-14 code; filled in phase 2 |
  | `figures/learning_curves.png` | P1-13 |

  Smoke outputs never go here.
- **`results/metrics.json`** is the single source for `REPLICATION.md` and the demo's model card. P1-13 fixes its
  schema:
  - top level: `config_hash`, `git_commit`, `verdict`;
  - `tests.a`, `tests.b`, `tests.stage2`: point estimate, sheet CI, block CI, threshold, pass, n, per-fold, per-seed,
    and breakdowns;
  - `baselines`, `paired_diffs`, `secondary`, `spike_claims`, `discrepancies`.

### 4.4 Module ownership (who edits what)

| Files | Owner |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.gitignore`, `src/landslide/{config,paths,net,d4,logs}.py`, `scripts/check_env.py` | P1-01 |
| `configs/landslide.json`, `PREREGISTRATION.md`, `tests/test_config_appendix_b.py` | P1-02 (P1-16 at the freeze) |
| `src/landslide/sheets.py`, `scripts/fetch_{dtm,register,sheets,geology}.py`, `scripts/provenance_check.py` | P1-03 |
| `src/landslide/{dtm,channels,sat}.py` | P1-04 |
| `src/landslide/metrics.py` | P1-05 |
| `src/landslide/{model,train,predict}.py` | P1-06 |
| `src/landslide/susceptibility.py`, `scripts/fetch_{susceptibility,zabaged,weights}.py` | P1-07 |
| `src/landslide/{geo,splits}.py`, `scripts/build_geo.py` | P1-08 |
| `src/landslide/sampling.py`, `scripts/build_samples.py` | P1-09 |
| `src/landslide/{patches,features}.py`, `scripts/{build_patches,verify_dataset}.py` | P1-10 |
| `src/landslide/baselines.py`, `scripts/run_baselines.py` | P1-11 |
| `src/landslide/stage2.py`, `scripts/stage2_baselines.py` | P1-12 |
| `src/landslide/{evaluate,verdict,figures}.py`, `scripts/evaluate.py` | P1-13 |
| `src/landslide/{runner,preflight,winawake}.py`, `scripts/run_overnight.py` | P1-14 |
| `demo/`, `.github/workflows/demo.yml` | P1-15 (stub only); the demo itself later |

---

## 5. Reference numbers from the spike (checks, not targets)

- **Use them** to catch bugs, and to choose between readings of genuinely ambiguous method text. If you use one that
  way, say so in your decisions file.
- **Never tune sampling or splits** to hit them.
- **Report every difference** in the phase-1 report.

| Quantity | Spike | Checked in |
| --- | --- | --- |
| Register / surveyed-sheet feature counts (returnCountOnly) | 21,356 / 860 | P1-03 |
| Per-sheet DTM grids (recipe vs `fetch_log.jsonl`) | identical for 860 sheets | P1-03 |
| `in_mapped` polygons | 18,649 of 21,356 | P1-08 |
| Eligible positives (skupina 1, in_mapped, ≥ 3,000 m²) | 15,082 | P1-08 |
| Sheet rasters with 0 m cells (outside the border) | 96 of 860 | P1-08 |
| Flysch outline area | 6,586 km² | P1-08 |
| Flysch sheets (≥ 50 % inside) | 300 of 860 | P1-08 |
| Stage-1 sheets | 751 | P1-08 |
| Strata (sheets / blocks / eligible polygons) | F_ZLMS 242/34/8,000 · F_other 42/18/575 · N_ZLMS 39/18/372 · N_other 428/96/5,670 | P1-08 |
| Eligible stage-1 polygons | 14,617 | P1-08 |
| (a) folds | 10–11 blocks, 1,714–1,716 eligible polygons each | P1-08 |
| Stage-2 targets | 465 (278 flysch, 187 non-flysch) | P1-08 |
| Eligible stage-1 polygons with no feasible patch | 870 | P1-09 |
| Positives selected → kept | 3,613 (caps 8 flysch / 3 non-flysch per sheet) → 3,584 | P1-09 |

---

## 6. Decision register (ambiguities in Appendix A, with a recommended default)

The owner decides, records the choice in `docs/decisions/<owner>.md` and may override a default with a written reason.
Every choice is final at the freeze.

| ID | Ambiguity | Recommended default | Owner |
| --- | --- | --- | --- |
| D01 | Copy or read raw data in place | Read everything in place from `data/raw_data/`, read-only (the brief allows this). Check the sha256 of every file against `MANIFEST.csv` once. | P1-01, P1-03 |
| D02 | RNG | `numpy.random.default_rng` everywhere. Each step's stream is `SeedSequence([seed, stream_id])`; the stream ids are listed in the config. No `random` module. | P1-01 |
| D03 | Sheet geometry in EPSG:5514 | pyproj `Transformer.from_crs(4326, 5514, always_xy=True)` with the default operation. Record the PROJ version and the operation. Verify against the fetch-log grids. | P1-03 |
| D04 | `in_mapped` sheet union | All 860 sheets, partial ones included | P1-08 |
| D05 | Polygon area | Geometry area in EPSG:5514. Report the count using `st_area(shape)` too. | P1-08 |
| D06 | Sheet kraj | First token of `gen_kraj` split on ",". Majority over all register polygons that intersect the sheet. Ties go to the larger summed intersection area, then alphabetical order. | P1-08 |
| D07 | "2 km morphological closing" | `buffer(+2000).buffer(−2000)` (the config's `closing_m` is the buffer distance). Then fill the holes, then drop parts under 1 km². Report the area against 6,586 km² and the other reading's area as well. | P1-08 |
| D08 | 15 km block grid | Cells `floor(x/15000)`, `floor(y/15000)` of the sheet centroid from the EPSG:5514 origin. Block = stratum + cell. Opaque ids in committed files. | P1-08 |
| D09 | (a) fold assignment | Shuffle the flysch blocks with rng(11), stable-sort them by eligible-polygon count (descending), and assign each to the fold with the fewest polygons (ties to the lowest index). | P1-08 |
| D10 | Inner-validation rule | §4 rule for every run, s2 included: shuffle the training blocks with rng(1000+f / 2000 / 3000) and add blocks until they hold ≥ 15 % of the run's training eligible polygons. Note that the config's s2 text says "15 % of blocks". | P1-08 |
| D11 | Patch geometry | Centre on a pixel corner (r, c), covering rows r−75…r+74. "Inside the sheet" means the square lies within the sheet polygon (screened on the pixel grid, exact shapely check on the selected patches). The square must not intersect any stage-2 sheet buffered by 150 m. | P1-09 |
| D12 | Positive points | Sample pixel corners inside polygon ∩ feasible set, greedy random, ≥ 100 m apart. k = clip(⌊area/40,000⌋, 1, 3), reduced if there isn't room. | P1-09 |
| D13 | Water-filling | Split the budget between the pools by eligible-polygon share (rounded). The per-sheet cap counts points. Each sheet's polygons go in seeded random order; a polygon is taken whole if it fits under the cap, otherwise skipped. Use the largest integer cap whose total ≤ the pool budget. | P1-09 |
| D14 | Jittered 60 m grid | Grid anchored at the raster origin, jitter U(−15, 15) m per axis, snapped to pixel corners | P1-09 |
| D15 | Look-alike matching | Quota = round(0.12 × positives). Round-robin over types (quarry, spoil, rock, cut). Exact bin, levels 1 → 2 → 5. Point geometries in layer 130 are used as points. Cut points every 100 m along the line, kept if the p90 slope in a 21 px window is ≥ 20°. | P1-09 |
| D16 | Matching order | Level-by-level passes (all positives try level 1, then the unmatched try level 2, …). Seeded random order within a pass, seeded random choice among candidates. A chosen negative must be ≥ 150 m from every negative already chosen. | P1-09 |
| D17 | Background sample | Uniform over the unused random candidates, ≥ 150 m from the selected negatives and from each other | P1-09 |
| D18 | Channel numerics | Float64. Slope by central differences at 2 m. LRM by `uniform_filter`. Gaussian σ = 1, truncate 4. Profile curvature −(p²r+2pqs+q²t)/((p²+q²)(1+p²+q²)^1.5); tangential −(q²r−2pqs+p²t)/((p²+q²)(1+p²+q²)^0.5), 0 where p²+q² < ε. SVF over 16 directions k·22.5°, r = 1…20 px, bilinear sampling, 1 − mean sin(max(0, horizon)). detr: least-squares plane on the 150 × 150 window. | P1-04 |
| D19 | Patch statistics | Over the 150 × 150 patch (not the context), from float64 channels. TRI = mean over the patch of the mean \|z − 8 neighbours\| (Riley). Susceptibility stats over the patch footprint. | P1-10 |
| D20 | Training details | Per-step LR (linear warm-up over 2 epochs, then cosine to 0 at epoch 40). Normalisation from the training patches without the inner-val ones. `drop_last=True`. Ties for the best epoch go to the earliest. | P1-06 |
| D21 | Degenerate bootstrap replicate (one class) | Drop it and report the count | P1-05 |
| D22 | Stage-2 lattice | Window centres at pixel corners with row, col ≡ 0 (mod 25) from the raster origin. Keep a window if its centre is inside the sheet polygon. | P1-12 |
| D23 | Distance edges | NMS suppresses d ≤ 150 m. TP: the centre lies within target.buffer(50). Ignored: distance to any register polygon ≤ 150 m. The jitter seed is in the config. | P1-12 |
| D24 | Smoke subset | 5 % of `pair_id`s plus 5 % of the background, seeded. The smoke test runs stage 2 on 1 sheet whatever (a) and (b) give. | P1-14 |
| D25 | Verdict words | REPLICATED / NOT REPLICATED / INCOMPLETE, with Appendix A §1 precedence: a completed test below its threshold means NOT REPLICATED even when other runs are missing | P1-13 |
| D26 | (withdrawn) Demo pair seed | Deferred with the rest of the demo. If that view is built, its seed is fixed and committed before any pair is drawn. | — |

---

## 7. Phase 1 task cards

Sizes are rough agent effort: S < 1 h, M 1–3 h, L 3 h or more.

### P1-01 · Repo scaffold and environment · S
- **Depends on:** M0.
- **Do:**
  1. The repo already exists locally with this plan at `docs/PLAN.md` on `phase-1`. Create branch `p1-01-scaffold`
     from `phase-1`.
  2. **uv project, Python 3.13.**
     - Pin exact versions in `pyproject.toml` and commit `uv.lock`.
     - Target `torch==2.14.1+cu130` and the matching torchvision from the `https://download.pytorch.org/whl/cu130`
       index.
     - Also: numpy, scipy, pandas, pyarrow, geopandas, shapely, pyproj, rasterio, pyogrio, scikit-learn, matplotlib,
       pillow, requests, tqdm, pytest.
     - If that torch build isn't installable, pin the nearest and record it as a deviation.
  3. **`.gitignore`:** `data/`, `logs/`, `*.pt`, `*.npy`, `*.tif`, `*.gpkg`, `*.parquet`, `demo/node_modules`,
     `demo/dist`.
  4. **`landslide.config`:** load the config and compute `config_hash` (sha256 of the canonical JSON).
  5. **`landslide.paths`:** every path in §4.2. Smoke mode redirects to `data/smoke/`. `LANDSLIDE_RAW_DIR` defaults to
     `data/raw_data`.
  6. **`landslide.net`:** a requests session with User-Agent
     `landslide-lidar-replication (+https://github.com/Mark-Bugden/landslide-lidar)`, ≥ 0.3 s between requests,
     retries with backoff, and a hard error if any parameter is `outSR`.
  7. **`landslide.d4`:** the 8 dihedral transforms and their inverses for numpy and torch (batched), with tests.
  8. **`landslide.logs`:** a logger to file. tqdm goes to the console only.
  9. **`scripts/check_env.py`:** writes `results/environment.json` (versions, CUDA, GPU name, bf16 support).
  10. Create `README.md` (purpose, data credits, licence notes for rows 3–5), `docs/websearch_log.md` (count 0) and
      `docs/decisions/README.md` (the format).
- **Done when:**
  - `uv run python scripts/check_env.py` reports `cuda: true` on the RTX 3060;
  - `uv run pytest` passes;
  - `data/raw_data/` can be read;
  - the branch is ready to merge into `phase-1`.

### P1-02 · Frozen config and pre-registration draft · S
- **Depends on:** P1-01.
- **Do:**
  1. **`configs/landslide.json`:**
     - Every Appendix B key and value verbatim, except `name`, `status` and `paths`.
     - Add a `replication` block for our own fixed choices: seed-stream ids, provenance sample seed and size,
       negative jitter, NMS tie seed, random-baseline seed, smoke subset seed, and the
       verdict words.
  2. **`tests/test_config_appendix_b.py`:** embed Appendix B and assert equality on every key except paths, name and
     status.
  3. **`PREREGISTRATION.md` (status: DRAFT)** contains:
     - purpose and an independence statement;
     - the spike's claims table;
     - the replication criteria, verbatim from the brief (REPLICATED needs (a) ≥ 0.75, (b) ≥ 0.65 and stage-2
       P@R50 ≥ 0.30; the point estimate decides; INCOMPLETE is never a pass; a point estimate outside the spike's CI
       is a discrepancy and doesn't change the verdict);
     - the method: a pointer to Appendix A and B of `docs/BRIEF.md`, which is in the repo;
     - data sources and licences;
     - empty "Decisions" and "Deviations" sections, which P1-16 fills.
- **Done when:** the test passes, the criteria match the brief word for word, and the branch is merged into
  `phase-1`. This is the brief's "write and commit the pre-registration" step; P1-16 freezes it.

### P1-03 · Fetch scripts for rows 1–4 and the provenance check · M · ■ G1
- **Depends on:** P1-01.
- **Do:**
  1. **`landslide.sheets`:**
     - Load the surveyed sheets and reproject them locally (D03).
     - `dtm_grid(sheet)`: bounds + 250 m, snapped outwards to x ≡ 0.4 and y ≡ −0.12 (mod 2 m), giving (xmin, ymax,
       w, h).
     - Compare with `data/raw_data/dmr5g/fetch_log.jsonl` and each raster's own transform for **all 860** sheets.
  2. **Fetch scripts.** First read each layer's metadata (spatialReference, maxRecordCount, geometry type) and record
     it.
     - `fetch_dtm.py` (row 1): the exact recipe in the brief's note 1. Options `--sheets`, `--sample N --seed`. It
       writes the unmodified F32 TIFF to `data/cache/provenance/`.
     - `fetch_register.py` (row 2): layer 1, all fields, native SR, paginated by objectid, saved as GeoPackage.
     - `fetch_sheets.py` (row 3): layer 0, EPSG:4326 without `outSR` (for example `f=geojson`).
     - `fetch_geology.py` (row 4): layer 7, `subregion IN (…)`, fields `objectid, region, subregion, hor_karto`.
     - Every script has `--count-only`. For rows 2–4, run **only** the count mode. Don't download them in full.
  3. **Provenance run.**
     - Re-download **≥ 10 randomly drawn sheets** (12 recommended; seed from the config).
     - Compare each with the package file: georeferencing equal, values bitwise equal wherever the package isn't NaN,
       and every package NaN explained (NoData, or outside [−100, 2000] m in the download).
     - Run `returnCountOnly=true` on row 2 (expect 21,356) and row 3 (expect 860).
     - Check the sha256 of every `data/raw_data/` file against `MANIFEST.csv`.
  4. Write `results/provenance.json` and a short `docs/provenance.md`.
- **Gate:** on any disagreement, **stop**. Tell the orchestrator, who tells Mark. Wave 3 doesn't start.

### P1-04 · Terrain channels library · M
- **Depends on:** P1-01. The tests are synthetic. Add one real-DTM integration test after G1.
- **Do:**
  1. **`landslide.dtm`:**
     - Open rasters read-only.
     - Invalid rule: z ≤ 0, z > 2000 or NaN becomes NaN.
     - Pixel ↔ world transforms, using the pixel-corner centre convention (D11).
  2. **`landslide.channels`** (D18):
     - `compute_channels(z_block) -> (7, H, W)` in float64.
     - `patch_channels(raster, r, c)`: compute on the patch plus a 28 px margin, then crop.
     - `full_channels(raster)` for stage 2: channels 0–5 on the full raster. detr is computed per window by the
       stage-2 code.
  3. **`landslide.sat`:** summed-area tables for patch means and validity counts at any pixel corner.
- **Tests:**
  - D4 equivariance, channels(T z) = T channels(z), to 1e-9 for all 8 transforms.
  - Patch-wise vs full raster equal to ≤ 5e-9.
  - Analytic cases: a plane (known slope, zero curvature, zero detr, SVF = 1 when flat) and a paraboloid (curvature
    signs).
  - NaN propagation.
  - Benchmark: a full 2,850 × 2,400 sheet and a single patch. Record both times.
- **Watch out:** the 28 px margin must cover every kernel (LRM 25 px, SVF 20 px + 1 for interpolation, Gaussian
  4 px + 1). Patches and stage 2 use one code path.

### P1-05 · Metrics and bootstrap · S
- **Depends on:** P1-01.
- **Do:** `landslide.metrics`:
  - rank-based ROC AUC, with ties averaged;
  - a cluster bootstrap (unit = sheet or block, 2,000 replicates, rng(123), percentile 95 %). It returns the point
    estimate, the CI and the degenerate count (D21);
  - paired differences on the **same** replicate indices;
  - a seed-ensemble helper;
  - same-sheet pair accuracy;
  - a precision–recall sweep with P@R50 (precision at the highest threshold whose recall is ≥ 0.5), interpolated max
    precision over recall ≥ 0.5, and max recall.
- **Tests:** known AUCs, bootstrap determinism, identical draws for paired differences, and recall that never reaches
  0.5 (returns "not reached", which counts as a fail).

### P1-06 · Model and training loop (synthetic data only) · M
- **Depends on:** P1-01.
- **Do:**
  1. **`landslide.model`:**
     - `resnet18(weights=None)` with the state dict loaded from the local cached file. No network.
     - conv1 becomes a 7-channel conv whose filters are the mean of the RGB filters, replicated and scaled by 3/7.
     - fc becomes `Linear(512, 1)`.
  2. **`landslide.train.train_unit(run, seed, cfg, paths)`:**
     - **Data:** patches selected from the memory map, held as fp16 on the GPU. No DataLoader.
     - **Normalisation:** float64 statistics, saved to `norm.json` (D20).
     - **Augmentation:** seeded shuffle and a per-sample random D4 transform on the GPU.
     - **Optimisation:** AdamW (lr 3e-4, weight decay 0.01), batch 64, bf16 autocast, BCE-with-logits.
     - **Schedule:** per-step warm-up and cosine (D20).
     - **Early stopping:** on inner-val AUC with 8-view TTA, patience 8, max 40 epochs, best weights restored.
     - **Outputs:** `history.csv` with every field in Appendix A §7, `best.pt`, `preds_val.csv`, `preds_test.csv`,
       `done.json` (§4.3).
     - **Determinism:** seed python, numpy and torch; `cudnn.deterministic=True`, `benchmark=False`,
       `use_deterministic_algorithms(True)`; the entry point sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
       starts.
  3. **`landslide.predict.tta_predict(model, x)`:** the mean sigmoid over the 8 views. Stage 2 reuses it.
- **Tests (random tensors only):**
  - output shapes;
  - the conv1 initialisation formula;
  - TTA on a D4-invariant input;
  - LR values at the schedule boundaries;
  - two 1-epoch runs give bit-identical results;
  - early-stopping logic with scripted AUCs.
- **Hard rule:** nothing in this task touches `patches.npy`.

### P1-07 · Fetch susceptibility, ZABAGED and ImageNet weights · M (long network runs)
- **Depends on:** G1.
- **Do:**
  1. **`fetch_susceptibility.py`** (row 5):
     - Requests: per sheet and per class (mid, high), the `export` recipe from the brief's note 5, on that sheet's
       DTM grid.
     - Assert that each PNG's size equals the DTM's.
     - Classes: alpha ≥ 128. Combine into a uint8 raster (0 low/no data, 1 mid, 2 high) and save it as a deflate
       GeoTIFF on the DTM grid. Count any pixels that are both mid and high.
     - Resumable, with a jsonl log. About 1,720 requests, roughly 1 h.
     - Fetch all 860 sheets so the inputs can be rebuilt.
  2. **`fetch_zabaged.py`** (row 6):
     - Layers 118, 119, 130, 129, 79, 75.
     - Envelope queries per sheet grid (`inSR=5514`, no `outSR`), paginated, deduplicated by objectid.
     - One GeoPackage per layer in `data/cache/zabaged/`. Record the geometry types.
  3. **`fetch_weights.py`** (row 8): `ResNet18_Weights.IMAGENET1K_V1` with `TORCH_HOME=data/cache/torch`. Record the
     sha256. Check that it loads with the network disabled.
  4. Write `results/sources.json`: retrieval dates, service versions, layer metadata, file counts and sizes.
- **Done when:** every sheet has a susceptibility raster, every layer is cached, the weights load offline, and the
  branch (scripts only, no data) is ready to merge.

### P1-08 · Geo preparation and splits · M · ■ G2 (soft)
- **Depends on:** G1.
- **Do:**
  1. **Polygons:** `in_mapped` (D04), `eligible` (D05) and `kraj1`.
  2. **Flysch outline** (D07): its area, each sheet's `flysch_frac`, and the flysch flag.
  3. **DTM validity per sheet:** stream the 860 rasters and set `has_zero_cells` and `dtm_all_valid`.
  4. **Polygon → sheet:** the sheet holding the largest share of the polygon's area.
  5. **Sheet attributes:** kraj (D06), group_b, stratum and block (D08).
  6. **Stage-2 sheets:** use the brief's 20. Verify that each is full, has ≥ 5 eligible polygons, has a fully valid
     DTM and has the expected flysch flag. Report any failure. Don't swap a sheet.
  7. **Stage-1 sheets and splits:** stage-1 sheets, (a) folds (D09), (b) groups, and inner-val blocks for a_f0–a_f4,
     b and s2 (D10).
  8. **Outputs:** write the files in §4.3. Put the comparison of every count with §5 in `docs/decisions/P1-08.md`.
- **Watch out:** the outline, the sheet geometry and the susceptibility stay in `data/`. If `in_mapped` or the
  eligible count is off, check D03 and D05 first, and report before P1-09 starts.

### P1-09 · Sampling and slope matching · L
- **Depends on:** P1-04, P1-07, P1-08.
- **Do** (Appendix A §5; D11–D17):
  1. **Per stage-1 sheet,** one at a time:
     - slope raster (with `landslide.channels`);
     - validity SAT (patch plus 28 px context all valid);
     - patch-mean slope at every pixel corner (SAT);
     - feasibility masks:
       - the square lies inside the sheet;
       - the square doesn't touch a stage-2 sheet buffered by 150 m;
       - the centre is ≥ 150 m from every register polygon (all 21,356) and from ZABAGED landslide/scree (layer 129).
  2. **Positives:** feasible points per polygon (D12), then water-filling to 4,000 split by pool (D13). Count the
     infeasible polygons.
  3. **Random-negative candidates** (D14) and **look-alike candidates** (D15), each with its slope bin.
  4. **Matching:**
     - look-alikes first, then random negatives through levels 1–6 (D16);
     - store `match_level` and `pair_id`;
     - drop and count the unmatched positives.
  5. **Background:** 4,000 patches (D17).
  6. **Outputs:** `data/dataset/samples.parquet` and `results/sampling_summary.json`.
- **Assert:**
  - every negative is in its positive's stratum;
  - the slope-bin rule holds for each match level;
  - negatives are ≥ 150 m apart;
  - every centre is ≥ 150 m from every polygon;
  - the patch rule holds (exact shapely check);
  - reruns are deterministic.
- **Watch out:** the 16 GB RAM limit, so stream sheets. Expect "millions of candidates", so vectorise.

### P1-10 · Patch extraction, statistics and dataset check · M
- **Depends on:** P1-09.
- **Do:**
  1. **Patches:** stream sheet by sheet into `patches.npy` (`open_memmap`, float16) with `patch_channels`.
  2. **`meta.csv`:** with the statistics (D19), computed in float64 before the cast.
  3. **`results/dataset_summary.json`:** counts by label, source, kraj, flysch flag, fold, group, stratum and match
     level. These are the counts the phase-1 report needs.
  4. **`figures/slope_distributions.png`:** positives vs negatives per test set. No maps.
  5. **`scripts/verify_dataset.py`** checks:
     - shapes and dtype;
     - no NaN or inf, and no float16 overflow;
     - unique ids, and meta ↔ patches alignment;
     - one negative per `pair_id`;
     - the patch rule rechecked from the coordinates;
     - the file size.
- **Done when:** `verify_dataset.py` passes, and the brief's "no patch is missing or contains NaN" is confirmed [E].

### P1-11 · Stage-1 baselines (CPU) · M
- **Depends on:** P1-10, P1-05, P1-08.
- **Do:**
  1. **Baselines** (Appendix A §10):
     - unfitted: `slope_only`, `susceptibility`;
     - fitted (StandardScaler + LogisticRegression, fixed `random_state`): `slope_susc_lr`, `terrain_stats_lr`;
     - `terrain_stats_gbm`: HistGradientBoosting with defaults and fixed `random_state`.
  2. **Fitting:** each fitted baseline trains on train + inner-val blocks. For (a), fit per fold and pool the
     out-of-fold predictions. For (b), fit once.
  3. **Metrics:** AUC with sheet and block CIs; per fold; (b) by kraj and by flysch flag; positives vs each negative
     source; slope-only AUC as the matching check.
  4. **Outputs:** per-patch predictions to `data/baselines/` (for the paired differences later). Commit
     `results/baselines.csv` (with a `kind` column) and `baselines.json`.
- **Watch out:** no tuning. Use the same test patches as the CNN. Report the numbers next to the spike's best non-CNN
  (0.695 / 0.624) without acting on them.

### P1-12 · Stage-2 pipeline and stage-2 baselines · L
- **Depends on:** P1-04, P1-05, P1-07, P1-08, P1-10. Uses the predict API from P1-06.
- **Do:**
  1. **Lattice per reserved sheet:** full-raster channels and the window lattice (D22); per-window means of slope and
     susceptibility via SAT.
  2. **CNN scorer** (GPU, batched):
     - windows built from the full channels, with detr per window;
     - normalised with the `s2` run's `norm.json`;
     - D4 TTA, then the mean over seeds.
     - Unit-test it on a toy raster. It runs on real data only in the smoke test and in phase 2.
  3. **NMS:** greedy, radius 150 m, with seeded 1e-9 jitter (D23).
  4. **Matching** (Appendix A §9, D23), pooled over the 20 sheets:
     - **TP:** inside a still-unmatched target buffered 50 m; the lowest objectid wins if there are several.
     - **Duplicate:** inside an already-matched target.
     - **Ignored:** within 150 m of any register polygon.
     - **FP:** anything else.
  5. **Metrics:**
     - P@R50, interpolated precision, max recall;
     - per stratum;
     - post hoc: the ring counted as FP.
  6. **Baselines, run now on the CPU:**
     - window-mean slope;
     - window-mean susceptibility;
     - logistic regression on (slope mean, slope std, susc mean, susc high fraction), fitted on stage-1 positives vs
       the background sample;
     - random (seeded).
     - Commit `results/stage2_baselines.csv`.
- **Tests:** hand-built cases for every outcome, the objectid tie, and "50 % recall never reached = FAIL".
- **Watch out:**
  - If a baseline can't reach 50 % recall, report it; the rule doesn't change.
  - Save lattices to `data/stage2/` only.
  - Estimate the CNN cost: windows × 8 views × 3 seeds.

### P1-13 · Evaluation and verdict · M
- **Depends on:** P1-05, P1-06 (the prediction format), P1-11, P1-12. It can start against synthetic predictions.
- **Do:** `scripts/evaluate.py`, re-runnable from saved predictions only:
  1. **Integrity:** check every `done.json` (config hash = the frozen hash) and list the missing and failed units.
  2. **CNN scores:**
     - the seed ensemble for each run, and the pooled (a) out-of-fold predictions;
     - AUC with sheet and block CIs;
     - paired differences against each baseline on the same replicates.
  3. **Breakdowns:** per fold and per seed; (b) by kraj and flysch flag; per negative source; the slope-only check;
     same-sheet pair accuracy; the stage-2 metrics when present.
  4. **Verdict:** D25, tested with a truth table.
  5. **Learning curves:** `figures/learning_curves.png` as in Appendix A §7 (one row per run, BCE and AUC panels,
     faint per-seed curves, bold mean, best-epoch dots).
  6. **Outputs:**
     - `results/metrics.json`, with its schema fixed in this task;
     - `results/metrics_table.md`: our numbers next to the spike's claims, with a discrepancy flag when our point
       estimate is outside the spike's CI.
- **Watch out:** in smoke mode, read and write only `data/smoke/`.

### P1-14 · Overnight runner · M
- **Depends on:** P1-06, P1-12, P1-13.
- **Do:** `scripts/run_overnight.py`.
  1. **Entry point:**
     - wrapped in `if __name__ == "__main__":`;
     - sets `CUBLAS_WORKSPACE_CONFIG` and `TORCH_HOME` before importing torch;
     - options `--smoke`, `--dry-run` (preflight only) and `--max-hours`.
  2. **Preflight, under 60 s:**
     - CUDA available, ≥ 7 GB free GPU memory, ≥ 10 GB free disk;
     - every cached file present (a required-files list with sizes; no full hashing);
     - the `patches.npy` header shape matches the rows of `meta.csv`;
     - the config hash equals `configs/landslide.sha256`.
     - Then make socket connections raise for the rest of the run.
  3. **Keep Windows awake:** `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` through ctypes, released
     in `finally`. Don't change power settings.
  4. **Order of work:**
     - a_f0…a_f4 × seeds 0–2, then b × seeds;
     - then evaluate;
     - if both (a) and (b) pass: s2 × seeds, then stage-2 inference and baselines;
     - then the final evaluation and the verdict.
  5. **Resume, robustness, cap:**
     - **Resume:** skip any unit that has `done.json`.
     - **Robustness:** an exception in a unit is logged with its traceback and `failed.json`, and the runner moves on.
     - **Wall-clock cap:** 6 h, with 15 min kept for evaluation. When the cap is reached, stop cleanly between
       epochs and evaluate what has finished (INCOMPLETE).
  6. **Logging:**
     - `logs/overnight_<timestamp>.log` (gitignored; the progress bars stay out of it);
     - the tracked `results/run_summary.json`: units with their status and duration, the config hash, the git commit,
       the verdict and the missing units.
  7. **Smoke mode:** the Appendix B `smoke` block (D24). It writes only under `data/smoke/`.
- **Tests (mocked):**
  - a missing file, low disk and no CUDA each exit within 60 s with a clear message;
  - resume skips finished units;
  - one failing unit doesn't stop the run;
  - the cap triggers a clean stop;
  - `--smoke` never writes to `results/`.

### P1-15 · Demo stub release (optional) · S · on hold
- **Status:** deferred to 2026-11-27, when Mark works on it and the remote exists. Don't start it before then.
- **Depends on:** the M0 names (title "Locating the Landslides"), P1-01 and G1 (for the still). Independent of
  everything else.
- **Do:** follow the site contract's appendix.
  1. **Kit tag:** `git ls-remote` for v0.5.0. Clone the kit outside the repo, copy `template/` to `demo/`, then
     delete the clone.
  2. **Template rules:**
     - keep `vite.config.js` as it is;
     - depend on `github:Mark-Bugden/demo-kit#v0.5.0`;
     - package name `landslide-lidar-demo`, version `0.1.0`;
     - commit `package-lock.json`.
  3. **Manifest:**
     - `status: "stub"`, `kitVersion: "0.5.0"`, `writeup: "/projects/landslide-lidar/"`;
     - **no model card and no results**. The summary describes the replication as pending.
  4. **Files:** `main.js` and `demo.css` from the appendix; `index.html` with `data-demo="landslide-lidar"`.
  5. **Stills:**
     - Render a hillshade of a 3 × 3 km square wholly inside one sheet, from DMR 5G only.
     - Run the contract's `make_stills.py`, saved outside the repo.
     - If the square was chosen for its looks, the alt text says "chosen for illustration".
  6. **Workflow:** copy `.github/workflows/demo.yml` unchanged from zombie-outbreak.
  7. **Checks:** `npm ci`, `npm run build`, `npx demo-kit validate dist/` (must print OK), and `npx vite preview` with
     no console errors.
- **Done when:** the PR is open and Mark is told that merging it releases v0.1.0. Never push tags. The stub
  touches only `demo/` and the workflow, so it can go to `main` as its own PR, separate from `phase-1`, if Mark
  prefers.

### P1-16 · Freeze the pre-registration · S · ■ G3
- **Depends on:** P1-02 … P1-14 merged.
- **Do:**
  1. Fold every `docs/decisions/*.md` into `PREREGISTRATION.md`. "Decisions" and "Deviations" each list the
     reason, the owning task and the evidence, including the count comparison with §5.
  2. Finalise the config `replication` block.
  3. Set `status` to `"FROZEN <date> at the pre-registration commit (before any CNN training, including the smoke
     test)"`.
  4. Write `configs/landslide.sha256`.
  5. Run the full test suite. Confirm that no run directory exists.
  6. Commit "Freeze pre-registration" on `phase-1`. Record the commit SHA.
- **Done when:** the freeze commit is on `phase-1`, and every later commit (smoke-test fixes, report) comes after it.
  The run summary records the freeze SHA.

### P1-17 · Smoke test · S
- **Depends on:** P1-16.
- **Do:**
  1. Run `uv run python scripts/run_overnight.py --smoke`:
     - units a_f0, b and s2, seed 0, 2 epochs, 5 % of patches, 1 stage-2 sheet;
     - everything under `data/smoke/`.
  2. **Measure:**
     - seconds per epoch per unit, and peak GPU memory;
     - stage-2 windows per second;
     - disk use of `data/`.
  3. **Estimate the full runtime:**
     - training: Σ over units of epoch time × (full size / smoke size) × epochs, with 40 epochs as the upper bound;
     - stage 2: 20 sheets × windows × 8 views × 3 seeds ÷ throughput;
     - compare with the spike's 0.83 h and the 6 h cap.
  4. **Runner checks:**
     - re-run once to confirm that resume skips the finished units and that the history is bit-identical;
     - check one preflight failure by hand (exits in under 60 s).
- **Rules:**
  - Smoke metrics never change anything.
  - Only crash fixes are allowed. Each fix goes in `docs/decisions/P1-17-fixes.md` and the report, and none may change
    the method or the config.

### P1-18 · Phase-1 report · S · then ■ STOP
- **Do:** first, if the remote exists, push `phase-1` and open the `phase-1` → `main` PR (§2 item 3). The report
  links it. Then write the report: at most 500 words, tagged [V], [E] and [I], covering:
  - the provenance check;
  - dataset counts by label, source, kraj, flysch flag and fold;
  - baseline AUCs with CIs for (a) and (b);
  - smoke result, time per epoch, peak GPU memory, runtime estimate and disk use;
  - every decision and deviation, as a short list pointing to `PREREGISTRATION.md`;
  - the commits (`git log --oneline`) and the WebSearch count;
  - the command for Mark:

    ```powershell
    cd C:\Users\Mark\Code\personal_website\landslide-lidar
    uv run python scripts/run_overnight.py
    # progress:  Get-Content (Get-ChildItem logs\overnight_*.log | Sort-Object LastWriteTime | Select-Object -Last 1) -Tail 30 -Wait
    # summary:   Get-Content results\run_summary.json
    # resume:    run the same command again (finished units are skipped)
    ```

---

## 8. Phase 2 (Mark)

Mark runs the command above and leaves it unattended. Expected time is about 1 h (the spike took 0.83 h); P1-17 gives
our estimate. The run writes `results/run_summary.json`, `results/metrics.json` and the logs.

---

## 9. Phase 3 task card

### P3-01 · Verify the run and write `REPLICATION.md` · M · then ■ STOP
- **Do:**
  1. **Audit first.** Check `run_summary.json` and every `done.json` (config hash, git commit, status). Report any
     failed or missing unit, or any hash mismatch, **before anything else**.
  2. **Reproduce the metrics.** Re-run `scripts/evaluate.py` from the saved predictions; `metrics.json` must come out
     identical.
  3. **Write `REPLICATION.md`:**
     - the verdict;
     - our numbers with CIs next to the spike's claims and our baselines (tests a, b and stage 2);
     - per-fold and per-seed spread;
     - secondary results: block CIs, (b) by kraj and flysch flag (the non-flysch Jihomoravský AUC), per negative
       source, the slope-only check, same-sheet pair accuracy, stage 2 with the ring counted as FP;
     - every discrepancy (a point estimate outside the spike's CI) and what we found when we investigated it (dataset
       counts, block structure, base rates). Discrepancies don't change the verdict;
     - caveats: circularity, label noise, spatial autocorrelation, the stage-2 base rate.
  4. Commit it with the tracked results and open a PR.
  5. Final message to Mark, at most 400 words.
- **If NOT REPLICATED or INCOMPLETE:** report and stop. The demo doesn't present the spike's claims, and Mark decides
  what happens next.

---

## 10. Phase 4: the demo (planned later)

Deliberately not planned yet. The demo is designed and split into tasks only after a REPLICATED verdict and Mark's go
(G6), working from the brief's "Phase 4" section and the site contract. Until then, no agent spends effort on demo
details. Phases 1–3 need nothing beyond what they already produce: `results/metrics.json` and the stage-2 outputs in
`data/stage2/`.

## 10a. Model improvement (planned after verification)

Not split into tasks yet; planned with Mark after phase 3. Agreed principles (2026-11-27):

- **Start from the verified baseline.** The frozen replication run is the reference every change is compared with.
  It is never re-run with changed settings; improvements are new, separately recorded runs.
- **Honest evaluation.**
  - Nested spatial cross-validation: all tuning (hyperparameters, architectures, augmentation, sample sizes) happens
    inside the training folds; the outer folds give the performance estimate. Folds are spatial blocks, never random
    patches, because nearby terrain is correlated.
  - Several seeds for every compared configuration, with their spread reported.
  - The 20 reserved stage-2 sheets stay untouched until one final, pre-registered check of the chosen model. They are
    also where the demo's map explorer lives.
- **Candidate directions.** More of the 15,000 eligible landslides (the spike's recipe uses about 3,600); tuned
  optimisation and longer training; more augmentation or larger models; optimising for the whole-sheet detection task
  (stage 2), the closest to real use.
- **Claims.** The register was mapped on the same LiDAR hillshade, so the model learns to agree with the mappers;
  any real-world claim says so (brief: "Claims").

---

## 11. Estimates and risks

| Item | Estimate |
| --- | --- |
| New disk use in `data/` | About 6 GB: patches 3.5 GB, checkpoints 1 GB, susceptibility and ZABAGED under 1 GB, stage 2 under 1 GB. This is on top of the 11.4 GB of hand-over data already in `data/raw_data/`. Free now: about 109 GB. |
| Network in phase 1 | About 12 provenance DTMs (~330 MB uncompressed F32), about 1,720 susceptibility renders (~1 h), ZABAGED queries (~0.5–1.5 h), 45 MB of weights |
| Patch set | About 7,200 matched + 4,000 background ≈ 11,200 × 315 KB ≈ 3.5 GB float16. It fits on the GPU (12 GB) for the s2 run. |
| Full run | About 1 h expected (the spike took 0.83 h). Stage-2 inference is about 20 sheets × ~11k windows × 8 views × 3 seeds ≈ 5M forward passes. The 6 h cap gives a wide margin. |

| Risk | Mitigation |
| --- | --- |
| A source changed since 2026-11-25 (counts, DTM values, susceptibility, ZABAGED) | G1 catches rows 1–3. For rows 5 and 6, record the retrieval date and metadata; a change is reported, never silently absorbed. |
| pyproj or PROJ differences shift the sheet geometry, and with it `in_mapped` and the grids | P1-03 compares all 860 grids with the fetch log; P1-08 checks the label counts (G2). |
| Ambiguous method text leads to a different dataset than the spike's | §6 defaults, recorded and frozen. §5 numbers are used only as checks. The phase-3 discrepancy investigation explains the gaps. |
| 16 GB RAM during the build | Stream per sheet, memory-map `patches.npy`, never load all the rasters |
| The torch 2.14.1+cu130 wheel is unavailable for Python 3.13 on Windows | Pin the nearest CUDA build and record it as a deviation before training |
| Stage-2 baselines can't reach 50 % recall | Report it; the rule stays fixed (it is already the spike's final rule) |
| Parallel agents conflict in shared files | §4.4 ownership, per-task decision files, worktrees |
| WebSearch budget overrun | Shared log, ask the orchestrator from 8 calls |
| A licence leak (sheet outlines, susceptibility or geology in figures or the demo) | §3 rule 3. Review every figure before merging a task branch. |
