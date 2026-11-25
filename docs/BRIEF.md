# Task: independently reproduce the landslide kill test, then build its demo

You work in a **new, empty, public repo** that Mark creates. Mark (Mark Bugden) owns it and reviews through pull
requests. A research spike in a separate private repo found that a CNN recognises mapped landslides in Czech LiDAR
terrain. Your job comes in two parts, in this order:

1. **Reproduce the result independently.** Use this repo's own code, the inputs listed under "Data sources" below,
   and a **newly trained model**.
2. **Then build the website demo**, using only this repo's own model and outputs.

The spike's results are claims to verify. They are not results you may reuse.

## What you are given
1. **This brief.** It includes the spike's method (Appendix A) and its frozen configuration (Appendix B).
2. **The site contract,** `C:\Users\Mark\Code\personal_website\site-contract-landslide-lidar.md`. It is binding for
   the repo layout, the demo bundle and releases; on anything technical about the site, it wins over this brief.
3. **Raw data as downloaded from the sources below,** in `C:\Users\Mark\Code\personal_website\landslide-demo-handoff\raw_data\`
   (called `raw_data/` from here on). Its `MANIFEST.csv` lists each file's source, retrieval date, sha256 and any
   change from the download.

Nothing else from the spike is part of the hand-over: no code, no results, no derived data. Don't look for or read the
spike's private repo. Mark compares the spike's full results with yours later.

## How this task is split
| Phase | Who | What |
| --- | --- | --- |
| **1 (now)** | You | Set up the repo. Write and commit the replication pre-registration. Get the data (see "Data sources"), build the dataset and all code, run the baselines and a smoke test. Then **stop and report**. Do **not** run the full training. |
| **2** | Mark | Runs one command. Training, evaluation and stage 2 run unattended. |
| **3 (when told)** | You | Check the run. Write the replication report: verdict, your numbers next to the spike's claims, discrepancies. Stop and report. |
| **4 (when told, and only if REPLICATED)** | You | Build the demo from this repo's outputs, as the site contract requires. |

## The spike's claims (to verify)
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

## Data sources
These eight inputs are the **only** data you may use. Each row gives the source and how you get it: **use** the file
from `raw_data/`, or **fetch** it yourself from the source. All sources are public and need no account.

| # | Input | Source | How you get it |
| --- | --- | --- | --- |
| 1 | DMR 5G terrain, 2 m, one raster per surveyed sheet | ČÚZK, `https://ags.cuzk.gov.cz/arcgis2/rest/services/dmr5g/ImageServer/exportImage` | **Use** `raw_data/dmr5g/<sheet>.tif` (all 860 sheets, downloaded 2026-11-25), after the provenance check below. |
| 2 | Landslide polygons (the labels) | ČGS slope-deformation register, `https://mapy.geology.cz/arcgis/rest/services/Geohazardy/svahove_nestability/MapServer/1` | **Use** `raw_data/cgs_register_polygons.gpkg` (all 21,356 polygons). Derive `in_mapped` yourself (note 2 below). |
| 3 | Surveyed map sheets: where unlabelled terrain counts as negative | ČGS, `https://mapy.geology.cz/arcgis/rest/services/Prozkoumanost/prehled_mapovani_svahove_nestability/MapServer/0` | **Use** `raw_data/cgs_surveyed_sheets.geojson` (860 sheets, EPSG:4326). |
| 4 | Geological map 1:500k, for the flysch outline | ČGS, `https://mapy.geology.cz/arcgis/rest/services/Geologie/geologicka_mapa500_CR/MapServer/7` | **Use** `raw_data/cgs_geol500_carpathian_units.gpkg` (the subregions in Appendix A §3). Build the outline yourself. |
| 5 | Landslide susceptibility map (a baseline input) | ČGS, `https://mapy.geology.cz/arcgis/rest/services/Geohazardy/sesuvna_nachylnost/MapServer` | **Fetch** (note 5 below). |
| 6 | Look-alike negatives and the landslide/scree exclusion | ČÚZK ZABAGED, `https://ags.cuzk.gov.cz/arcgis/rest/services/ZABAGED_POLOHOPIS/MapServer`, layers 118 quarry, 119 spoil heap, 130 rock formation, 129 landslide/scree, 79 road, 75 railway | **Fetch**. |
| 7 | Orthophoto (demo only, phase 4) | ČÚZK, `https://ags.cuzk.gov.cz/arcgis1/rest/services/ORTOFOTO/MapServer/export` | **Fetch**. |
| 8 | ImageNet weights for ResNet-18 | torchvision, `ResNet18_Weights.IMAGENET1K_V1` | **Download** through torchvision. |

**Notes on each source** (what the spike learned about them):

1. **DMR 5G.**
   - **Recipe.** One `exportImage` request per surveyed sheet:
     - `bbox` = the sheet's bounds in EPSG:5514 plus 250 m, snapped outwards to the native grid, whose pixel edges
       sit at x ≡ 0.4 and y ≡ −0.12 (mod 2 m);
     - `bboxSR=imageSR=5514`, `size` = bbox / 2 m, `format=tiff`, `pixelType=F32`,
       `interpolation=RSP_NearestNeighbor`.
   - **Border.** The service returns **0 m, not NoData, outside the national border**. Treat z ≤ 0 as invalid.
   - **Changes in the package files.** NoData and values outside [−100, 2000] m were set to NaN, and each raster was
     re-encoded as deflate-compressed GeoTIFF on the same grid. `raw_data/dmr5g/fetch_log.jsonl` gives each sheet's
     grid.
2. **Register.**
   - **`in_mapped`** is not a source field. Set it True when a polygon's `representative_point()` lies within the union
     of the surveyed sheets (row 3, reprojected locally to EPSG:5514). That gives 18,649 of 21,356 polygons.
   - **Eligible positives:** `skupina == 1`, `in_mapped`, area ≥ 3,000 m², which gives 15,082 polygons.
   - **The package file** is the download (layer 1, all features, native SR) minus the spike's `in_mapped` column.
4. **Geology.** The package file is the result of a query on layer 7 with `subregion IN (…)`, using the subregions in
   Appendix A §3, and fields `objectid, region, subregion, hor_karto`.
5. **Susceptibility.**
   - **Why not query it.** The vector layer stores huge dissolved polygons, so a per-sheet query returns 12 MB or
     HTTP 500.
   - **Recipe.** Render each class with `…/sesuvna_nachylnost/MapServer/export` on each sheet's DTM grid: `bbox` and
     `size` of the DTM raster, `bboxSR=imageSR=5514`, `format=png32`, `transparent=true`, `layers=show:0`, `dpi=96`,
     and `layerDefs={"0": "struktura LIKE 'Třída střední%'"}` for mid or `'Třída vysoké%'` for high.
   - **Classes.** A pixel belongs to the class if its alpha is ≥ 128. Code 0 = low or no data, 1 = mid, 2 = high.
6. **ZABAGED** has no typed cut/embankment layer. Derive cuts from road and rail lines on steep terrain
   (Appendix A §5).

**Rules:**
- **Nothing else.** In particular, derive yourself everything the spike derived: patches, `in_mapped`, the flysch
  outline, susceptibility rasters, ZABAGED subsets, splits and baseline scores.
- **Provenance check** (phase 1, before you use rows 1–4):
  - **DTM.** Re-download at least 10 randomly chosen sheets from row 1's source with the recipe in note 1. They must
    equal the package files everywhere except where the package has NaN.
  - **Counts.** Count the features of rows 2 and 3 with `returnCountOnly=true`. On 2026-11-25 they were 21,356 and 860.
  - **On any disagreement,** stop and report to Mark. Don't switch sources on your own.
- **Rebuildable without the hand-over.** Write a fetch script for every row, rows 1–4 included, so that anyone can
  rebuild your inputs from the sources. For rows 1–4 you only run it for the provenance check (a full DTM fetch is
  about 11 GB and 45 min).
- **Requests.**
  - Never pass `outSR`: query in the native EPSG:5514 or in EPSG:4326, and transform locally with pyproj.
  - Send an explicit User-Agent and pause about 0.3 s between requests.
- **Storage.** Keep every input under a gitignored `data/`. Copy files from `raw_data/` or read them in place, but
  never modify `raw_data/`. No raw data is committed.
- **Licences.**
  - Rows 1, 2, 6 and 7 are CC BY 4.0 (ČÚZK, ČGS); credit them.
  - Rows 3, 4 and 5 have no stated or verified licence; see "Constraints".
  - Row 8 falls under torchvision's terms.

## Independence rules
- **Data.** Only the inputs in "Data sources".
- **Code.** Write your own pipeline from Appendix A: dataset build, baselines, training, evaluation (including the
  bootstrap) and the stage-2 detection and matching. No spike code is provided. Where the method is ambiguous, decide,
  document the decision and commit it before training.
- **Same method and values.**
  - The method is Appendix A.
  - The values are Appendix B; your frozen config must match them. The file paths in Appendix B are the spike's and
    don't apply to you.
- **Splits.**
  - Use **the same 20 reserved stage-2 sheets** (listed in Appendix A §4), and exclude them from all stage-1 data.
  - Build everything else yourself with the rules and seeds of Appendix A §4: each sheet's kraj and flysch flag,
    blocks, (a) folds, (b) groups and inner-validation blocks. Your folds will not match the spike's exactly; that's
    expected.
- **Deviations.** Allowed only before any training (smoke test included), recorded with reasons in your
  pre-registration. A source may have changed since 2026-11-25, for example; if so, report it.
- **Pre-registration.** Commit `PREREGISTRATION.md` and the frozen config before any training. It points to (or
  restates) the method, lists every decision and deviation, and fixes the replication criteria below.

## Replication criteria (fixed now; never change them after training)
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

## Phase 1: what to build (then stop)
1. **Environment.** Python with a CUDA build of PyTorch; confirm `torch.cuda.is_available()`. Pin exact versions (the
   spike used torch 2.14.1+cu130 on Python 3.13). The hardware is an RTX 3060 (12 GB), 16 GB RAM, Windows.
2. **Pre-registration.** Commit `PREREGISTRATION.md` and the frozen config before any CNN training, the smoke test
   included.
3. **Data.** Run the provenance check, then get every input as "Data sources" says. Build the dataset per
   Appendix A §5 and verify that no patch is missing or contains NaN.
4. **Baselines.** They are CPU-only. Run them on your splits and commit their AUCs with sheet-level bootstrap CIs.
5. **Code.**
   - Training: 5 sheet-grouped folds for (a), 1 run for (b), and the stage-2 model, each with early stopping on an
     inner validation split.
   - Evaluation: re-runnable on its own from saved predictions.
   - Stage 2.
6. **Overnight runner.** One command: training → evaluation → stage 2 (only if (a) and (b) pass) → replication
   verdict. It must be:
   - **Offline:** it reads only cached data, and checks this at start-up.
   - **Fail-fast:** it checks CUDA, free disk and every cached file at start-up, and on any problem exits within 60 s
     with a clear message.
   - **Awake:** it keeps Windows awake with `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` via ctypes,
     without changing power settings.
   - **Resumable:** re-running the command skips finished units.
   - **Robust:** an exception in one unit is logged and the runner moves on. Missing units are listed and never pass.
   - **Logged:** progress goes to gitignored logs, plus a small tracked run summary.
   - **Capped:** a wall-clock limit (default 6 h) after which it stops gracefully and evaluates what has finished.
   - **Windows-safe:** `if __name__ == "__main__":`, and no DataLoader worker problems.
7. **Smoke test.** Run the whole pipeline on a tiny subset (one fold, about 5 % of patches, 1–2 epochs) with
   `--smoke`, writing to a separate directory, never to the tracked results.
   - Measure time per epoch and peak GPU memory, and estimate the full runtime (the spike's full run took 0.83 h).
   - Run no GPU job longer than the smoke test in phase 1, and never use smoke metrics to change anything.
8. **Demo stub (optional).** Ship the stub release from the site contract's appendix so the site can list the project
   early.
9. **Commit** code, config, the pre-registration and small results. Data, caches and checkpoints stay out of git.

**Phase-1 report** (at most 500 words):
- the provenance check (DTM re-downloads, feature counts);
- dataset counts by label, source, kraj, flysch flag and fold;
- baseline AUCs with CIs for (a) and (b);
- the smoke-test result, time per epoch, peak GPU memory, estimated runtime and disk use;
- the exact PowerShell command for Mark, plus how to check progress and resume;
- every decision and deviation;
- the commits made and the number of WebSearch calls used.

## Phase 3: verify (when told)
- **Check the run.** If any run failed, is missing or has a config-hash mismatch, report that first.
- **Write `REPLICATION.md`** and commit it with the tracked results. It contains:
  - the verdict (REPLICATED, NOT REPLICATED or INCOMPLETE);
  - a table of your numbers with CIs next to the spike's claims and your baselines;
  - per-fold and per-seed spread;
  - discrepancies and what you found about them;
  - caveats.
- **Final message** (at most 400 words). Then stop.

## Phase 4: the demo (when told, and only if REPLICATED)
The design is yours and Mark's; this is a recommendation.
- **Main view: a held-out map explorer.** The viewer picks a 3 × 3 km area inside one of the 20 reserved sheets
  (which your model never trained on).
  - Layers: a hillshade, optionally the ČÚZK orthophoto, your model's window-score heatmap, the mapped polygons, and
    detections above a threshold slider, coloured by outcome under the pre-registered rule (hit, false alarm,
    ignored, duplicate).
  - A status line with counts in view.
- **Optional: "Spot the landslide."** The viewer sees two 300 m patches from the same sheet with the same slope and
  guesses which is the mapped landslide. Then your model's answer and the truth are revealed.
  - Draw the pairs at random from your held-out predictions, with a seed you fix before you look.
  - Show your model's accuracy on the drawn pairs and on all such pairs, from your run (the spike found 0.76–0.82).
- **Model card.** Fill it from your own `metrics.json` only. It must be true (see the site contract).

### Bundle constraints
- **Size and formats.** The bundle must stay under 2 MB, and `data/` may hold only `.json`, `.csv` and `.js` files.
  The validator of the kit tag you pin enforces this.
- **No runtime fetching.**
- **Images.** An image that is not one of the contract's named files travels as data: base64 inside a `.json`/`.js`
  file, or a numeric grid drawn in a canvas.
- **Measured costs:**
  - a 3 × 3 km hillshade at 2 m is about 250 KB as WebP, or 330 KB as base64;
  - at 4 m it is about 41 KB;
  - a sheet's 50 m score lattice is about 50 KB of JSON;
  - one sheet's simplified polygons are about 75 KB.
- **Generated data.** Commit the script that derives `demo/public/data/` from your outputs, and commit the derived
  files; CI builds without your caches.

### Claims
- **Allowed** (with your numbers): "agrees with ČGS's mapped landslides on map sheets and regions it never saw"; "beats
  slope, the official susceptibility map and simple terrain statistics"; "reproduces an earlier pre-registered result".
- **Not allowed:**
  - "detects landslides" without qualification;
  - metrics computed on a hand-picked area;
  - "finds unmapped landslides" as a fact.

  The register's polygons were drawn on the same LiDAR hillshade the model sees, so agreement with the mappers is not
  field truth.
- **Labelling.**
  - Counts inside one area are illustrative; label them as such.
  - Areas or examples chosen because they look good must be labelled "chosen for illustration".

## Constraints (all phases)
- **CC BY 4.0 inputs** (rows 1, 2, 6 and 7 of "Data sources": DMR 5G, the register, ZABAGED, the orthophoto).
  Credit ČÚZK and ČGS in the repo README and in the demo.
- **Inputs with no stated or verified licence** (rows 3, 4 and 5: the surveyed sheets, the 1:500k geology, the
  susceptibility map).
  - Keep them only in the gitignored `data/`. Never commit them, or any raster or geometry derived from them, and
    never show them in a figure or the demo.
  - Aggregate results that use them are fine: AUCs, a per-sheet flysch flag, map-sheet numbers.
  - Never show map-sheet outlines, and never derive one (a score map clipped at a sheet edge draws one). Show areas as
    axis-aligned squares wholly inside their sheet.
- **Never committed:** caches, patches and checkpoints. Commit code, configs and small results.
- **Fixed settings.** No hyperparameter search, fixed seeds, fixed thresholds.
- **No outreach.** Don't contact organisations, register accounts or accept licences.
- **WebSearch.** At most 10 calls across all phases, only for method details. Count them.
- **Honest reporting.** Report negative or borderline results plainly; a failed replication is a useful result. Tag
  claims [V] verified, [E] empirical (your run) or [I] inference.

## Decisions to settle with Mark early
1. Slug, title and repo name. Proposed: `landslide-lidar`, "Spotting the Slides", `Mark-Bugden/landslide-lidar`.
2. Whether the "Spot the landslide" view goes in the first release.

## Appendix A: the spike's method (pre-registration §1–§12, verbatim)

This is the method you reproduce, copied unchanged from the spike's pre-registration (only the heading levels are
demoted). File and script names in it (`data/11_landslide/…`, `scripts/11_*.py`, `results/11/…`) refer to the
spike's private repo and are not available to you. The spike's dataset summary and results (§13 onwards) are left out
on purpose; its headline claims are in "The spike's claims" above.

### 1. Hypothesis and pre-registered decision rule

A ResNet-18 trained on 300 m multi-channel patches of DMR 5G (2 m) recognises landslide morphology well enough to beat
simple baselines once slope is controlled. Those baselines gave AUC 0.43–0.57 on slope-matched samples in earlier quick
tests [E, `notes/00_empirical_landslides.md`].

| Test | Definition | PASS if |
| --- | --- | --- |
| (a) held-out flysch | 5 spatial-block folds over flysch sheets; train on 4 folds of flysch, test on the 5th; pooled out-of-fold predictions | slope-matched ROC AUC **≥ 0.75** |
| (b) transfer | train on sheets in Zlínský + Moravskoslezský kraj, test on all sheets of all other kraje | slope-matched ROC AUC **≥ 0.65** |
| stage 2 | run **only if (a) and (b) both pass**: sliding-window detection on 20 reserved sheets, unmatched background | precision at 50 % object recall **≥ 0.30** |

- If (a) or (b) fails, the verdict is **FAIL**. If both pass and stage 2 fails, the verdict is **FAIL**. **PASS** requires all three.
- The **point estimate** decides. The 95 % CI is always reported, and a result whose CI straddles its threshold is called
  "borderline" in the write-up. Borderline does not change the verdict.
- A test with any missing fold or run is **INCOMPLETE**, which is never a pass. If a completed test is below its
  threshold, the verdict is FAIL even when other runs are missing.
- No hyperparameter search. One frozen configuration with fixed seeds. Thresholds and configuration are never changed
  after seeing CNN results.

### 2. Data (all cached under `data/11_landslide/`, gitignored; the overnight run makes no network requests)

| Input | Source | Use |
| --- | --- | --- |
| Labels | ČGS slope-deformation register, `svahove_nestability_polygons.gpkg` (21,356 polygons, EPSG:5514, CC BY 4.0) | positives: `skupina == 1`, `in_mapped == True`, area ≥ 3,000 m² (15,082 polygons) |
| True-negative area | ČGS surveyed-sheet layer (860 ZM10 sheets, EPSG:4326, reprojected locally with pyproj; **licence not stated, kept out of published outputs**) | sampling frame, split units |
| Terrain | DMR 5G ImageServer, native 2 m. One request per sheet, sheet bbox + 250 m, aligned to the native grid (x ≡ 0.4, y ≡ −0.12 mod 2 m), nearest neighbour | all channels. An aligned bilinear request returns identical values [E]. The service returns **0 m outside the border** (96 of 860 sheet rasters) [E], so z ≤ 0 is treated as NaN |
| Susceptibility | ČGS `Geohazardy/sesuvna_nachylnost` MapServer, rendered per class (mid, high) with `layerDefs` on each sheet's DTM grid in the native SR (wkid 102067 = 5514, no reprojection); everything else = low | baselines. 81/81 of the critical review's point queries agree with the rasters [E] |
| Look-alikes | ZABAGED polohopis (ČÚZK, CC BY 4.0): 118 quarries/pits, 119 spoil heaps, 130 rock formations, 79 roads + 75 railways (for cuts/embankments) | hard negatives |
| Exclusion | ZABAGED 129 "Sesuv půdy, suť" (landslide/scree) | no negative within 150 m |
| Flysch | ČGS geological map 1:500,000 (`Geologie/geologicka_mapa500_CR`, layer 7) | flysch flag |

Why the susceptibility is rendered rather than queried: the vector layer stores dissolved polygons of up to 17,587 km²
(low) and 1,209 km² (mid). A per-sheet query returns 12 MB of geometry or HTTP 500 [E]. Low + mid + high cover
77,959 km² of 78,871 km², so treating the ~1 % without data as "low" changes little [E].

### 3. Flysch definition

The Outer Western Carpathian flysch is the union of the 1:500k geological-map polygons whose `subregion` is one of:
*Marinní terciér vnějších Karpat*, *Mezozoické sedimenty Karpat*, *Marinní mezozoikum a terciér vnějších a vnitřních
Karpat*, *Vulkanity vnějších Karpat*, *Vulkanity vnějších a vnitřních Karpat*. The Carpathian foredeep and the
intramontane Neogene basins (*Terciér Alpsko–karpatské předhlubně…*) are excluded. The union is closed with a 2 km
morphological closing, interior holes (Quaternary valley fills) are filled, and parts under 1 km² are dropped.
Result: 6,586 km² [E] (Pánek et al. 2019 give ~7,539 km² for the Czech flysch Carpathians [V, abstract]).
A sheet is **flysch** if ≥ 50 % of its area lies inside the outline: 300 of 860 sheets.
This is the ČGS outline the prompt prefers, not the kraj proxy.

### 4. Units, splits and leakage control

- **Sheet attributes.**
  - Kraj: the majority first-listed `gen_kraj` of all register polygons that intersect the sheet.
  - Transfer group: "train" if the kraj is Zlínský or Moravskoslezský, otherwise "test".
  - Stratum: flysch flag × transfer group.
  - Spatial block: a 15 km grid cell of the sheet centroid, nested inside the stratum.
- **Polygon → sheet.** Each eligible polygon belongs to the single sheet that holds the largest part of its area.
  Its points are sampled only inside that sheet.
- **Patch rule.** Every patch square (300 m) lies entirely inside its own sheet, so no patch can overlap a patch of
  another sheet, and therefore of another fold. Patches also never touch any stage-2 sheet buffered by 150 m, so
  stage-2 windows (centre inside the sheet) never overlap a stage-1 patch.
- **Stage-2 sheets (chosen before anything else, seed 2026).**
  - 10 flysch + 10 non-flysch full sheets ("list celý"), drawn at random from the sheets with ≥ 5 eligible polygons
    **whose whole cached DTM raster (sheet bbox + 250 m) is valid**.
  - Why the validity criterion: outside the national border the ImageServer returns 0 m rather than NoData, and full
    ZM10 rectangles at the border extend abroad. In the first draw one sheet (26-11-13) was 70 % outside Czechia, so
    the criterion was added and the draw repeated with the same seed. This was before any training.
  - They are excluded from all stage-1 data, so from all training and evaluation in stage 1.
  - Flysch: 24-43-19, 24-44-17, 25-14-17, 25-23-01, 25-23-04, 25-24-01, 25-32-22, 25-33-01, 25-41-05, 35-12-16.
  - Non-flysch: 02-41-01, 02-41-18, 03-34-01, 03-34-04, 14-13-14, 14-14-22, 24-12-21, 24-14-16, 24-42-23, 25-12-22.
  - These are map-sheet numbers only; no sheet geometry is published.
- **Stage-1 sheets.** Every other sheet with ≥ 1 eligible polygon: 751 sheets.
  - F_ZLMS: 242 sheets, 34 blocks, 8,000 eligible polygons.
  - F_other: 42 sheets, 18 blocks, 575 eligible polygons.
  - N_ZLMS: 39 sheets, 18 blocks, 372 eligible polygons.
  - N_other: 428 sheets, 96 blocks, 5,670 eligible polygons.
- **Test (a) folds.** Flysch blocks are shuffled (seed 11), sorted by eligible-polygon count, and assigned greedily
  to the fold with the fewest polygons. This gives 5 folds of 10–11 blocks with 1,714–1,716 eligible polygons each.
  Training uses flysch only.
- **Test (b).** Train on all ZLMS-stratum blocks (flysch and non-flysch), test on all "other" blocks. Note that the 42
  F_other sheets (flysch in Jihomoravský/Olomoucký) are part of the (b) test set.
- **Inner validation (early stopping).** Random training blocks are added until they hold ≥ 15 % of the run's
  training eligible polygons (seed 1000 + fold for (a), 2000 for (b), 3000 for the stage-2 model). They are carved
  from the training blocks only.
- All split lists are in `data/11_landslide/splits.json`; the counts are in `results/11/splits_summary.json`.

### 5. Sampling (`scripts/11_build_dataset.py`)

- **Patches.** 300 m × 300 m at 2 m (150 × 150 px), centred on a native pixel corner.
- **Positives.**
  - Random interior points of eligible polygons, k = clip(⌊area / 40,000 m²⌋, 1, 3) points per polygon, ≥ 100 m
    apart. This caps how much a large polygon can contribute.
  - The budget is **4,000** points, split between flysch and non-flysch sheets in proportion to their eligible
    polygons.
  - Within each pool, polygons are water-filled across sheets in random order up to a common per-sheet cap. This
    flattens Zlín's density and maximises the number of independent sheets.
- **Random negatives.**
  - Candidates lie on a jittered 60 m grid inside the same stage-1 sheets.
  - Each candidate's centre is ≥ 150 m from every register polygon (all 21,356, any group) and from ZABAGED
    landslide/scree polygons, and its patch obeys the same patch rule as positives.
- **Look-alikes (target 12 % of negatives).**
  - Sources: up to 2 random interior points per ZABAGED quarry, spoil heap or rock-formation polygon, and points every
    100 m on ZABAGED roads and railways where the 90th-percentile slope in a 42 m window is ≥ 20° (cuts and
    embankments).
  - They have the same distance and patch rules as random negatives.
  - They are matched first, round-robin over types, in the exact slope bin.
- **Slope matching (1:1).**
  - Each positive gets one negative from the same 2° bin of patch-mean slope.
  - Search order:
    1. same sheet;
    2. same block;
    3. adjacent bin, same sheet;
    4. adjacent bin, same block;
    5. same stratum, exact bin;
    6. same stratum, adjacent bin.
  - Selected negatives are ≥ 150 m apart. A positive with no match is dropped (counted).
  - The match level is stored per patch. Strata nest the (a) and (b) test sets, so the pooled test sets stay matched.
- **Background sample (stage-2 model and secondary reference only).** 4,000 unused random-negative candidates, drawn
  uniformly (unmatched).
- **Storage.**
  - `patches.npy`: (N, 7, 150, 150) float16, memory-mappable.
  - `meta.csv`: id, label, x, y, sheet, kraj, flysch, source, fold, group_b, block, stratum, slope_bin, match_level,
    pair_id, terrain and susceptibility statistics.
  - Validity: the patch plus its 28 px channel context is NaN-free (checked with summed-area tables before matching,
    asserted at extraction).

### 6. Channels (`scripts/11_common.py`)

All seven channels are invariant (or equivariant) under the dihedral group D4 of 90° rotations and flips, so D4
augmentation is exact on the pixel grid and needs no hillshade recomputation. The pitfall about rotating a
directional hillshade does not apply.

| # | Channel | Definition |
| --- | --- | --- |
| 0 | slope | degrees, central differences at 2 m |
| 1 | lrm_s | z − 11 × 11 px (22 m) moving mean |
| 2 | lrm_l | z − 51 × 51 px (102 m) moving mean |
| 3 | curv_prof | profile curvature from a Gaussian-smoothed DTM (σ = 1 px), asinh(k / 0.001 m⁻¹) |
| 4 | curv_plan | tangential (plan) curvature, same smoothing and scaling |
| 5 | svf | sky-view factor, 16 directions, radius 20 px (40 m) |
| 6 | detr | patch elevation minus its least-squares plane |

Computed patch-wise with a 28 px margin. This equals the full-raster computation used in stage 2 to ≤ 5 × 10⁻⁹ [E].

### 7. Model and training (`scripts/11_train.py`)

- **Model.** torchvision ResNet-18 with ImageNet (IMAGENET1K_V1) initialisation, cached locally.
  - conv1 is replaced by a 7-channel conv whose filters are the mean of the RGB filters, replicated and scaled by 3/7.
  - The head is a single logit.
- **Normalisation.** Per-channel mean and std from the run's training patches.
- **Optimisation.** BCE-with-logits; AdamW (lr 3e-4, weight decay 0.01); batch 64; bf16 autocast.
  - Schedule: linear warm-up for 2 epochs, then cosine to 0 over 40 epochs.
- **Epochs and early stopping.** At most **40 epochs**. Early stopping on inner-validation ROC AUC (with D4 TTA),
  patience 8; the weights from the best epoch are kept (the model is rolled back to them).
- **Augmentation.** A random D4 transform per sample. No other augmentation.
- **Prediction.** Mean of the sigmoid over the 8 D4 views (TTA).
- **Seeds.** 0, 1, 2 for every run. The run prediction is the mean of the seed probabilities (ensemble). cuDNN is
  deterministic.
- **Monitoring (added before the full run; no effect on training).**
  - Every epoch records running train loss and running train AUC (on the augmented batches as trained), validation
    loss (mean BCE over the 8 views) and validation AUC (8-view TTA), plus learning rate and seconds. These go to the
    run log and to `runs/<run>_s<seed>/history.csv`.
  - The evaluation draws `figures/11_learning_curves.png`: one row per run (a folds 0–4, b, stage-2 model) with a
    BCE panel and an AUC panel. Each seed's train (blue) and validation (orange) curves are faint. The bold curve is
    their mean, drawn while all seeds are still training. Dots mark each seed's best epoch (the one kept). The bold mean is the
    mean of the seeds' curves, not the ensemble's score.
  - Console progress bars show the unit counter, epochs out of the maximum (early stopping may end sooner), batches
    within the epoch, validation progress and stage-2 windows. The bars are never written to the log.
  - A smoke rerun reproduced the earlier losses and AUCs bit for bit.
- **Data handling.** Training patches are held on the GPU, so there is no DataLoader and no worker processes
  (Windows-safe).
- **Runs.**
  - (a): 5 folds × 3 seeds.
  - (b): 1 × 3 seeds.
  - Stage-2 model (only if (a) and (b) pass): 3 seeds, trained on all stage-1 patches **plus** the 4,000 unmatched
    background patches as extra negatives.
  - Why the extra negatives: a model trained only on slope-matched pairs never sees flat or ordinary terrain, and
    stage 2 scores unmatched background.

### 8. Metrics

- **Primary (a).** ROC AUC of the pooled out-of-fold ensemble predictions over all flysch test patches.
- **Primary (b).** ROC AUC on all test patches of the "other" kraje.
- **CI.** Sheet-level cluster bootstrap: resample test sheets with replacement, 2,000 replicates, percentile 95 %,
  seed 123.
- **Paired differences.** CNN − baseline uses the same resamples.
- **Secondary (reported, not decisive).**
  - Block-level bootstrap CI.
  - Per-fold and per-seed AUC.
  - (b) by kraj and by flysch flag.
  - Positives against each negative source (random vs each look-alike type).
  - Slope-only AUC on each test set as a check that matching worked (≈ 0.5 expected).

### 9. Stage 2 (`scripts/11_stage2.py`)

- **Windows.** 150 px windows on a 25 px (50 m) lattice of each reserved sheet's DTM grid, centre inside the sheet.
  Channels are computed on the full sheet raster, and the plane is removed per window on the GPU.
- **Score.** The mean over the stage-2 model's seeds of the D4-TTA probability.
- **Objects.** Greedy non-maximum suppression of the 50 m score lattice with radius **150 m**, half a window, which
  is the localisation scale of a 300 m patch classifier. Ties are broken by a seeded 10⁻⁹ jitter. Every lattice cell
  ends up selected or suppressed, so there is no threshold: the precision–recall curve sweeps all detections in
  descending score.
- **Object-matching rule.**
  - Targets: eligible polygons (skupina 1, in_mapped, ≥ 3,000 m²) assigned to the sheet; 465 in total (278 on flysch
    sheets, 187 on non-flysch sheets).
  - Detections from all 20 sheets are processed in descending score.
    - **TP**: a detection inside a target buffered by **50 m** that is still unmatched; the target becomes matched
      (the lowest objectid if several).
    - **Duplicate (ignored)**: a detection inside an already-matched target.
    - **Ignored**: a detection within **150 m of any register polygon**, targets included. This is the ring that
      training never labelled, because negatives are ≥ 150 m from every polygon.
    - **FP**: every other detection.
  - Object recall = matched targets / all targets; precision = TP / (TP + FP).
- **Rule change before training (from baselines only).** The first draft used 3 × 3 local maxima. On the smooth
  window-score maps this left so few detections that slope and slope+susceptibility could not reach 50 % recall
  (maximum recall 0.41–0.45). The CNN map is built from the same overlapping 300 m windows, so stage 2 would have
  failed by construction. Two alternatives were compared on baseline maps only (NMS radius 100 vs 150 m; ignore
  buffer 50 vs 150 m), and the training-consistent variant above was adopted on principle, not on baseline scores.
  Under all four variants every baseline stayed ≤ 0.14 and random ≤ 0.085.
- **Metric.** Primary **P@R50** = precision at the highest threshold whose recall ≥ 0.50. If 50 % recall is never
  reached, stage 2 FAILs. Secondary: interpolated (max) precision over recall ≥ 0.5, per-stratum results, and the
  maximum recall.
- **Comparisons (same pipeline).**
  - Non-ML baseline: window-mean slope.
  - Non-ML baseline: window-mean susceptibility class.
  - Classical-ML comparator: logistic regression on (slope mean, slope std, susceptibility mean, high-class fraction), fitted on stage-1
    positives vs the unmatched background sample.
  - Random scores (chance level).
  - These are CPU-only and were run in phase 1: `results/11/stage2_baselines.csv`.

### 10. Non-ML baselines and classical-ML comparators (`scripts/11_baselines.py`; same splits, same test patches)

**Terminology (decided after review, before any CNN training; labels only, nothing else changes).** Only the unfitted
scores are baselines in the strict sense. The fitted models are classical machine learning: they are automated and
trained on the same blocks as the CNN, but they use hand-chosen features. Tests (a) and (b) therefore ask two
questions: does the CNN beat the non-ML baselines, and does deep learning beat classical ML on these inputs? The
pass thresholds are absolute AUCs and do not depend on either comparison. The CSVs carry a `kind` column.

Non-ML baselines:
- **slope_only**: patch-mean slope.
- **susceptibility**: patch-mean class (0/1/2) of the expert-made ČGS map. This is a reference for current practice,
  not an automated method.

Classical-ML comparators:
- **slope_susc_lr**: logistic regression on (slope mean, slope std, susc mean, susc high fraction).
- **terrain_stats_lr**: logistic regression on 12 simple statistics: slope mean, std and p90; LRM std at both
  scales; curvature stds; SVF mean and std; detrended std and 2–98 % range; TRI.
- **terrain_stats_gbm** (secondary): histogram gradient boosting on the same 12 statistics.
- Fitted baselines train on the same training blocks (train + inner-val) as the CNN.

### 11. Runner rules (`scripts/11_run_overnight.py`)

- **Order.** (a) folds, then (b), then evaluation. If both pass: the stage-2 model, then stage-2 inference with
  baselines, then the final evaluation.
- **Preflight.** CUDA, ≥ 7 GB free GPU memory, ≥ 10 GB free disk, every cached input present, and patch-array
  shapes consistent. Socket connections are then disabled for the rest of the run.
- **Wall-clock cap.** 6 h (config), with 15 min reserved for evaluation. On reaching it the runner stops gracefully;
  missing units are listed and the verdict is INCOMPLETE (never PASS).
- **Resume.** A unit with `done.json` is skipped.
- **Failures.** An exception in one unit is logged and the runner moves on.

### 12. Deviations from the prompt and known limitations

**Deviations (decided before any CNN training)**

1. **"At least one point per polygon".** There are 15,082 eligible polygons but a 4,000-point budget, so I read this
   as "every *selected* polygon gives 1–3 points". Selection is by per-sheet water-filling (§5). Roughly a quarter of
   the eligible polygons are used.
2. **DTM is cached per surveyed sheet, not per patch.** All 860 sheets plus 250 m margin are cached (~12 GB).
   Benefits:
   - exact slope matching over millions of candidates;
   - identical channel code for patches and stage-2 windows;
   - an offline stage 2.
3. **Susceptibility is rendered per class with MapServer `export`** in the native SR, not queried as vectors
   (reasons in §2). Low and "no data" are merged.
4. **Road and rail cuts or embankments are derived from geometry.** They are ZABAGED road/rail points with a steep
   local slope, because ZABAGED polohopis has no typed cut/embankment layer.
5. **Slope-matching fallbacks.** Matching falls back to block and then stratum level for steep positives that have no
   same-sheet negative. Unmatched positives are dropped and counted (§13).
6. **Test (a) trains on flysch sheets only** ("flysch first"). Test (b) training includes the 39 non-flysch
   Zlínský/Moravskoslezský sheets, because the prompt defines (b) by kraj.
7. **Stage-2 model.** It uses an extra 4,000-patch unmatched background sample as negatives (§7). Without it the
   detector would never have seen flat terrain.
8. **The bootstrap resamples sheets, as specified.** Folds are built from 15 km blocks, so a block-level CI (wider)
   is reported as secondary.
9. **No DataLoader.** Patches are GPU-resident tensors, which is equivalent to `num_workers = 0` and avoids Windows
   multiprocessing entirely.
10. **Hyperparameters were fixed before training.** These are lr, weight decay 0.01, 40 epochs and patience 8. The
    weight decay was set to 0.01 in this phase (1e-4 would be a no-op with AdamW); this was before any training.

11. **Positives: 3,584 against the "about 4,000" cap.** Water-filling with an integer per-sheet cap (8 per flysch
    sheet, 3 per non-flysch sheet) selected 3,613 points; raising either cap by one would overshoot the budget.
    29 positives were then dropped because their patch context touched the border (0 m) cells. I kept this rather
    than add a top-up round after the build had started.
12. **870 of 14,617 eligible stage-1 polygons have no feasible patch.** Their 300 m square cannot lie inside their
    own sheet, or it would touch the stage-2 buffer. These are mostly polygons along sheet edges, a mild selection
    against border-straddling landslides.

**Known limitations and risks [I]**

- **Circularity.** ČGS mappers drew the polygons on this same DMR 5G hillshade. A pass means "agrees with the ČGS
  mappers on held-out sheets", not "finds field-verified landslides".
- **Label noise.** Unmapped landslides may sit among the negatives (1 of 8 in the earlier visual check). Outside the
  flysch, positives include failures of engineered slopes. Both depress the measured AUC.
- **Test (b) mixes geologies.** Its test set includes 42 flysch sheets outside ZL/MS, which are expected to be easier.
  It is reported by flysch flag and kraj.
- **Spatial autocorrelation.** Folds are 15 km blocks, but landslide-prone terrain is regionally clustered. The
  sheet-level CI may be optimistic.
- **Stage-2 base rate.** Precision at fixed recall depends on how densely landslides cover the sheet. The baselines
  in `results/11/stage2_baselines.csv` show what slope, susceptibility and random scores reach on the same 20 sheets.
  If a simple baseline already reaches 0.30, a stage-2 pass alone says little; the CNN-vs-baseline gap then matters.
  The threshold itself is not changed.

## Appendix B: the spike's frozen configuration (`configs/11_landslide.json`, verbatim)

Your frozen config must use the same values. The `paths` entries refer to the spike's repo and don't apply to you.

```json
{
  "name": "11_landslide_kill_test",
  "status": "FROZEN 2026-11-25 at the pre-registration commit (before any CNN training, including the smoke test)",
  "crs": "EPSG:5514",
  "paths": {
    "labels": "data/01_geohazards_mining/svahove_nestability_polygons.gpkg",
    "surveyed_sheets": "data/01_geohazards_mining/prehled_mapovani_svahove_nestability.geojson",
    "data_dir": "data/11_landslide",
    "results_dir": "results/11",
    "smoke_dir": "data/11_landslide/smoke"
  },
  "dtm": {
    "service": "https://ags.cuzk.gov.cz/arcgis2/rest/services/dmr5g/ImageServer (DMR 5G, native 2 m)",
    "cell_m": 2.0,
    "margin_m": 250,
    "invalid_rule": "z <= 0 (ImageServer returns 0.0 outside the national border) or z > 2000 -> NaN"
  },
  "labels": {
    "skupina": 1,
    "in_mapped": true,
    "min_area_m2": 3000
  },
  "patch": {
    "size_m": 300,
    "px": 150
  },
  "channels": [
    "slope",
    "lrm_s",
    "lrm_l",
    "curv_prof",
    "curv_plan",
    "svf",
    "detr"
  ],
  "channel_params": {
    "lrm_small_px": 11,
    "lrm_large_px": 51,
    "curv_sigma_px": 1.0,
    "curv_k0": 0.001,
    "svf_radius_px": 20,
    "svf_dirs": 16
  },
  "flysch": {
    "source": "ČGS geologická mapa 1:500 000 (Geologie/geologicka_mapa500_CR, layer 7 Horniny)",
    "subregions": [
      "Marinní terciér vnějších Karpat",
      "Mezozoické sedimenty Karpat",
      "Marinní mezozoikum a terciér vnějších a vnitřních Karpat",
      "Vulkanity vnějších Karpat",
      "Vulkanity vnějších a vnitřních Karpat"
    ],
    "closing_m": 2000,
    "sheet_rule": "sheet is flysch if >= 50% of its area lies inside the closed flysch outline",
    "min_sheet_frac": 0.5
  },
  "sampling": {
    "seed": 11,
    "max_pos": 4000,
    "max_neg": 4000,
    "pts_area_unit_m2": 40000,
    "max_pts_per_poly": 3,
    "min_pt_spacing_m": 100,
    "neg_min_dist_to_polygon_m": 150,
    "neg_exclude_zabaged_landslide_m": 150,
    "neg_candidate_spacing_m": 60,
    "neg_min_spacing_m": 150,
    "slope_bin_deg": 2.0,
    "lookalike_frac": 0.12,
    "lookalike_types": [
      "quarry",
      "spoil",
      "rock",
      "cut"
    ],
    "lookalike_max_per_polygon": 2,
    "cut_point_step_m": 100,
    "cut_window_px": 21,
    "cut_slope_p90_deg": 20.0
  },
  "splits": {
    "block_km": 15,
    "n_folds_a": 5,
    "inner_val_frac": 0.15,
    "transfer_train_kraje": [
      "Zlínský",
      "Moravskoslezský"
    ],
    "sheet_kraj_rule": "majority gen_kraj (first-listed kraj) of all register polygons intersecting the sheet",
    "stage2": {
      "n_flysch": 10,
      "n_nonflysch": 10,
      "min_eligible_polygons": 5,
      "full_sheets_only": true,
      "seed": 2026,
      "require_valid_dtm": "whole cached raster (sheet bbox + 250 m) valid: no 0-m cells (outside the border) or NaN"
    }
  },
  "model": {
    "arch": "resnet18",
    "init": "torchvision IMAGENET1K_V1; conv1 = mean of RGB filters replicated over channels x 3/C",
    "in_channels": 7,
    "head": "single logit"
  },
  "train": {
    "loss": "BCEWithLogits",
    "optimizer": "AdamW",
    "lr": 0.0003,
    "weight_decay": 0.01,
    "batch_size": 64,
    "max_epochs": 40,
    "warmup_epochs": 2,
    "schedule": "linear warmup then cosine to 0 over max_epochs",
    "early_stopping": {
      "metric": "inner-val ROC AUC",
      "patience": 8,
      "restore_best": true
    },
    "amp": "bf16 autocast",
    "augmentation": "random dihedral D4 (8 rotations/flips, exact on the pixel grid)",
    "normalisation": "per-channel mean/std from the run's training patches",
    "tta": "mean of sigmoid over the 8 D4 views",
    "seeds": [
      0,
      1,
      2
    ],
    "ensemble": "mean of seed probabilities",
    "deterministic": true
  },
  "eval": {
    "primary_a": "pooled out-of-fold ROC AUC over all held-out flysch patches (5 folds)",
    "primary_b": "ROC AUC on all patches of non-Zlínský/non-Moravskoslezský sheets",
    "bootstrap": {
      "unit": "sheet",
      "n": 2000,
      "ci": 0.95,
      "method": "percentile",
      "seed": 123
    },
    "thresholds": {
      "a_auc_min": 0.75,
      "b_auc_min": 0.65,
      "stage2_recall": 0.5,
      "stage2_precision_min": 0.3
    }
  },
  "stage2": {
    "model": "same configuration retrained on all stage-1 patches (matched) PLUS the unmatched background sample (4,000 patches, uniform over the stage-1 sheets' feasible area, >= 150 m from polygons) as extra negatives; 15% of blocks as inner validation; seeds [0,1,2]; ensemble",
    "background_n": 4000,
    "stride_px": 25,
    "nms_radius_m": 150,
    "match_buffer_m": 50,
    "ignore_buffer_m": 150,
    "object_rule": "detections = greedy NMS (radius 150 m) of the 50 m window-score lattice; TP = first detection inside a target (eligible polygon assigned to the sheet) buffered 50 m; duplicate on a matched target = ignored; within 150 m of any register polygon = ignored; else FP",
    "precision_rule": "precision at the highest threshold whose object recall >= 0.5 (primary); interpolated max precision at recall >= 0.5 (secondary)",
    "rule_change_note": "3x3 local maxima (first draft) capped max recall at 0.41-0.45 for smooth baseline maps; replaced before any CNN training using baseline maps only"
  },
  "runner": {
    "wall_clock_h": 6.0,
    "min_free_disk_gb": 10,
    "num_workers": 0
  },
  "smoke": {
    "patch_frac": 0.05,
    "max_epochs": 2,
    "folds": [
      0
    ],
    "seeds": [
      0
    ],
    "runs": [
      "a_f0",
      "b",
      "s2"
    ],
    "stage2_sheets": 1,
    "note": "proves the code runs end to end; smoke metrics are never used to change the configuration"
  }
}
```
