# Locating the Landslides

An independent replication of a pre-registered test: does a CNN trained on Czech DMR 5G LiDAR terrain recognise the
landslides mapped in the ČGS slope-deformation register? A demo follows only if the replication succeeds.

**Status: work in progress (phase 1: pipeline and pre-registration). No CNN has been trained yet.**

## What this repo does

An earlier research spike reported that a ResNet-18 trained on 300 m × 300 m patches of 2 m LiDAR terrain separates
mapped landslides from slope-matched terrain on map sheets and regions it never saw. This repo re-implements that
method from its written description, rebuilds every input from public sources, trains a new model and checks the
spike's three pre-registered tests. The spike's numbers are claims to verify; nothing from the spike produces a result
here. The method, the frozen configuration and the replication criteria are in [`docs/BRIEF.md`](docs/BRIEF.md) and,
once frozen, `PREREGISTRATION.md`.

## Setup

Python 3.13 with [uv](https://docs.astral.sh/uv/). The lock file pins a CUDA 13.0 build of PyTorch.

```powershell
uv sync
uv run python scripts/check_env.py   # writes results/environment.json; needs a CUDA GPU
uv run pytest
```

All data lives in the gitignored `data/` (override with `LANDSLIDE_DATA_DIR`). The `scripts/fetch_*.py` scripts
rebuild every input from its public source.

## Data credits

- **ČÚZK** (Český úřad zeměměřický a katastrální): DMR 5G digital terrain model, ZABAGED® polohopis and orthophoto,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **ČGS** (Česká geologická služba): slope-deformation register (Registr svahových nestabilit),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **ImageNet weights** for ResNet-18 from torchvision (`ResNet18_Weights.IMAGENET1K_V1`), under torchvision's terms.

Three further ČGS inputs have no stated or verified licence: the overview of surveyed map sheets, the 1:500,000
geological map and the landslide susceptibility map. They are used only inside the gitignored `data/` directory.
Neither they nor anything derived from them (sheet outlines, the flysch outline, susceptibility rasters) is committed,
drawn in a figure or shown in the demo. Only aggregate results that use them appear here: AUCs, per-sheet flysch flags
and map-sheet numbers.

## Licence

The code is MIT-licensed (see [`LICENSE`](LICENSE)). The data keeps its own licences, listed above.
