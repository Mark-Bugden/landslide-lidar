"""A synthetic dataset world for the P1-10 tests.

P1-09's synthetic world (`sampling_synth.make_world`: 6 sheets, polygons, ZABAGED, DTMs with an invalid border corner
and holes) is sampled with `landslide.sampling.build_samples`, and its rasters are written as GeoTIFFs in the layout of
`data/` (`LANDSLIDE_DATA_DIR`):

- `raw_data/dmr5g/<sheet>.tif`: the synthetic DTM, float64 (so read_dtm returns exactly what P1-09 sampled from);
- `cache/susceptibility/<sheet>.tif`: uint8 class codes 0/1/2 (blobs of a smooth random field) on the DTM grid;
- `derived/sheets.parquet`: the sheet table with `dtm_path`, EPSG:5514;
- `dataset/samples.parquet`: the sampled table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

from landslide import sampling as S
from landslide.patches import SheetSource
from landslide.sheets import grid_origin
from landslide.paths import sheet_to_stem
from sampling_synth import make_world


@dataclass
class World:
    cfg: dict
    inputs: S.Inputs
    samples: pd.DataFrame
    data: Path
    sheets: gpd.GeoDataFrame
    z: dict[str, np.ndarray]
    susc: dict[str, np.ndarray]

    def sources(self) -> dict[str, SheetSource]:
        cell = float(self.cfg["dtm"]["cell_m"])
        susc_dir = self.data / "cache" / "susceptibility"
        return {row.sheet: SheetSource(sheet=row.sheet, dtm_path=self.data / "raw_data" / row.dtm_path,
                                       susc_path=susc_dir / f"{sheet_to_stem(row.sheet)}.tif",
                                       grid=S.grid_of_sheet(row, cell, grid_origin(self.cfg)))
                for row in self.sheets.itertuples(index=False)}


def susc_codes(shape: tuple[int, int], seed: int) -> np.ndarray:
    """0 / 1 / 2 blobs: a smooth random field cut at its 45th and 75th percentiles."""
    f = ndi.gaussian_filter(np.random.default_rng(seed).normal(size=shape), 25.0, mode="wrap")
    a, b = np.percentile(f, [45, 75])
    return np.where(f < a, 0, np.where(f < b, 1, 2)).astype(np.uint8)


def _write(path: Path, arr: np.ndarray, transform, nodata=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype.name, crs="EPSG:5514", transform=transform, nodata=nodata) as dst:
        dst.write(arr, 1)


def write_world(root: Path, seed: int = 7) -> World:
    inputs, cfg = make_world(seed)
    res = S.build_samples(cfg, inputs)
    data = Path(root) / "data"
    z, susc = {}, {}
    dtm_paths = []
    for k, sheet in enumerate(inputs.sheets["sheet"]):
        d = inputs.read_dtm(sheet)
        stem = sheet_to_stem(sheet)
        _write(data / "raw_data" / "dmr5g" / f"{stem}.tif", d.z.astype(np.float64), d.transform, nodata=np.nan)
        codes = susc_codes(d.z.shape, 500 + k)
        _write(data / "cache" / "susceptibility" / f"{stem}.tif", codes, d.transform)
        z[sheet], susc[sheet] = d.z, codes
        dtm_paths.append(f"dmr5g/{stem}.tif")
    sheets = gpd.GeoDataFrame(inputs.sheets.assign(dtm_path=dtm_paths), geometry="geometry", crs="EPSG:5514")
    (data / "derived").mkdir(parents=True, exist_ok=True)
    sheets.to_parquet(data / "derived" / "sheets.parquet")
    (data / "dataset").mkdir(parents=True, exist_ok=True)
    res.samples.to_parquet(data / "dataset" / "samples.parquet", index=False)
    return World(cfg=cfg, inputs=inputs, samples=res.samples, data=data, sheets=sheets, z=z, susc=susc)
