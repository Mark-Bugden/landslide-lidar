"""A small synthetic world for the P1-09 sampling tests: 6 sheets, register polygons, ZABAGED layers and DTMs.

Sheets are a 3 x 2 block of rotated rectangles (2.0 km x 1.6 km, turned 5 degrees, sharing edges exactly), with grids
built like the brief's recipe (bounds + 250 m, snapped outwards to the DMR 5G lattice x = 0.4, y = -0.12 mod 2 m).
Sheet (1, 1) is the stage-2 sheet. Strata: (0, 0) F_ZLMS, (1, 0) and (2, 0) F_other (one block), (0, 1) N_ZLMS,
(2, 1) N_other. Polygons include ones that straddle sheet edges or sit next to the stage-2 sheet (infeasible), large
ones (k = 2, 3), small and skupina-2 ones (ineligible, but still exclusions). One sheet has an invalid "border"
corner and one has invalid holes, so some positives fail validity.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from landslide.config import load_config
from landslide.dtm import Dtm
from landslide.geo import assign_to_sheets
from landslide.sampling import Inputs
from landslide.sheets import dtm_grid
from terrain_synth import synthetic_dtm

# DTM transforms differ from the stored grid by 3e-10 m, as a raster's may differ from P1-08's rounded corner.
ORIGIN = (-600_000.0, -1_150_000.0)
W_M, H_M, THETA = 2000.0, 1600.0, math.radians(5.0)
STAGE2 = (1, 1)
ATTRS = {  # (i, j): flysch, kraj, block, fold
    (0, 0): (True, "Zlínský", "F_ZLMS-001", 0),
    (1, 0): (True, "Jihomoravský", "F_other-001", 1),
    (2, 0): (True, "Jihomoravský", "F_other-001", 1),
    (0, 1): (False, "Moravskoslezský", "N_ZLMS-001", None),
    (1, 1): (False, "Jihomoravský", None, None),
    (2, 1): (False, "Olomoucký", "N_other-001", None),
}


def synth_config(max_pos: int = 40, background_n: int = 30, **sampling) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["sampling"]["max_pos"] = max_pos
    cfg["stage2"]["background_n"] = background_n
    cfg["sampling"].update(sampling)
    return cfg


def _corner(i: float, j: float) -> tuple[float, float]:
    u = (W_M * math.cos(THETA), W_M * math.sin(THETA))
    v = (-H_M * math.sin(THETA), H_M * math.cos(THETA))
    return ORIGIN[0] + i * u[0] + j * v[0], ORIGIN[1] + i * u[1] + j * v[1]


def sheet_polygon(i: int, j: int) -> Polygon:
    return Polygon([_corner(i, j), _corner(i + 1, j), _corner(i + 1, j + 1), _corner(i, j + 1)])


def local_point(i: int, j: int, fu: float, fv: float) -> tuple[float, float]:
    """The point at fractions (fu, fv) of sheet (i, j)."""
    return _corner(i + fu, j + fv)


def ellipse(x: float, y: float, a: float, b: float, angle: float = 0.0, n: int = 48) -> Polygon:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ca, sa = math.cos(angle), math.sin(angle)
    px, py = a * np.cos(t), b * np.sin(t)
    return Polygon(np.column_stack([x + ca * px - sa * py, y + sa * px + ca * py]))


def make_world(seed: int = 7, cfg: dict | None = None) -> tuple[Inputs, dict]:
    cfg = cfg or synth_config()
    rng = np.random.default_rng(seed)
    rows = []
    dtms: dict[str, Dtm] = {}
    for (i, j), (flysch, kraj, block, fold) in ATTRS.items():
        sheet = f"90-{i + 1:02d}-{j + 1:02d}"
        geom = sheet_polygon(i, j)
        g = dtm_grid(geom.bounds, cfg)
        group = "train" if kraj in ("Zlínský", "Moravskoslezský") else "test"
        stratum = ("F_" if flysch else "N_") + ("ZLMS" if group == "train" else "other")
        rows.append({"sheet": sheet, "geometry": geom, "grid_xmin": g.xmin, "grid_ymax": g.ymax, "grid_w": g.w,
                     "grid_h": g.h, "flysch": flysch, "kraj": kraj, "group_b": group, "stratum": stratum,
                     "block": block, "fold_a": fold, "is_stage2": (i, j) == STAGE2, "ij": (i, j)})
        z = synthetic_dtm((g.h, g.w), seed=100 + 10 * i + j, border=(i, j) == (2, 1), holes=(i, j) == (0, 1),
                          lake=(i, j) == (0, 0))
        dtms[sheet] = Dtm(z=z, transform=Affine(2.0, 0.0, g.xmin + 3e-10, 0.0, -2.0, g.ymax - 3e-10), crs=None,
                          path=Path(f"synthetic/{sheet}.tif"))
    sheets = pd.DataFrame(rows)

    polys = []

    def add(geom: Polygon, skupina: int = 1) -> None:
        polys.append({"objectid": len(polys) + 1, "skupina": skupina, "geometry": geom})

    for (i, j) in ATTRS:
        if (i, j) == STAGE2:
            continue
        for _ in range(9):   # interior polygons, mostly small
            fu, fv = rng.uniform(0.12, 0.88), rng.uniform(0.14, 0.86)
            a = rng.uniform(25, 140)
            add(ellipse(*local_point(i, j, fu, fv), a, a * rng.uniform(0.4, 1.0), rng.uniform(0, np.pi)))
        add(ellipse(*local_point(i, j, 0.5, 0.5), 260, 170, 0.3))                       # large: k = 3
        add(ellipse(*local_point(i, j, rng.uniform(0.2, 0.8), 0.02), 60, 40))           # straddles the j edge
        add(ellipse(*local_point(i, j, 0.03, rng.uniform(0.3, 0.7)), 80, 30, 1.2))      # near the i edge
        add(ellipse(*local_point(i, j, rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)), 25, 20))  # < 3000 m2
        add(ellipse(*local_point(i, j, rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)), 70, 50), skupina=2)
    # next to the stage-2 sheet: inside (1, 0) and (2, 1), within 150 m + half a patch of (1, 1)
    add(ellipse(*local_point(1, 0, 0.5, 0.93), 70, 35))
    add(ellipse(*local_point(2, 1, 0.08, 0.5), 35, 70))
    add(ellipse(*local_point(1, 1, 0.5, 0.5), 90, 60))      # inside the stage-2 sheet (a stage-2 target)
    P = pd.DataFrame(polys)
    P["area_m2"] = shapely.area(P["geometry"].to_numpy())
    P["in_mapped"] = True
    P["eligible"] = (P["skupina"] == 1) & (P["area_m2"] >= cfg["labels"]["min_area_m2"])
    sheet_of, _ = assign_to_sheets(P["geometry"].to_numpy(), sheets["geometry"].to_numpy(), sheets["sheet"])
    P["sheet"] = np.where(P["eligible"], sheet_of, None)
    n_elig = P.loc[P["eligible"]].groupby("sheet").size()
    sheets["n_eligible"] = sheets["sheet"].map(n_elig).fillna(0).astype(int)
    sheets["is_stage1"] = ~sheets["is_stage2"] & (sheets["n_eligible"] >= 1)
    sheets["fold_a"] = pd.array(sheets["fold_a"], dtype="Int64")

    # ZABAGED
    zab: dict[str, pd.DataFrame] = {}
    oid = iter(range(5001, 10_000))

    def frame(geoms: list) -> pd.DataFrame:
        return pd.DataFrame({"objectid": [next(oid) for _ in geoms], "geometry": geoms})

    stage1 = [ij for ij in ATTRS if ij != STAGE2]
    zab["quarry"] = frame([ellipse(*local_point(i, j, rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)),
                                   rng.uniform(30, 90), rng.uniform(20, 60), rng.uniform(0, 3)) for i, j in stage1
                           for _ in range(3)])
    zab["spoil"] = frame([ellipse(*local_point(i, j, rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)), 50, 30)
                          for i, j in stage1 for _ in range(2)]
                         + [ellipse(*local_point(0, 0, 0.99, 0.5), 60, 40)])   # straddles (0,0) | (1,0)
    zab["rock"] = frame([Point(*local_point(i, j, rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)))
                         for i, j in stage1 for _ in range(3)]
                        + [MultiPoint([local_point(2, 0, 0.4, 0.4), local_point(2, 0, 0.42, 0.4)]),
                           ellipse(*local_point(0, 1, 0.6, 0.6), 40, 40)])
    zab["landslide"] = frame([ellipse(*local_point(i, j, rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)), 60, 40)
                              for i, j in stage1])
    zab["road"] = frame([LineString([local_point(0, j, 0.02, 0.1 + 0.15 * k), local_point(3, j, 0.98, 0.15 * k + 0.2)])
                         for j in (0, 1) for k in range(5)])
    zab["railway"] = frame([LineString([local_point(0, 0, 0.3, 0.05), local_point(0, 1, 0.35, 0.95),
                                        local_point(2, 1, 0.5, 0.9)])])
    inputs = Inputs(sheets=sheets.drop(columns="ij"), polygons=P, zabaged=zab, read_dtm=lambda s: dtms[s])
    return inputs, cfg
