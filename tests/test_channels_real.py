"""Real-DTM integration test for the channels (P1-04).

Runs only after gate G1: it skips unless `results/provenance.json` records `gate_G1.pass == true` (the P1-03
schema) **and** the DMR 5G rasters are present under the data directory (`LANDSLIDE_DATA_DIR` or `data/`). The
brief forbids using rows 1-4 before the provenance check passes, so nothing here reads a raster before both checks.

It reads one sheet (the first raster in sorted order), applies the invalid rule, computes the full-raster channels
and checks patch-wise vs full raster (<= 5e-9), D4 equivariance on a real patch (<= 1e-9) and plausible ranges.
"""

import json
import time

import numpy as np
import pytest

from landslide import channels as ch
from landslide import d4, dtm
from landslide.config import load_config
from landslide.paths import Paths
from landslide.sat import ValueSat

pytestmark = pytest.mark.needs_data

CFG = load_config()
P = ch.channel_params(CFG)
IDX = {n: i for i, n in enumerate(P.names)}
HALF = P.patch_px // 2


def _gate_g1(paths: Paths) -> tuple[bool, str]:
    p = paths.results / "provenance.json"
    if not p.exists():
        return False, f"{p} is missing"
    try:
        result = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, f"{p} is unreadable: {e}"
    if result.get("gate_G1", {}).get("pass") is not True:
        return False, f"{p} does not record gate_G1.pass == true"
    return True, ""


@pytest.fixture(scope="module")
def sheet():
    paths = Paths()
    ok, why = _gate_g1(paths)
    if not ok:
        pytest.skip(f"gate G1 not passed: {why}")
    if not paths.dmr5g_dir.is_dir():
        pytest.skip(f"no DTM rasters at {paths.dmr5g_dir}")
    tifs = sorted(paths.dmr5g_dir.glob("*.tif"))
    if not tifs:
        pytest.skip(f"no DTM rasters at {paths.dmr5g_dir}")
    return dtm.read_dtm(tifs[0], CFG)


@pytest.fixture(scope="module")
def full(sheet):
    t = time.perf_counter()
    out = ch.full_channels(sheet.z, CFG)
    print(f"\nfull_channels on {sheet.path.name} {sheet.shape}: {time.perf_counter() - t:.2f} s")
    return out


def test_invalid_rule_and_grid(sheet):
    lo, hi = dtm.invalid_bounds(CFG)
    z = sheet.z[sheet.valid]
    assert sheet.z.dtype == np.float64 and z.size > 0
    assert z.min() > lo and z.max() <= hi
    x0, y0 = dtm.corner_xy(sheet.transform, 0, 0)
    cell = CFG["dtm"]["cell_m"]
    assert sheet.transform.a == cell and sheet.transform.e == -cell
    assert min((x0 - 0.4) % cell, cell - (x0 - 0.4) % cell) < 1e-6    # brief note 1: x = 0.4 (mod 2 m)
    assert min((y0 + 0.12) % cell, cell - (y0 + 0.12) % cell) < 1e-6  # y = -0.12 (mod 2 m)


def test_patch_vs_full_on_real_sheet(sheet, full):
    counts = ValueSat(sheet.z).count_grid(HALF + P.context_px)
    need = (2 * (HALF + P.context_px)) ** 2
    rng = np.random.default_rng(0)
    clean = np.argwhere(counts == need)
    assert len(clean) > 0, "no fully valid patch on this sheet"
    picks = [tuple(clean[i]) for i in rng.choice(len(clean), size=min(12, len(clean)), replace=False)]
    partial = np.argwhere((counts > 0) & (counts < need))
    H, W = sheet.shape
    partial = [tuple(rc) for rc in partial if HALF <= rc[0] <= H - HALF and HALF <= rc[1] <= W - HALF]
    if partial:
        picks += [partial[i] for i in rng.choice(len(partial), size=min(4, len(partial)), replace=False)]
    terrain = {n: j for j, n in enumerate(P.terrain_names)}
    for r, c in picks:
        pc = ch.patch_channels(sheet.z, r, c, CFG)
        sl = (slice(r - HALF, r + HALF), slice(c - HALF, c + HALF))
        for name, j in terrain.items():
            a, b = pc[IDX[name]], full[j][sl]
            assert np.array_equal(np.isnan(a), np.isnan(b)), name
            if np.isfinite(a).any():
                assert np.nanmax(np.abs(a - b)) <= 5e-9, name
        if counts[r, c] == need:
            assert np.isfinite(pc).all()
            assert 0 <= pc[IDX["slope"]].min() and pc[IDX["slope"]].max() < 90
            assert 0 <= pc[IDX["svf"]].min() and pc[IDX["svf"]].max() <= 1
            assert np.abs(pc[IDX["detr"]].mean()) < 1e-6


def test_d4_on_real_patch(sheet):
    counts = ValueSat(sheet.z).count_grid(HALF + P.context_px)
    r, c = np.argwhere(counts == (2 * (HALF + P.context_px)) ** 2)[0]
    m = HALF + P.context_px
    block = sheet.z[r - m:r + m, c - m:c + m]
    ref = ch.compute_channels(block, CFG)
    for k in range(8):
        got = ch.compute_channels(np.ascontiguousarray(d4.transform(block, k)), CFG)
        want = d4.transform(ref, k)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.nanmax(np.abs(got - want)) <= 1e-9
