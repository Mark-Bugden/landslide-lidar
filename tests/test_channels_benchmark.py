"""Channel benchmark on a synthetic sheet-sized raster (P1-04). Opt-in: set LANDSLIDE_BENCH=1.

    $env:LANDSLIDE_BENCH = "1"; uv run pytest tests/test_channels_benchmark.py -s

Times `full_channels` on a 2,850 x 2,400 raster (the size of a DMR 5G sheet plus its 250 m margin) with the default
workers and with one worker, and `patch_channels` on single patches; checks the traced peak memory and that tiled,
direct whole-array and patch-wise results agree at full size. The decisions file records the numbers.
"""

import os
import time
import tracemalloc

import numpy as np
import pytest

from landslide import channels as ch
from landslide.config import load_config
from terrain_synth import synthetic_dtm

pytestmark = pytest.mark.skipif(os.environ.get("LANDSLIDE_BENCH") != "1", reason="benchmark: set LANDSLIDE_BENCH=1")

CFG = load_config()
SHAPE = (2850, 2400)


def test_benchmark(capsys):
    p = ch.channel_params(CFG)
    half = p.patch_px // 2
    z = synthetic_dtm(SHAPE, seed=2026)
    ch.full_channels(z[:400, :400], CFG)  # warm-up

    tracemalloc.start()
    t = time.perf_counter()
    full = ch.full_channels(z, CFG)
    t_full = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t = time.perf_counter()
    full1 = ch.full_channels(z, CFG, workers=1)
    t_full1 = time.perf_counter() - t
    assert np.array_equal(full, full1, equal_nan=True)

    rng = np.random.default_rng(0)
    corners = list(zip(rng.integers(half + 40, SHAPE[0] - half - 40, 30),
                       rng.integers(half + 40, SHAPE[1] - half - 40, 30)))
    times, worst_patch = [], 0.0
    for r, c in corners:
        t = time.perf_counter()
        pc = ch.patch_channels(z, r, c, CFG)
        times.append(time.perf_counter() - t)
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        for j, name in enumerate(p.terrain_names):
            a, b = pc[p.names.index(name)], full[j][sl]
            assert np.array_equal(np.isnan(a), np.isnan(b))
            if np.isfinite(a).any():
                worst_patch = max(worst_patch, float(np.nanmax(np.abs(a - b))))

    direct = ch.terrain_channels(z, CFG)  # one whole-array pass, no tiles (memory-heavy; benchmark only)
    assert np.array_equal(np.isnan(direct), np.isnan(full))
    worst_direct = float(np.nanmax(np.abs(direct - full)))

    with capsys.disabled():
        print(f"\nfull_channels {SHAPE}: {t_full:.2f} s (default workers), {t_full1:.2f} s (1 worker); "
              f"traced peak {peak / 1e6:.0f} MB, output {full.nbytes / 1e6:.0f} MB")
        print(f"patch_channels: median {1e3 * np.median(times):.1f} ms, max {1e3 * np.max(times):.1f} ms "
              f"over {len(times)} patches")
        print(f"max |patch - full| = {worst_patch:.3g}; max |direct - tiled| = {worst_direct:.3g}")
    assert peak < 1.0e9
    assert worst_patch <= 5e-9 and worst_direct <= 5e-9
