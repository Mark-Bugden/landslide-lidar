"""The D19 patch statistics (`landslide.features`) against brute-force formulas."""

from __future__ import annotations

import math

import numpy as np
import pytest

from landslide.channels import full_channels, patch_channels
from landslide.config import load_config
from landslide.dtm import extract_block
from landslide.features import NEIGHBOURS, STAT_COLUMNS, patch_stats, susc_stats, tri
from landslide.sat import ValueSat
from terrain_synth import synthetic_dtm

NAMES = ("slope", "lrm_s", "lrm_l", "curv_prof", "curv_plan", "svf", "detr")


def _mean(v):
    return math.fsum(v) / len(v)


def _std(v):
    m = _mean(v)
    return math.sqrt(math.fsum((x - m) ** 2 for x in v) / len(v))


def _pct(v, q):
    """Linear interpolation between closest ranks (numpy's default), written out."""
    s = sorted(v)
    h = (len(s) - 1) * q / 100.0
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def _tri_brute(z):
    H, W = z.shape
    per = []
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            per.append(sum(abs(z[i, j] - z[i + di, j + dj]) for di, dj in NEIGHBOURS) / 8.0)
    return _mean(per)


def _random_patch(rng, px=12):
    ch = rng.normal(size=(7, px, px)) * np.array([10, 2, 5, 1, 1, 0.1, 3])[:, None, None]
    ch[0] = np.abs(ch[0])
    ch[5] = rng.uniform(0.5, 1.0, size=(px, px))
    z = 400 + rng.normal(size=(px + 2, px + 2)).cumsum(axis=0)
    susc = rng.integers(0, 3, size=(px, px)).astype(np.uint8)
    return ch, z, susc


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stats_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    ch, z, susc = _random_patch(rng, px=13)
    st = patch_stats(ch, NAMES, z, susc)
    assert tuple(st) == STAT_COLUMNS
    v = {n: ch[i].ravel().tolist() for i, n in enumerate(NAMES)}
    want = {
        "slope_mean": _mean(v["slope"]), "slope_std": _std(v["slope"]), "slope_p90": _pct(v["slope"], 90),
        "lrm_s_std": _std(v["lrm_s"]), "lrm_l_std": _std(v["lrm_l"]), "curv_prof_std": _std(v["curv_prof"]),
        "curv_plan_std": _std(v["curv_plan"]), "svf_mean": _mean(v["svf"]), "svf_std": _std(v["svf"]),
        "detr_std": _std(v["detr"]), "detr_range_2_98": _pct(v["detr"], 98) - _pct(v["detr"], 2),
        "tri": _tri_brute(z),
        "susc_mean": sum(int(x) for x in susc.ravel()) / susc.size,
        "susc_high_frac": sum(int(x == 2) for x in susc.ravel()) / susc.size,
    }
    for k in STAT_COLUMNS:
        assert st[k] == pytest.approx(want[k], rel=1e-12, abs=1e-12), k


def test_channel_order_follows_names():
    """Statistics follow the channel names, not positions: a permuted stack gives the same statistics."""
    rng = np.random.default_rng(5)
    ch, z, susc = _random_patch(rng)
    perm = [6, 0, 5, 1, 4, 2, 3]
    a = patch_stats(ch, NAMES, z, susc)
    b = patch_stats(ch[perm], tuple(NAMES[i] for i in perm), z, susc)
    assert a == b


def test_tri_brute_force_and_plane():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(20, 17)) * 3 + 500
    assert tri(z) == pytest.approx(_tri_brute(z), rel=1e-13)
    # a plane z = a*col + b*row: every pixel's 8 differences are |a|, |a|, |b|, |b|, |a+b| x 2, |a-b| x 2
    a, b = 0.7, -1.9
    ii, jj = np.mgrid[0:30, 0:25].astype(np.float64)
    plane = 300 + a * jj + b * ii
    assert tri(plane) == pytest.approx((abs(a) + abs(b) + abs(a + b) + abs(a - b)) / 4, rel=1e-12)
    assert tri(np.full((5, 5), 412.5)) == 0.0
    zn = z.copy()
    zn[0, 0] = np.nan                        # a ring corner is a neighbour of interior pixel (1, 1)
    assert np.isnan(tri(zn))
    with pytest.raises(ValueError):
        tri(np.zeros((2, 5)))


def test_tri_uses_the_ring():
    """The patch-edge pixels' neighbours come from the 1 px ring: changing only a ring cell changes the TRI."""
    z = np.full((8, 8), 100.0)
    z[0, 3] = 101.0                          # ring cell, neighbour of patch pixels (1, 2), (1, 3), (1, 4)
    assert tri(z) == pytest.approx(3 * (1 / 8) / 36)


def test_susceptibility_stats():
    s = np.zeros((10, 10), dtype=np.uint8)
    s[:3] = 2                                # 30 high
    s[3:5] = 1                               # 20 mid
    assert susc_stats(s) == (pytest.approx((30 * 2 + 20) / 100), pytest.approx(0.3))
    assert susc_stats(np.zeros((4, 4), dtype=np.uint8)) == (0.0, 0.0)
    assert susc_stats(np.full((4, 4), 2, dtype=np.uint8)) == (2.0, 1.0)
    bad = s.copy()
    bad[0, 0] = 3
    with pytest.raises(ValueError, match="unexpected susceptibility codes"):
        susc_stats(bad)


def test_patch_stats_rejects_bad_input():
    rng = np.random.default_rng(1)
    ch, z, susc = _random_patch(rng)
    with pytest.raises(TypeError):
        patch_stats(ch.astype(np.float16), NAMES, z, susc)          # statistics come from float64, never float16
    bad = ch.copy()
    bad[2, 3, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        patch_stats(bad, NAMES, z, susc)
    zb = z.copy()
    zb[0, 5] = np.nan
    with pytest.raises(ValueError, match="ring"):
        patch_stats(ch, NAMES, zb, susc)
    with pytest.raises(ValueError):
        patch_stats(ch, NAMES, z[1:, 1:], susc)
    with pytest.raises(ValueError):
        patch_stats(ch[:6], NAMES, z, susc)


def test_slope_mean_of_patches_equals_the_sampling_sat_mean():
    """P1-09 stores slope_mean as a summed-area-table mean of `full_channels` slope; the patch statistic (mean of the
    float64 `patch_channels` slope) must equal it to 1e-9 at every corner, as the build asserts."""
    cfg = load_config()
    z = synthetic_dtm((700, 640), seed=12, border=False, holes=False, lake=True)
    sat = ValueSat(full_channels(z, cfg, names=["slope"])[0])
    rng = np.random.default_rng(4)
    worst = 0.0
    for r, c in zip(rng.integers(103, 700 - 103, 25), rng.integers(103, 640 - 103, 25)):
        ch = patch_channels(z, int(r), int(c), cfg)
        st = patch_stats(ch, cfg["channels"], extract_block(z, r - 76, c - 76, r + 76, c + 76),
                         np.zeros((150, 150), dtype=np.uint8))
        worst = max(worst, abs(st["slope_mean"] - float(sat.mean(r, c, 75))))
        assert 0.0 < st["svf_mean"] <= 1.0 and st["slope_std"] > 0 and st["tri"] > 0
    assert worst < 1e-9
