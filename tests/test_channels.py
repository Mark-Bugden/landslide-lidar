"""Terrain channels (P1-04): D4 equivariance, patch-wise vs full raster, analytic cases, NaN propagation.

Synthetic data only (no real DTM before gate G1).
"""

import copy
import math

import numpy as np
import pytest
from scipy import ndimage as ndi

from landslide import channels as ch
from landslide import d4
from landslide.config import load_config
from terrain_synth import synthetic_dtm

CFG = load_config()
P = ch.channel_params(CFG)
NAMES = P.names
IDX = {n: i for i, n in enumerate(NAMES)}
TERRAIN_IDX = {n: i for i, n in enumerate(P.terrain_names)}
HALF = P.patch_px // 2


def _cfg(**changes):
    cfg = copy.deepcopy(CFG)
    for path, value in changes.items():
        node = cfg
        *head, last = path.split("__")
        for part in head:
            node = node.setdefault(part, {})
        node[last] = value
    return cfg


def _assert_close(a, b, atol, what=""):
    """Identical NaN masks and max |a - b| <= atol on the finite cells."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, what
    na, nb = np.isnan(a), np.isnan(b)
    assert np.array_equal(na, nb), f"{what}: NaN masks differ ({int((na != nb).sum())} cells)"
    if (~na).any():
        err = float(np.max(np.abs(a[~na] - b[~na])))
        assert err <= atol, f"{what}: max abs diff {err:.3g} > {atol:g}"


# ---- parameters and the margin budget ----

def test_params_come_from_config():
    cp = CFG["channel_params"]
    assert P.names == tuple(CFG["channels"])
    assert P.cell_m == CFG["dtm"]["cell_m"] and P.patch_px == CFG["patch"]["px"]
    assert (P.lrm_small_px, P.lrm_large_px, P.svf_radius_px, P.svf_dirs) == (
        cp["lrm_small_px"], cp["lrm_large_px"], cp["svf_radius_px"], cp["svf_dirs"])
    assert (P.curv_sigma_px, P.curv_k0) == (cp["curv_sigma_px"], cp["curv_k0"])


def test_margin_budget():
    radii = ch.check_margin(P)
    # PLAN.md P1-04: LRM 25 px, SVF 20 px (+ interpolation), Gaussian 4 px + 1 for the derivatives
    assert radii == {"slope": 1, "lrm_s": 5, "lrm_l": 25, "curv_prof": 5, "curv_plan": 5, "svf": 20}
    assert max(radii.values()) <= P.context_px == 28


@pytest.mark.parametrize("changes", [
    {"channel_params__svf_radius_px": 29},
    {"channel_params__lrm_large_px": 59},
    {"channel_params__curv_sigma_px": 7.0},
    {"replication__channels__context_px": 24},
])
def test_margin_violation_raises(changes):
    with pytest.raises(ValueError, match="context"):
        ch.channel_params(_cfg(**changes))


@pytest.mark.parametrize("changes", [
    {"channels": ["slope", "lrm_s", "lrm_l", "curv_prof", "curv_plan", "svf"]},
    {"channels": ["slope", "lrm_s", "lrm_l", "curv_prof", "curv_plan", "svf", "tri"]},
    {"channel_params__lrm_small_px": 10},
    {"channel_params__svf_dirs": 10},
    {"patch__px": 151},
])
def test_bad_config_raises(changes):
    with pytest.raises(ValueError):
        ch.channel_params(_cfg(**changes))


def test_replication_block_overrides_method_defaults():
    p = ch.channel_params(_cfg(replication__channels__curv_flat_eps=1e-9))
    assert p.curv_flat_eps == 1e-9 and p.context_px == ch.METHOD_DEFAULTS["context_px"]


def test_svf_directions_exactly_closed_under_d4():
    vecs = ch._unit_vectors(P.svf_dirs)
    assert len(vecs) == P.svf_dirs
    for k, (c, s) in enumerate(vecs):
        a = 2 * math.pi * k / P.svf_dirs
        assert abs(c - math.cos(a)) < 1e-15 and abs(s - math.sin(a)) < 1e-15
    key = {(c + 0.0, s + 0.0) for c, s in vecs}           # + 0.0 folds -0.0 into 0.0
    for c, s in vecs:
        for img in ((s, c), (-c, s), (c, -s), (-s, c)):  # transpose, flips, quarter turn
            assert (img[0] + 0.0, img[1] + 0.0) in key


# ---- output layout ----

def test_compute_channels_layout_and_dtype():
    z = synthetic_dtm((120, 110), seed=3, border=False, holes=False)
    out = ch.compute_channels(z.astype(np.float32), CFG)
    assert out.shape == (7, 120, 110) and out.dtype == np.float64
    order = ["svf", "detr", "slope", "curv_plan", "lrm_l", "curv_prof", "lrm_s"]
    perm = ch.compute_channels(z, _cfg(channels=order))
    for i, name in enumerate(order):
        np.testing.assert_array_equal(perm[i], ch.compute_channels(z, CFG)[IDX[name]])


def test_patch_and_full_layout():
    z = synthetic_dtm((330, 300), seed=4)
    pc = ch.patch_channels(z, 160, 150, CFG)
    assert pc.shape == (7, P.patch_px, P.patch_px) and pc.dtype == np.float64
    full = ch.full_channels(z, CFG, workers=1)
    assert full.shape == (6, 330, 300) and full.dtype == np.float64
    slope = ch.full_channels(z, CFG, names=["slope"], workers=1)
    np.testing.assert_array_equal(slope[0], full[TERRAIN_IDX["slope"]])
    with pytest.raises(ValueError):
        ch.full_channels(z, CFG, names=["detr"])


# ---- D4 equivariance ----

@pytest.mark.parametrize("k", range(8))
def test_d4_equivariance_block(k):
    z = synthetic_dtm((200, 200), seed=5)
    ref = ch.compute_channels(z, CFG)
    got = ch.compute_channels(np.ascontiguousarray(d4.transform(z, k)), CFG)
    want = d4.transform(ref, k)
    for i, name in enumerate(NAMES):
        _assert_close(got[i], want[i], 1e-9, f"{name}, transform {k}")
    assert np.isfinite(ref[IDX["svf"]]).sum() > 20000  # the test exercises real values


def _corner_image(n: int, r: int, c: int, k: int) -> tuple[int, int]:
    """Where corner (r, c) of an n x n raster goes under transform k (corners form an (n+1) x (n+1) lattice)."""
    lattice = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    (rr, cc), = np.argwhere(d4.transform(lattice, k) == lattice[r, c])
    return int(rr), int(cc)


@pytest.mark.parametrize("k", range(8))
def test_d4_equivariance_patch(k):
    n = 300
    z = synthetic_dtm((n, n), seed=6)
    zk = np.ascontiguousarray(d4.transform(z, k))
    for r, c in [(150, 150), (110, 190), (190, 104)]:
        rk, ck = _corner_image(n, r, c, k)
        got = ch.patch_channels(zk, rk, ck, CFG)
        want = d4.transform(ch.patch_channels(z, r, c, CFG), k)
        for i, name in enumerate(NAMES):
            _assert_close(got[i], want[i], 1e-9, f"{name}, transform {k}, corner {(r, c)}")


# ---- patch-wise vs full raster ----

@pytest.fixture(scope="module")
def raster():
    return synthetic_dtm((520, 470), seed=7)


@pytest.fixture(scope="module")
def full(raster):
    return ch.full_channels(raster, CFG, tile_px=96, workers=3)


def test_tiled_equals_direct(raster, full):
    direct = ch.terrain_channels(raster, CFG)  # one whole-array computation, no tiles
    for i, name in enumerate(P.terrain_names):
        _assert_close(full[i], direct[i], 5e-9, name)


def test_tiles_and_workers_do_not_change_the_result(raster, full):
    np.testing.assert_array_equal(ch.full_channels(raster, CFG, tile_px=96, workers=1), full)
    other = ch.full_channels(raster, CFG, tile_px=1000, workers=2)
    for i, name in enumerate(P.terrain_names):
        _assert_close(other[i], full[i], 5e-9, name)


def test_patch_equals_full(raster, full):
    H, W = raster.shape
    rng = np.random.default_rng(8)
    corners = [(HALF, HALF), (H - HALF, W - HALF), (HALF + 3, W - HALF - 10), (300, 300), (400, 380)]
    corners += list(zip(rng.integers(HALF, H - HALF + 1, 25), rng.integers(HALF, W - HALF + 1, 25)))
    n_nan_patches = 0
    for r, c in corners:
        pc = ch.patch_channels(raster, r, c, CFG)
        sl = (slice(r - HALF, r + HALF), slice(c - HALF, c + HALF))
        for name, j in TERRAIN_IDX.items():
            _assert_close(pc[IDX[name]], full[j][sl], 5e-9, f"{name} at corner {(r, c)}")
        np.testing.assert_array_equal(pc[IDX["detr"]], ch.detrend(raster[sl]))
        n_nan_patches += bool(np.isnan(pc).any())
    assert 0 < n_nan_patches < len(corners)  # both clean patches and patches touching invalid cells were checked


# ---- analytic cases ----

def _grid(shape):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    return yy * P.cell_m, xx * P.cell_m


def _svf_plane(gx, gy):
    t = np.array([gx * math.cos(a) + gy * math.sin(a)
                  for a in 2 * np.pi * np.arange(P.svf_dirs) / P.svf_dirs])
    t = np.maximum(t, 0.0)
    return 1.0 - np.mean(t / np.sqrt(1.0 + t * t))


@pytest.mark.parametrize("gx, gy", [(0.3, 0.0), (0.0, -0.45), (0.2, 0.25), (-0.7, 0.4), (1e-7, 0.0)])
def test_plane(gx, gy):
    Y, X = _grid((160, 150))
    z = 420.0 + gx * X + gy * Y
    out = ch.compute_channels(z, CFG)
    inner = (slice(25, -25), slice(25, -25))
    expect = {
        "slope": math.degrees(math.atan(math.hypot(gx, gy))),
        "lrm_s": 0.0, "lrm_l": 0.0, "curv_prof": 0.0, "curv_plan": 0.0, "detr": 0.0,
        "svf": _svf_plane(gx, gy),
    }
    for name, value in expect.items():
        np.testing.assert_allclose(out[IDX[name]][inner], value, rtol=0, atol=1e-9, err_msg=name)
    if gx == 1e-7:  # gradient below the flat threshold: curvature exactly 0
        assert gx * gx < P.curv_flat_eps
        assert np.all(out[IDX["curv_prof"]][inner] == 0) and np.all(out[IDX["curv_plan"]][inner] == 0)


def test_flat():
    out = ch.compute_channels(np.full((140, 140), 312.5), CFG)
    inner = (slice(25, -25), slice(25, -25))
    for name in ("slope", "lrm_s", "lrm_l", "curv_prof", "curv_plan", "detr"):
        assert np.all(out[IDX[name]][inner] == 0.0), name
    # 1 up to rounding: the four bilinear weights don't sum to exactly 1 in floating point
    np.testing.assert_allclose(out[IDX["svf"]][inner], 1.0, rtol=0, atol=1e-12)
    assert np.all(out[IDX["svf"]][inner] <= 1.0)


@pytest.mark.parametrize("kx, ky, sign_prof, sign_plan", [
    (0.004, 0.004, -1, -1),     # bowl: concave, both curvatures negative
    (-0.004, -0.004, 1, 1),     # dome: convex, both positive
    (0.004, -0.004, None, None),  # saddle: signs depend on the gradient direction
])
def test_paraboloid(kx, ky, sign_prof, sign_plan):
    H = W = 161
    Y, X = _grid((H, W))
    X -= X[0, W // 2]
    Y -= Y[H // 2, 0]
    z = 400.0 + 0.5 * kx * X ** 2 + 0.5 * ky * Y ** 2
    out = ch.compute_channels(z, CFG)
    # Gaussian smoothing adds a constant to a quadratic and central differences are exact for it, so
    # p = kx X, q = ky Y, r = kx, t = ky, s = 0.
    p, q = kx * X, ky * Y
    g2 = p * p + q * q
    with np.errstate(divide="ignore", invalid="ignore"):
        prof = -(p * p * kx + q * q * ky) / (g2 * (1 + g2) ** 1.5)
        plan = -(q * q * kx + p * p * ky) / (g2 * np.sqrt(1 + g2))
    prof[g2 < P.curv_flat_eps] = 0.0
    plan[g2 < P.curv_flat_eps] = 0.0
    inner = (slice(10, -10), slice(10, -10))
    for name, k in (("curv_prof", prof), ("curv_plan", plan)):
        np.testing.assert_allclose(out[IDX[name]][inner], np.arcsinh(k / P.curv_k0)[inner], rtol=0, atol=1e-9,
                                   err_msg=name)
    c = (H // 2, W // 2)
    assert out[IDX["curv_prof"]][c] == 0.0 and out[IDX["curv_plan"]][c] == 0.0  # apex: flat rule
    off = (H // 2 + 12, W // 2 - 7)
    if sign_prof is not None:
        assert np.sign(out[IDX["curv_prof"]][off]) == sign_prof
        assert np.sign(out[IDX["curv_plan"]][off]) == sign_plan
    else:  # on the x axis of the saddle: profile follows kx (concave), plan follows -ky
        on_x = (H // 2, W // 2 + 15)
        assert out[IDX["curv_prof"]][on_x] < 0 < out[IDX["curv_plan"]][on_x]
    svf_c = out[IDX["svf"]][c]
    if kx > 0 and ky > 0:
        assert 0.9 < svf_c < 1.0
    if kx < 0 and ky < 0:
        assert svf_c == 1.0


def test_svf_matches_bruteforce():
    """An independent SVF: scipy map_coordinates (order 1) at r = 1..20 px in 16 directions, atan, sin."""
    z = synthetic_dtm((140, 140), seed=9, border=False, holes=False, lake=False)
    svf = ch.compute_channels(z, CFG)[IDX["svf"]]
    rng = np.random.default_rng(10)
    radii = np.arange(P.svf_r_min_px, P.svf_radius_px + 1)
    for i, j in zip(rng.integers(20, 120, 40), rng.integers(20, 120, 40)):
        sins = []
        for a in 2 * np.pi * np.arange(P.svf_dirs) / P.svf_dirs:
            vals = ndi.map_coordinates(z, [i + radii * np.sin(a), j + radii * np.cos(a)], order=1, prefilter=False)
            h = np.max(np.arctan((vals - z[i, j]) / (radii * P.cell_m)))
            sins.append(np.sin(max(0.0, h)))
        assert abs(svf[i, j] - (1.0 - np.mean(sins))) <= 1e-9


def test_detrend_matches_lstsq():
    rng = np.random.default_rng(11)
    w = 300.0 + rng.normal(size=(4, 30, 26)) * 5 + np.arange(26) * 0.3
    got = ch.detrend(w)
    yy, xx = np.mgrid[0:30, 0:26]
    A = np.column_stack([np.ones(yy.size), xx.ravel(), yy.ravel()])
    for n in range(4):
        coef, *_ = np.linalg.lstsq(A, w[n].ravel(), rcond=None)
        np.testing.assert_allclose(got[n], w[n] - (A @ coef).reshape(30, 26), rtol=0, atol=1e-9)
    w[2, 5, 5] = np.nan
    out = ch.detrend(w)
    assert np.isnan(out[2]).all() and np.isfinite(out[[0, 1, 3]]).all()


# ---- NaN propagation ----

def test_nan_propagation_single_cell():
    H, W, i0, j0 = 150, 140, 70, 80
    z = synthetic_dtm((H, W), seed=12, border=False, holes=False, lake=False)
    z_other = z.copy()
    z_other[i0, j0] += 50.0
    z_nan = z.copy()
    z_nan[i0, j0] = np.nan
    a = ch.terrain_channels(z_nan, CFG)
    b = ch.terrain_channels(z_other, CFG)
    ii, jj = np.mgrid[0:H, 0:W]
    radii = P.footprint_radii()
    for name, k in TERRAIN_IDX.items():
        R = radii[name]
        expect = (np.maximum(abs(ii - i0), abs(jj - j0)) <= R) | (ii < R) | (jj < R) | (ii >= H - R) | (jj >= W - R)
        np.testing.assert_array_equal(np.isnan(a[k]), expect, err_msg=name)
        # outside the footprint the invalid cell has no influence at all
        assert np.max(np.abs(a[k][~expect] - b[k][~expect])) <= 1e-9, name
    assert np.isnan(ch.compute_channels(z_nan, CFG)[IDX["detr"]]).all()


def test_nan_and_raster_edge_in_patch_context():
    z = synthetic_dtm((400, 400), seed=13, border=False, holes=False, lake=False)
    clean = ch.patch_channels(z, 200, 200, CFG)
    assert np.isfinite(clean).all()
    z2 = z.copy()
    z2[200 - HALF - 20, 200] = np.nan          # in the context, 20 px above the patch
    pc = ch.patch_channels(z2, 200, 200, CFG)
    radii = P.footprint_radii()
    for name, R in radii.items():
        hit = np.isnan(pc[IDX[name]])
        assert hit.any() == (R >= 20), name    # only kernels reaching 20 px into the context see it
        assert not hit[max(R - 19, 0):].any()  # and only in the first R - 19 patch rows
    assert np.isfinite(pc[IDX["detr"]]).all()  # the cell is outside the patch itself
    edge = ch.patch_channels(z, HALF + 10, 200, CFG)  # context leaves the raster 18 px above the patch
    for name, R in radii.items():
        assert np.isnan(edge[IDX[name]]).any() == (R > 10), name
