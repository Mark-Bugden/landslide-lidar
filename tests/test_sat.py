"""Summed-area tables (P1-04): sums, means and validity counts against brute force."""

import numpy as np
import pytest

from landslide import sat


@pytest.fixture
def field():
    rng = np.random.default_rng(0)
    a = 400.0 + 50.0 * rng.normal(size=(47, 39))
    a[rng.random(a.shape) < 0.005] = np.nan
    a[:6, 30:] = np.nan
    return a


def _brute_corner(a, r, c, half):
    return a[r - half:r + half, c - half:c + half]


def test_integral_matches_cumsum():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, size=(9, 7))
    S = sat.integral(a)
    assert S.shape == (10, 8) and S.dtype == np.int64
    for i in range(10):
        for j in range(8):
            assert S[i, j] == a[:i, :j].sum()
    assert sat.integral(a.astype(float)).dtype == np.float64
    assert sat.integral(a > 2).dtype == np.int64


def test_box_sums_and_bounds():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(20, 17))
    S = sat.integral(a)
    r0 = rng.integers(0, 10, 50)
    c0 = rng.integers(0, 8, 50)
    r1 = r0 + rng.integers(0, 11, 50)
    c1 = c0 + rng.integers(0, 10, 50)
    got = sat.box_sums(S, r0, c0, r1, c1)
    want = [a[i:k, j:l].sum() for i, j, k, l in zip(r0, c0, r1, c1)]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    for bad in [(-1, 0, 3, 3), (0, 0, 21, 3), (0, 0, 3, 18), (5, 0, 3, 3)]:
        with pytest.raises(ValueError):  # never wraps round with negative indices
            sat.box_sums(S, *bad)


def test_corner_box_sums():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(30, 30))
    S = sat.integral(a)
    r, c = rng.integers(5, 26, 40), rng.integers(5, 26, 40)
    np.testing.assert_allclose(sat.corner_box_sums(S, r, c, 5),
                               [_brute_corner(a, i, j, 5).sum() for i, j in zip(r, c)], rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        sat.corner_box_sums(S, 4, 10, 5)


@pytest.mark.parametrize("half", [1, 3, 7])
def test_corner_box_grid(half):
    rng = np.random.default_rng(4)
    a = rng.integers(0, 9, size=(16, 13))
    S = sat.integral(a)
    g = sat.corner_box_grid(S, half, fill=-1)
    assert g.shape == (17, 14)
    for r in range(17):
        for c in range(14):
            fits = half <= r <= 16 - half and half <= c <= 13 - half
            assert g[r, c] == (_brute_corner(a, r, c, half).sum() if fits else -1)


@pytest.mark.parametrize("radius", [0, 1, 4])
def test_pixel_box_grid(radius):
    rng = np.random.default_rng(5)
    a = rng.integers(0, 9, size=(15, 12))
    S = sat.integral(a)
    g = sat.pixel_box_grid(S, radius, fill=-1)
    assert g.shape == (15, 12)
    for i in range(15):
        for j in range(12):
            fits = radius <= i < 15 - radius and radius <= j < 12 - radius
            want = a[i - radius:i + radius + 1, j - radius:j + radius + 1].sum() if fits else -1
            assert g[i, j] == want


def test_value_sat_point_queries(field):
    vs = sat.ValueSat(field)
    rng = np.random.default_rng(6)
    half = 6
    r = rng.integers(half, field.shape[0] - half + 1, 200)
    c = rng.integers(half, field.shape[1] - half + 1, 200)
    blocks = [_brute_corner(field, i, j, half) for i, j in zip(r, c)]
    counts = np.array([np.isfinite(b).sum() for b in blocks])
    np.testing.assert_array_equal(vs.valid_count(r, c, half), counts)
    np.testing.assert_allclose(vs.sum(r, c, half), [np.nansum(b) for b in blocks], rtol=0, atol=1e-9)
    with np.errstate(invalid="ignore"):
        any_mean = np.array([np.nanmean(b) if np.isfinite(b).any() else np.nan for b in blocks])
    full_mean = np.where(counts == (2 * half) ** 2, any_mean, np.nan)
    np.testing.assert_allclose(vs.mean(r, c, half), full_mean, rtol=0, atol=1e-9)
    np.testing.assert_allclose(vs.mean(r, c, half, require_all_valid=False), any_mean, rtol=0, atol=1e-9)
    assert np.isnan(full_mean).any() and np.isfinite(full_mean).any()


def test_value_sat_grids(field):
    vs = sat.ValueSat(field)
    half = 4
    H, W = field.shape
    counts = vs.count_grid(half)
    means = vs.mean_grid(half)
    loose = vs.mean_grid(half, require_all_valid=False)
    assert counts.shape == means.shape == (H + 1, W + 1)
    for r in range(H + 1):
        for c in range(W + 1):
            if not (half <= r <= H - half and half <= c <= W - half):
                assert counts[r, c] == -1 and np.isnan(means[r, c]) and np.isnan(loose[r, c])
                continue
            b = _brute_corner(field, r, c, half)
            n = np.isfinite(b).sum()
            assert counts[r, c] == n
            if n == b.size:
                assert abs(means[r, c] - b.mean()) <= 1e-9
            else:
                assert np.isnan(means[r, c])
            if n:
                assert abs(loose[r, c] - np.nanmean(b)) <= 1e-9
            else:
                assert np.isnan(loose[r, c])


def test_value_sat_precision_on_large_raster():
    """Patch means of elevations around 1,000 m on a sheet-sized raster stay accurate (offset subtraction)."""
    rng = np.random.default_rng(7)
    a = 1000.0 + rng.normal(size=(1500, 1400))
    vs = sat.ValueSat(a)
    r, c = np.array([75, 1425, 700]), np.array([75, 1325, 600])
    want = [_brute_corner(a, i, j, 75).mean() for i, j in zip(r, c)]
    np.testing.assert_allclose(vs.mean(r, c, 75), want, rtol=0, atol=1e-10)


def test_value_sat_all_invalid():
    vs = sat.ValueSat(np.full((10, 10), np.nan))
    assert vs.valid_count(5, 5, 3) == 0
    assert np.isnan(vs.mean(5, 5, 3)) and np.isnan(vs.mean(5, 5, 3, require_all_valid=False))
