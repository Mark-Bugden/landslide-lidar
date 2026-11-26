"""DTM access (P1-04): invalid rule, read-only reads, pixel-corner transforms. Synthetic rasters only."""

import copy
import hashlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from landslide import dtm
from landslide.config import load_config

CFG = load_config()
# A grid like the brief's DMR 5G recipe: pixel edges at x = 0.4 and y = -0.12 (mod 2 m), EPSG:5514
XMIN, YMAX, CELL = -548_001.6, -1_150_000.12, 2.0
TRANSFORM = Affine(CELL, 0.0, XMIN, 0.0, -CELL, YMAX)


def test_invalid_bounds_from_config():
    assert dtm.invalid_bounds(CFG) == (0.0, 2000.0)
    for rule in ("z < 0 or z > 2000", "z <= 0", "z <= 5 or z > 1 -> NaN", "z <= 0 or z <= 1 or z > 2000"):
        cfg = copy.deepcopy(CFG)
        cfg["dtm"]["invalid_rule"] = rule
        with pytest.raises(ValueError):
            dtm.invalid_bounds(cfg)


def test_apply_invalid():
    z = np.array([-5.0, 0.0, 1e-3, 512.25, 1999.9, 2000.0, 2000.01, np.nan, np.inf, -9999.0], dtype=np.float32)
    out = dtm.apply_invalid(z, CFG, nodata=-9999.0)
    assert out.dtype == np.float64
    expect_valid = [False, False, True, True, True, True, False, False, False, False]
    np.testing.assert_array_equal(np.isfinite(out), expect_valid)
    np.testing.assert_array_equal(out[np.isfinite(out)], z[expect_valid].astype(np.float64))
    assert np.isfinite(z[1])  # the input is not modified


@pytest.fixture
def tif(tmp_path):
    rng = np.random.default_rng(0)
    z = (300.0 + 20.0 * rng.random((40, 30))).astype(np.float32)
    z[0, :5] = 0.0             # outside the border: the service returns 0 m
    z[3, 3] = np.nan           # package NoData
    z[4, 4] = 2500.0           # out of range
    z[5, 5] = -3.0e38          # the file's nodata value
    path = tmp_path / "02_23_04.tif"
    with rasterio.open(path, "w", driver="GTiff", width=30, height=40, count=1, dtype="float32",
                       crs="EPSG:5514", transform=TRANSFORM, nodata=-3.0e38, compress="deflate") as dst:
        dst.write(z, 1)
    return path, z


def test_read_dtm_full(tif):
    path, z = tif
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    d = dtm.read_dtm(path, CFG)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before  # read-only
    assert d.z.dtype == np.float64 and d.shape == (40, 30)
    assert d.transform == TRANSFORM and d.crs.to_epsg() == 5514 and (d.row_off, d.col_off) == (0, 0)
    invalid = ~d.valid
    assert invalid[0, :5].all() and invalid[3, 3] and invalid[4, 4] and invalid[5, 5]
    assert invalid.sum() == 8
    np.testing.assert_array_equal(d.z[d.valid], z.astype(np.float64)[d.valid])


def test_read_dtm_window_and_boundless(tif):
    path, z = tif
    full = dtm.read_dtm(path, CFG)
    w = dtm.read_dtm(path, CFG, window=(10, 5, 12, 8))
    np.testing.assert_array_equal(w.z, full.z[10:22, 5:13])
    assert dtm.corner_xy(w.transform, 0, 0) == dtm.corner_xy(TRANSFORM, 10, 5)
    edge = dtm.read_dtm(path, CFG, window=(-3, 25, 10, 10))  # sticks out above and to the right
    assert edge.shape == (10, 10)
    assert np.isnan(edge.z[:3]).all() and np.isnan(edge.z[:, 5:]).all()
    np.testing.assert_array_equal(edge.z[3:, :5], full.z[0:7, 25:30])
    assert dtm.corner_xy(edge.transform, 0, 0) == dtm.corner_xy(TRANSFORM, -3, 25)
    np.testing.assert_array_equal(dtm.extract_block(full.z, -3, 25, 7, 35), edge.z)


def test_corner_convention():
    x, y = dtm.corner_xy(TRANSFORM, 0, 0)
    assert (x, y) == (XMIN, YMAX)
    x, y = dtm.corner_xy(TRANSFORM, 10, 7)
    assert x == pytest.approx(XMIN + 14.0, abs=1e-9) and y == pytest.approx(YMAX - 20.0, abs=1e-9)
    # every corner keeps the native alignment x = 0.4, y = -0.12 (mod 2 m)
    assert (x - 0.4) % 2.0 == pytest.approx(0.0, abs=1e-6) or (x - 0.4) % 2.0 == pytest.approx(2.0, abs=1e-6)
    assert (y + 0.12) % 2.0 == pytest.approx(0.0, abs=1e-6) or (y + 0.12) % 2.0 == pytest.approx(2.0, abs=1e-6)
    xc, yc = dtm.pixel_center_xy(TRANSFORM, 10, 7)
    assert (xc, yc) == pytest.approx((x + 1.0, y - 1.0))


def test_corner_roundtrip_vectorised():
    rng = np.random.default_rng(1)
    r, c = rng.integers(0, 3000, 100), rng.integers(0, 3000, 100)
    x, y = dtm.corner_xy(TRANSFORM, r, c)
    rr, cc = dtm.xy_corner(TRANSFORM, x, y)
    np.testing.assert_array_equal(rr, r)
    np.testing.assert_array_equal(cc, c)
    with pytest.raises(ValueError):
        dtm.xy_corner(TRANSFORM, x[:1] + 0.5, y[:1])
    pr, pc = dtm.xy_pixel(TRANSFORM, x + 0.3, y - 1.9)  # inside pixel (r, c)
    np.testing.assert_array_equal(pr, r)
    np.testing.assert_array_equal(pc, c)


def test_patch_geometry():
    px = CFG["patch"]["px"]
    assert dtm.patch_slices(200, 300, px) == (125, 225, 275, 375)   # rows r-75 .. r+74
    assert dtm.patch_slices(200, 300, px, margin=28) == (97, 197, 303, 403)
    with pytest.raises(ValueError):
        dtm.patch_slices(200, 300, 151)
    xmin, ymin, xmax, ymax = dtm.patch_bounds(TRANSFORM, 200, 300, px)
    cx, cy = dtm.corner_xy(TRANSFORM, 200, 300)
    size = CFG["patch"]["size_m"]
    assert (xmax - xmin, ymax - ymin) == pytest.approx((size, size))
    assert ((xmin + xmax) / 2, (ymin + ymax) / 2) == pytest.approx((cx, cy))


def test_extract_block():
    z = np.arange(20.0).reshape(4, 5)
    np.testing.assert_array_equal(dtm.extract_block(z, 1, 1, 3, 4), z[1:3, 1:4])
    b = dtm.extract_block(z, -1, -2, 2, 1)
    assert b.shape == (3, 3)
    assert np.isnan(b[0]).all() and np.isnan(b[:, :2]).all()
    np.testing.assert_array_equal(b[1:, 2], z[0:2, 0])
    with pytest.raises(ValueError):
        dtm.extract_block(z, 2, 2, 2, 4)
