"""Synthetic DTMs for the channel tests (no real data before gate G1)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _field(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    f = ndi.gaussian_filter(rng.normal(size=shape), sigma, mode="wrap")
    return f / f.std()


def synthetic_dtm(shape: tuple[int, int], seed: int = 0, *, border: bool = True, holes: bool = True,
                  lake: bool = True) -> np.ndarray:
    """Hilly terrain around 400 m (float64, invalid cells NaN).

    Multi-scale smooth relief (hills, slopes up to ~40 degrees, some roughness), a scarp, and optionally:
    a "lake" (an exactly flat disc, to exercise the flat-gradient rule), an invalid "outside the border" corner
    (z set to NaN as the invalid rule would) and a few isolated invalid cells.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    z = (400.0 + 60.0 * _field(rng, shape, 40.0) + 12.0 * _field(rng, shape, 10.0)
         + 1.5 * _field(rng, shape, 2.5) + 0.05 * rng.normal(size=shape))
    # a scarp: a smoothed 8 m step along a slanted line
    d = (xx - 0.6 * W) * 0.8 + (yy - 0.4 * H) * 0.6
    z += 8.0 / (1.0 + np.exp(-d / 3.0))
    if lake:
        lake_mask = (yy - 0.7 * H) ** 2 + (xx - 0.3 * W) ** 2 < (0.08 * min(H, W)) ** 2
        z[lake_mask] = 385.25
    if border:
        z[(yy / H + xx / W) > 1.75] = np.nan
    if holes:
        n = max(1, H * W // 40000)
        z[rng.integers(0, H, n), rng.integers(0, W, n)] = np.nan
    return z
