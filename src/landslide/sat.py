"""Summed-area tables (integral images) for box sums, means and validity counts.

`integral(a)` has shape (H + 1, W + 1) with `S[i, j] = a[:i, :j].sum()`, so the sum over pixel rows r0..r1-1 and
columns c0..c1-1 is `S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]`.

Two box conventions:
- **corner-centred** (patches, PLAN.md D11): the box of half-size h at pixel corner (r, c) covers pixel rows
  r-h .. r+h-1, i.e. table rows r-h and r+h. A 150 px patch is h = 75; the patch plus its 28 px channel context is
  h = 103. Corners run over 0..H and 0..W.
- **pixel-centred** (filters): the box of radius R at pixel (i, j) covers rows i-R .. i+R.

Boxes that leave the array are never wrapped round: point queries raise, and dense grids put `fill` there.

`ValueSat` holds a float64 table of the valid values (NaN counted as 0, after subtracting an offset for precision)
and an integer table of valid-pixel counts. Counts are exact; means are exact to ~1e-12 relative.
"""

from __future__ import annotations

import numpy as np


def integral(a: np.ndarray, dtype=None) -> np.ndarray:
    """Summed-area table of a 2-D array, shape (H + 1, W + 1), first row and column zero.

    Default dtype: int64 for bool/integer input, float64 otherwise. NaN propagates; use `ValueSat` for data with
    NaN.
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"need a 2-D array, got shape {a.shape}")
    if dtype is None:
        dtype = np.int64 if a.dtype.kind in "biu" else np.float64
    H, W = a.shape
    S = np.zeros((H + 1, W + 1), dtype=dtype)
    np.cumsum(a, axis=0, dtype=dtype, out=S[1:, 1:])
    np.cumsum(S[1:, 1:], axis=1, dtype=dtype, out=S[1:, 1:])
    return S


def box_sums(S: np.ndarray, r0, c0, r1, c1) -> np.ndarray:
    """Sums over pixel rows r0..r1-1 and cols c0..c1-1 (half-open), vectorised. Raises if a box leaves the table."""
    r0, c0, r1, c1 = (np.asarray(v, dtype=np.int64) for v in (r0, c0, r1, c1))
    H, W = S.shape[0] - 1, S.shape[1] - 1
    if (np.any(r0 < 0) or np.any(c0 < 0) or np.any(r1 > H) or np.any(c1 > W)
            or np.any(r1 < r0) or np.any(c1 < c0)):
        raise ValueError(f"box outside the {H} x {W} array or inverted")
    return S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]


def corner_box_sums(S: np.ndarray, r, c, half: int) -> np.ndarray:
    """Sums over the corner-centred boxes of half-size `half` at corners (r, c). Raises if a box leaves the array."""
    r = np.asarray(r, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    return box_sums(S, r - half, c - half, r + half, c + half)


def _dense(S: np.ndarray, top: int, bottom: int, out_shape: tuple[int, int], fill) -> np.ndarray:
    """Box sums for a dense lattice of anchors (i, j), i < out_shape[0], j < out_shape[1]: the box of anchor i
    runs from table index i - top to table index i + bottom (same for columns). Anchors whose box leaves the table
    get `fill`."""
    Hs, Ws = S.shape
    oh, ow = out_shape
    dtype = np.result_type(S.dtype, np.asarray(fill).dtype)
    out = np.full(out_shape, fill, dtype=dtype)
    i0, i1 = top, min(oh, Hs - bottom)          # anchors with i - top >= 0 and i + bottom <= Hs - 1
    j0, j1 = top, min(ow, Ws - bottom)
    if i1 <= i0 or j1 <= j0:
        return out
    a, b = i0 - top, i1 - top
    c, d = j0 - top, j1 - top
    a2, b2 = i0 + bottom, i1 + bottom
    c2, d2 = j0 + bottom, j1 + bottom
    out[i0:i1, j0:j1] = S[a2:b2, c2:d2] - S[a:b, c2:d2] - S[a2:b2, c:d] + S[a:b, c:d]
    return out


def corner_box_grid(S: np.ndarray, half: int, fill=0) -> np.ndarray:
    """Corner-centred box sums at every corner, shape (H + 1, W + 1); `fill` where the box leaves the array."""
    if half < 1:
        raise ValueError("half must be >= 1")
    return _dense(S, half, half, S.shape, fill)


def pixel_box_grid(S: np.ndarray, radius: int, fill=0) -> np.ndarray:
    """Pixel-centred box sums of radius R at every pixel, shape (H, W); `fill` where the box leaves the array."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    H, W = S.shape[0] - 1, S.shape[1] - 1
    return _dense(S, radius, radius + 1, (H, W), fill)


class ValueSat:
    """Box means and valid-pixel counts of a 2-D field with NaN as invalid.

    `mean(..., require_all_valid=True)` is NaN unless every pixel of the box is valid; with False it is the mean
    of the valid pixels (NaN if none).
    """

    def __init__(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"need a 2-D array, got shape {a.shape}")
        valid = np.isfinite(a)
        self.shape = a.shape
        self.offset = float(a[valid].mean()) if valid.any() else 0.0
        count_dtype = np.int32 if a.size < 2**31 else np.int64
        self.counts = integral(valid, dtype=count_dtype)
        self.sums = integral(np.where(valid, a - self.offset, 0.0))

    # ---- point queries at corners ----
    def valid_count(self, r, c, half: int) -> np.ndarray:
        return corner_box_sums(self.counts, r, c, half)

    def sum(self, r, c, half: int) -> np.ndarray:
        """Sum of the valid values in each box."""
        n = self.valid_count(r, c, half)
        return corner_box_sums(self.sums, r, c, half) + self.offset * n

    def mean(self, r, c, half: int, require_all_valid: bool = True) -> np.ndarray:
        n = self.valid_count(r, c, half).astype(np.float64)
        s = corner_box_sums(self.sums, r, c, half)
        return self._mean(s, n, (2 * half) ** 2, require_all_valid)

    # ---- dense grids over all corners, shape (H + 1, W + 1) ----
    def count_grid(self, half: int) -> np.ndarray:
        """Valid-pixel count of the box at every corner; -1 where the box leaves the array."""
        return corner_box_grid(self.counts, half, fill=-1)

    def mean_grid(self, half: int, require_all_valid: bool = True) -> np.ndarray:
        """Box mean at every corner; NaN where the box leaves the array (or is not all valid, if required)."""
        n = corner_box_grid(self.counts, half, fill=0).astype(np.float64)
        s = corner_box_grid(self.sums, half, fill=np.nan)
        return self._mean(s, n, (2 * half) ** 2, require_all_valid)

    def _mean(self, s, n, full: int, require_all_valid: bool):
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / n + self.offset
        bad = (n < full) if require_all_valid else (n == 0)
        return np.where(bad, np.nan, m)
