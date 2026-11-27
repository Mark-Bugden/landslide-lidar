"""DMR 5G rasters: read-only access, the invalid rule, and pixel <-> world transforms.

**Invalid rule** (Appendix B `dtm.invalid_rule`, brief note 1): a cell is invalid if it is NaN or non-finite, if it
is the raster's nodata value, if z <= 0 m (the ImageServer returns 0 m outside the national border) or if z > 2000 m.
Invalid cells become NaN. The two thresholds are parsed from the config string, so the config stays the only source
of the values.

**Pixel-corner convention** (PLAN.md D11). Pixel (r, c) covers rows r..r+1 and columns c..c+1 of the continuous
grid; *corner* (r, c) is the top-left corner of pixel (r, c), at world `transform * (c, r)`. A patch of `px` pixels
(px even) is centred on corner (r, c) and covers pixel rows r - px/2 .. r + px/2 - 1 (Python slice
`r - px/2 : r + px/2`), and the same for columns. So a 150 px patch at corner (r, c) covers rows r-75..r+74.

Every raster is opened with mode "r"; nothing here writes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

_RULE_LE = re.compile(r"\bz\s*<=\s*(-?\d+(?:\.\d+)?)")
_RULE_GT = re.compile(r"\bz\s*>\s*(-?\d+(?:\.\d+)?)")


# ---- invalid rule ----

def invalid_bounds(cfg: dict[str, Any]) -> tuple[float, float]:
    """(le, gt) from `cfg["dtm"]["invalid_rule"]`: z <= le or z > gt is invalid.

    The rule string must contain exactly one `z <= <number>` and one `z > <number>`; anything else raises, so a
    reworded rule can't be silently misread.
    """
    rule = cfg["dtm"]["invalid_rule"]
    le, gt = _RULE_LE.findall(rule), _RULE_GT.findall(rule)
    if len(le) != 1 or len(gt) != 1:
        raise ValueError(f"cannot parse dtm.invalid_rule {rule!r}: need one 'z <= a' and one 'z > b'")
    lo, hi = float(le[0]), float(gt[0])
    if not lo < hi:
        raise ValueError(f"dtm.invalid_rule bounds out of order: {lo} >= {hi}")
    return lo, hi


def apply_invalid(z: np.ndarray, cfg: dict[str, Any], nodata: float | None = None) -> np.ndarray:
    """A float64 copy of `z` with every invalid cell (non-finite, == nodata, z <= le, z > gt) set to NaN."""
    lo, hi = invalid_bounds(cfg)
    out = np.array(z, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    with np.errstate(invalid="ignore"):
        bad |= (out <= lo) | (out > hi)
        if nodata is not None and np.isfinite(nodata):
            bad |= out == nodata
    out[bad] = np.nan
    return out


# ---- reading ----

@dataclass(frozen=True)
class Dtm:
    """A DTM (or a window of one) in float64 with invalid cells as NaN.

    `transform` maps (col, row) of *this array* to world coordinates, so corner (0, 0) of `z` is at
    `transform * (0, 0)`. `row_off`/`col_off` give the array's position in the full raster.
    """

    z: np.ndarray
    transform: Affine
    crs: Any
    path: Path
    row_off: int = 0
    col_off: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape  # type: ignore[return-value]

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.z)


def read_dtm(path: Path | str, cfg: dict[str, Any],
             window: tuple[int, int, int, int] | None = None) -> Dtm:
    """Read band 1 of a single-band raster read-only, as float64 with the invalid rule applied.

    `window` = (row_off, col_off, height, width) in pixels of the full raster. It may extend past the raster;
    the part outside is NaN (boundless read), and the returned transform is the window's own.
    """
    path = Path(path)
    with rasterio.open(path, mode="r") as src:
        if src.count != 1:
            raise ValueError(f"{path}: expected 1 band, found {src.count}")
        if window is None:
            r0, c0, h, w = 0, 0, src.height, src.width
        else:
            r0, c0, h, w = (int(v) for v in window)
            if h <= 0 or w <= 0:
                raise ValueError(f"empty window {window}")
        z = np.full((h, w), np.nan, dtype=np.float64)
        ir0, ic0 = max(r0, 0), max(c0, 0)
        ir1, ic1 = min(r0 + h, src.height), min(c0 + w, src.width)
        if ir1 > ir0 and ic1 > ic0:
            part = src.read(1, window=Window(ic0, ir0, ic1 - ic0, ir1 - ir0), masked=True)
            z[ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0] = part.astype(np.float64).filled(np.nan)
        z = apply_invalid(z, cfg, nodata=src.nodata)
        transform = src.transform if window is None else window_transform(Window(c0, r0, w, h), src.transform)
        return Dtm(z=z, transform=transform, crs=src.crs, path=path, row_off=r0, col_off=c0)


# ---- pixel <-> world (D11) ----

def corner_xy(transform: Affine, r, c) -> tuple[np.ndarray, np.ndarray]:
    """World (x, y) of pixel corner (r, c): the top-left corner of pixel (r, c). Vectorised; r, c may be float."""
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    x = transform.c + transform.a * c + transform.b * r
    y = transform.f + transform.d * c + transform.e * r
    return x, y


def pixel_center_xy(transform: Affine, r, c) -> tuple[np.ndarray, np.ndarray]:
    """World (x, y) of the centre of pixel (r, c)."""
    return corner_xy(transform, np.asarray(r, dtype=np.float64) + 0.5, np.asarray(c, dtype=np.float64) + 0.5)


def _world_to_rc(transform: Affine, x, y) -> tuple[np.ndarray, np.ndarray]:
    inv = ~transform
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    col = inv.c + inv.a * x + inv.b * y
    row = inv.f + inv.d * x + inv.e * y
    return row, col


def xy_corner(transform: Affine, x, y, tol_px: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """The pixel corner (r, c) at world (x, y). Raises if (x, y) is more than `tol_px` pixels from a corner."""
    row, col = _world_to_rc(transform, x, y)
    r, c = np.rint(row), np.rint(col)
    off = np.maximum(np.abs(row - r), np.abs(col - c))
    if np.any(off > tol_px):
        raise ValueError(f"point(s) are not on a pixel corner (max offset {float(np.max(off)):.3g} px)")
    return r.astype(np.int64), c.astype(np.int64)


def xy_pixel(transform: Affine, x, y) -> tuple[np.ndarray, np.ndarray]:
    """The pixel (r, c) containing world (x, y) (half-open: a point on a pixel edge belongs to the pixel below/right
    of it in array order)."""
    row, col = _world_to_rc(transform, x, y)
    return np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)


def patch_slices(r: int, c: int, px: int, margin: int = 0) -> tuple[int, int, int, int]:
    """Pixel rows/cols (r0, c0, r1, c1), half-open, of a px-square centred on corner (r, c), grown by `margin`."""
    if px % 2:
        raise ValueError(f"a patch centred on a pixel corner needs an even size, got {px}")
    h = px // 2 + int(margin)
    return int(r) - h, int(c) - h, int(r) + h, int(c) + h


def patch_bounds(transform: Affine, r, c, px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """World bounds (xmin, ymin, xmax, ymax) of the px-square centred on corner (r, c). Vectorised."""
    if px % 2:
        raise ValueError(f"a patch centred on a pixel corner needs an even size, got {px}")
    h = px // 2
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    xa, ya = corner_xy(transform, r - h, c - h)
    xb, yb = corner_xy(transform, r + h, c + h)
    return np.minimum(xa, xb), np.minimum(ya, yb), np.maximum(xa, xb), np.maximum(ya, yb)


def extract_block(z: np.ndarray, r0: int, c0: int, r1: int, c1: int, fill: float = np.nan) -> np.ndarray:
    """A float64 copy of z[r0:r1, c0:c1]; the part outside the array is `fill` (NaN by default = invalid)."""
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"empty block rows {r0}:{r1}, cols {c0}:{c1}")
    H, W = z.shape
    out = np.full((r1 - r0, c1 - c0), fill, dtype=np.float64)
    ir0, ic0, ir1, ic1 = max(r0, 0), max(c0, 0), min(r1, H), min(c1, W)
    if ir1 > ir0 and ic1 > ic0:
        out[ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0] = z[ir0:ir1, ic0:ic1]
    return out
