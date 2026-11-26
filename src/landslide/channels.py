"""The seven terrain channels (BRIEF.md Appendix A §6, PLAN.md D18), in float64.

Every channel is computed from the DTM `z` (metres, invalid cells NaN; see `landslide.dtm`) on a grid of
`cfg["dtm"]["cell_m"]` metres. Array axes are (row, col); x runs along columns, y along rows (downwards). All
formulas below are invariant under that choice, and under every D4 grid transform.

| channel   | definition                                                                                   | radius R |
| --------- | -------------------------------------------------------------------------------------------- | -------- |
| slope     | degrees(arctan(sqrt(p^2 + q^2))), p, q central differences of z at the cell size              | 1        |
| lrm_s     | z - uniform_filter(z, lrm_small_px)                                                          | 5        |
| lrm_l     | z - uniform_filter(z, lrm_large_px)                                                          | 25       |
| curv_prof | asinh(k / curv_k0), k = -(p^2 r + 2pqs + q^2 t) / ((p^2+q^2)(1+p^2+q^2)^1.5) on the smoothed z | 4 + 1    |
| curv_plan | asinh(k / curv_k0), k = -(q^2 r - 2pqs + p^2 t) / ((p^2+q^2)(1+p^2+q^2)^0.5) on the smoothed z | 4 + 1    |
| svf       | 1 - mean_d sin(max(0, horizon_d)), 16 directions, r = 1..20 px, bilinear sampling              | 20       |
| detr      | z minus its least-squares plane over the window (patch or stage-2 window)                     | window   |

Curvatures: z is smoothed with a Gaussian (sigma = curv_sigma_px, truncate 4, so radius 4 px), then p, q, r, s, t
are the Zevenbergen-Thorne central differences of the smoothed grid (3 x 3 stencil, radius 1), and k = 0 where
p^2 + q^2 < curv_flat_eps (the gradient direction is undefined there). SVF: the horizon in direction d is
atan(max_r (z(r) - z0) / (r * cell)), and sin(atan(t)) = t / sqrt(1 + t^2) is used directly.

**Validity.** Channel c at pixel (i, j) is finite iff every pixel within Chebyshev radius R_c of (i, j) lies inside
the array and is valid; otherwise it is NaN. Invalid cells are filled with 0 before filtering (so NaN never enters
arithmetic) and the result is masked with a summed-area table of the invalid mask. detr is NaN over the whole window
if any pixel of the window is invalid.

**Margin budget.** Every R_c must be <= the channel context (28 px, Appendix A §5-§6); `channel_params` checks
this and raises otherwise. Then a channel value depends only on z within 28 px, so `patch_channels` (patch plus
28 px context, cropped) and `full_channels` (tiles plus 28 px halo, stitched) run the same code on the same inputs
and agree with a direct whole-array computation to rounding (only the running sums of `uniform_filter` differ,
by ~1e-12).

**D4.** The 16 SVF directions are built to be exactly closed under the 8 grid symmetries, the Gaussian is applied
in both axis orders and averaged, and the stencils are written so that slope and the curvatures are bitwise
D4-equivariant. LRM, SVF and detr are equivariant to rounding (~1e-12).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from scipy import ndimage as ndi

from landslide.dtm import extract_block, patch_slices
from landslide.sat import integral, pixel_box_grid

#: The seven channels Appendix A §6 defines. `cfg["channels"]` must list exactly these, in the order used everywhere.
KNOWN_CHANNELS = ("slope", "lrm_s", "lrm_l", "curv_prof", "curv_plan", "svf", "detr")
#: The channel computed per window (patch or stage-2 window) rather than per pixel.
WINDOW_CHANNEL = "detr"

#: Method values stated in Appendix A's prose or decided in D18 that Appendix B has no key for. They are read from
#: `cfg["replication"]["channels"]` when that block exists (requested from P1-02); these are the values used
#: until then, and the decisions file records them.
METHOD_DEFAULTS: dict[str, float] = {
    "context_px": 28,        # Appendix A §5 "the patch plus its 28 px channel context", §6 "a 28 px margin"
    "curv_truncate": 4.0,    # D18: Gaussian truncate 4 (radius int(4 * 1.0 + 0.5) = 4 px)
    "curv_flat_eps": 1e-12,  # D18: curvature 0 where p^2 + q^2 < eps (|grad z| < 1e-6 m/m); our value
    "svf_r_min_px": 1,       # D18: r = 1 .. svf_radius_px
}

#: Threads for `full_channels` (performance only; results don't depend on it). Measured on the i5-11600K (6 cores,
#: 12 MB L3): 1 -> 8.7 s, 2 -> 4.8 s, 3 or 4 -> 3.9 s, 6 -> 6.7 s per 2,850 x 2,400 sheet (cache contention above 4).
DEFAULT_WORKERS = 3


# ---- parameters and the margin budget ----

@dataclass(frozen=True)
class ChannelParams:
    names: tuple[str, ...]
    cell_m: float
    patch_px: int
    context_px: int
    lrm_small_px: int
    lrm_large_px: int
    curv_sigma_px: float
    curv_truncate: float
    curv_k0: float
    curv_flat_eps: float
    svf_radius_px: int
    svf_dirs: int
    svf_r_min_px: int

    @property
    def terrain_names(self) -> tuple[str, ...]:
        """The per-pixel channels (all but detr), in config order."""
        return tuple(n for n in self.names if n != WINDOW_CHANNEL)

    @property
    def gauss_radius(self) -> int:
        """scipy's Gaussian kernel radius, int(truncate * sigma + 0.5)."""
        return int(self.curv_truncate * self.curv_sigma_px + 0.5)

    def footprint_radii(self) -> dict[str, int]:
        """Chebyshev radius (px) of the input each per-pixel channel reads."""
        return {
            "slope": 1,
            "lrm_s": (self.lrm_small_px - 1) // 2,
            "lrm_l": (self.lrm_large_px - 1) // 2,
            "curv_prof": self.gauss_radius + 1,
            "curv_plan": self.gauss_radius + 1,
            "svf": _svf_plan(self.svf_dirs, self.svf_r_min_px, self.svf_radius_px)[1],
        }


def channel_params(cfg: dict[str, Any]) -> ChannelParams:
    """Collect and validate every channel setting from the config, and check the margin budget."""
    names = tuple(cfg["channels"])
    if sorted(names) != sorted(KNOWN_CHANNELS) or len(set(names)) != len(names):
        raise ValueError(f"cfg['channels'] must be a permutation of {KNOWN_CHANNELS}, got {names}")
    cp = cfg["channel_params"]
    extra = {**METHOD_DEFAULTS, **cfg.get("replication", {}).get("channels", {})}
    p = ChannelParams(
        names=names,
        cell_m=float(cfg["dtm"]["cell_m"]),
        patch_px=int(cfg["patch"]["px"]),
        context_px=int(extra["context_px"]),
        lrm_small_px=int(cp["lrm_small_px"]),
        lrm_large_px=int(cp["lrm_large_px"]),
        curv_sigma_px=float(cp["curv_sigma_px"]),
        curv_truncate=float(extra["curv_truncate"]),
        curv_k0=float(cp["curv_k0"]),
        curv_flat_eps=float(extra["curv_flat_eps"]),
        svf_radius_px=int(cp["svf_radius_px"]),
        svf_dirs=int(cp["svf_dirs"]),
        svf_r_min_px=int(extra["svf_r_min_px"]),
    )
    if p.cell_m <= 0 or p.curv_k0 <= 0 or p.curv_sigma_px <= 0 or p.curv_flat_eps < 0:
        raise ValueError("cell_m, curv_k0 and curv_sigma_px must be > 0 and curv_flat_eps >= 0")
    if p.patch_px <= 0 or p.patch_px % 2:
        raise ValueError(f"patch px must be even (centred on a pixel corner), got {p.patch_px}")
    for k in ("lrm_small_px", "lrm_large_px"):
        v = getattr(p, k)
        if v < 1 or v % 2 == 0:
            raise ValueError(f"{k} must be odd so the moving mean is centred, got {v}")
    if p.svf_dirs < 4 or p.svf_dirs % 4:
        raise ValueError(f"svf_dirs must be a multiple of 4 (closed under D4), got {p.svf_dirs}")
    if not 1 <= p.svf_r_min_px <= p.svf_radius_px:
        raise ValueError("need 1 <= svf_r_min_px <= svf_radius_px")
    check_margin(p)
    return p


def check_margin(p: ChannelParams) -> dict[str, int]:
    """The margin budget: every per-pixel kernel must fit inside the channel context. Raises if one doesn't."""
    radii = p.footprint_radii()
    worst = max(radii, key=radii.get)
    if radii[worst] > p.context_px:
        raise ValueError(f"channel {worst!r} reads {radii[worst]} px around each pixel, more than the "
                         f"{p.context_px} px channel context; patch-wise and full-raster channels would differ")
    return radii


# ---- SVF sampling plan ----

def _unit_vectors(n: int) -> list[tuple[float, float]]:
    """(cos, sin) of k * 360/n degrees, k = 0..n-1, built so the set is exactly closed under the D4 symmetries:
    the first quadrant is computed once (and mirrored about the diagonal), the rest are exact 90-degree turns."""
    q = n // 4
    base: list[tuple[float, float]] = []
    for j in range(q):
        if 2 * j < q:
            a = 2.0 * math.pi * j / n
            base.append((math.cos(a), math.sin(a)) if j else (1.0, 0.0))
        elif 2 * j == q:
            base.append((math.sqrt(0.5), math.sqrt(0.5)))
        else:
            c, s = base[q - j]
            base.append((s, c))
    out = []
    for turn in range(4):
        for c, s in base:
            for _ in range(turn):
                c, s = -s, c
            out.append((c, s))
    return out


def _axis_taps(o: float) -> tuple[tuple[int, float], ...]:
    """Linear-interpolation taps (integer offset, weight) at offset o; one tap when o is an integer."""
    f = math.floor(o)
    fr = o - f
    if fr == 0.0:
        return ((int(f), 1.0),)
    return ((int(f), 1.0 - fr), (int(f) + 1, fr))


@lru_cache(maxsize=8)
def _svf_plan(n_dirs: int, r_min: int, r_max: int):
    """For each direction, for each radius r: (r, taps), taps = ((drow, dcol, weight), ...). Also the footprint."""
    plan = []
    radius = 0
    for c, s in _unit_vectors(n_dirs):
        samples = []
        for r in range(r_min, r_max + 1):
            taps = tuple((dy, dx, wy * wx) for dy, wy in _axis_taps(r * s) for dx, wx in _axis_taps(r * c))
            radius = max(radius, *(max(abs(dy), abs(dx)) for dy, dx, _ in taps))
            samples.append((r, taps))
        plan.append(tuple(samples))
    return tuple(plan), radius


# ---- the per-pixel kernels; each computes on rows/cols [a, H - a) of the filled block ----

def _slope(zf: np.ndarray, cell: float, a: int) -> np.ndarray:
    H, W = zf.shape
    p = (zf[a:H - a, a + 1:W - a + 1] - zf[a:H - a, a - 1:W - a - 1]) / (2.0 * cell)
    q = (zf[a + 1:H - a + 1, a:W - a] - zf[a - 1:H - a - 1, a:W - a]) / (2.0 * cell)
    g = p * p
    g += q * q
    np.sqrt(g, out=g)
    np.arctan(g, out=g)
    return np.degrees(g, out=g)


def _lrm(zf: np.ndarray, size: int, a: int) -> np.ndarray:
    H, W = zf.shape
    m = ndi.uniform_filter(zf, size=size, mode="nearest")
    return zf[a:H - a, a:W - a] - m[a:H - a, a:W - a]


def _smooth(zf: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    """Separable Gaussian, applied in both axis orders and averaged, so it commutes exactly with transposition."""
    g = lambda x, ax: ndi.gaussian_filter1d(x, sigma, axis=ax, mode="nearest", radius=radius)  # noqa: E731
    out = g(g(zf, 0), 1)
    out += g(g(zf, 1), 0)
    out *= 0.5
    return out


def _curvatures(zf: np.ndarray, p: ChannelParams, a: int) -> tuple[np.ndarray, np.ndarray]:
    """(asinh(k_prof / k0), asinh(k_plan / k0)) on [a, H - a)."""
    H, W = zf.shape
    zs = _smooth(zf, p.curv_sigma_px, p.gauss_radius)
    g = p.cell_m
    v = lambda di, dj: zs[a + di:H - a + di, a + dj:W - a + dj]  # noqa: E731
    c, e, w, s_, n = v(0, 0), v(0, 1), v(0, -1), v(1, 0), v(-1, 0)
    zp = (e - w) / (2.0 * g)                                       # dz/dx
    zq = (s_ - n) / (2.0 * g)                                      # dz/dy
    zr = ((e + w) - 2.0 * c) / (g * g)                             # d2z/dx2
    zt = ((s_ + n) - 2.0 * c) / (g * g)                            # d2z/dy2
    zs_ = ((v(1, 1) + v(-1, -1)) - (v(1, -1) + v(-1, 1))) / (4.0 * g * g)  # d2z/dxdy
    pp, qq = zp * zp, zq * zq
    g2 = pp + qq
    pq2s = 2.0 * (zp * zq) * zs_
    one = 1.0 + g2
    root = np.sqrt(one)
    flat = g2 < p.curv_flat_eps
    with np.errstate(divide="ignore", invalid="ignore"):
        prof = -((pp * zr + qq * zt) + pq2s) / (g2 * one * root)
        plan = -((qq * zr + pp * zt) - pq2s) / (g2 * root)
    prof[flat] = 0.0
    plan[flat] = 0.0
    return np.arcsinh(prof / p.curv_k0), np.arcsinh(plan / p.curv_k0)


def _svf(zf: np.ndarray, p: ChannelParams, a: int) -> np.ndarray:
    H, W = zf.shape
    plan, _ = _svf_plan(p.svf_dirs, p.svf_r_min_px, p.svf_radius_px)
    shape = (H - 2 * a, W - 2 * a)
    z0 = zf[a:H - a, a:W - a]
    total = np.zeros(shape)
    acc, tmp, tmp2 = np.empty(shape), np.empty(shape), np.empty(shape)
    v = lambda di, dj: zf[a + di:H - a + di, a + dj:W - a + dj]  # noqa: E731
    for samples in plan:
        for k, (r, taps) in enumerate(samples):
            (di, dj, wt), rest = taps[0], taps[1:]
            if not rest:                       # integer offset: weight is exactly 1
                np.subtract(v(di, dj), z0, out=tmp)
            else:
                np.multiply(v(di, dj), wt, out=tmp)
                for di, dj, wt in rest:
                    np.multiply(v(di, dj), wt, out=tmp2)
                    tmp += tmp2
                tmp -= z0
            tmp *= 1.0 / (r * p.cell_m)        # tan of the elevation angle to the sample
            if k == 0:
                np.copyto(acc, tmp)
            else:
                np.maximum(acc, tmp, out=acc)
        np.maximum(acc, 0.0, out=acc)          # t = tan(max(0, horizon))
        np.multiply(acc, acc, out=tmp)
        tmp += 1.0
        np.sqrt(tmp, out=tmp)
        acc /= tmp                             # sin(atan(t)) = t / sqrt(1 + t^2)
        total += acc
    total *= -1.0 / len(plan)
    total += 1.0
    return total


# ---- public API ----

def _resolve_names(p: ChannelParams, names: Iterable[str] | None) -> tuple[str, ...]:
    if names is None:
        return p.terrain_names
    names = tuple(names)
    bad = [n for n in names if n not in p.terrain_names]
    if bad:
        raise ValueError(f"not per-pixel channels: {bad} (choose from {p.terrain_names}; detr is per window)")
    return names


def _terrain(z: np.ndarray, p: ChannelParams, names: tuple[str, ...], margin: int) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"need a 2-D DTM block, got shape {z.shape}")
    H, W = z.shape
    m = int(margin)
    if m < 0 or H <= 2 * m or W <= 2 * m:
        raise ValueError(f"block {H} x {W} is too small for margin {m}")
    valid = np.isfinite(z)
    zf = np.where(valid, z, 0.0)
    inv_sat = integral(~valid, dtype=np.int32 if z.size < 2**31 else np.int64)
    radii = p.footprint_radii()
    out = np.full((len(names), H - 2 * m, W - 2 * m), np.nan)
    curv: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        R = radii[name]
        a = max(m, R)
        if H - 2 * a <= 0 or W - 2 * a <= 0:
            continue                           # no pixel has its whole footprint inside the block
        if name == "slope":
            vals = _slope(zf, p.cell_m, a)
        elif name == "lrm_s":
            vals = _lrm(zf, p.lrm_small_px, a)
        elif name == "lrm_l":
            vals = _lrm(zf, p.lrm_large_px, a)
        elif name in ("curv_prof", "curv_plan"):
            if not curv:
                curv["curv_prof"], curv["curv_plan"] = _curvatures(zf, p, a)
            vals = curv[name]
        elif name == "svf":
            vals = _svf(zf, p, a)
        else:  # pragma: no cover - _resolve_names guards this
            raise ValueError(name)
        d = a - m
        out[i, d:d + vals.shape[0], d:d + vals.shape[1]] = vals
        bad = pixel_box_grid(inv_sat, R, fill=1)[m:H - m, m:W - m] > 0
        out[i][bad] = np.nan
    return out


def terrain_channels(z: np.ndarray, cfg: dict[str, Any], names: Iterable[str] | None = None,
                     margin: int = 0) -> np.ndarray:
    """The per-pixel channels (all but detr, config order unless `names` is given) of a DTM block.

    Returns (n, H - 2*margin, W - 2*margin) float64 for the block's interior. NaN where a pixel's footprint
    leaves the block or touches an invalid cell (module docstring). For a whole sheet use `full_channels`: this
    one-pass version needs ~1.4 GB and ~24 s on a 2,850 x 2,400 raster.
    """
    p = channel_params(cfg)
    return _terrain(z, p, _resolve_names(p, names), margin)


def detrend(z: np.ndarray) -> np.ndarray:
    """z minus its least-squares plane over each window, for (..., H, W) windows.

    On a full rectangular grid the centred coordinates u (columns) and v (rows) are orthogonal to each other and to
    the constant, so the fit is exact in closed form: plane = mean + a*u + b*v with a = sum(u*z)/sum(u^2) and
    b = sum(v*z)/sum(v^2). Any NaN in a window makes that whole window NaN.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim < 2 or z.shape[-1] < 2 or z.shape[-2] < 2:
        raise ValueError(f"need windows of at least 2 x 2, got shape {z.shape}")
    H, W = z.shape[-2:]
    u = np.arange(W, dtype=np.float64) - (W - 1) / 2.0
    v = (np.arange(H, dtype=np.float64) - (H - 1) / 2.0)[:, None]
    zc = z - z.mean(axis=(-2, -1), keepdims=True)
    a = (zc * u).sum(axis=(-2, -1), keepdims=True) / (H * float((u * u).sum()))
    b = (zc * v).sum(axis=(-2, -1), keepdims=True) / (W * float((v * v).sum()))
    return zc - a * u - b * v


def _assemble(p: ChannelParams, terrain: np.ndarray, detr: np.ndarray) -> np.ndarray:
    out = np.empty((len(p.names), *detr.shape))
    t = iter(terrain)
    for i, name in enumerate(p.names):
        out[i] = detr if name == WINDOW_CHANNEL else next(t)
    return out


def compute_channels(z_block: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """All seven channels of a block, (7, H, W) float64 in config order; detr is fitted over the whole block."""
    p = channel_params(cfg)
    z_block = np.asarray(z_block, dtype=np.float64)
    return _assemble(p, _terrain(z_block, p, p.terrain_names, 0), detrend(z_block))


def patch_channels(z: np.ndarray, r: int, c: int, cfg: dict[str, Any]) -> np.ndarray:
    """The (7, px, px) float64 channels of the patch centred on pixel corner (r, c) of the raster `z`.

    The per-pixel channels are computed on the patch plus the channel context (28 px) on every side and cropped;
    detr is the patch's own plane residual. Cells outside the raster count as invalid, so a patch whose context
    leaves the raster or touches an invalid cell has NaN where the footprints reach it.
    """
    p = channel_params(cfg)
    m = p.context_px
    r0, c0, r1, c1 = patch_slices(r, c, p.patch_px, margin=m)
    block = extract_block(np.asarray(z), r0, c0, r1, c1)
    terrain = _terrain(block, p, p.terrain_names, m)
    return _assemble(p, terrain, detrend(block[m:m + p.patch_px, m:m + p.patch_px]))


def full_channels(z: np.ndarray, cfg: dict[str, Any], names: Iterable[str] | None = None,
                  tile_px: int = 256, workers: int | None = None) -> np.ndarray:
    """Per-pixel channels of a whole raster, (n, H, W) float64; by default all but detr, in config order.

    Stage 2 removes the plane per window itself (`detrend`). The raster is processed in tiles of `tile_px` with the
    channel context as halo, through the same code as `patch_channels`; tiles run on `workers` threads (numpy
    releases the GIL). Tile size and worker count don't change the result (tested). Memory: the output (55 MB per
    channel for a 2,850 x 2,400 sheet) plus ~15 MB per worker; traced peak 373 MB for all six channels.
    """
    p = channel_params(cfg)
    names = _resolve_names(p, names)
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError(f"need a 2-D raster, got shape {z.shape}")
    if tile_px < 1:
        raise ValueError("tile_px must be >= 1")
    H, W = z.shape
    m = p.context_px
    out = np.empty((len(names), H, W))
    tiles = [(r0, c0, min(r0 + tile_px, H), min(c0 + tile_px, W))
             for r0 in range(0, H, tile_px) for c0 in range(0, W, tile_px)]

    def run(t: tuple[int, int, int, int]) -> None:
        r0, c0, r1, c1 = t
        block = extract_block(z, r0 - m, c0 - m, r1 + m, c1 + m)
        out[:, r0:r1, c0:c1] = _terrain(block, p, names, m)

    n_workers = workers if workers is not None else min(DEFAULT_WORKERS, os.cpu_count() or 1)
    if n_workers <= 1:
        for t in tiles:
            run(t)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for _ in ex.map(run, tiles):
                pass
    return out
