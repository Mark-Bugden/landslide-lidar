"""The dihedral group D4 on the pixel grid: 4 rotations x optional flip, acting on the last two axes.

Transform k (0..7) is `rot90(flip(x) if k >= 4 else x, k % 4)`, where flip mirrors the last axis and rot90 turns
counter-clockwise from the last axis towards the second-last. Every channel is a scalar field invariant under
grid rotations and flips, so augmentation and TTA need no recomputation (BRIEF.md Appendix A §6).

The numpy functions work on any array with at least 2 dims. The torch functions are imported lazily so that CPU
scripts don't load torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

N_TRANSFORMS = 8


def _check_k(k: int) -> tuple[int, bool]:
    if not 0 <= int(k) < N_TRANSFORMS:
        raise ValueError(f"D4 transform index must be 0..7, got {k}")
    return int(k) % 4, int(k) >= 4


def inverse_index(k: int) -> int:
    """Index of the inverse transform. Reflections are their own inverse; rotations invert by angle."""
    r, f = _check_k(k)
    return k if f else (4 - r) % 4


def compose_index(a: int, b: int) -> int:
    """Index of `transform(transform(x, b), a)`, found by acting on a probe array."""
    probe = np.arange(9).reshape(3, 3)
    target = transform(transform(probe, b), a)
    for k in range(N_TRANSFORMS):
        if np.array_equal(transform(probe, k), target):
            return k
    raise AssertionError("D4 is closed under composition")


# ---- numpy ----

def transform(x: np.ndarray, k: int) -> np.ndarray:
    r, f = _check_k(k)
    if f:
        x = np.flip(x, axis=-1)
    return np.rot90(x, r, axes=(-2, -1))


def inverse(x: np.ndarray, k: int) -> np.ndarray:
    return transform(x, inverse_index(k))


# ---- torch ----

def t_transform(x: torch.Tensor, k: int) -> torch.Tensor:
    import torch

    r, f = _check_k(k)
    if f:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, r, dims=(-2, -1))


def t_inverse(x: torch.Tensor, k: int) -> torch.Tensor:
    return t_transform(x, inverse_index(k))


def t_apply_per_sample(x: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """Apply transform `ks[i]` to sample `x[i]` of a batch (B, ..., H, W) with H == W."""
    import torch

    if x.shape[-1] != x.shape[-2]:
        raise ValueError("per-sample D4 needs square samples")
    if ks.shape != (x.shape[0],):
        raise ValueError("need one transform index per sample")
    out = torch.empty_like(x)
    for k in range(N_TRANSFORMS):
        idx = (ks == k).nonzero(as_tuple=True)[0]
        if idx.numel():
            out[idx] = t_transform(x[idx], k)
    return out


def t_all_views(x: torch.Tensor) -> torch.Tensor:
    """All 8 views of a batch, concatenated view-major: shape (8 * B, ...). View k is rows k*B .. (k+1)*B - 1."""
    import torch

    return torch.cat([t_transform(x, k) for k in range(N_TRANSFORMS)], dim=0)
