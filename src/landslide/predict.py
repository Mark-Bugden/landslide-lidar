"""Inference: per-channel normalisation and 8-view D4 test-time augmentation (BRIEF.md Appendix A §7).

`tta_predict(model, x)` is the mean over the 8 D4 views of the sigmoid of the logit; training (inner-val and test
predictions) and stage 2 both use it. Inputs are in physical units, in any float dtype and on any device; each chunk
is moved to the model's device, normalised in float32 with a `Normaliser` (the run's `norm.json`) and run under the
same bf16 autocast as training. All 8 views of a chunk go through the network in one batch, view-major
(`landslide.d4.t_all_views`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from landslide import d4
from landslide.logs import progress

# Samples per forward chunk when the caller gives none (8 x this many images per forward pass). Training passes
# cfg["train"]["batch_size"]. The chunk size has no effect on the method, but cuDNN may pick different kernels for
# different batch shapes, so fix it per use to keep reruns bit-identical.
DEFAULT_CHUNK = 64


class Normaliser:
    """Per-channel (x - mean) / std in float32. The statistics are kept in float64 as computed."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        m = np.asarray(mean, dtype=np.float64)
        s = np.asarray(std, dtype=np.float64)
        if m.ndim != 1 or m.shape != s.shape:
            raise ValueError("mean and std must be 1-D and of equal length")
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(s)) and np.all(s > 0)):
            raise ValueError("normalisation statistics must be finite with std > 0")
        self.mean = m
        self.std = s
        self._cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def n_channels(self) -> int:
        return len(self.mean)

    def _tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._cache:
            shape = (1, self.n_channels, 1, 1)
            self._cache[device] = (
                torch.tensor(self.mean, dtype=torch.float32, device=device).reshape(shape),
                torch.tensor(self.std, dtype=torch.float32, device=device).reshape(shape),
            )
        return self._cache[device]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) in any float dtype -> float32 normalised, on x's device."""
        if x.shape[1] != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {x.shape[1]}")
        m, s = self._tensors(x.device)
        return (x.float() - m) / s

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_json(cls, path: Path | str) -> Normaliser:
        with Path(path).open(encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["mean"], d["std"])


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def tta_logits(model: nn.Module, x: torch.Tensor | np.ndarray, *, batch_size: int | None = None,
               norm: Normaliser | None = None, amp: bool = True,
               progress_desc: str | None = None) -> torch.Tensor:
    """Logits of every D4 view: float32 on the CPU, shape (8, N); row k is view k (`landslide.d4.t_transform`).

    The model runs in eval mode (its previous mode is restored). `amp` uses the bf16 autocast of training.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim != 4:
        raise ValueError(f"expected (N, C, H, W), got shape {tuple(x.shape)}")
    chunk = int(batch_size) if batch_size is not None else DEFAULT_CHUNK
    if chunk < 1:
        raise ValueError("batch_size must be >= 1")
    device = _model_device(model)
    n = x.shape[0]
    out = torch.empty((d4.N_TRANSFORMS, n), dtype=torch.float32)
    was_training = model.training
    model.eval()
    try:
        bar = progress(total=n, desc=progress_desc, leave=False, disable=progress_desc is None, unit="patch")
        for i in range(0, n, chunk):
            xb = x[i:i + chunk].to(device, non_blocking=False)
            xb = norm.apply(xb) if norm is not None else xb.float()
            views = d4.t_all_views(xb)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(views)
            out[:, i:i + xb.shape[0]] = logits.float().reshape(d4.N_TRANSFORMS, xb.shape[0]).cpu()
            bar.update(xb.shape[0])
        bar.close()
    finally:
        model.train(was_training)
    return out


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """(8, N) view logits -> (N,) mean of the sigmoids, in float64 on the CPU."""
    return torch.sigmoid(logits.detach().cpu().double()).mean(dim=0)


def tta_predict(model: nn.Module, x: torch.Tensor | np.ndarray, *, batch_size: int | None = None,
                norm: Normaliser | None = None, amp: bool = True,
                progress_desc: str | None = None) -> torch.Tensor:
    """Mean over the 8 D4 views of sigmoid(logit): float64 on the CPU, shape (N,)."""
    return probs_from_logits(tta_logits(model, x, batch_size=batch_size, norm=norm, amp=amp,
                                        progress_desc=progress_desc))
