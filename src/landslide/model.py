"""The CNN: torchvision ResNet-18 with ImageNet initialisation, a 7-channel conv1 and a single-logit head.

BRIEF.md Appendix A §7 and Appendix B `model`:

- the ImageNet state dict is loaded from a local file (P1-07 caches torchvision's `IMAGENET1K_V1` file under
  `TORCH_HOME = data/cache/torch`, i.e. `hub/checkpoints/<name>.pth`). Nothing here touches the network;
- conv1 becomes a C-channel conv whose filters are the mean of the RGB filters, replicated over the C channels and
  scaled by 3 / C, so a C-channel input whose channels are all equal to one image gives exactly the response the RGB
  conv gives to that image in all three channels;
- fc becomes a single-logit linear layer. It is computed in float32 even under bf16 autocast (decision in
  `docs/decisions/P1-06.md`), so the logits that AUC, the seed ensemble and stage 2 rank are not rounded to bf16.

The model returns logits of shape (N, 1).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
import torchvision
from torch import nn

from landslide.paths import Paths

# torch.hub names a cached checkpoint `<name>-<first hex digits of its sha256>.<ext>`; same pattern as torch.hub.
_HUB_HASH = re.compile(r"-([a-f0-9]{6,})\.")
_WEIGHTS_TAG = re.compile(r"\b(IMAGENET1K_V\d+)\b")


class Float32Linear(nn.Linear):
    """`nn.Linear` that always runs in float32, also inside an autocast region. State-dict keys are unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float())


def _check_model_cfg(cfg: dict[str, Any]) -> int:
    m = cfg["model"]
    if not str(m["arch"]).startswith("resnet"):
        raise ValueError(f"model.arch {m['arch']!r}: only torchvision ResNets are implemented")
    if m["head"] != "single logit":
        raise ValueError(f"model.head {m['head']!r}: only 'single logit' is implemented")
    c = int(m["in_channels"])
    if c != len(cfg["channels"]):
        raise ValueError(f"model.in_channels = {c} but the config lists {len(cfg['channels'])} channels")
    return c


def imagenet_weights(cfg: dict[str, Any]) -> torchvision.models.WeightsEnum:
    """The torchvision weights enum named in `cfg['model']['init']` (e.g. `ResNet18_Weights.IMAGENET1K_V1`)."""
    tag = _WEIGHTS_TAG.search(cfg["model"]["init"])
    if tag is None:
        raise ValueError(f"no IMAGENET1K_V* weights named in model.init {cfg['model']['init']!r}")
    return torchvision.models.get_model_weights(cfg["model"]["arch"])[tag.group(1)]


def imagenet_weights_path(cfg: dict[str, Any], paths: Paths) -> Path:
    """Where torchvision caches the weights file with `TORCH_HOME = paths.torch_home` (`hub/checkpoints/<file>`).

    The file name is the config's `sources.weights.file`; it must be the name torchvision gives the weights that
    `model.init` names, or the config and the download would disagree.
    """
    name = str(cfg["sources"]["weights"]["file"])
    torchvision_name = Path(urlparse(imagenet_weights(cfg).url).path).name
    if name != torchvision_name:
        raise ValueError(f"sources.weights.file {name!r} is not torchvision's file for {cfg['model']['init']!r} "
                         f"({torchvision_name!r})")
    return paths.torch_home / "hub" / "checkpoints" / name


def load_state_dict_file(path: Path | str) -> dict[str, torch.Tensor]:
    """Load a state dict from a local file (CPU, `weights_only`).

    If the file name carries a torch.hub hash prefix (`resnet18-f37072fd.pth`), the file's sha256 must start with it.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"weights file not found: {p} (P1-07's fetch_weights.py caches it; no download here)")
    m = _HUB_HASH.search(p.name)
    if m:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if not h.hexdigest().startswith(m.group(1)):
            raise ValueError(f"{p.name}: sha256 {h.hexdigest()[:16]}... does not start with {m.group(1)}")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(f"{p}: expected a state dict, got {type(sd).__name__}")
    return sd


def conv1_from_rgb(w_rgb: torch.Tensor, in_channels: int) -> torch.Tensor:
    """(O, 3, k, k) RGB filters -> (O, C, k, k): the mean over the RGB filters, replicated C times, times 3 / C."""
    if w_rgb.ndim != 4:
        raise ValueError(f"expected conv weights (O, I, k, k), got shape {tuple(w_rgb.shape)}")
    n_rgb = w_rgb.shape[1]
    return w_rgb.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * (n_rgb / in_channels)


def build_model(cfg: dict[str, Any], weights_path: Path | str | None) -> nn.Module:
    """ResNet with the ImageNet state dict from `weights_path`, conv1 widened to C channels and a 1-logit head.

    `weights_path=None` builds the same architecture from random initialisation; use it only to load a trained
    checkpoint (`load_checkpoint`). The random draws come from torch's global generator, so seed it first.
    """
    c = _check_model_cfg(cfg)
    net = torchvision.models.get_model(cfg["model"]["arch"], weights=None)
    if weights_path is not None:
        net.load_state_dict(load_state_dict_file(weights_path), strict=True)
    old = net.conv1
    conv1 = nn.Conv2d(c, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding,
                      dilation=old.dilation, groups=old.groups, bias=old.bias is not None)
    with torch.no_grad():
        conv1.weight.copy_(conv1_from_rgb(old.weight.detach(), c))
        if old.bias is not None:
            conv1.bias.copy_(old.bias.detach())
    net.conv1 = conv1
    net.fc = Float32Linear(net.fc.in_features, 1)
    return net


def load_checkpoint(path: Path | str, cfg: dict[str, Any],
                    map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from a `best.pt` written by `landslide.train` and return (model in eval mode, its metadata)."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
    with torch.random.fork_rng(devices=[]):  # the throw-away random init must not move the caller's CPU generator
        model = build_model(cfg, None)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(map_location).eval()
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}
