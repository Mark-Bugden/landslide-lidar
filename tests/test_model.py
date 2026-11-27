"""landslide.model on fake weights (a randomly initialised resnet18's state dict); no network, no real data."""

import copy
import hashlib
import shutil
import socket

import pytest
import torch
import torchvision
from torch import nn

from landslide.config import load_config
from landslide.model import (Float32Linear, build_model, conv1_from_rgb, imagenet_weights, imagenet_weights_path,
                             load_checkpoint, load_state_dict_file)
from landslide.paths import Paths

CFG = load_config()
C = CFG["model"]["in_channels"]


@pytest.fixture(scope="module")
def fake_sd():
    torch.manual_seed(1234)
    return torchvision.models.resnet18(weights=None).state_dict()


@pytest.fixture
def fake_weights(tmp_path, fake_sd):
    p = tmp_path / "fake_resnet18.pth"
    torch.save(fake_sd, p)
    return p


def test_output_shape_and_dtype(fake_weights):
    model = build_model(CFG, fake_weights).eval()
    x = torch.randn(2, C, CFG["patch"]["px"], CFG["patch"]["px"])
    with torch.no_grad():
        assert model(x).shape == (2, 1)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(x)
    assert out.shape == (2, 1)
    assert out.dtype == torch.float32  # the head runs in float32 under autocast


def test_conv1_init_formula(fake_weights, fake_sd):
    model = build_model(CFG, fake_weights)
    w_rgb = fake_sd["conv1.weight"]
    assert w_rgb.shape[1] == 3
    w = model.conv1.weight.detach()
    assert w.shape == (w_rgb.shape[0], C, *w_rgb.shape[2:])
    # mean of the RGB filters, replicated to C channels, times 3 / C
    expected = (w_rgb[:, 0] + w_rgb[:, 1] + w_rgb[:, 2]) / 3 * (3 / C)
    for c in range(C):
        torch.testing.assert_close(w[:, c], expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(conv1_from_rgb(w_rgb, C), w, rtol=0, atol=0)
    old = torchvision.models.resnet18().conv1
    conv = model.conv1
    assert (conv.in_channels, conv.kernel_size, conv.stride, conv.padding, conv.bias) == \
        (C, old.kernel_size, old.stride, old.padding, None)


def test_conv1_on_identical_channels_matches_rgb_conv(fake_weights, fake_sd):
    """C copies of an image through the new conv1 = 3 copies through the RGB conv1 (the point of the 3/C scale)."""
    model = build_model(CFG, fake_weights)
    g = torch.randn(2, 1, 40, 40, dtype=torch.float64)
    w_rgb = fake_sd["conv1.weight"].double()
    rgb = nn.functional.conv2d(g.expand(-1, 3, -1, -1), w_rgb, stride=2, padding=3)
    new = nn.functional.conv2d(g.expand(-1, C, -1, -1), model.conv1.weight.detach().double(), stride=2, padding=3)
    torch.testing.assert_close(new, rgb, rtol=1e-5, atol=1e-6)


def test_other_layers_loaded_and_head_replaced(fake_weights, fake_sd):
    model = build_model(CFG, fake_weights)
    sd = model.state_dict()
    for k, v in fake_sd.items():
        if k.startswith(("conv1.", "fc.")):
            continue
        assert torch.equal(sd[k], v), k
    assert isinstance(model.fc, Float32Linear)
    assert (model.fc.in_features, model.fc.out_features) == (512, 1)


def test_weights_path_is_torchvision_cache_file():
    paths = Paths()
    p = imagenet_weights_path(CFG, paths)
    assert imagenet_weights(CFG) is torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    assert p.parent == paths.torch_home / "hub" / "checkpoints"
    assert p.name == torchvision.models.ResNet18_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]


def test_no_network_used(fake_weights, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", boom)
    build_model(CFG, fake_weights)


def test_missing_weights_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no download"):
        build_model(CFG, tmp_path / "absent.pth")


def test_hub_hash_prefix_is_checked(tmp_path, fake_weights):
    digest = hashlib.sha256(fake_weights.read_bytes()).hexdigest()
    good = tmp_path / f"resnet18-{digest[:8]}.pth"
    bad = tmp_path / "resnet18-deadbeef.pth"
    shutil.copy(fake_weights, good)
    shutil.copy(fake_weights, bad)
    assert set(load_state_dict_file(good)) == set(torch.load(fake_weights, weights_only=True))
    with pytest.raises(ValueError, match="sha256"):
        load_state_dict_file(bad)


def test_cfg_mismatch_rejected(fake_weights):
    cfg = copy.deepcopy(CFG)
    cfg["model"]["in_channels"] = C + 1
    with pytest.raises(ValueError, match="in_channels"):
        build_model(cfg, fake_weights)


def test_load_checkpoint_roundtrip(tmp_path, fake_weights):
    torch.manual_seed(0)
    model = build_model(CFG, fake_weights).eval()
    torch.save({"state_dict": model.state_dict(), "epoch": 3, "val_auc": 0.7}, tmp_path / "best.pt")
    state = torch.random.get_rng_state()
    loaded, info = load_checkpoint(tmp_path / "best.pt", CFG)
    assert torch.equal(torch.random.get_rng_state(), state)  # the caller's generator is untouched
    assert info == {"epoch": 3, "val_auc": 0.7}
    assert not loaded.training
    x = torch.randn(2, C, 48, 48)
    with torch.no_grad():
        assert torch.equal(loaded(x), model(x))
