"""landslide.predict: 8-view TTA and normalisation, on random tensors (CPU)."""

import json

import numpy as np
import pytest
import torch

from landslide import d4
from landslide.config import load_config
from landslide.model import build_model
from landslide.predict import Normaliser, probs_from_logits, tta_logits, tta_predict

CFG = load_config()
C = CFG["model"]["in_channels"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(CFG, None).eval()


def _invariant_batch(n=3, size=24):
    """Fields that depend only on the squared distance from the patch centre, so every D4 view is identical."""
    c = (size - 1) / 2
    i, j = np.mgrid[0:size, 0:size]
    r2 = (i - c) ** 2 + (j - c) ** 2
    rng = np.random.default_rng(5)
    a = rng.uniform(0.01, 0.2, size=(n, C, 1, 1))
    b = rng.normal(size=(n, C, 1, 1))
    return torch.from_numpy(np.cos(a * r2) + b).float()


def test_shape_and_dtype(model):
    x = torch.randn(5, C, 24, 24)
    logits = tta_logits(model, x, batch_size=2)
    assert logits.shape == (8, 5) and logits.dtype == torch.float32
    p = tta_predict(model, x, batch_size=2)
    assert p.shape == (5,) and p.dtype == torch.float64
    assert ((p > 0) & (p < 1)).all()


@pytest.mark.parametrize("amp", [False, True])
def test_d4_invariant_input(model, amp):
    x = _invariant_batch()
    views = d4.t_all_views(x)
    for k in range(8):
        assert torch.equal(views[k * 3:(k + 1) * 3], x)
    logits = tta_logits(model, x, amp=amp)
    torch.testing.assert_close(logits, logits[:1].expand(8, -1), rtol=1e-5, atol=1e-5)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=amp):
        single = torch.sigmoid(model(x)[:, 0].double())
    torch.testing.assert_close(tta_predict(model, x, amp=amp), single, rtol=1e-5, atol=1e-6)


def test_equals_mean_sigmoid_over_views(model):
    x = torch.randn(4, C, 24, 24)
    with torch.no_grad():
        per_view = torch.stack([model(d4.t_transform(x, k))[:, 0] for k in range(8)])
    logits = tta_logits(model, x, amp=False)
    torch.testing.assert_close(logits, per_view, rtol=1e-5, atol=1e-5)  # row k is view k
    expected = torch.sigmoid(per_view.double()).mean(0)
    torch.testing.assert_close(tta_predict(model, x, amp=False), expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(logits[0], logits[1])  # a generic input is not D4-invariant


def test_batching_does_not_change_results(model):
    x = torch.randn(5, C, 24, 24)
    full = tta_predict(model, x, batch_size=5, amp=False)
    for bs in (1, 2, 3):
        torch.testing.assert_close(tta_predict(model, x, batch_size=bs, amp=False), full, rtol=1e-5, atol=1e-6)


def test_normalisation_applied(model, tmp_path):
    rng = np.random.default_rng(2)
    mean, std = rng.normal(size=C) * 10, rng.uniform(1, 5, size=C)
    norm = Normaliser(mean, std)
    raw = torch.randn(3, C, 24, 24, dtype=torch.float64) * torch.tensor(std).reshape(1, C, 1, 1) + \
        torch.tensor(mean).reshape(1, C, 1, 1)
    z = (raw - torch.tensor(mean).reshape(1, C, 1, 1)) / torch.tensor(std).reshape(1, C, 1, 1)
    torch.testing.assert_close(norm.apply(raw.half()), z.float(), rtol=2e-3, atol=2e-2)
    torch.testing.assert_close(tta_predict(model, raw.float(), norm=norm, amp=False),
                               tta_predict(model, z.float(), amp=False), rtol=1e-4, atol=1e-5)
    (tmp_path / "norm.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}), encoding="utf-8")
    again = Normaliser.from_json(tmp_path / "norm.json")
    np.testing.assert_array_equal(again.mean, mean)
    np.testing.assert_array_equal(again.std, std)


def test_normaliser_rejects_bad_stats():
    with pytest.raises(ValueError):
        Normaliser([0.0, 1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        Normaliser([0.0], [1.0, 2.0])


def test_restores_training_mode(model):
    model.train()
    try:
        tta_predict(model, torch.randn(2, C, 24, 24))
        assert model.training
    finally:
        model.eval()


def test_probs_from_logits():
    logits = torch.tensor([[0.0, 2.0], [0.0, -2.0]] + [[0.0, 0.0]] * 6)
    p = probs_from_logits(logits)
    torch.testing.assert_close(p, torch.tensor([0.5, (1 / (1 + np.exp(-2)) + 1 / (1 + np.exp(2)) + 3) / 8],
                                               dtype=torch.float64))
