import numpy as np
import pytest
import torch

from landslide import d4


@pytest.fixture
def x():
    return np.random.default_rng(0).normal(size=(2, 3, 5, 5))


def test_eight_distinct_transforms(x):
    views = [d4.transform(x, k) for k in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            assert not np.array_equal(views[i], views[j])


@pytest.mark.parametrize("k", range(8))
def test_inverse_roundtrip(x, k):
    np.testing.assert_array_equal(d4.inverse(d4.transform(x, k), k), x)


def test_group_closure_and_identity(x):
    assert d4.compose_index(0, 0) == 0
    for a in range(8):
        assert d4.compose_index(a, d4.inverse_index(a)) == 0
        for b in range(8):
            c = d4.compose_index(a, b)
            np.testing.assert_array_equal(d4.transform(d4.transform(x, b), a), d4.transform(x, c))


def test_rotation_direction():
    a = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(d4.transform(a, 1), [[2, 4], [1, 3]])
    np.testing.assert_array_equal(d4.transform(a, 4), [[2, 1], [4, 3]])


def test_bad_index():
    with pytest.raises(ValueError):
        d4.transform(np.zeros((2, 2)), 8)


@pytest.mark.parametrize("k", range(8))
def test_torch_matches_numpy(x, k):
    t = torch.from_numpy(x.copy())
    np.testing.assert_array_equal(d4.t_transform(t, k).numpy(), d4.transform(x, k))
    np.testing.assert_array_equal(d4.t_inverse(d4.t_transform(t, k), k).numpy(), x)


def test_torch_per_sample(x):
    t = torch.from_numpy(np.random.default_rng(1).normal(size=(16, 3, 5, 5)))
    ks = torch.arange(16) % 8
    out = d4.t_apply_per_sample(t, ks)
    for i in range(16):
        np.testing.assert_array_equal(out[i].numpy(), d4.transform(t[i].numpy(), int(ks[i])))


def test_torch_per_sample_rejects_non_square():
    with pytest.raises(ValueError):
        d4.t_apply_per_sample(torch.zeros(2, 1, 3, 4), torch.zeros(2, dtype=torch.long))


def test_torch_all_views_layout(x):
    t = torch.from_numpy(x.copy())
    v = d4.t_all_views(t)
    assert v.shape == (8 * x.shape[0], *x.shape[1:])
    for k in range(8):
        np.testing.assert_array_equal(v[k * 2:(k + 1) * 2].numpy(), d4.transform(x, k))


def test_invariant_input_has_identical_views():
    # A radially symmetric field is fixed by every D4 element.
    yy, xx = np.mgrid[-3:4, -3:4]
    z = torch.from_numpy((xx**2 + yy**2).astype(float))[None, None]
    v = d4.t_all_views(z)
    for k in range(8):
        assert torch.equal(v[k], v[0])
