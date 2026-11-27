"""landslide.train on synthetic meta tables, splits and random patches only (gate G3: no real patch is read).

The training tests use tiny patches (cfg patch.px = 32) and the real config otherwise. They run on the GPU when
there is one and on the CPU otherwise; the bit-identity test needs CUDA and is skipped without it.
"""

import copy
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before anything starts CUDA

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision

from landslide import train as T
from landslide.config import load_config
from landslide.model import load_checkpoint
from landslide.paths import Paths
from landslide.predict import Normaliser, tta_predict

CFG = load_config()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PX = 32
N_PER_BLOCK = 24
N_BG = 2

# 10 flysch blocks (fold = i % 5; group_b train for F0..F5) and 4 non-flysch blocks (fold -1; train for N0, N1)
BLOCKS = [(f"F{i}", i % 5, "train" if i < 6 else "test") for i in range(10)] + \
         [(f"N{i}", -1, "train" if i < 2 else "test") for i in range(4)]
INNER_VAL = {"a_f0": ["F1"], "a_f1": ["F2"], "a_f2": ["F3"], "a_f3": ["F4"], "a_f4": ["F0"], "b": ["F2", "N1"],
             "s2": ["F7", "N3"]}


def synth_meta(n_per_block=N_PER_BLOCK, n_bg=N_BG):
    rows = []
    for b, fold, g in BLOCKS:
        for j in range(n_per_block):
            pos = j % 2 == 1
            rows.append({"label": int(pos), "source": "positive" if pos else ("random", "quarry", "cut")[j // 2 % 3],
                         "fold": fold, "group_b": g, "block": b, "sheet": f"{b}-{j % 3}", "flysch": fold >= 0})
        for _ in range(n_bg):
            rows.append({"label": 0, "source": "background", "fold": fold, "group_b": g, "block": b,
                         "sheet": f"{b}-0", "flysch": fold >= 0})
    meta = pd.DataFrame(rows)
    meta.insert(0, "id", np.arange(len(meta)) + 1000)
    return meta


def synth_splits():
    return {
        "stage2": {"flysch": [], "nonflysch": []},
        "folds_a": {str(k): {"blocks": [b for b, f, _ in BLOCKS if f == k], "sheets": [], "n_eligible": 0}
                    for k in range(5)},
        "group_b": {"train_blocks": [b for b, _, g in BLOCKS if g == "train"],
                    "test_blocks": [b for b, _, g in BLOCKS if g == "test"]},
        "inner_val": INNER_VAL,
        "config_hash": "synthetic",
    }


def synth_patches(meta, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(meta), len(CFG["channels"]), PX, PX)) * 3 + 10
    x[meta["label"].to_numpy() == 1, 0] += 0.3  # a weak signal, so the AUCs are not all 0.5
    return x.astype(np.float16)


@pytest.fixture(scope="module")
def meta():
    return synth_meta()


@pytest.fixture(scope="module")
def splits():
    return synth_splits()


@pytest.fixture(scope="module")
def patches(meta):
    return synth_patches(meta)


@pytest.fixture
def cfg():
    c = copy.deepcopy(CFG)
    c["patch"]["px"] = PX
    return c


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    torch.manual_seed(1234)
    p = tmp_path_factory.mktemp("weights") / "fake_resnet18.pth"
    torch.save(torchvision.models.resnet18(weights=None).state_dict(), p)
    return p


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.delenv("LANDSLIDE_DATA_DIR", raising=False)
    return Paths(root=tmp_path)


def run(cfg, paths, meta, splits, patches, weights, unit="a_f0", seed=0, **kw):
    kw.setdefault("epoch_limit", 1)
    return T.train_unit(unit, seed, cfg, paths, data=patches, meta=meta, splits=splits, weights_path=weights,
                        device=DEV, show_progress=False, **kw)


# ---------------------------------------------------------------------------------------------------------------
# Unit selection
# ---------------------------------------------------------------------------------------------------------------

def _rows(meta, mask):
    return np.flatnonzero(mask.to_numpy())


@pytest.mark.parametrize("k", range(5))
def test_select_a_folds(meta, splits, k):
    r = T.select_unit_rows(f"a_f{k}", meta, splits)
    bg = meta["source"] == "background"
    np.testing.assert_array_equal(r.test, _rows(meta, ~bg & (meta["fold"] == k)))
    pool = ~bg & (meta["fold"] >= 0) & (meta["fold"] != k)
    inner = meta["block"].isin(INNER_VAL[f"a_f{k}"])
    np.testing.assert_array_equal(r.val, _rows(meta, pool & inner))
    np.testing.assert_array_equal(r.train, _rows(meta, pool & ~inner))
    assert not bg.iloc[np.concatenate([r.train, r.val, r.test])].any()
    assert (meta["flysch"].iloc[np.concatenate([r.train, r.val, r.test])]).all()
    assert len(set(r.train) | set(r.val) | set(r.test)) == len(r.train) + len(r.val) + len(r.test)
    assert len(r.val) == N_PER_BLOCK and len(r.test) == 2 * N_PER_BLOCK


def test_select_b(meta, splits):
    r = T.select_unit_rows("b", meta, splits)
    bg = meta["source"] == "background"
    np.testing.assert_array_equal(r.test, _rows(meta, ~bg & (meta["group_b"] == "test")))
    pool = ~bg & (meta["group_b"] == "train")
    inner = meta["block"].isin(INNER_VAL["b"])
    np.testing.assert_array_equal(r.val, _rows(meta, pool & inner))
    np.testing.assert_array_equal(r.train, _rows(meta, pool & ~inner))
    assert set(meta["block"].iloc[r.train]) == {"F0", "F1", "F3", "F4", "F5", "N0"}  # flysch and non-flysch


def test_select_s2_uses_background(meta, splits):
    r = T.select_unit_rows("s2", meta, splits)
    assert len(r.test) == 0
    inner = meta["block"].isin(INNER_VAL["s2"])
    np.testing.assert_array_equal(r.val, _rows(meta, inner))
    np.testing.assert_array_equal(r.train, _rows(meta, ~inner))
    assert len(r.train) + len(r.val) == len(meta)
    bg = (meta["source"] == "background").to_numpy()
    assert bg[r.train].sum() == N_BG * (len(BLOCKS) - 2) and bg[r.val].sum() == N_BG * 2


def test_select_keep_subset(meta, splits):
    keep = np.flatnonzero(np.arange(len(meta)) % 3 != 0)
    full = T.select_unit_rows("a_f1", meta, splits)
    sub = T.select_unit_rows("a_f1", meta, splits, keep=keep)
    for a, b in ((full.train, sub.train), (full.val, sub.val), (full.test, sub.test)):
        np.testing.assert_array_equal(b, a[a % 3 != 0])
    mask = np.zeros(len(meta), dtype=bool)
    mask[keep] = True
    np.testing.assert_array_equal(T.select_unit_rows("a_f1", meta, splits, keep=mask).train, sub.train)


def test_select_rejects_leakage_and_inconsistency(meta, splits):
    bad = copy.deepcopy(splits)
    bad["inner_val"]["a_f0"] = ["F0"]  # a test block as inner-val
    with pytest.raises(ValueError, match="inner-val"):
        T.select_unit_rows("a_f0", meta, bad)
    bad = copy.deepcopy(splits)
    bad["folds_a"]["0"]["blocks"] = ["F0"]  # meta says F5 is fold 0 too
    with pytest.raises(ValueError, match="fold 0"):
        T.select_unit_rows("a_f0", meta, bad)
    bad = copy.deepcopy(splits)
    bad["group_b"]["train_blocks"].remove("N0")
    with pytest.raises(ValueError, match="group_b"):
        T.select_unit_rows("b", meta, bad)
    m3 = meta.copy()
    m3.loc[(m3["block"] == "F1") & (m3["source"] == "positive"), "label"] = 0
    with pytest.raises(ValueError, match="both classes"):
        T.select_unit_rows("a_f0", m3, splits)
    for name in ("a_f9", "c"):
        with pytest.raises((KeyError, ValueError)):
            T.select_unit_rows(name, meta, splits)


# ---------------------------------------------------------------------------------------------------------------
# Schedule and early stopping
# ---------------------------------------------------------------------------------------------------------------

def test_lr_schedule_boundaries():
    spe = 10
    s = T.LRSchedule.from_cfg(CFG, spe)
    base, W, Tt = CFG["train"]["lr"], CFG["train"]["warmup_epochs"] * spe, CFG["train"]["max_epochs"] * spe
    assert (s.base_lr, s.warmup_steps, s.total_steps) == (base, W, Tt)
    assert s(0) == pytest.approx(base / W, rel=1e-12)          # first step of warm-up
    assert s(W // 2 - 1) == pytest.approx(base / 2, rel=1e-12)  # half way through warm-up
    assert s(W - 1) == base                                     # last warm-up step: the peak
    assert s(W) == pytest.approx(base * 0.5 * (1 + math.cos(math.pi / (Tt - W))), rel=1e-12)
    assert s(W - 1 + (Tt - W) // 2) == pytest.approx(base / 2, rel=1e-12)  # cosine half way
    assert s(Tt - 1) == 0.0                                     # last step of max_epochs: exactly 0
    lrs = [s(t) for t in range(Tt)]
    assert all(a < b for a, b in zip(lrs[:W - 1], lrs[1:W]))
    assert all(a > b for a, b in zip(lrs[W - 1:-1], lrs[W:]))
    with pytest.raises(ValueError):
        s(Tt)
    with pytest.raises(ValueError):
        s(-1)


def test_lr_schedule_without_warmup():
    s = T.LRSchedule(1.0, 0, 4)
    assert [s(t) for t in range(4)] == pytest.approx([0.5 * (1 + math.cos(math.pi * k / 4)) for k in (1, 2, 3, 4)])


def test_early_stopping_scripted():
    es = T.EarlyStopping(patience=3)
    script = [0.60, 0.70, 0.70, 0.65, float("nan"), 0.69]
    improved = [es.update(e, v) for e, v in enumerate(script, start=1)]
    assert improved == [True, True, False, False, False, False]
    assert es.best_epoch == 2 and es.best_value == 0.70  # the tie at epoch 3 goes to the earliest
    assert es.should_stop  # epochs 3, 4, 5 without improvement
    es = T.EarlyStopping(patience=8)
    for e, v in enumerate([0.5, 0.6, 0.55, 0.58, 0.59, 0.6, 0.61, 0.6, 0.6, 0.6, 0.6], start=1):
        es.update(e, v)
        assert es.should_stop == (e >= 7 + 8)
    assert es.best_epoch == 7
    with pytest.raises(ValueError):
        es.update(3, 0.9)


# ---------------------------------------------------------------------------------------------------------------
# train_unit
# ---------------------------------------------------------------------------------------------------------------

def test_outputs_and_contract(cfg, paths, meta, splits, patches, weights):
    res = run(cfg, paths, meta, splits, patches, weights, epoch_limit=2)
    d = paths.run_dir("a_f0", 0)
    assert res["status"] == "done"
    assert sorted(p.name for p in d.iterdir()) == sorted(
        ["best.pt", "done.json", "history.csv", "norm.json", "preds_test.csv", "preds_val.csv"])
    rows = T.select_unit_rows("a_f0", meta, splits)

    hist = pd.read_csv(d / "history.csv")
    assert list(hist.columns) == ["epoch", "lr", "train_loss", "train_auc", "val_loss", "val_auc", "seconds"]
    assert hist["epoch"].tolist() == [1, 2]
    spe = len(rows.train) // cfg["train"]["batch_size"]
    sched = T.LRSchedule.from_cfg(cfg, spe)
    assert hist["lr"].tolist() == [sched(spe - 1), sched(2 * spe - 1)]  # the lr of each epoch's last step
    assert np.isfinite(hist.drop(columns="epoch").to_numpy()).all()

    done = json.loads((d / "done.json").read_text(encoding="utf-8"))
    for key in ("unit", "seed", "config_hash", "git_commit", "best_epoch", "best_val_auc", "n_train", "n_val",
                "n_test", "seconds", "peak_gpu_mem_gb"):
        assert key in done
    assert (done["n_train"], done["n_val"], done["n_test"]) == (len(rows.train), len(rows.val), len(rows.test))
    assert done["best_val_auc"] == hist["val_auc"].max()
    assert done["best_epoch"] == int(hist["epoch"][hist["val_auc"] == hist["val_auc"].max()].min())
    assert done["stop_reason"] == "epoch_limit"
    if DEV == "cuda":
        assert 0 < done["peak_gpu_mem_gb"] < 12

    # norm.json: float64 statistics of the training rows only (inner-val excluded)
    norm = json.loads((d / "norm.json").read_text(encoding="utf-8"))
    xt = patches[rows.train].astype(np.float64)
    np.testing.assert_allclose(norm["mean"], xt.mean(axis=(0, 2, 3)), rtol=1e-12)
    np.testing.assert_allclose(norm["std"], xt.std(axis=(0, 2, 3)), rtol=1e-10)
    xtv = patches[np.concatenate([rows.train, rows.val])].astype(np.float64)
    assert not np.allclose(norm["mean"], xtv.mean(axis=(0, 2, 3)), rtol=1e-12)

    pv, pt = pd.read_csv(d / "preds_val.csv"), pd.read_csv(d / "preds_test.csv")
    assert list(pv.columns) == ["id", "prob"] == list(pt.columns)
    np.testing.assert_array_equal(pv["id"], meta["id"].to_numpy()[rows.val])
    np.testing.assert_array_equal(pt["id"], meta["id"].to_numpy()[rows.test])
    assert pv["prob"].between(0, 1).all() and pt["prob"].between(0, 1).all()

    # best.pt reproduces the predictions; the restored weights give the best epoch's inner-val AUC
    model, info = load_checkpoint(d / "best.pt", cfg, map_location=DEV)
    assert info["epoch"] == done["best_epoch"] and info["config_hash"] == done["config_hash"]
    with T.deterministic_mode():
        p = tta_predict(model, torch.from_numpy(patches[rows.test]), batch_size=cfg["train"]["batch_size"],
                        norm=Normaliser.from_json(d / "norm.json")).numpy()
    np.testing.assert_allclose(p, pt["prob"].to_numpy(), rtol=1e-12, atol=0)
    assert done["val_auc_recomputed"] == done["best_val_auc"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bit-identity is checked on the GPU; CUDA is absent")
def test_two_runs_bit_identical(cfg, tmp_path, monkeypatch, meta, splits, patches, weights):
    monkeypatch.delenv("LANDSLIDE_DATA_DIR", raising=False)
    dirs = []
    for name, seed in (("one", 0), ("two", 0), ("other_seed", 1)):
        p = Paths(root=tmp_path / name)
        run(cfg, p, meta, splits, patches, weights, seed=seed, epoch_limit=1)
        dirs.append(p.run_dir("a_f0", seed))
    a, b, c = dirs
    ha, hb, hc = (pd.read_csv(x / "history.csv").drop(columns="seconds") for x in dirs)
    pd.testing.assert_frame_equal(ha, hb, check_exact=True)
    for f in ("preds_val.csv", "preds_test.csv", "norm.json"):
        if f == "norm.json":
            na, nb = (json.loads((x / f).read_text(encoding="utf-8")) for x in (a, b))
            assert na["mean"] == nb["mean"] and na["std"] == nb["std"]
        else:
            assert (a / f).read_bytes() == (b / f).read_bytes()
    sa, sb, sc = (torch.load(x / "best.pt", weights_only=True)["state_dict"] for x in dirs)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)  # another seed trains differently
    assert not ha.equals(hc)


def test_early_stopping_in_train_unit(cfg, paths, meta, splits, patches, weights, monkeypatch):
    cfg["train"]["early_stopping"]["patience"] = 3
    script = iter([0.50, 0.80, 0.80, 0.70, 0.79, 0.60, 0.99])  # best epoch 2 (tie at 3); stop after epoch 5
    seen = {}

    def scripted(model, x_val, y_val, norm, batch_size, show_progress):
        epoch = len(seen) + 1
        seen[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return 0.5, next(script)

    monkeypatch.setattr(T, "_evaluate_val", scripted)
    res = run(cfg, paths, meta, splits, patches, weights, epoch_limit=None)
    d = paths.run_dir("a_f0", 0)
    assert res["status"] == "done" and res["stop_reason"] == "early_stopping"
    assert (res["best_epoch"], res["best_val_auc"], res["epochs_run"]) == (2, 0.80, 5)
    assert pd.read_csv(d / "history.csv")["val_auc"].tolist() == [0.50, 0.80, 0.80, 0.70, 0.79]
    best = torch.load(d / "best.pt", weights_only=True)
    assert best["epoch"] == 2
    assert all(torch.equal(best["state_dict"][k], seen[2][k]) for k in seen[2])
    assert not all(torch.equal(seen[2][k], seen[5][k]) for k in seen[2])  # rolled back, not the last weights


def test_interruption_hook(cfg, paths, meta, splits, patches, weights):
    calls = []

    def stop_after_first_epoch():
        calls.append(1)
        return len(calls) >= 2  # asked before epoch 1 (False), then after epoch 1 (True)

    res = run(cfg, paths, meta, splits, patches, weights, epoch_limit=3, should_stop=stop_after_first_epoch)
    d = paths.run_dir("a_f0", 0)
    assert res["status"] == "interrupted" and res["epochs_completed"] == 1
    assert not (d / "done.json").exists() and not (d / "best.pt").exists()
    assert (d / "interrupted.json").exists()
    assert len(pd.read_csv(d / "history.csv")) == 1

    res = run(cfg, paths, meta, splits, patches, weights, epoch_limit=1)  # a rerun starts afresh
    assert res["status"] == "done"
    assert (d / "done.json").exists() and not (d / "interrupted.json").exists()

    with pytest.raises(FileExistsError):
        run(cfg, paths, meta, splits, patches, weights)
    res = run(cfg, paths, meta, splits, patches, weights, overwrite=True,
              should_stop=lambda: True)  # stopped before the first epoch
    assert res["status"] == "interrupted" and res["epochs_completed"] == 0
    assert not (d / "done.json").exists()


def test_failure_writes_failed_json(cfg, paths, meta, splits, patches, weights):
    bad = patches.copy()
    rows = T.select_unit_rows("a_f0", meta, splits)
    bad[rows.train[3], 2, 5, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        run(cfg, paths, meta, splits, bad, weights)
    d = paths.run_dir("a_f0", 0)
    failed = json.loads((d / "failed.json").read_text(encoding="utf-8"))
    assert failed["unit"] == "a_f0_s0" and "non-finite" in failed["error"] and "Traceback" in failed["traceback"]
    assert not (d / "done.json").exists()


def test_smoke_mode_caps_epochs_and_redirects(cfg, tmp_path, monkeypatch, meta, splits, patches, weights):
    monkeypatch.delenv("LANDSLIDE_DATA_DIR", raising=False)
    p = Paths(root=tmp_path, smoke=True)
    res = run(cfg, p, meta, splits, patches, weights, unit="s2", epoch_limit=None)
    assert res["epochs_run"] == cfg["smoke"]["max_epochs"] and res["stop_reason"] == "epoch_limit"
    d = p.run_dir("s2", 0)
    assert (tmp_path / "data" / "smoke" / "runs") in d.parents
    assert pd.read_csv(d / "preds_test.csv").empty and res["n_test"] == 0
    assert res["n_train"] + res["n_val"] == len(meta)  # background included in s2


def test_rejects_config_it_does_not_implement(cfg, paths, meta, splits, patches, weights):
    cfg["train"]["optimizer"] = "SGD"
    with pytest.raises(ValueError, match="optimizer"):
        run(cfg, paths, meta, splits, patches, weights)


def test_too_few_training_rows(cfg, paths, meta, splits, patches, weights):
    cfg["train"]["batch_size"] = 10_000
    with pytest.raises(ValueError, match="less than one batch"):
        run(cfg, paths, meta, splits, patches, weights)


def test_patch_array_shape_checked(cfg, paths, meta, splits, patches, weights):
    cfg["patch"]["px"] = PX + 2
    with pytest.raises(ValueError, match="shape"):
        run(cfg, paths, meta, splits, patches, weights)


def test_memmap_path_source(cfg, paths, meta, splits, patches, weights, tmp_path):
    f = tmp_path / "synthetic_patches.npy"
    np.save(f, patches)
    res = run(cfg, paths, meta, splits, f, weights)
    assert res["status"] == "done"
