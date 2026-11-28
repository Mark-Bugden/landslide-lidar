"""Diagnostic for the flat-looking validation curves (phase 3; changes no frozen result).

Usage: uv run python scripts/diagnose_first_epoch.py [out.json]   (default results/diag_first_epoch.json)

1. Leakage: do inner-val or test patches of a_f0 overlap (or nearly touch) its training patches?
2. Simple-feature reference: terrain-stats LR fitted on a_f0's training rows, AUC on its inner-val rows.
3. Fine-grained training: re-train a_f0 seed 0 exactly as train_unit does for 2 epochs, scoring the inner-val set
   (8-view TTA) before training and every 4 steps.
"""

import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from landslide import d4
from landslide.config import load_config
from landslide.model import build_model, imagenet_weights_path
from landslide.patches import read_meta
from landslide.paths import Paths
from landslide.predict import Normaliser, tta_predict
from landslide.train import (LRSchedule, deterministic_mode, norm_stats, rows_to_device, seed_everything,
                             select_unit_rows)

OUT = sys.argv[1] if len(sys.argv) > 1 else str(Paths().results / "diag_first_epoch.json")
cfg, paths = load_config(), Paths()
meta = read_meta(paths.meta_csv)
splits = json.loads(paths.splits_json.read_text(encoding="utf-8"))
arr = np.load(paths.patches_npy, mmap_mode="r")
run, seed = "a_f0", 0
rows = select_unit_rows(run, meta, splits)
y = meta["label"].to_numpy().astype(np.float32)
res = {"run": run, "seed": seed, "n_train": len(rows.train), "n_val": len(rows.val), "n_test": len(rows.test)}

# 1. leakage: Chebyshev distance between patch centres; squares overlap iff < 300 m
xy = meta[["x", "y"]].to_numpy(float)
tree = cKDTree(xy[rows.train])
for name, r in (("val", rows.val), ("test", rows.test)):
    d, _ = tree.query(xy[r], p=np.inf)
    res[f"{name}_to_train_min_m"] = float(d.min())
    res[f"{name}_overlapping_train"] = int((d < 300).sum())
    res[f"{name}_within_600m_of_train"] = int((d < 600).sum())
res["val_sheets_in_train"] = len(set(meta.sheet.iloc[rows.val]) & set(meta.sheet.iloc[rows.train]))
res["val_blocks_in_train"] = len(set(meta.block.iloc[rows.val]) & set(meta.block.iloc[rows.train]))

# 2. simple-feature reference on the same inner-val rows
feats = [s["features"] for s in cfg["replication"]["baselines"]["scorers"] if s["name"] == "terrain_stats_lr"][0]
lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=int(cfg["replication"]["seed"])))
lr.fit(meta.iloc[rows.train][feats], y[rows.train])
res["terrain_stats_lr_val_auc"] = float(roc_auc_score(y[rows.val], lr.predict_proba(meta.iloc[rows.val][feats])[:, 1]))
res["slope_mean_val_auc"] = float(roc_auc_score(y[rows.val], meta.iloc[rows.val]["slope_mean"]))

# 3. fine-grained re-training, same recipe as train_unit
tr = cfg["train"]
batch = int(tr["batch_size"])
dev = torch.device("cuda")
curve = []
with deterministic_mode():
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    x_train, x_val = rows_to_device(arr, rows.train, dev), rows_to_device(arr, rows.val, dev)
    y_train, y_val = torch.from_numpy(y[rows.train]).to(dev), y[rows.val]
    norm = Normaliser(*norm_stats(x_train))
    model = build_model(cfg, imagenet_weights_path(cfg, paths)).to(dev)
    spe = len(rows.train) // batch
    sched = LRSchedule.from_cfg(cfg, spe)
    opt = torch.optim.AdamW(model.parameters(), lr=sched(0), weight_decay=float(tr["weight_decay"]))

    def score(step, epoch):
        p = tta_predict(model, x_val, batch_size=batch, norm=norm).numpy()
        q = tta_predict(model, x_train[:len(rows.val)], batch_size=batch, norm=norm).numpy()
        curve.append({"step": step, "epoch": epoch, "samples_seen": step * batch, "val_auc": roc_auc_score(y_val, p),
                      "train_subset_auc": roc_auc_score(y[rows.train][:len(rows.val)], q), "lr": sched(max(step - 1, 0))})
        print(curve[-1], flush=True)

    score(0, 0)
    step = 0
    for epoch in (1, 2):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(rows.train))).to(dev)
        ks = torch.from_numpy(rng.integers(0, 8, size=len(rows.train))).to(dev)
        for s in range(spe):
            for g in opt.param_groups:
                g["lr"] = sched(step)
            sl = slice(s * batch, (s + 1) * batch)
            idx = perm[sl]
            xb = norm.apply(d4.t_apply_per_sample(x_train[idx], ks[sl]))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(xb)[:, 0]
            loss = F.binary_cross_entropy_with_logits(logits.float(), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 4 == 0 or s == spe - 1:
                score(step, epoch)
res["curve"] = curve
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))
