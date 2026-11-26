"""Figures drawn by the evaluation (PLAN.md P1-13; BRIEF.md Appendix A §7 "Monitoring").

`learning_curves` draws `figures/learning_curves.png`: one row per run (a_f0..a_f4, b, s2, in that order), with a BCE
panel and an ROC AUC panel against the epoch.

- Each seed's train curve (blue) and validation curve (orange) is drawn faint.
- The bold curve of each colour is the mean of the seeds' curves, drawn only over the epochs that every seed has run
  ("drawn while all seeds are still training"). It is the mean of the seeds' curves, not the ensemble's score.
- A dot marks each seed's best epoch (the one kept), on its validation curve in both panels. Only finished units
  (with `done.json`) have a best epoch; a failed or interrupted unit's history is drawn without a dot.

No map, sheet geometry or susceptibility is ever drawn here (licence rule, PLAN.md §3 rule 3).

The figure is rendered with matplotlib's Agg canvas (no pyplot state) and saved without the "Software" metadata
entry, so the same inputs give the same bytes.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

__all__ = ["TRAIN_COLOUR", "VAL_COLOUR", "learning_curves_png", "learning_curves"]

#: Appendix A §7 fixes train = blue, validation = orange (a colour-blind-safe pair).
TRAIN_COLOUR = "#2a78d6"
VAL_COLOUR = "#eb6834"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e4e3de"
_FAINT_ALPHA = 0.35
_FAINT_LW = 1.0
_BOLD_LW = 2.2
_DPI = 100

_PANELS = (("BCE loss", "train_loss", "val_loss"), ("ROC AUC", "train_auc", "val_auc"))


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=_INK_2, labelsize=8, length=3, width=0.6)


def _mean_curve(curves: Sequence[np.ndarray]) -> np.ndarray:
    """Mean over seeds, over the epochs every seed has run."""
    n = min(len(c) for c in curves)
    return np.nanmean(np.stack([c[:n] for c in curves]), axis=0) if n else np.empty(0)


def learning_curves_png(histories: Mapping[str, Mapping[int, pd.DataFrame]],
                        best_epochs: Mapping[str, Mapping[int, int]]) -> bytes:
    """Render the learning-curve figure and return the PNG bytes.

    `histories[run][seed]` is that unit's `history.csv` (columns epoch, train_loss, train_auc, val_loss, val_auc);
    runs are drawn in the mapping's order, seeds in ascending order. `best_epochs[run][seed]` is the epoch kept
    (done units only).
    """
    runs = [r for r, h in histories.items() if any(len(df) for df in h.values())]
    if not runs:
        raise ValueError("no history to draw")
    n = len(runs)
    fig = Figure(figsize=(9.0, 1.85 * n + 0.9), dpi=_DPI, facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = np.atleast_2d(fig.subplots(n, 2, squeeze=False))
    for i, run in enumerate(runs):
        seeds = sorted(s for s, df in histories[run].items() if len(df))
        for j, (title, train_col, val_col) in enumerate(_PANELS):
            ax = axes[i, j]
            _style(ax)
            for col, colour in ((train_col, TRAIN_COLOUR), (val_col, VAL_COLOUR)):
                curves = []
                for s in seeds:
                    df = histories[run][s].sort_values("epoch")
                    ep, y = df["epoch"].to_numpy(dtype=float), df[col].to_numpy(dtype=float)
                    ax.plot(ep, y, color=colour, alpha=_FAINT_ALPHA, linewidth=_FAINT_LW)
                    curves.append(y)
                mean = _mean_curve(curves)
                ax.plot(np.arange(1, mean.size + 1), mean, color=colour, linewidth=_BOLD_LW)
            for s in seeds:
                be = best_epochs.get(run, {}).get(s)
                if be is None:
                    continue
                df = histories[run][s]
                row = df.loc[df["epoch"] == int(be), val_col]
                if len(row):
                    ax.plot([int(be)], [float(row.iloc[0])], marker="o", markersize=6, color=VAL_COLOUR,
                            markeredgecolor=_SURFACE, markeredgewidth=1.2, linestyle="none", zorder=5)
            if i == 0:
                ax.set_title(title, fontsize=10, color=_INK, loc="left")
            if i == n - 1:
                ax.set_xlabel("epoch", fontsize=9, color=_INK_2)
            if j == 0:
                ax.set_ylabel(run, fontsize=10, color=_INK, rotation=0, ha="right", va="center", labelpad=8)
    handles = [
        Line2D([], [], color=TRAIN_COLOUR, alpha=_FAINT_ALPHA, linewidth=_FAINT_LW, label="train, per seed"),
        Line2D([], [], color=VAL_COLOUR, alpha=_FAINT_ALPHA, linewidth=_FAINT_LW, label="validation, per seed"),
        Line2D([], [], color=TRAIN_COLOUR, linewidth=_BOLD_LW, label="train, mean of seeds"),
        Line2D([], [], color=VAL_COLOUR, linewidth=_BOLD_LW, label="validation, mean of seeds"),
        Line2D([], [], color=VAL_COLOUR, marker="o", markersize=6, markeredgecolor=_SURFACE, linestyle="none",
               label="best epoch (kept)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=5, fontsize=8, frameon=False, labelcolor=_INK_2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, facecolor=_SURFACE, metadata={"Software": None})
    return buf.getvalue()


def learning_curves(histories: Mapping[str, Mapping[int, pd.DataFrame]],
                    best_epochs: Mapping[str, Mapping[int, int]], out: Path | str) -> Path:
    """Draw the learning-curve figure to `out` (PNG), written atomically. Returns the path."""
    data = learning_curves_png(histories, best_epochs)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)
    return p
