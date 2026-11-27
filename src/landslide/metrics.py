"""Metrics for tests (a), (b) and stage 2 (Appendix A §8 and §9; PLAN.md P1-05).

Every rule here is recorded in `docs/decisions/P1-05.md`. In short:

- **ROC AUC** (`roc_auc`) is the rank (Mann-Whitney) AUC with ties averaged: a positive-negative pair counts 1 if
  the positive scores higher, 1/2 if they tie, 0 otherwise. It is computed in exact integer arithmetic from the score
  tie groups and equals `sklearn.metrics.roc_auc_score`.
- **Cluster bootstrap.** `draw_clusters` draws the replicates once: the cluster labels (sheet or block) are sorted
  with `np.unique`, and replicate b draws K of the K clusters with replacement,
  `default_rng(seed).integers(0, K, size=(B, K))[b]`. A replicate is the concatenation of every drawn cluster's
  items, so an item's weight in replicate b is the number of times its cluster was drawn. Every scorer and every
  statistic bootstrapped from one `ClusterDraws` uses the same resamples, which is what makes paired differences
  paired. The replicate AUCs are computed from those weights, vectorised over replicates; this equals the AUC of
  the concatenated resample exactly.
- **Degenerate replicates (D21).** A replicate whose resample holds only one class has no AUC. It is dropped and
  counted. It depends only on the labels and the draws, so the same replicates are dropped for every scorer.
- **CI.** Percentile: the (1 - ci)/2 and 1 - (1 - ci)/2 quantiles of the used replicates, numpy's default
  ("linear") quantile method.
- **Paired differences** are reference minus other, replicate by replicate, on the same draws.
- **Seed ensemble** is the arithmetic mean of the seed probabilities, aligned on patch id; the id sets must agree.
- **Pair accuracy.** Per complete `pair_id` (>= 0), credit 1 if the positive scores above its matched negative,
  1/2 on a tie (consistent with the AUC), 0 otherwise. With `sheet` given, only same-sheet pairs count.
- **Precision-recall sweep (stage 2).** The operating points are the distinct scores of all detections, highest
  first; the point at score t contains every detection with score >= t, so tied detections enter together. TP and
  FP count; ignored and duplicate detections occupy thresholds but add to neither, so a threshold holding only them
  repeats the previous precision and recall. P@R is the precision at the highest threshold whose object recall
  (TP / targets) is >= the recall level; if no threshold reaches it, the result is explicitly "not reached", which
  `PRSweep.passes` reports as a fail.

Nothing here reads the config implicitly: values come from explicit arguments, and `bootstrap_spec(cfg)` reads the
`eval.bootstrap` block for callers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "STAGE2_OUTCOMES",
    "REACHED",
    "NOT_REACHED",
    "BootstrapSpec",
    "BootstrapResult",
    "ClusterDraws",
    "PairAccuracy",
    "PRSweep",
    "bootstrap_spec",
    "roc_auc",
    "draw_clusters",
    "draw_clusters_from_cfg",
    "percentile_ci",
    "replicate_aucs",
    "bootstrap_auc",
    "bootstrap_aucs",
    "bootstrap_statistic",
    "paired_differences",
    "seed_ensemble",
    "pair_accuracy",
    "pr_sweep",
]

STAGE2_OUTCOMES = ("tp", "fp", "ignored", "duplicate")
REACHED = "reached"
NOT_REACHED = "not_reached"

# Replicates are processed in chunks of about this many (replicate x item) cells, which keeps the int64
# temporaries near 16 MB each. An implementation detail, not a method value.
_CHUNK_CELLS = 2_000_000


# ------------------------------------------------------------------------------------------------ input checks


def _labels(y_true: ArrayLike) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y.shape}")
    if y.dtype == bool:
        return y
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0/1 (or bool)")
    return y.astype(bool)


def _scores(score: ArrayLike, n: int, name: str = "score") -> np.ndarray:
    s = np.asarray(score, dtype=np.float64)
    if s.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {s.shape}")
    if not np.isfinite(s).all():
        raise ValueError(f"{name} contains NaN or inf")
    return s


def _weights(weights: ArrayLike | None, n: int) -> np.ndarray:
    """int64 ones by default; integer weights stay int64 (exact), others become float64."""
    if weights is None:
        return np.ones(n, dtype=np.int64)
    w = np.asarray(weights)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    w = w.astype(np.int64) if np.issubdtype(w.dtype, np.integer) else w.astype(np.float64)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative")
    return w


# ------------------------------------------------------------------------------------------------ ROC AUC


def _tie_groups(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ascending stable order of `s` and the start index of each run of equal scores in that order."""
    order = np.argsort(s, kind="stable")
    ss = s[order]
    starts = np.flatnonzero(np.concatenate(([True], ss[1:] != ss[:-1])))
    return order, starts


def _u2(wpos: np.ndarray, wneg: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Twice the Mann-Whitney U, and the positive and negative weight totals, along the last axis.

    `wpos`, `wneg` hold each item's weight as a positive / negative in ascending score order (shape (..., n)).
    For tie group g with positive weight P_g, negative weight N_g and cumulative negative weight C_g,
    2U = sum_g P_g (2 C_g - N_g): every negative strictly below counts 1, every tied negative 1/2.
    """
    p = np.add.reduceat(wpos, starts, axis=-1)
    q = np.add.reduceat(wneg, starts, axis=-1)
    c = np.cumsum(q, axis=-1)
    return np.sum(p * (2 * c - q), axis=-1), p.sum(axis=-1), q.sum(axis=-1)


def roc_auc(y_true: ArrayLike, score: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Rank-based ROC AUC with ties averaged (= Mann-Whitney U / (n_pos n_neg)).

    `weights` are optional non-negative item multiplicities (integer weights are exact). Raises ValueError if
    either class is absent (or has zero total weight), or a score is NaN or inf.
    """
    y = _labels(y_true)
    s = _scores(score, y.size)
    w = _weights(weights, y.size)
    if y.size == 0:
        raise ValueError("ROC AUC of an empty set")
    order, starts = _tie_groups(s)
    yo, wo = y[order], w[order]
    u2, w1, w0 = _u2(np.where(yo, wo, 0), np.where(yo, 0, wo), starts)
    if w1 == 0 or w0 == 0:
        raise ValueError("ROC AUC needs both classes (with non-zero weight)")
    return float(u2 / (2 * w1 * w0))


# ------------------------------------------------------------------------------------------------ bootstrap


@dataclass(frozen=True)
class BootstrapSpec:
    """The `eval.bootstrap` block of the config."""

    n_replicates: int
    ci: float
    seed: int
    method: str
    unit: str


def bootstrap_spec(cfg: Mapping[str, Any]) -> BootstrapSpec:
    """Read `cfg["eval"]["bootstrap"]`. Only the percentile method is implemented."""
    b = cfg["eval"]["bootstrap"]
    if b["method"] != "percentile":
        raise ValueError(f"bootstrap method {b['method']!r} is not implemented (only 'percentile')")
    return BootstrapSpec(n_replicates=int(b["n"]), ci=float(b["ci"]), seed=int(b["seed"]),
                         method=str(b["method"]), unit=str(b["unit"]))


@dataclass(frozen=True, eq=False)
class ClusterDraws:
    """Cluster-bootstrap replicates, drawn once and shared by every scorer and statistic on the same items.

    `labels` are the sorted unique cluster labels (K), `item_cluster[i]` the index of item i's cluster,
    `draw_idx[b]` the K cluster indices drawn (with replacement) for replicate b, and `counts[b, k]` how often
    cluster k was drawn in replicate b. `key` fingerprints the draws so results can prove they share them.
    """

    labels: np.ndarray
    item_cluster: np.ndarray
    draw_idx: np.ndarray
    counts: np.ndarray
    seed: int
    key: str

    @property
    def n_replicates(self) -> int:
        return int(self.draw_idx.shape[0])

    @property
    def n_clusters(self) -> int:
        return int(self.labels.size)

    @property
    def n_items(self) -> int:
        return int(self.item_cluster.size)

    def weights(self, b: int) -> np.ndarray:
        """Item multiplicities in replicate b (int64, shape (n_items,))."""
        return self.counts[b, self.item_cluster]

    def replicate_items(self, b: int) -> np.ndarray:
        """Item indices of replicate b as an explicit resample: each drawn cluster's items, in draw order."""
        members = [np.flatnonzero(self.item_cluster == k) for k in range(self.n_clusters)]
        return np.concatenate([members[k] for k in self.draw_idx[b]])


def draw_clusters(clusters: ArrayLike, n_replicates: int, seed: int) -> ClusterDraws:
    """Draw `n_replicates` cluster-bootstrap resamples of the items labelled by `clusters`.

    Clusters are sorted with `np.unique`, so the draws do not depend on row order. With K clusters, the draws are
    `np.random.default_rng(seed).integers(0, K, size=(n_replicates, K))`: replicate b resamples K clusters with
    replacement.
    """
    lab = np.asarray(clusters)
    if lab.ndim != 1 or lab.size == 0:
        raise ValueError("clusters must be a non-empty 1-D array of labels")
    if pd.isna(lab).any():
        raise ValueError("clusters contains missing labels")
    if int(n_replicates) < 1:
        raise ValueError("n_replicates must be >= 1")
    labels, inv = np.unique(lab, return_inverse=True)
    inv = inv.reshape(-1).astype(np.int64)
    k, b = labels.size, int(n_replicates)
    draw_idx = np.random.default_rng(int(seed)).integers(0, k, size=(b, k))
    flat = (draw_idx + k * np.arange(b)[:, None]).ravel()
    counts = np.bincount(flat, minlength=b * k).reshape(b, k).astype(np.int64)
    h = hashlib.sha256()
    h.update(f"{int(seed)}|{b}|{k}|".encode())
    h.update("\x1f".join(map(str, labels.tolist())).encode("utf-8"))
    h.update(inv.tobytes())
    h.update(draw_idx.astype(np.int64).tobytes())
    return ClusterDraws(labels=labels, item_cluster=inv, draw_idx=draw_idx, counts=counts, seed=int(seed),
                        key=h.hexdigest()[:16])


def draw_clusters_from_cfg(clusters: ArrayLike, cfg: Mapping[str, Any]) -> ClusterDraws:
    """`draw_clusters` with n and seed from `cfg["eval"]["bootstrap"]`. The caller chooses the unit column."""
    spec = bootstrap_spec(cfg)
    return draw_clusters(clusters, spec.n_replicates, spec.seed)


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with its percentile CI. `replicates` has one value per replicate; NaN = degenerate."""

    estimate: float
    ci_low: float
    ci_high: float
    ci: float
    n_replicates: int
    n_used: int
    n_degenerate: int
    draws_key: str
    replicates: np.ndarray = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"estimate": self.estimate, "ci_low": self.ci_low, "ci_high": self.ci_high, "ci": self.ci,
                "n_replicates": self.n_replicates, "n_used": self.n_used, "n_degenerate": self.n_degenerate}


def percentile_ci(values: ArrayLike, ci: float) -> tuple[float, float]:
    """Percentile interval of the non-NaN `values`: quantiles (1 - ci)/2 and 1 - (1 - ci)/2, numpy "linear".

    The quantile levels are rounded to 12 decimals, so ci = 0.95 uses exactly 0.025 and 0.975 (not
    0.025000000000000022, the float result of (1 - 0.95)/2). NaN, NaN if no value is left.
    """
    if not 0.0 < float(ci) < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    a = round((1.0 - float(ci)) / 2.0, 12)
    lo, hi = np.quantile(v, [a, round(1.0 - a, 12)], method="linear")
    return float(lo), float(hi)


def _summarise(estimate: float, reps: np.ndarray, ci: float, draws_key: str) -> BootstrapResult:
    lo, hi = percentile_ci(reps, ci)
    n_deg = int(np.isnan(reps).sum())
    return BootstrapResult(estimate=float(estimate), ci_low=lo, ci_high=hi, ci=float(ci),
                           n_replicates=int(reps.size), n_used=int(reps.size - n_deg), n_degenerate=n_deg,
                           draws_key=draws_key, replicates=reps)


def replicate_aucs(y_true: ArrayLike, score: ArrayLike, draws: ClusterDraws) -> np.ndarray:
    """The AUC of every replicate in `draws` (shape (B,)); NaN where the resample holds one class only."""
    y = _labels(y_true)
    n = draws.n_items
    if y.size != n:
        raise ValueError(f"y_true has {y.size} items but the draws were made for {n}")
    s = _scores(score, n)
    order, starts = _tie_groups(s)
    ypos = y[order].astype(np.int64)
    ic = draws.item_cluster[order]
    out = np.full(draws.n_replicates, np.nan)
    step = max(1, _CHUNK_CELLS // max(1, n))
    for lo in range(0, draws.n_replicates, step):
        w = np.take(draws.counts[lo:lo + step], ic, axis=1)
        wp = w * ypos
        u2, w1, w0 = _u2(wp, w - wp, starts)
        ok = (w1 > 0) & (w0 > 0)
        chunk = out[lo:lo + step]
        chunk[ok] = u2[ok] / (2 * w1[ok] * w0[ok])
    return out


def bootstrap_auc(y_true: ArrayLike, score: ArrayLike, draws: ClusterDraws, *, ci: float) -> BootstrapResult:
    """Point AUC with its cluster-bootstrap percentile CI. Degenerate replicates are dropped and counted (D21)."""
    return _summarise(roc_auc(y_true, score), replicate_aucs(y_true, score, draws), ci, draws.key)


def bootstrap_aucs(y_true: ArrayLike, scores: Mapping[str, ArrayLike], draws: ClusterDraws, *,
                   ci: float) -> dict[str, BootstrapResult]:
    """`bootstrap_auc` for several scorers of the same items on the same draws (ready for `paired_differences`)."""
    return {name: bootstrap_auc(y_true, s, draws, ci=ci) for name, s in scores.items()}


def bootstrap_statistic(statistic: Callable[[np.ndarray, np.ndarray], float], draws: ClusterDraws, *,
                        ci: float) -> BootstrapResult:
    """Bootstrap any statistic on the same draws.

    `statistic(item_weights, cluster_counts)` gets the item multiplicities (n_items,) and the cluster counts
    (n_clusters,) of one replicate and returns a float, or NaN when the replicate is degenerate (dropped and
    counted). The point estimate is `statistic(ones, ones)` and must be finite.
    """
    est = float(statistic(np.ones(draws.n_items, dtype=np.int64), np.ones(draws.n_clusters, dtype=np.int64)))
    if not np.isfinite(est):
        raise ValueError("the statistic is not finite on the full sample")
    reps = np.array([float(statistic(draws.weights(b), draws.counts[b])) for b in range(draws.n_replicates)])
    return _summarise(est, reps, ci, draws.key)


def paired_differences(results: Mapping[str, BootstrapResult], reference: str) -> dict[str, BootstrapResult]:
    """`reference` minus each other scorer, replicate by replicate (e.g. CNN - baseline).

    Every result must come from the same draws (same `draws_key`) and CI level. The point estimate is the
    difference of the point estimates; the CI is the percentile CI of the per-replicate differences. Degenerate
    replicates are the same for every AUC scorer, so they drop out of the differences consistently.
    """
    if reference not in results:
        raise KeyError(f"reference {reference!r} is not among the results")
    ref = results[reference]
    out: dict[str, BootstrapResult] = {}
    for name, r in results.items():
        if name == reference:
            continue
        if r.draws_key != ref.draws_key:
            raise ValueError(f"{name!r} and {reference!r} were bootstrapped on different draws")
        if r.ci != ref.ci:
            raise ValueError(f"{name!r} and {reference!r} use different CI levels")
        if not np.array_equal(np.isnan(r.replicates), np.isnan(ref.replicates)):
            raise ValueError(f"{name!r} and {reference!r} have different degenerate replicates")
        out[name] = _summarise(ref.estimate - r.estimate, ref.replicates - r.replicates, ref.ci, ref.draws_key)
    return out


# ------------------------------------------------------------------------------------------------ seed ensemble


def seed_ensemble(preds: Mapping[Any, pd.DataFrame] | Sequence[pd.DataFrame], *, id_col: str = "id",
                  prob_col: str = "prob") -> pd.DataFrame:
    """Mean of the seed probabilities per patch id (the run prediction, Appendix A §7).

    `preds` maps seed -> DataFrame (or is a sequence of DataFrames) with columns `id_col`, `prob_col`, as in
    `preds_test.csv`. Every seed must hold the same set of unique ids with finite probabilities in [0, 1];
    anything else raises. The output keeps the first seed's row order.
    """
    items = list(preds.items()) if isinstance(preds, Mapping) else list(enumerate(preds))
    if not items:
        raise ValueError("no seed predictions given")
    ref_ids = pd.Index(items[0][1][id_col].to_numpy())
    stacked = []
    for seed, df in items:
        ids = pd.Index(df[id_col].to_numpy())
        if not ids.is_unique:
            raise ValueError(f"seed {seed!r}: duplicate {id_col} values")
        if len(ids) != len(ref_ids) or not ids.isin(ref_ids).all():
            raise ValueError(f"seed {seed!r}: its {id_col} set differs from seed {items[0][0]!r}")
        p = pd.Series(df[prob_col].to_numpy(dtype=np.float64), index=ids).reindex(ref_ids).to_numpy()
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f"seed {seed!r}: {prob_col} must be finite and in [0, 1]")
        stacked.append(p)
    return pd.DataFrame({id_col: ref_ids.to_numpy(), prob_col: np.mean(np.stack(stacked), axis=0)})


# ------------------------------------------------------------------------------------------------ pair accuracy


@dataclass(frozen=True)
class PairAccuracy:
    """Share of matched pairs whose positive scores above its negative (ties count 1/2)."""

    accuracy: float
    n_pairs: int
    n_correct: int
    n_ties: int
    n_wrong: int
    n_cross_sheet: int
    n_incomplete: int
    pair_ids: np.ndarray = field(repr=False, compare=False)
    credit: np.ndarray = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"accuracy": self.accuracy, "n_pairs": self.n_pairs, "n_correct": self.n_correct,
                "n_ties": self.n_ties, "n_wrong": self.n_wrong, "n_cross_sheet": self.n_cross_sheet,
                "n_incomplete": self.n_incomplete}


def _pair_rows(pid: np.ndarray, rows: np.ndarray, common: np.ndarray) -> np.ndarray:
    o = np.argsort(pid[rows], kind="stable")
    return rows[o][np.searchsorted(pid[rows][o], common)]


def pair_accuracy(y_true: ArrayLike, score: ArrayLike, pair_id: ArrayLike,
                  sheet: ArrayLike | None = None) -> PairAccuracy:
    """Matched-pair accuracy over the items given.

    - Items with `pair_id < 0` (background, -1 in `meta.csv`) are ignored.
    - A pair is complete when exactly one positive and one negative carry its id. A pair id with only one member
      among the items is dropped and counted (`n_incomplete`); two positives, two negatives or more than two
      members raise ValueError.
    - With `sheet` given, a pair counts only if both members lie on the same sheet; others are dropped and counted
      (`n_cross_sheet`). This gives the same-sheet pair accuracy.
    - Credit per pair: 1 if score(positive) > score(negative), 1/2 if equal, 0 otherwise. Accuracy is the mean
      credit, NaN when no pair counts.
    """
    y = _labels(y_true)
    s = _scores(score, y.size)
    raw = np.asarray(pair_id)
    if raw.shape != y.shape:
        raise ValueError("pair_id must have the same shape as y_true")
    pid = raw.astype(np.int64)
    if not np.array_equal(pid, raw):
        raise ValueError("pair_id must hold integers")
    keep = pid >= 0
    pos_rows = np.flatnonzero(keep & y)
    neg_rows = np.flatnonzero(keep & ~y)
    for rows, what in ((pos_rows, "positives"), (neg_rows, "negatives")):
        if np.unique(pid[rows]).size != rows.size:
            raise ValueError(f"a pair_id has two or more {what}")
    common = np.intersect1d(pid[pos_rows], pid[neg_rows])
    n_incomplete = int(np.union1d(pid[pos_rows], pid[neg_rows]).size - common.size)
    rp = _pair_rows(pid, pos_rows, common)
    rn = _pair_rows(pid, neg_rows, common)
    n_cross = 0
    if sheet is not None:
        sh = np.asarray(sheet)
        if sh.shape != y.shape:
            raise ValueError("sheet must have the same shape as y_true")
        same = sh[rp] == sh[rn]
        n_cross = int((~same).sum())
        common, rp, rn = common[same], rp[same], rn[same]
    sp, sn = s[rp], s[rn]
    credit = np.where(sp > sn, 1.0, np.where(sp == sn, 0.5, 0.0))
    n = int(credit.size)
    return PairAccuracy(accuracy=float(credit.mean()) if n else float("nan"), n_pairs=n,
                        n_correct=int((credit == 1.0).sum()), n_ties=int((credit == 0.5).sum()),
                        n_wrong=int((credit == 0.0).sum()), n_cross_sheet=n_cross, n_incomplete=n_incomplete,
                        pair_ids=common, credit=credit)


# ------------------------------------------------------------------------------------------------ stage-2 PR sweep


@dataclass(frozen=True)
class PRSweep:
    """Object-level precision-recall sweep of stage-2 detections.

    `status` is REACHED or NOT_REACHED. When the recall level is never reached, `precision_at_recall`,
    `threshold_at_recall` and `interpolated_precision` are None and `passes()` is False whatever the threshold.
    The curve arrays have one entry per distinct score (operating point), highest score first; `precision` is NaN
    where no TP or FP has entered yet.
    """

    status: str
    recall_level: float
    n_targets: float
    precision_at_recall: float | None
    threshold_at_recall: float | None
    interpolated_precision: float | None
    max_recall: float
    n_tp: float
    n_fp: float
    n_ignored: float
    n_duplicate: float
    thresholds: np.ndarray = field(repr=False, compare=False)
    tp: np.ndarray = field(repr=False, compare=False)
    fp: np.ndarray = field(repr=False, compare=False)
    precision: np.ndarray = field(repr=False, compare=False)
    recall: np.ndarray = field(repr=False, compare=False)

    @property
    def reached(self) -> bool:
        return self.status == REACHED

    def passes(self, min_precision: float) -> bool:
        """True only if the recall level is reached and P@R >= `min_precision`. Not reached is a fail."""
        return self.reached and self.precision_at_recall is not None and self.precision_at_recall >= min_precision

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "recall_level": self.recall_level, "n_targets": self.n_targets,
                "precision_at_recall": self.precision_at_recall, "threshold_at_recall": self.threshold_at_recall,
                "interpolated_precision": self.interpolated_precision, "max_recall": self.max_recall,
                "n_tp": self.n_tp, "n_fp": self.n_fp, "n_ignored": self.n_ignored, "n_duplicate": self.n_duplicate,
                "n_thresholds": int(self.thresholds.size)}


def _num(x: Any) -> float | int:
    x = x.item() if isinstance(x, np.generic) else x
    return int(x) if isinstance(x, (int, np.integer)) else float(x)


def pr_sweep(scores: ArrayLike, outcomes: ArrayLike, n_targets: float, recall_level: float,
             weights: ArrayLike | None = None) -> PRSweep:
    """Precision-recall sweep over detections with known outcomes (geometry-free; P1-12 does the matching).

    - `outcomes[i]` is one of "tp", "fp", "ignored", "duplicate" for detection i; `scores[i]` its score. Input
      order does not matter.
    - Operating points are the distinct scores, highest first. The point at score t holds every detection with
      score >= t, so tied detections enter together.
    - TP and FP accumulate; ignored and duplicate detections occupy thresholds but count in neither, so a threshold
      holding only them repeats the previous point's precision and recall.
    - precision = TP / (TP + FP) (NaN while both are 0); recall = TP / n_targets.
    - P@R (primary, P@R50 for recall_level 0.5) = precision at the highest threshold with TP >= recall_level *
      n_targets. Interpolated precision (secondary) = the maximum precision over all points with recall >= the
      level. Max recall = total TP / n_targets.
    - If no point reaches the level, the result's status is NOT_REACHED (a fail).
    - `weights` (optional) are detection multiplicities, e.g. a sheet bootstrap; pass the matching weighted
      `n_targets`.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = s.size
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or inf")
    o = np.asarray(outcomes).astype(str).reshape(-1)
    if o.size != n:
        raise ValueError("scores and outcomes differ in length")
    bad = ~np.isin(o, STAGE2_OUTCOMES)
    if bad.any():
        raise ValueError(f"unknown outcomes: {sorted(set(o[bad].tolist()))}; expected {STAGE2_OUTCOMES}")
    if not (np.isfinite(n_targets) and n_targets > 0):
        raise ValueError("n_targets must be > 0")
    if not 0.0 < float(recall_level) <= 1.0:
        raise ValueError("recall_level must be in (0, 1]")
    w = _weights(weights, n)

    order = np.argsort(-s, kind="stable")
    ss, oo, ww = s[order], o[order], w[order]
    ends = np.flatnonzero(np.concatenate((ss[1:] != ss[:-1], [True]))) if n else np.zeros(0, dtype=np.int64)
    tp = np.cumsum(np.where(oo == "tp", ww, 0))[ends]
    fp = np.cumsum(np.where(oo == "fp", ww, 0))[ends]
    n_tp = w[o == "tp"].sum()
    if n_tp > n_targets:
        raise ValueError(f"{n_tp} true positives but only {n_targets} targets")
    thresholds = ss[ends]
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    recall = tp / n_targets
    hit = tp >= float(recall_level) * n_targets
    reached = bool(hit.any())
    if reached:
        i = int(np.argmax(hit))
        p_at, t_at, p_int = float(precision[i]), float(thresholds[i]), float(np.max(precision[hit]))
    else:
        p_at = t_at = p_int = None
    return PRSweep(status=REACHED if reached else NOT_REACHED, recall_level=float(recall_level),
                   n_targets=_num(n_targets), precision_at_recall=p_at, threshold_at_recall=t_at,
                   interpolated_precision=p_int, max_recall=float(n_tp / n_targets), n_tp=_num(n_tp),
                   n_fp=_num(w[o == "fp"].sum()), n_ignored=_num(w[o == "ignored"].sum()),
                   n_duplicate=_num(w[o == "duplicate"].sum()), thresholds=thresholds, tp=tp, fp=fp,
                   precision=precision, recall=recall)
