"""Sheet split attributes and the split lists (task P1-08; Appendix A §4, decisions D08-D10).

- **Sheet attributes.** Transfer group (`group_b`: "train" if the sheet's kraj is in `splits.transfer_train_kraje`,
  else "test"), stratum (flysch flag x transfer group: F_ZLMS, F_other, N_ZLMS, N_other) and the 15 km block (D08).
- **Stage-2 sheets.** The 20 sheets of Appendix A §4, checked against the stage-2 criteria, never redrawn.
- **Stage-1 sheets.** Every other sheet with at least one eligible polygon.
- **(a) folds** (D09), **(b) groups**, and the **inner-validation blocks** of every run (D10).

Random numbers follow D02: where the method names a seed ("shuffled (seed 11)", "seed 1000 + fold"), the generator
is `numpy.random.default_rng(seed)`. Every shuffle starts from the blocks sorted by id, so the result depends only on
the block ids, their counts and the seed.

The Appendix A §4 values that Appendix B has no key for come from `replication.splits`: the 20 stage-2 sheet numbers
(`stage2_sheets`), the fold seed (`fold_seed`, 11) and the inner-validation seeds (`inner_val_seeds`: a_base 1000,
b 2000, s2 3000).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd

STRATA = ("F_ZLMS", "F_other", "N_ZLMS", "N_other")
GROUP_TRAIN, GROUP_TEST = "train", "test"


# ======================================================================================================================
# Config values (replication.splits)
# ======================================================================================================================


def stage2_sheets(cfg: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Appendix A §4's 20 reserved sheets: `replication.splits.stage2_sheets` ({flysch, nonflysch})."""
    s2 = cfg["replication"]["splits"]["stage2_sheets"]
    return {"flysch": tuple(str(v) for v in s2["flysch"]), "nonflysch": tuple(str(v) for v in s2["nonflysch"])}


def fold_seed(cfg: Mapping[str, Any]) -> int:
    """Appendix A §4 "Flysch blocks are shuffled (seed 11)": `replication.splits.fold_seed`."""
    return int(cfg["replication"]["splits"]["fold_seed"])


def kraj_tie_rule(cfg: Mapping[str, Any]) -> str:
    """D06 tie rule: `replication.splits.kraj_tie_rule` (checked by `landslide.geo`)."""
    return str(cfg["replication"]["splits"]["kraj_tie_rule"])


# ======================================================================================================================
# Sheet attributes
# ======================================================================================================================


def _missing(v: Any) -> bool:
    """None, NaN or pandas NA."""
    return v is None or bool(pd.isna(v))


def transfer_group(kraj: str | None, train_kraje: Iterable[str]) -> str | None:
    """Appendix A §4: "train" if the kraj is one of `train_kraje`, otherwise "test"; None without a kraj."""
    if _missing(kraj):
        return None
    return GROUP_TRAIN if kraj in set(train_kraje) else GROUP_TEST


def stratum(flysch: bool, group: str | None) -> str | None:
    """Flysch flag x transfer group: F_ZLMS, F_other, N_ZLMS, N_other (None without a group)."""
    if _missing(group):
        return None
    if group not in (GROUP_TRAIN, GROUP_TEST):
        raise ValueError(f"unknown transfer group {group!r}")
    return ("F_" if bool(flysch) else "N_") + ("ZLMS" if group == GROUP_TRAIN else "other")


def block_cell(x: float, y: float, block_m: float) -> tuple[int, int]:
    """D08: the grid cell (floor(x / block_m), floor(y / block_m)) of a point, from the EPSG:5514 origin."""
    return int(math.floor(x / block_m)), int(math.floor(y / block_m))


def block_ids(sheet: Sequence[str], strat: Sequence[str | None], cell_x: Sequence[int],
              cell_y: Sequence[int]) -> list[str | None]:
    """Opaque block ids. A block is (stratum, cell); its id is `<stratum>-<k>`, k = 1, 2, ... numbering the stratum's
    blocks by their smallest member sheet number. The ids carry no coordinates. None where the stratum is None."""
    df = pd.DataFrame({"sheet": list(sheet), "stratum": list(strat), "cx": list(cell_x), "cy": list(cell_y)})
    out: list[str | None] = [None] * len(df)
    ok = df["stratum"].map(lambda v: not _missing(v))
    if not ok.any():
        return out
    d = df[ok]
    first = d.groupby(["stratum", "cx", "cy"], sort=True)["sheet"].min().reset_index()
    first = first.sort_values(["stratum", "sheet"], kind="stable")
    first["k"] = first.groupby("stratum").cumcount() + 1
    width = max(3, len(str(int(first["k"].max()))))
    key = {(r.stratum, r.cx, r.cy): f"{r.stratum}-{r.k:0{width}d}" for r in first.itertuples()}
    for idx, r in zip(d.index, d.itertuples()):
        out[idx] = key[(r.stratum, r.cx, r.cy)]
    return out


# ======================================================================================================================
# Folds and inner validation
# ======================================================================================================================


def shuffled(items: Iterable[str], seed: int) -> list[str]:
    """The items sorted, then permuted with `default_rng(seed).permutation`."""
    s = sorted(items)
    rng = np.random.default_rng(seed)
    return [s[k] for k in rng.permutation(len(s))]


def assign_folds(block_n: Mapping[str, int], n_folds: int, seed: int) -> dict[str, int]:
    """D09: shuffle the blocks (seed), stable-sort by eligible-polygon count descending, then give each block to the
    fold with the fewest polygons so far (ties to the lowest fold index)."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    order = sorted(shuffled(block_n, seed), key=lambda b: -int(block_n[b]))  # Python's sort is stable
    totals = [0] * n_folds
    out: dict[str, int] = {}
    for b in order:
        f = min(range(n_folds), key=lambda k: (totals[k], k))
        out[b] = f
        totals[f] += int(block_n[b])
    return out


def inner_val_blocks(block_n: Mapping[str, int], frac: float, seed: int) -> list[str]:
    """D10: random training blocks (shuffled with `seed`) are added until they hold >= `frac` of the run's training
    eligible polygons. Returns the chosen blocks in the order they were added.

    The comparison is exact (`frac` as a decimal fraction), so 15 of 100 polygons counts as reaching 0.15.
    """
    total = sum(int(v) for v in block_n.values())
    f = Fraction(str(frac))
    chosen: list[str] = []
    acc = 0
    for b in shuffled(block_n, seed):
        if acc * f.denominator >= f.numerator * total:
            break
        chosen.append(b)
        acc += int(block_n[b])
    return chosen


def inner_val_seed(run: str, cfg: Mapping[str, Any]) -> int:
    """Appendix A §4 (`replication.splits.inner_val_seeds`): a_base + fold for a_f<k>, b for b, s2 for s2."""
    seeds = cfg["replication"]["splits"]["inner_val_seeds"]
    if run.startswith("a_f"):
        return int(seeds["a_base"]) + int(run[3:])
    if run == "b":
        return int(seeds["b"])
    if run == "s2":
        return int(seeds["s2"])
    raise ValueError(f"unknown run {run!r}")


def training_blocks(run: str, block_table: pd.DataFrame) -> list[str]:
    """The blocks a run trains on (train + inner-val), from a table with columns block, stratum, fold_a, group_b.

    - a_f<k>: flysch blocks (F_ZLMS, F_other) not in fold k (Appendix A §4: "Training uses flysch only");
    - b: blocks of the ZLMS strata (group "train");
    - s2: every stage-1 block (the stage-2 model trains on all stage-1 patches).
    """
    t = block_table
    if run.startswith("a_f"):
        k = int(run[3:])
        sel = t["stratum"].isin(["F_ZLMS", "F_other"]) & (t["fold_a"] != k)
    elif run == "b":
        sel = t["group_b"] == GROUP_TRAIN
    elif run == "s2":
        sel = pd.Series(True, index=t.index)
    else:
        raise ValueError(f"unknown run {run!r}")
    return sorted(t.loc[sel, "block"])


# ======================================================================================================================
# Stage-2 sheets
# ======================================================================================================================


def check_stage2(sheets: pd.DataFrame, expected: Mapping[str, Sequence[str]],
                 cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Check each reserved sheet against Appendix A §4 / config `splits.stage2`. One record per sheet; never swaps.

    `sheets` needs columns sheet, full, n_eligible, dtm_all_valid, flysch. A sheet passes if it exists, is full
    (when `full_sheets_only`), has >= `min_eligible_polygons` eligible polygons, a fully valid DTM and the flysch
    flag of its list.
    """
    s2 = cfg["splits"]["stage2"]
    by = sheets.set_index("sheet")
    out = []
    for kind in ("flysch", "nonflysch"):
        for s in expected[kind]:
            rec: dict[str, Any] = {"sheet": s, "list": kind, "exists": s in by.index}
            if rec["exists"]:
                r = by.loc[s]
                rec.update({"full": bool(r["full"]), "n_eligible": int(r["n_eligible"]),
                            "dtm_all_valid": bool(r["dtm_all_valid"]), "flysch": bool(r["flysch"])})
                checks = {"full": rec["full"] or not bool(s2["full_sheets_only"]),
                          "min_eligible": rec["n_eligible"] >= int(s2["min_eligible_polygons"]),
                          "dtm_all_valid": rec["dtm_all_valid"],
                          "flysch_flag": rec["flysch"] == (kind == "flysch")}
            else:
                checks = {"full": False, "min_eligible": False, "dtm_all_valid": False, "flysch_flag": False}
            rec["checks"] = checks
            rec["pass"] = all(checks.values())
            rec["failed"] = sorted(k for k, v in checks.items() if not v)
            out.append(rec)
    return out


def stage2_list_sizes_ok(expected: Mapping[str, Sequence[str]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The list sizes against `splits.stage2.n_flysch` / `n_nonflysch`, and that the 20 sheets are distinct."""
    s2 = cfg["splits"]["stage2"]
    allv = list(expected["flysch"]) + list(expected["nonflysch"])
    return {"n_flysch": len(expected["flysch"]), "n_nonflysch": len(expected["nonflysch"]),
            "sizes_match_config": (len(expected["flysch"]) == int(s2["n_flysch"])
                                   and len(expected["nonflysch"]) == int(s2["n_nonflysch"])),
            "distinct": len(set(allv)) == len(allv)}
