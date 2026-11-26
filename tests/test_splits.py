"""Synthetic tests for `landslide.splits`: transfer group and stratum, block cells and opaque block ids (D08), the (a)
fold assignment (D09), the inner-validation rule (D10), training blocks per run and the stage-2 checks. One
`needs_data` test checks the built split files for internal consistency."""

import json

import numpy as np
import pandas as pd
import pytest

from landslide import splits as SP
from landslide.config import load_config
from landslide.paths import RUN_NAMES, Paths

CFG = {"splits": {"stage2": {"n_flysch": 2, "n_nonflysch": 2, "min_eligible_polygons": 5,
                             "full_sheets_only": True}}}
TRAIN = ["Zlínský", "Moravskoslezský"]


# ---- group, stratum ----

def test_transfer_group_and_stratum():
    assert SP.transfer_group("Zlínský", TRAIN) == "train"
    assert SP.transfer_group("Moravskoslezský", TRAIN) == "train"
    assert SP.transfer_group("Olomoucký", TRAIN) == "test"
    assert SP.transfer_group(None, TRAIN) is None and SP.transfer_group(float("nan"), TRAIN) is None
    assert SP.stratum(True, "train") == "F_ZLMS" and SP.stratum(True, "test") == "F_other"
    assert SP.stratum(False, "train") == "N_ZLMS" and SP.stratum(False, "test") == "N_other"
    assert SP.stratum(True, None) is None and SP.stratum(False, float("nan")) is None
    with pytest.raises(ValueError):
        SP.stratum(True, "val")


# ---- blocks (D08) ----

@pytest.mark.parametrize("x, y, want", [(0.0, 0.0, (0, 0)), (14999.9, 15000.0, (0, 1)), (-0.1, -15000.0, (-1, -1)),
                                        (-15000.1, -29999.9, (-2, -2)), (-600123.0, -1100456.0, (-41, -74))])
def test_block_cell_floor_from_origin(x, y, want):
    assert SP.block_cell(x, y, 15000.0) == want


def test_block_ids_are_opaque_and_nested_in_stratum():
    sheet = ["25-13-02", "25-13-01", "25-13-03", "14-11-01", "14-11-02", "02-01-01"]
    strat = ["F_ZLMS", "F_ZLMS", "F_other", "F_ZLMS", "N_other", None]
    cx = [-40, -40, -40, -50, -50, 0]
    cy = [-75, -75, -75, -70, -70, 0]
    ids = SP.block_ids(sheet, strat, cx, cy)
    # Same stratum and cell -> same block; same cell, other stratum -> other block; numbered by smallest sheet.
    assert ids[0] == ids[1]
    assert ids[2] != ids[0] and ids[2].startswith("F_other-")
    assert ids[3] == "F_ZLMS-001" and ids[0] == "F_ZLMS-002"   # "14-11-01" < "25-13-01"
    assert ids[4] == "N_other-001" and ids[5] is None
    assert not any(str(c) in "".join(i for i in ids if i) for c in (40, 50, 70, 75))  # no coordinates


# ---- (a) folds (D09) ----

def test_assign_folds_hand_example():
    # Sorted by count: a 10, b 9, c 8, then d and e (1 each, in shuffled order).
    # a -> 0 (tie 0/0 -> lowest), b -> 1, c -> 1 (9 < 10), d -> 0 (10 < 17), e -> 0 (11 < 17).
    n = {"a": 10, "b": 9, "c": 8, "d": 1, "e": 1}
    for seed in range(5):
        assert SP.assign_folds(n, 2, seed) == {"a": 0, "b": 1, "c": 1, "d": 0, "e": 0}


def test_assign_folds_equal_counts_follow_the_seeded_shuffle():
    n = {f"blk{i}": 7 for i in range(5)}
    for seed in (11, 12, 13):
        order = SP.shuffled(n, seed)
        got = SP.assign_folds(n, 5, seed)
        assert [got[b] for b in order] == [0, 1, 2, 3, 4]  # stable sort keeps the shuffle; ties -> lowest fold


def test_shuffle_is_seeded_and_input_order_free():
    items = [f"b{i:02d}" for i in range(20)]
    assert SP.shuffled(items, 11) == SP.shuffled(list(reversed(items)), 11)
    assert SP.shuffled(items, 11) == [items[k] for k in np.random.default_rng(11).permutation(20)]
    assert SP.shuffled(items, 11) != SP.shuffled(items, 12)


@pytest.mark.parametrize("seed", range(6))
def test_assign_folds_greedy_balance(seed):
    rng = np.random.default_rng(seed)
    n = {f"F-{i:03d}": int(v) for i, v in enumerate(rng.integers(1, 400, 52))}
    folds = SP.assign_folds(n, 5, 11)
    assert set(folds) == set(n) and set(folds.values()) == set(range(5))
    tot = [sum(v for b, v in n.items() if folds[b] == k) for k in range(5)]
    assert max(tot) - min(tot) <= max(n.values())  # greedy (LPT) bound
    assert SP.assign_folds(n, 5, 11) == folds  # deterministic


# ---- inner validation (D10) ----

def test_inner_val_adds_shuffled_blocks_until_fraction_reached():
    n = {f"b{i}": v for i, v in enumerate([40, 30, 20, 5, 3, 2])}  # total 100, target 15
    for seed in range(10):
        order = SP.shuffled(n, seed)
        got = SP.inner_val_blocks(n, 0.15, seed)
        assert got == order[:len(got)]                       # a prefix of the seeded shuffle
        assert sum(n[b] for b in got) >= 15                  # reaches the target
        assert sum(n[b] for b in got[:-1]) < 15              # and stops as soon as it does


def test_inner_val_exact_boundary():
    # 20 blocks of 5 polygons: 3 blocks hold exactly 15 % of 100. In floats 0.15 * 100 = 15.000000000000002, which
    # would wrongly ask for a fourth block.
    n = {f"b{i:02d}": 5 for i in range(20)}
    assert len(SP.inner_val_blocks(n, 0.15, 1000)) == 3


def test_inner_val_seeds():
    assert [SP.inner_val_seed(r) for r in RUN_NAMES] == [1000, 1001, 1002, 1003, 1004, 2000, 3000]
    with pytest.raises(ValueError):
        SP.inner_val_seed("c")


def test_training_blocks_per_run():
    bt = pd.DataFrame({"block": ["F_ZLMS-001", "F_ZLMS-002", "F_other-001", "N_ZLMS-001", "N_other-001"],
                       "stratum": ["F_ZLMS", "F_ZLMS", "F_other", "N_ZLMS", "N_other"],
                       "fold_a": pd.array([0, 1, 2, pd.NA, pd.NA], dtype="Int64"),
                       "group_b": ["train", "train", "test", "train", "test"]})
    assert SP.training_blocks("a_f1", bt) == ["F_ZLMS-001", "F_other-001"]
    assert SP.training_blocks("a_f3", bt) == ["F_ZLMS-001", "F_ZLMS-002", "F_other-001"]
    assert SP.training_blocks("b", bt) == ["F_ZLMS-001", "F_ZLMS-002", "N_ZLMS-001"]
    assert SP.training_blocks("s2", bt) == sorted(bt["block"])


# ---- stage 2 ----

def test_stage2_constants_match_config():
    cfg = load_config()
    s = SP.stage2_list_sizes_ok(SP.STAGE2_SHEETS, cfg)
    assert s == {"n_flysch": 10, "n_nonflysch": 10, "sizes_match_config": True, "distinct": True}


def test_check_stage2_reports_every_failure_and_never_swaps():
    sheets = pd.DataFrame({
        "sheet": ["f-ok", "f-partial", "n-ok", "n-few", "n-border", "n-flysch", "spare"],
        "full": [True, False, True, True, True, True, True],
        "n_eligible": [5, 9, 7, 4, 8, 8, 50],
        "dtm_all_valid": [True, True, True, True, False, True, True],
        "flysch": [True, True, False, False, False, True, False]})
    expected = {"flysch": ["f-ok", "f-partial", "f-missing"], "nonflysch": ["n-ok", "n-few", "n-border", "n-flysch"]}
    out = SP.check_stage2(sheets, expected, CFG)
    assert [r["sheet"] for r in out] == expected["flysch"] + expected["nonflysch"]  # the list, never a substitute
    fails = {r["sheet"]: r["failed"] for r in out}
    assert fails == {"f-ok": [], "f-partial": ["full"], "f-missing": ["dtm_all_valid", "flysch_flag", "full",
                                                                       "min_eligible"],
                     "n-ok": [], "n-few": ["min_eligible"], "n-border": ["dtm_all_valid"], "n-flysch": ["flysch_flag"]}
    assert [r["pass"] for r in out] == [True, False, False, True, False, False, False]
    assert not out[2]["exists"]


# ---- built outputs (skipped without data/derived) ----

@pytest.mark.needs_data
@pytest.mark.skipif(not Paths().splits_json.exists() or not Paths().sheets_parquet.exists(),
                    reason="data/derived is not built")
def test_built_splits_are_consistent():
    import geopandas as gpd

    p = Paths()
    sp = json.loads(p.splits_json.read_text(encoding="utf-8"))
    sh = gpd.read_parquet(p.sheets_parquet)
    po = gpd.read_parquet(p.polygons_parquet)
    assert len(sh) == 860 and sh["sheet"].is_unique
    s2 = set(sp["stage2"]["flysch"]) | set(sp["stage2"]["nonflysch"])
    assert s2 == set(sh.loc[sh["is_stage2"], "sheet"]) and len(s2) == 20
    s1 = sh[sh["is_stage1"]]
    assert set(sp["stage1_sheets"]) == set(s1["sheet"]) and not (set(s1["sheet"]) & s2)
    assert (s1["n_eligible"] >= 1).all() and s1["block"].notna().all() and s1["stratum"].notna().all()
    # Eligible polygons all go to a stage-1 or a stage-2 sheet; counts agree with the sheets.
    el = po[po["eligible"]]
    assert el["sheet"].notna().all() and el["sheet"].isin(set(s1["sheet"]) | s2).all()
    assert int(sh["n_eligible"].sum()) == len(el)
    # Folds partition the flysch stage-1 blocks; groups partition all stage-1 blocks.
    blocks = sp["blocks"]
    fly = {b for b, v in blocks.items() if v["stratum"].startswith("F_")}
    fold_blocks = [b for f in sp["folds_a"].values() for b in f["blocks"]]
    assert len(fold_blocks) == len(set(fold_blocks)) and set(fold_blocks) == fly
    assert (s1["fold_a"].notna() == s1["stratum"].str.startswith("F_")).all()
    gb = sp["group_b"]
    assert not (set(gb["train_blocks"]) & set(gb["test_blocks"]))
    assert set(gb["train_blocks"]) | set(gb["test_blocks"]) == set(blocks) == set(s1["block"])
    # Inner-val blocks come from each run's training blocks only.
    bt = pd.DataFrame([{"block": b, **{k: v[k] for k in ("stratum", "group_b")},
                        "fold_a": v["fold_a"] if v["fold_a"] is not None else pd.NA} for b, v in blocks.items()])
    bt["fold_a"] = bt["fold_a"].astype("Int64")
    for run, iv in sp["inner_val"].items():
        assert set(iv) <= set(SP.training_blocks(run, bt)), run
        for k in range(5):
            if run == f"a_f{k}":
                assert not (set(iv) & set(sp["folds_a"][str(k)]["blocks"]))
