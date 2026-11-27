"""P1-10 on a synthetic world (`patches_synth.py`): extraction, alignment, float16 checks, the meta table, the summary,
the figure, the dataset check and determinism. Nothing here reads or writes the real `data/`."""

from __future__ import annotations

import importlib.util
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import rasterio

from landslide import patches as P
from landslide.channels import patch_channels
from landslide.dtm import extract_block
from landslide.features import STAT_COLUMNS, patch_stats
from landslide.paths import Paths
from patches_synth import write_world


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Paths().root / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return write_world(tmp_path_factory.mktemp("world"))


@pytest.fixture(autouse=True)
def _data_dir(world, monkeypatch):
    monkeypatch.setenv("LANDSLIDE_DATA_DIR", str(world.data))
    monkeypatch.delenv("LANDSLIDE_RAW_DIR", raising=False)


@pytest.fixture(scope="module")
def built(world, tmp_path_factory):
    """The dataset built by `scripts/build_patches.py --out OUT` (env set by hand: module scope)."""
    out = tmp_path_factory.mktemp("out")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LANDSLIDE_DATA_DIR", str(world.data))
        mp.delenv("LANDSLIDE_RAW_DIR", raising=False)
        assert _script("build_patches").main(["--out", str(out), "--workers", "3"]) == 0
    return out


def _extract(world, workers=1, open_window=None, samples=None):
    cfg = world.cfg
    samples = world.samples if samples is None else samples
    src = world.sources()
    out = np.zeros((len(samples), 7, 150, 150), dtype=np.float16)
    ow = open_window or (lambda s, r, c: P.read_sheet_window(src[s], r, c, cfg))
    ext = P.extract_dataset(cfg, samples, ow, out, workers=workers)
    return out, ext


def _verify(world, ds, **kw):
    kw.setdefault("n_check", 0)
    return P.verify_dataset(world.cfg, ds / "patches.npy", ds / "meta.csv", world.sheets, world.sources(),
                            samples=kw.pop("samples", world.samples), **kw)


def _failed(checks):
    return {c.name for c in P.failures(checks)}


# ---- extraction ----

def test_patches_equal_the_channels_of_the_sampled_dtm(world, built):
    """patches[i] = float16(patch_channels) at sample i's corner of the DTM P1-09 sampled from (in memory), and the
    statistics equal patch_stats on the same arrays: the file reading, windows and offsets change nothing."""
    cfg, S = world.cfg, world.samples
    pt = np.load(built / "dataset" / "patches.npy", mmap_mode="r")
    meta = P.read_meta(built / "dataset" / "meta.csv")
    assert pt.shape == (len(S), 7, 150, 150) and pt.dtype == np.float16
    for i, s, r, c in zip(S.index, S["sheet"], S["row"], S["col"]):
        z, su = world.z[s], world.susc[s]
        ch = patch_channels(z, r, c, cfg)
        assert np.isfinite(ch).all()
        assert np.array_equal(pt[i].view(np.uint16), ch.astype(np.float16).view(np.uint16)), i
        st = patch_stats(ch, cfg["channels"], extract_block(z, r - 76, c - 76, r + 76, c + 76),
                         su[r - 75:r + 75, c - 75:c + 75])
        for k in STAT_COLUMNS:
            want = S.at[i, "slope_mean"] if k == "slope_mean" else st[k]
            assert meta.at[i, k] == want, (i, k)
        assert abs(st["slope_mean"] - S.at[i, "slope_mean"]) < 1e-9


def test_bounding_window_equals_per_patch_windows(world):
    """The sheet's bounding window (one read per raster) and a per-patch read_dtm window (patch + 28 px context) give
    bit-identical float64 channels and statistics."""
    cfg, S = world.cfg, world.samples
    src = world.sources()
    for s, g in S.groupby("sheet"):
        rows, cols = g["row"].to_numpy(), g["col"].to_numpy()
        big = P.read_sheet_window(src[s], rows, cols, cfg)
        for r, c in zip(rows[:4], cols[:4]):
            one = P.read_sheet_window(src[s], np.array([r]), np.array([c]), cfg)
            assert one.z.shape == (206, 206)
            a, sa = P.extract_patch(big.z, big.susc, r - big.row_off, c - big.col_off, cfg)
            b, sb = P.extract_patch(one.z, one.susc, r - one.row_off, c - one.col_off, cfg)
            assert np.array_equal(a, b) and sa == sb


def test_each_raster_is_opened_once_per_sheet(world, monkeypatch):
    opened = []
    real = rasterio.open

    def counting(path, *a, **k):
        opened.append(str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(rasterio, "open", counting)
    _extract(world)
    n_sheets = world.samples["sheet"].nunique()
    assert len(opened) == 2 * n_sheets and len(set(opened)) == 2 * n_sheets


def test_workers_do_not_change_the_result(world):
    a, ea = _extract(world, workers=1)
    b, eb = _extract(world, workers=4)
    assert np.array_equal(a.view(np.uint16), b.view(np.uint16))
    pd.testing.assert_frame_equal(ea.stats, eb.stats, check_exact=True)
    assert ea.info == eb.info


def test_extraction_info(world):
    _, ext = _extract(world)
    info = ext.info
    assert info["n_patches"] == len(world.samples)
    assert info["max_slope_mean_diff_deg"] < 1e-9
    assert 0 <= info["channel_min"]["slope"] and info["channel_max"]["slope"] <= 90
    assert 0 <= info["channel_min"]["svf"] and info["channel_max"]["svf"] <= 1
    assert info["float16_headroom"] > 0
    # float16 rounding error: at most half a float16 spacing at each channel's largest magnitude
    for n in info["channels"]:
        m = max(abs(info["channel_min"][n]), abs(info["channel_max"][n]))
        assert info["float16_max_abs_error"][n] <= float(np.spacing(np.float16(m))) / 2 + 1e-12


# ---- extraction assertions ----

def test_nan_in_the_context_stops_the_build(world):
    src = world.sources()
    first = world.samples.iloc[0]

    def poisoned(s, r, c):
        w = P.read_sheet_window(src[s], r, c, world.cfg)
        if s == first["sheet"]:
            w.z[first["row"] - w.row_off - 100, first["col"] - w.col_off] = np.nan   # 25 px into the context
        return w

    with pytest.raises(P.PatchError, match="NaN or inf in"):
        _extract(world, open_window=poisoned)


def test_float16_overflow_stops_the_build(world):
    src = world.sources()
    first = world.samples.iloc[0]

    def spiked(s, r, c):
        w = P.read_sheet_window(src[s], r, c, world.cfg)
        if s == first["sheet"]:
            w.z[first["row"] - w.row_off, first["col"] - w.col_off] += 1e5        # detr and lrm exceed 65504
        return w

    with pytest.raises(P.PatchError, match="float16 overflow"):
        _extract(world, open_window=spiked)


def test_slope_mean_mismatch_stops_the_build(world):
    S = world.samples.copy()
    S.loc[5, "slope_mean"] += 2e-9
    with pytest.raises(P.PatchError, match="slope_mean of sample id 5"):
        _extract(world, samples=S)


def test_float16_cast_checks():
    ok = np.array([[[0.0, -65504.0, 65504.0, 1e-3]]])
    assert P.to_float16(ok).dtype == np.float16 and np.isfinite(P.to_float16(ok)).all()
    for bad in (65504.5, -7e4, 1e9, np.inf, -np.inf, np.nan):
        with pytest.raises(P.PatchError):
            P.to_float16(np.array([[[1.0, bad]]]))


def test_window_outside_the_raster_and_misaligned_grids(world, tmp_path):
    cfg = world.cfg
    src = world.sources()
    s = world.samples["sheet"].iloc[0]
    with pytest.raises(P.PatchError, match="leaves"):
        P.read_sheet_window(src[s], np.array([50]), np.array([500]), cfg)
    # a susceptibility raster shifted by one pixel is refused
    with rasterio.open(src[s].susc_path) as ds:
        codes, t = ds.read(1), ds.transform
    shifted = tmp_path / "shifted.tif"
    with rasterio.open(shifted, "w", driver="GTiff", height=codes.shape[0], width=codes.shape[1], count=1,
                       dtype="uint8", crs="EPSG:5514", transform=t @ rasterio.Affine.translation(1, 0)) as dst:
        dst.write(codes, 1)
    bad = P.SheetSource(sheet=s, dtm_path=src[s].dtm_path, susc_path=shifted, grid=src[s].grid)
    g = world.samples[world.samples["sheet"] == s]
    with pytest.raises(P.PatchError, match="susceptibility raster is not on"):
        P.read_sheet_window(bad, g["row"].to_numpy(), g["col"].to_numpy(), cfg)


# ---- meta.csv, summary, figure ----

def test_meta_columns_order_and_roundtrip(world, built):
    meta = P.read_meta(built / "dataset" / "meta.csv")
    assert tuple(meta.columns) == P.META_COLUMNS
    assert len(meta) == len(world.samples)
    assert (meta["id"].to_numpy() == np.arange(len(meta))).all()
    for k in P.META_SAMPLE_COLUMNS + ("slope_mean",):
        assert (meta[k].to_numpy() == world.samples[k].to_numpy()).all(), k
    # round trip of every float, bit for bit
    _, ext = _extract(world)
    again = P.build_meta(world.samples, ext.stats)
    pd.testing.assert_frame_equal(meta, again, check_dtype=False, check_exact=True)
    text = (built / "dataset" / "meta.csv").read_bytes()
    assert b"\r\n" not in text and text.decode("utf-8")


def test_summary_counts_and_contents(world, built):
    summ = json.loads((built / "results" / "dataset_summary.json").read_text(encoding="utf-8"))
    S = world.samples
    n = len(S)
    assert summ["n_patches"] == n and len(summ["config_hash"]) == 64
    assert summ["by_source"] == {str(k): int(v) for k, v in S["source"].value_counts().items()}
    n_pos = int((S["label"] == 1).sum())
    assert summ["by_kind"] == {"background": int((S["pair_id"] < 0).sum()), "negative": n_pos, "positive": n_pos}
    for key in ("by_kraj", "by_flysch", "by_fold", "by_group_b", "by_stratum"):
        assert sum(v["total"] for v in summ[key].values()) == n, key
    assert sum(summ["by_match_level"].values()) == n_pos
    a = summ["test_sets"]["a_flysch_folds_pooled"]
    m = (S["pair_id"] >= 0) & (S["fold"] >= 0)
    assert a["positive"] == a["negative"] == int((m & (S["label"] == 1)).sum()) > 0
    b = summ["test_sets"]["b_test"]
    assert b["positive"] == b["negative"] > 0
    assert summ["pairs"]["n"] == n_pos
    txt = json.dumps(summ, ensure_ascii=False)
    for bad in ('"x"', '"y"', '"row"', '"col"', "geometry", "grid_", "susc", "90-0"):   # no coordinates, sheets, susc
        assert bad not in txt, bad


def test_figure_is_written(built):
    png = built / "figures" / "slope_distributions.png"
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 20_000


def test_build_info(built, world):
    info = json.loads((built / "dataset" / "build_info.json").read_text(encoding="utf-8"))
    npy = built / "dataset" / "patches.npy"
    assert info["patches"]["shape"] == [len(world.samples), 7, 150, 150]
    n = len(world.samples)
    assert info["patches"]["bytes"] == npy.stat().st_size == info["patches"]["header_bytes"] + n * 315_000
    assert info["patches"]["sha256"] == P.sha256_file(npy)
    assert info["meta"]["sha256"] == P.sha256_file(built / "dataset" / "meta.csv")
    summ = json.loads((built / "results" / "dataset_summary.json").read_text("utf-8"))
    assert info["config_hash"] == summ["config_hash"]
    assert not list((built / "dataset").glob("*.part"))


def test_build_is_deterministic(world, built, tmp_path):
    """A second build (other worker count) gives byte-identical patches.npy, meta.csv and summary."""
    assert _script("build_patches").main(["--out", str(tmp_path), "--workers", "1"]) == 0
    for rel in ("dataset/patches.npy", "dataset/meta.csv", "results/dataset_summary.json"):
        assert (tmp_path / rel).read_bytes() == (built / rel).read_bytes(), rel


# ---- verification ----

def test_verify_passes_on_the_build(world, built):
    ds = built / "dataset"
    checks = _verify(world, ds, n_check=10_000,
                     build_info=json.loads((ds / "build_info.json").read_text("utf-8")),
                     summary=json.loads((built / "results" / "dataset_summary.json").read_text("utf-8")),
                     config_hash=json.loads((ds / "build_info.json").read_text("utf-8"))["config_hash"])
    assert all(c.ok for c in checks), [c.line() for c in checks if not c.ok]
    names = {c.name for c in checks}
    assert names == {"files", "shape_dtype", "file_size", "nan_inf", "float16_overflow", "channel_bounds",
                     "meta_columns", "unique_ids", "labels_sources", "pairs", "meta_vs_samples", "patch_rule",
                     "alignment", "build_info", "config_hash", "summary"}


def test_verify_script_exit_codes(world, built, tmp_path):
    ds = tmp_path / "dataset"
    shutil.copytree(built / "dataset", ds)
    shutil.copy(built / "results" / "dataset_summary.json", tmp_path / "summary.json")
    v = _script("verify_dataset")
    args = ["--dataset", str(ds), "--summary", str(tmp_path / "summary.json"), "--log", str(tmp_path / "v.log")]
    assert v.main(args) == 0
    mm = np.load(ds / "patches.npy", mmap_mode="r+")
    mm[3, 2, 10, 10] = np.nan
    mm.flush()
    del mm
    assert v.main(args + ["--no-hash"]) == 1


@pytest.fixture()
def copy(built, tmp_path):
    ds = tmp_path / "dataset"
    shutil.copytree(built / "dataset", ds)
    return ds


def _edit_patches(ds, f):
    mm = np.load(ds / "patches.npy", mmap_mode="r+")
    f(mm)
    mm.flush()
    del mm


def test_verify_catches_swapped_patches(world, copy):
    def swap(mm):
        a = np.array(mm[10])
        mm[10] = mm[11]
        mm[11] = a

    _edit_patches(copy, swap)
    assert _failed(_verify(world, copy, n_check=10_000)) == {"alignment"}


def test_verify_catches_swapped_meta_rows(world, copy):
    meta = P.read_meta(copy / "meta.csv")
    stats = meta.loc[[20, 21], list(STAT_COLUMNS)].to_numpy()
    meta.loc[[20, 21], list(STAT_COLUMNS)] = stats[::-1]
    P.write_meta(meta, copy / "meta.csv")
    assert _failed(_verify(world, copy, n_check=10_000, samples=None)) == {"alignment"}
    assert "meta_vs_samples" in _failed(_verify(world, copy))     # slope_mean no longer matches samples.parquet


@pytest.mark.parametrize("channel, value, failed", [
    (2, np.nan, {"nan_inf"}), (4, -np.inf, {"nan_inf"}),
    (2, 65504.0, {"float16_overflow"}),        # lrm_l at the float16 maximum: a saturated value
    (0, 91.0, {"channel_bounds"}),             # slope above 90 deg
    (5, -0.01, {"channel_bounds"}),            # svf below 0
])
def test_verify_catches_bad_values(world, copy, channel, value, failed):
    def put(mm):
        mm[7, channel, 3, 3] = value

    _edit_patches(copy, put)
    assert _failed(_verify(world, copy)) == failed


def test_verify_catches_meta_problems(world, copy):
    meta = P.read_meta(copy / "meta.csv")
    bad = meta.copy()
    bad.loc[5, "id"] = 4
    P.write_meta(bad, copy / "meta.csv")
    assert "unique_ids" in _failed(_verify(world, copy, samples=None))

    # a second negative for pair 0: a background row relabelled
    bad = meta.copy()
    j = bad.index[bad["source"] == "background"][0]
    bad.loc[j, ["source", "pair_id", "match_level"]] = ["random", 0, 1]
    P.write_meta(bad, copy / "meta.csv")
    assert _failed(_verify(world, copy, samples=None)) == {"pairs"}

    # meta out of step with samples.parquet
    bad = meta.copy()
    bad.loc[8, "slope_bin"] += 1
    P.write_meta(bad, copy / "meta.csv")
    assert "meta_vs_samples" in _failed(_verify(world, copy))


def test_verify_rechecks_the_patch_rule(world, copy):
    meta = P.read_meta(copy / "meta.csv")
    bad = meta.copy()
    bad.loc[2, "x"] += 2.0                      # no longer the corner (row, col)
    P.write_meta(bad, copy / "meta.csv")
    assert "patch_rule" in _failed(_verify(world, copy, samples=None))

    bad = meta.copy()                           # a corner whose square leaves the sheet
    sh = world.sheets.set_index("sheet")
    s = bad.at[2, "sheet"]
    from landslide.sampling import grid_of_sheet
    from landslide.config import load_config
    from landslide.sheets import grid_origin
    g = grid_of_sheet(sh.loc[s], 2.0, grid_origin(load_config()))
    bad.loc[2, ["row", "col"]] = [110, 110]
    bad.loc[2, "x"], bad.loc[2, "y"] = (float(v) for v in g.xy(110, 110))
    P.write_meta(bad, copy / "meta.csv")
    assert "patch_rule" in _failed(_verify(world, copy, samples=None))


def test_verify_catches_size_and_dtype(world, copy):
    data = (copy / "patches.npy").read_bytes()
    (copy / "patches.npy").write_bytes(data[:-2])
    assert "file_size" in _failed(_verify(world, copy))
    arr = np.zeros((len(world.samples), 7, 150, 150), dtype=np.float32)
    np.save(copy / "patches.npy", arr)
    assert "shape_dtype" in _failed(_verify(world, copy))


def test_verify_hash_mismatch_is_only_a_warning(world, copy, built):
    info = json.loads((copy / "build_info.json").read_text("utf-8"))
    checks = _verify(world, copy, build_info=info, config_hash="0" * 64, full_hash=False)
    warn = [c for c in checks if c.name == "config_hash"]
    assert len(warn) == 1 and not warn[0].ok and warn[0].level == "warn"
    assert not P.failures(checks)


def test_verify_rows_are_seeded(world):
    a = P.verify_rows(world.cfg, 1000, 50)
    b = P.verify_rows(world.cfg, 1000, 50)
    assert np.array_equal(a, b) and a[0] == 0 and a[-1] == 999 and 50 <= len(a) <= 52
    assert np.array_equal(P.verify_rows(world.cfg, 10, 500), np.arange(10))


def test_failed_build_leaves_no_partial_files(world, built, tmp_path):
    """A build that stops (here: a perturbed slope_mean) removes its partial patch file and any old build_info.json,
    so the verification cannot mistake a half-built dataset for a finished one."""
    S = world.samples.copy()
    S.loc[len(S) - 1, "slope_mean"] += 1e-6
    bad = tmp_path / "samples_bad.parquet"
    S.to_parquet(bad, index=False)
    shutil.copytree(built / "dataset", tmp_path / "dataset")      # an earlier finished build in the same place
    with pytest.raises(P.PatchError, match=f"sample id {len(S) - 1}"):
        _script("build_patches").main(["--out", str(tmp_path), "--samples", str(bad)])
    left = sorted(p.name for p in (tmp_path / "dataset").iterdir())
    assert "patches.npy.part" not in left and P.BUILD_INFO_NAME not in left
    checks = _verify(world, tmp_path / "dataset", require_build_info=True)
    assert "build_info" in _failed(checks)
