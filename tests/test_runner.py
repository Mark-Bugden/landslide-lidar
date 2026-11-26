"""The overnight runner (P1-14): preflight, network block, keep-awake, order, resume, robustness, the wall-clock cap
and smoke mode. Synthetic data on a temporary root only (gate G3: no CNN sees a real patch; `Paths(root=tmp_path)`
never honours the data-dir overrides).

Most runs use `runner_synth.FakeBackend` (synthetic run directories and CNN detections) with the real evaluation
and verdict. Two tests use the real training code on tiny random patches (the wall-clock cap, and an end-to-end
smoke run that also runs the real stage-2 code on a toy sheet).
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before anything starts CUDA

import copy
import json
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision

import runner_synth as RS
from eval_synth import cnn_probs, unit_test_ids, write_unit
from landslide import preflight as PF
from landslide import runner as R
from landslide import winawake as W
from landslide.config import config_hash, load_config, rng_for
from landslide.paths import REPO_ROOT, Paths

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class World:
    root: Path
    cfg: dict
    meta: pd.DataFrame
    paths: Paths


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """No data-dir override, env restored after main() sets it, keep-awake recorded instead of called."""
    monkeypatch.delenv("LANDSLIDE_DATA_DIR", raising=False)
    monkeypatch.delenv("LANDSLIDE_RAW_DIR", raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch_home_env"))
    awake: list[str] = []

    from contextlib import contextmanager

    @contextmanager
    def fake_awake():
        awake.append("enter")
        try:
            yield True
        finally:
            awake.append("exit")

    monkeypatch.setattr(R, "keep_awake", fake_awake)
    yield awake
    PF.unblock_network()


@pytest.fixture
def world(tmp_path):
    cfg = RS.make_cfg(px=8)
    meta = RS.full_meta()
    paths = RS.write_tree(tmp_path, cfg, meta)
    return World(tmp_path, cfg, meta, paths)


def run_main(w: World, *args: str, backend, gpu=None, disk=None, clock=time.monotonic) -> int:
    return R.main([*args, "--root", str(w.root), "--no-progress"], backend=backend,
                  gpu_probe=gpu or RS.gpu(), disk_usage=disk or RS.disk(), clock=clock)


def summary(w: World, smoke: bool = False) -> dict:
    return json.loads((Paths(smoke=smoke, root=w.root).results / R.SUMMARY_NAME).read_text(encoding="utf-8"))


def newest_log(w: World, smoke: bool = False) -> Path:
    return sorted(Paths(smoke=smoke, root=w.root).logs.glob("overnight_*.log"))[-1]


def full_units(cfg, runs=("a", "b")):
    plan = R.make_plan(cfg, False)
    rs = sum((getattr(plan, r) for r in runs), ())
    return [f"{r}_s{s}" for r, s in plan.units(rs)]


# ======================================================================================================================
# Module contracts
# ======================================================================================================================


def test_runner_import_does_not_load_torch():
    code = "import sys, landslide.runner, landslide.preflight, landslide.winawake; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_constants_match_the_modules():
    from landslide import evaluate as E
    from landslide import stage2 as S2
    from landslide import train as T

    assert R.CUBLAS_WORKSPACE_CONFIG == T.CUBLAS_WORKSPACE_CONFIG
    assert PF.STAGE2_BASELINES == tuple(S2.BASELINE_KINDS)
    assert set(T.META_COLUMNS) | set(E.META_COLUMNS) | set(S2.LR_FEATURES) <= set(PF.META_REQUIRED)
    cfg = load_config()
    for smoke in (False, True):
        plan, runs = R.make_plan(cfg, smoke), E.expected_runs(cfg, smoke)
        assert (list(plan.a), list(plan.b), list(plan.s2)) == (runs["a"], runs["b"], runs["stage2"])
        assert list(plan.seeds) == E.expected_seeds(cfg, smoke)
    smoke = R.make_plan(cfg, True)
    assert smoke.units(smoke.a + smoke.b + smoke.s2) == [("a_f0", 0), ("b", 0), ("s2", 0)]
    full = R.make_plan(cfg, False)
    assert full.units(full.a)[:4] == [("a_f0", 0), ("a_f0", 1), ("a_f0", 2), ("a_f1", 0)]
    assert len(full.units(full.a + full.b + full.s2)) == 21
    assert R.smoke_stage2_sheets(cfg) == ["02-41-01"]


def test_smoke_subset_rule():
    cfg = load_config()
    meta = RS.full_meta()
    keep, info = R.smoke_subset(cfg, meta)
    assert np.array_equal(keep, R.smoke_subset(cfg, meta)[0])  # deterministic
    is_bg = (meta["source"] == "background").to_numpy()
    pairs = np.unique(meta.loc[~is_bg, "pair_id"])
    assert info["n_pairs"] == pairs.size == 192 and info["n_background"] == is_bg.sum() == 64
    assert info["n_pairs_chosen"] == 10 and info["n_background_chosen"] == 3  # round half up of 9.6 and 3.2
    kept = meta[keep]
    kp = kept[kept["source"] != "background"]
    assert (kp.groupby("pair_id").size() == 2).all()  # whole pairs
    assert kp["pair_id"].nunique() == 10 and (kept["source"] == "background").sum() == 3
    assert info["n_rows"] == int(keep.sum()) == 23
    # The registered stream reproduces the draw: pairs first, then the background.
    rng = rng_for(cfg, int(cfg["replication"]["seed"]), "smoke_subset")
    want_pairs = np.sort(rng.choice(pairs, size=10, replace=False))
    want_bg = np.sort(rng.choice(np.sort(meta.loc[is_bg, "id"].to_numpy()), size=3, replace=False))
    assert np.array_equal(np.sort(kp["pair_id"].unique()), want_pairs)
    assert np.array_equal(np.sort(kept.loc[kept["source"] == "background", "id"]), want_bg)
    other = copy.deepcopy(cfg)
    other["replication"]["seed"] += 1
    assert not np.array_equal(R.smoke_subset(other, meta)[0], keep)
    whole = copy.deepcopy(cfg)
    whole["smoke"]["patch_frac"] = 1.0
    assert R.smoke_subset(whole, meta)[0].all()


# ======================================================================================================================
# Keep-awake
# ======================================================================================================================


def test_keep_awake_sets_and_releases_even_on_error():
    calls = []

    def setter(flags):
        calls.append(flags)
        return W.ES_CONTINUOUS

    with pytest.raises(RuntimeError, match="inside"):
        with W.keep_awake(setter=setter, platform="win32") as active:
            assert active is True and calls == [0x80000001]
            raise RuntimeError("inside")
    assert calls == [W.ES_CONTINUOUS | W.ES_SYSTEM_REQUIRED, W.ES_CONTINUOUS] == [0x80000001, 0x80000000]


def test_keep_awake_is_a_noop_off_windows():
    calls = []
    with W.keep_awake(setter=lambda f: calls.append(f) or 1, platform="linux") as active:
        assert active is False
    assert calls == []


def test_keep_awake_failure_is_not_fatal():
    calls = []
    with W.keep_awake(setter=lambda f: calls.append(f) or 0, platform="win32") as active:
        assert active is False
    assert calls == [0x80000001]


@pytest.mark.skipif(sys.platform != "win32", reason="SetThreadExecutionState exists on Windows only")
def test_keep_awake_real_call():
    with W.keep_awake() as active:
        assert active is True


# ======================================================================================================================
# Network block
# ======================================================================================================================


def test_socket_block():
    import requests

    orig = (socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto, socket.getaddrinfo,
            socket.gethostbyname, socket.gethostbyname_ex)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    PF.block_network()
    try:
        assert PF.network_blocked() and socket.getaddrinfo is not orig[3]  # patched before anything is tried
        with pytest.raises(PF.NetworkBlocked):
            socket.create_connection(("192.0.2.1", 80), timeout=1)  # TEST-NET-1, never routed
        with pytest.raises(PF.NetworkBlocked):
            socket.getaddrinfo("example.invalid", 443)
        with pytest.raises(PF.NetworkBlocked):
            socket.gethostbyname("example.invalid")
        s = socket.socket()
        with pytest.raises(PF.NetworkBlocked):
            s.connect(("192.0.2.1", 80))
        with pytest.raises(PF.NetworkBlocked):
            s.connect_ex(("192.0.2.1", 80))
        s.close()
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with pytest.raises(PF.NetworkBlocked):
            u.sendto(b"x", ("192.0.2.1", 53))
        u.close()
        with pytest.raises(PF.NetworkBlocked):
            requests.get("https://192.0.2.1/", timeout=1)
        # loopback never leaves the machine and stays usable
        c = socket.create_connection(srv.getsockname(), timeout=5)
        c.close()
        a, b = socket.socketpair()
        a.close()
        b.close()
    finally:
        PF.unblock_network()
        srv.close()
    assert not PF.network_blocked()
    assert (socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto, socket.getaddrinfo,
            socket.gethostbyname, socket.gethostbyname_ex) == orig


def test_offline_context_restores():
    with PF.offline():
        assert PF.network_blocked()
    assert not PF.network_blocked()


# ======================================================================================================================
# Preflight
# ======================================================================================================================


def _report(w: World, cfg=None, **kw):
    return PF.run_preflight(cfg or w.cfg, w.paths, gpu_probe=kw.pop("gpu", RS.gpu()),
                            disk_usage=kw.pop("disk", RS.disk()), **kw)


def _problems(rep) -> str:
    return " | ".join(c.detail for c in rep.problems)


def test_preflight_passes_on_a_complete_tree(world):
    rep = _report(world)
    assert rep.ok, rep.message()
    passed = {c.name for c in rep.checks if c.ok}
    assert passed >= {"freeze", "files", "dataset", "artefacts", "stage2_rasters", "disk", "gpu", "weights"}
    assert [c.name for c in rep.warnings] == ["freeze"]  # a temporary root is not a git checkout
    assert rep.info["frozen_hash"] == rep.info["config_hash"] == config_hash(world.cfg)
    assert rep.info["dataset"]["meta_rows"] == len(world.meta)
    assert rep.seconds < PF.TIME_BUDGET_S


CASES = {
    "missing_file": (lambda w: w.paths.patches_npy.unlink(), {}, "data/dataset/patches.npy is missing"),
    "low_disk": (None, {"disk": RS.disk(3.0)}, "only 3.0 GiB free on the drive"),
    "no_cuda": (None, {"gpu": RS.gpu(available=False)}, "CUDA is not available"),
    "missing_freeze": (lambda w: (w.root / "configs" / "landslide.sha256").unlink(), {},
                       "the pre-registration is not frozen"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_preflight_failure_exits_within_60s_with_a_clear_message(world, case, capsys):
    mutate, probes, text = CASES[case]
    if mutate is not None:
        mutate(world)
    backend = RS.FakeBackend(world.meta)
    t = time.monotonic()
    code = run_main(world, backend=backend, **probes)
    elapsed = time.monotonic() - t
    assert code == R.EXIT_PREFLIGHT == 2
    assert elapsed < 60
    err = capsys.readouterr().err
    log_text = newest_log(world).read_text(encoding="utf-8")
    for out in (err, log_text):
        assert "PREFLIGHT FAILED" in out and text in out and "nothing was trained" in out
    assert backend.calls == []
    assert not (world.root / "results" / R.SUMMARY_NAME).exists()
    assert not world.paths.runs.exists()
    assert not PF.network_blocked()


def test_missing_freeze_applies_to_smoke_too(world):
    (world.root / "configs" / "landslide.sha256").unlink()
    backend = RS.FakeBackend(world.meta)
    assert run_main(world, "--smoke", backend=backend) == R.EXIT_PREFLIGHT
    assert "not frozen" in newest_log(world, smoke=True).read_text(encoding="utf-8")
    assert backend.calls == [] and not Paths(smoke=True, root=world.root).runs.exists()


def test_dry_run_runs_the_preflight_only(world):
    backend = RS.FakeBackend(world.meta)
    assert run_main(world, "--dry-run", backend=backend) == R.EXIT_OK
    assert backend.calls == []
    assert not (world.root / "results" / R.SUMMARY_NAME).exists()
    assert "PREFLIGHT PASSED" in newest_log(world).read_text(encoding="utf-8")


def test_preflight_config_must_equal_the_frozen_hash(world):
    cfg = copy.deepcopy(world.cfg)
    cfg["train"]["lr"] = 1e-3
    assert "differs from the frozen hash" in _problems(_report(world, cfg))


def test_preflight_checks_the_artefact_hashes(world):
    p = world.paths
    d = json.loads(p.splits_json.read_text(encoding="utf-8"))
    d["config_hash"] = "0" * 64
    p.splits_json.write_text(json.dumps(d), encoding="utf-8")
    bi = json.loads((p.baselines_dir / "build_info.json").read_text(encoding="utf-8"))
    bi["inputs"]["meta_csv_sha256"] = "f" * 64
    (p.baselines_dir / "build_info.json").write_text(json.dumps(bi), encoding="utf-8")
    pd.DataFrame({"scorer": ["slope_only", "susceptibility", "random"], "config_hash": config_hash(world.cfg)}) \
        .to_csv(world.root / "results" / "stage2_baselines.csv", index=False)
    sheet = PF.stage2_sheet_list(world.cfg)[3]
    np.savez(p.stage2 / "lattice" / f"{sheet}.npz", row=np.arange(3), config_hash=np.str_("1" * 64))
    probs = _problems(_report(world))
    assert "data/derived/splits.json records config 000000000000..." in probs
    assert "was built from a meta.csv with sha256 ffffffffffff..." in probs
    assert "stage2_baselines.csv lacks the scorers ['slope_susc_lr']" in probs
    assert f"(e.g. {sheet}) were not built with the frozen config" in probs
    bi = json.loads((p.dataset / "build_info.json").read_text(encoding="utf-8"))
    bi["config_hash"] = "2" * 64
    (p.dataset / "build_info.json").write_text(json.dumps(bi), encoding="utf-8")
    assert "build_info.json records config 222222222222..." in _problems(_report(world))


def test_preflight_checks_the_dataset_shape(world):
    p = world.paths
    with p.patches_npy.open("ab") as f:  # a padded (or, with truncate, short) data block
        f.write(b"\0\0")
    assert "(truncated or padded)" in _problems(_report(world))
    world.meta.iloc[:-1].to_csv(p.meta_csv, index=False)
    probs = _problems(_report(world))
    assert f"patches.npy holds {len(world.meta):,} patches but meta.csv has {len(world.meta) - 1:,} rows" in probs
    np.save(p.patches_npy, np.zeros((len(world.meta), 7, 9, 9), dtype=np.float32))
    probs = _problems(_report(world))
    assert "dtype float32, expected float16" in probs and "expected (N, 7, 8, 8)" in probs
    world.meta.drop(columns=["susc_mean"]).to_csv(p.meta_csv, index=False)
    assert "meta.csv lacks columns ['susc_mean']" in _problems(_report(world))


def test_preflight_checks_the_stage2_rasters_and_weights(world):
    sheet = PF.stage2_sheet_list(world.cfg)[0]
    RS._write_tif(world.paths.dtm(sheet), np.zeros((4, 6), dtype=np.float32))
    (world.root / "results" / "sources.json").write_text(json.dumps({"weights": {"bytes": 999}}), encoding="utf-8")
    probs = _problems(_report(world))
    assert "02_41_01.tif is 4 x 6 px, sheets.parquet says 4 x 5" in probs
    assert "bytes, results/sources.json records 999" in probs
    pd.DataFrame({"sheet": ["01-01-01"], "grid_w": 5, "grid_h": 4, "is_stage2": True}).to_parquet(
        world.paths.sheets_parquet, index=False)
    assert "differ from the config's 20" in _problems(_report(world))


def test_preflight_gpu_memory_and_a_hung_probe(world, monkeypatch):
    assert "only 3.0 GiB of GPU memory free" in _problems(_report(world, gpu=RS.gpu(free_gib=3.0)))

    def hung():
        time.sleep(3)
        return RS.gpu()()

    monkeypatch.setattr(PF, "_MIN_PROBE_TIMEOUT_S", 0.3)
    t = time.monotonic()
    rep = _report(world, gpu=hung, t_start=time.monotonic() - 1000)  # the budget is used up
    assert time.monotonic() - t < 2.5
    probs = _problems(rep)
    assert "CUDA probe did not finish within" in probs and "not checked: torch did not load" in probs


def test_preflight_rejects_a_dataloader_config(world):
    cfg = copy.deepcopy(world.cfg)
    cfg["runner"]["num_workers"] = 2
    assert "runner.num_workers is 2" in _problems(_report(world, cfg))


def test_preflight_requires_a_committed_freeze(world):
    def git(*a):
        return subprocess.run(["git", "-C", str(world.root), *a], capture_output=True, text=True, timeout=60)

    if git("init", "-q").returncode != 0:
        pytest.skip("git is not available")
    git("add", "configs")
    if git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "freeze").returncode:
        pytest.skip("cannot commit in a temporary repository here")
    rep = _report(world)
    assert rep.ok, rep.message()
    assert rep.info["freeze_commit"] == git("rev-parse", "HEAD").stdout.strip()
    assert not rep.warnings
    sha = world.root / "configs" / "landslide.sha256"
    sha.write_text(sha.read_text(encoding="ascii") + "\n", encoding="ascii")  # same hash, uncommitted change
    assert "the frozen config is not committed" in _problems(_report(world))


def test_real_script_refuses_before_the_freeze(tmp_path):
    """The real entry point in a fresh process (real torch import and CUDA probe): exit 2 within 60 s."""
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO_ROOT / "configs" / "landslide.json", tmp_path / "configs" / "landslide.json")
    env = {k: v for k, v in os.environ.items() if k not in ("LANDSLIDE_DATA_DIR", "LANDSLIDE_RAW_DIR")}
    env["PYTHONIOENCODING"] = "utf-8"
    t = time.monotonic()
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_overnight.py"), "--dry-run", "--root",
                        str(tmp_path)], capture_output=True, text=True, encoding="utf-8", timeout=120, env=env,
                       cwd=tmp_path)
    elapsed = time.monotonic() - t
    assert r.returncode == 2, r.stderr[-2000:]
    assert elapsed < 60
    assert "PREFLIGHT FAILED" in r.stderr and "the pre-registration is not frozen" in r.stderr
    created = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert created[0] == "configs/landslide.json" and all(c.startswith("logs/overnight_") for c in created[1:])


# ======================================================================================================================
# The run (fake training, real evaluation)
# ======================================================================================================================


def test_full_run_order_and_stage2_when_a_and_b_pass(world, _isolate):
    backend = RS.FakeBackend(world.meta)
    assert run_main(world, backend=backend) == R.EXIT_OK
    s1, s2 = full_units(world.cfg, ("a", "b")), full_units(world.cfg, ("s2",))
    sheets = PF.stage2_sheet_list(world.cfg)
    assert backend.calls == ([("train", u) for u in s1] + [("evaluate", None)] + [("train", u) for u in s2]
                             + [("stage2_inputs", None), ("stage2_models", [0, 1, 2])]
                             + [("stage2_cnn_sheet", s) for s in sheets]
                             + [("stage2_evaluate_cnn", sheets), ("evaluate", None)])
    assert all(backend.network_blocked) and not PF.network_blocked()  # offline during the run only
    assert _isolate == ["enter", "exit"]  # kept awake, then released
    s = summary(world)
    assert s["status"] == "finished" and s["verdict"] == "REPLICATED" and s["missing_units"] == []
    assert s["config_hash"] == s["frozen_hash"] == config_hash(world.cfg)
    assert s["stage2"]["allowed"] is True and s["stage2"]["cnn"]["status"] == "done"
    assert [e["when"] for e in s["evaluations"]] == ["after_stage1", "final"] and s["evaluations"][1]["final"]
    assert list(s["units"]) == s1 + s2 and all(u["status"] == "done" for u in s["units"].values())
    assert s["log_file"].startswith("logs/overnight_") and s["cap"]["source"] == "runner.wall_clock_h"
    assert s["cap"]["wall_clock_h"] == world.cfg["runner"]["wall_clock_h"] and s["cap"]["reached"] is False
    assert s["measurements"]["units"]["a_f0_s0"]["seconds_per_epoch_mean"] == 1.0
    assert s["measurements"]["stage2"]["windows_per_s"] == 50.0
    assert json.loads((world.root / "results" / "metrics.json").read_text(encoding="utf-8"))["verdict"] \
        == "REPLICATED"
    raw = (world.root / "results" / R.SUMMARY_NAME).read_bytes()
    assert b"\r\n" not in raw and world.root.name.encode() not in raw  # LF, no absolute paths


@pytest.mark.parametrize("failing", ["a", "b"])
def test_stage2_is_skipped_when_a_or_b_fails(world, failing):
    backend = RS.FakeBackend(world.meta, strength_a=0.0 if failing == "a" else 2.0,
                             strength_b=0.0 if failing == "b" else 1.5)
    assert run_main(world, backend=backend) == R.EXIT_OK
    assert backend.trained() == full_units(world.cfg, ("a", "b"))  # no s2 unit
    assert [k for k, _ in backend.calls if k.startswith("stage2")] == []
    s = summary(world)
    assert s["verdict"] == "NOT REPLICATED"
    assert s["verdict_detail"][failing] == "FAIL" and s["verdict_detail"]["stage2"] == "NOT_RUN"
    assert s["stage2"]["allowed"] is False and "only if (a) and (b) both pass" in s["stage2"]["reason"]
    assert all(s["units"][u]["status"] == "not_run" for u in full_units(world.cfg, ("s2",)))
    assert len(s["evaluations"]) == 1 and s["evaluations"][0]["final"]


def test_one_failing_unit_does_not_stop_the_run(world):
    backend = RS.FakeBackend(world.meta, fail={"a_f1_s1"})
    assert run_main(world, backend=backend) == R.EXIT_OK
    assert backend.trained() == full_units(world.cfg, ("a", "b"))  # every unit attempted
    s = summary(world)
    u = s["units"]["a_f1_s1"]
    assert u["status"] == "failed" and u["error"] == "RuntimeError: boom in a_f1_s1"
    assert all(v["status"] == "done" for k, v in s["units"].items() if k != "a_f1_s1" and not k.startswith("s2"))
    assert s["verdict"] == "INCOMPLETE" and s["verdict_detail"]["a"] == "INCOMPLETE"
    assert "a_f1_s1" in s["missing_units"]
    log_text = newest_log(world).read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in log_text and "boom in a_f1_s1" in log_text
    assert "a_f1_s1 failed; moving on to the next unit" in log_text


def test_resume_skips_finished_units_and_retrains_failed_ones(world):
    p, meta, h = world.paths, world.meta, config_hash(world.cfg)
    for seed in (0, 1):
        ids = unit_test_ids(meta, "a_f0")
        write_unit(p, "a_f0", seed, ids, cnn_probs(meta, ids, seed, 2.0), h, epochs=3, best_epoch=2)
    write_unit(p, "a_f0", 2, np.empty(0), np.empty(0), h, status="failed")
    write_unit(p, "a_f1", 0, np.empty(0), np.empty(0), h, status="interrupted")
    d = p.run_dir("a_f1", 1)
    d.mkdir(parents=True)
    (d / "history.csv").write_text("epoch,seconds\n1,1.0\n", encoding="utf-8")  # killed mid-unit (Ctrl-C)
    done_before = (p.run_dir("a_f0", 0) / "done.json").read_bytes()

    backend = RS.FakeBackend(meta)
    assert run_main(world, backend=backend) == R.EXIT_OK
    trained = backend.trained()
    assert "a_f0_s0" not in trained and "a_f0_s1" not in trained
    assert trained[:3] == ["a_f0_s2", "a_f1_s0", "a_f1_s1"]
    assert (p.run_dir("a_f0", 0) / "done.json").read_bytes() == done_before  # never overwritten
    s = summary(world)
    assert s["units"]["a_f0_s0"]["status"] == "skipped" and s["units"]["a_f0_s0"]["seconds"] == 12.5
    assert s["units"]["a_f0_s2"]["status"] == "done"
    assert s["units"]["a_f0_s2"]["previous_attempt"]["status"] == "failed"
    assert s["units"]["a_f0_s2"]["previous_attempt"]["error"] == "RuntimeError: boom"
    assert s["units"]["a_f1_s0"]["previous_attempt"]["status"] == "interrupted"
    assert s["units"]["a_f1_s1"]["previous_attempt"]["status"].startswith("no status file")
    assert s["verdict"] == "REPLICATED"
    metrics = (world.root / "results" / "metrics.json").read_bytes()

    again = RS.FakeBackend(meta)
    assert run_main(world, backend=again) == R.EXIT_OK
    assert again.trained() == []  # everything was finished
    s2 = summary(world)
    assert all(u["status"] == "skipped" for u in s2["units"].values())
    assert s2["verdict"] == "REPLICATED" and s2["missing_units"] == []
    assert (world.root / "results" / "metrics.json").read_bytes() == metrics
    assert [(p["status"], p["verdict"]) for p in s2["previous_invocations"]] == [("finished", "REPLICATED")]
    assert s2["previous_invocations"][0]["started_utc"] == s["started_utc"]


def test_ctrl_c_writes_an_aborted_summary(world):
    backend = RS.FakeBackend(world.meta)
    train = backend.train

    def interrupted_by_user(run, seed, *a, **k):
        if (run, seed) == ("a_f0", 1):
            raise KeyboardInterrupt
        return train(run, seed, *a, **k)

    backend.train = interrupted_by_user
    assert run_main(world, backend=backend) == R.EXIT_ABORTED == 130
    s = summary(world)
    assert s["status"] == "aborted" and s["abort_reason"] == "KeyboardInterrupt"
    assert s["units"]["a_f0_s0"]["status"] == "done" and s["units"]["a_f0_s1"]["status"] == "aborted"
    assert not PF.network_blocked()


def test_a_failing_evaluation_is_logged_and_never_allows_stage2(world):
    backend = RS.FakeBackend(world.meta)

    def broken(cfg, paths):
        raise ValueError("bad metrics")

    backend.evaluate = broken
    assert run_main(world, backend=backend) == R.EXIT_ERROR
    assert backend.trained() == full_units(world.cfg, ("a", "b"))
    s = summary(world)
    assert s["evaluations"] == [{"when": "after_stage1", "error": "ValueError: bad metrics",
                                 "seconds": s["evaluations"][0]["seconds"], "final": True}]
    assert s["stage2"]["allowed"] is False and s["verdict"] is None and s["status"] == "finished"
    assert s["missing_units"] == full_units(world.cfg, ("s2",))
    assert "ValueError: bad metrics" in newest_log(world).read_text(encoding="utf-8")


def test_a_done_unit_with_another_hash_is_never_overwritten(world):
    ids = unit_test_ids(world.meta, "b")
    write_unit(world.paths, "b", 0, ids, cnn_probs(world.meta, ids, 0, 1.5), "0" * 64)
    backend = RS.FakeBackend(world.meta)
    assert run_main(world, backend=backend) == R.EXIT_OK
    assert "b_s0" not in backend.trained()
    s = summary(world)
    assert s["units"]["b_s0"]["status"] == "skipped" and "never overwritten" in s["units"]["b_s0"]["problem"]
    assert "b_s0" in s["missing_units"] and s["verdict"] == "INCOMPLETE"


def test_wall_clock_cap_stops_cleanly_and_the_verdict_is_incomplete(world):
    clock = RS.FakeClock()
    # --max-hours 1: work until start + 60 min - 15 min = 2,700 s. Each simulated epoch takes 400 s.
    backend = RS.FakeBackend(world.meta, epochs=3, tick=lambda: clock.advance(400))
    assert run_main(world, "--max-hours", "1", backend=backend, clock=clock) == R.EXIT_OK
    assert backend.trained() == ["a_f0_s0", "a_f0_s1", "a_f0_s2"]
    s = summary(world)
    st = {k: v["status"] for k, v in s["units"].items()}
    assert st["a_f0_s0"] == st["a_f0_s1"] == "done"
    assert st["a_f0_s2"] == "interrupted" and s["units"]["a_f0_s2"]["epochs_completed"] == 1
    assert all(st[u] == "not_started" for u in full_units(world.cfg, ("a", "b"))[3:])
    assert all(st[u] == "not_run" for u in full_units(world.cfg, ("s2",)))
    d = world.paths.run_dir("a_f0", 2)
    assert (d / "interrupted.json").exists() and not (d / "done.json").exists()
    assert s["status"] == "stopped_at_cap" and s["cap"]["reached"] is True and s["cap"]["source"] == "--max-hours"
    assert s["verdict"] == "INCOMPLETE" and s["verdict_detail"]["a"] == "INCOMPLETE"
    assert len(s["missing_units"]) == 21 - 2
    assert json.loads((world.root / "results" / "metrics.json").read_text(encoding="utf-8"))["verdict"] \
        == "INCOMPLETE"


def test_cap_shorter_than_the_evaluation_reserve_trains_nothing(world):
    backend = RS.FakeBackend(world.meta)
    assert run_main(world, "--max-hours", "0.1", backend=backend) == R.EXIT_OK  # 6 min < 15 min reserve
    assert backend.trained() == []
    s = summary(world)
    assert s["status"] == "stopped_at_cap" and s["verdict"] == "INCOMPLETE"
    assert {v["status"] for v in s["units"].values()} == {"not_started", "not_run"}


def test_cap_during_stage2_inference_is_incomplete(world):
    clock = RS.FakeClock()
    # 18 + 3 units x 3 epochs at 10 s, then 20 sheets at 100 s each: the cap falls inside stage 2.
    cap_h = (21 * 3 * 10 + 5 * 100 + 15 * 60 + 50) / 3600
    backend = RS.FakeBackend(world.meta, epochs=3, tick=lambda: clock.advance(10))
    orig = backend.stage2_cnn_sheet

    def slow_sheet(*a, **k):
        clock.advance(90)
        return orig(*a, **k)

    backend.stage2_cnn_sheet = slow_sheet
    assert run_main(world, "--max-hours", str(cap_h), backend=backend, clock=clock) == R.EXIT_OK
    s = summary(world)
    per = s["stage2"]["cnn"]["sheets"]
    assert 0 < sum(v["status"] == "done" for v in per.values()) < 20
    assert any(v["status"] == "not_started" for v in per.values())
    assert s["stage2"]["cnn"]["status"] == "incomplete" and "cnn_evaluation" not in s["stage2"]
    assert ("stage2_evaluate_cnn", PF.stage2_sheet_list(world.cfg)) not in backend.calls
    assert s["verdict"] == "INCOMPLETE" and s["verdict_detail"]["stage2"] == "INCOMPLETE"
    assert s["status"] == "stopped_at_cap"


def test_smoke_mode_with_a_fake_backend(world):
    """Smoke: a_f0, b, s2 x seed 0 on the D24 subset; stage 2 on one sheet even though (a) fails; nothing outside
    data/smoke/ changes."""
    before = RS.file_snapshot(world.root)
    backend = RS.FakeBackend(world.meta, strength_a=0.0)
    assert run_main(world, "--smoke", backend=backend) == R.EXIT_OK
    assert backend.trained() == ["a_f0_s0", "b_s0", "s2_s0"]
    keep, info = R.smoke_subset(world.cfg, world.meta)
    assert all(np.array_equal(k, keep) for k in backend.keeps.values())
    sheet = R.smoke_stage2_sheets(world.cfg)
    assert ("stage2_baselines", sheet) in backend.calls and ("stage2_cnn_sheet", sheet[0]) in backend.calls
    assert ("stage2_models", [0]) in backend.calls
    after = RS.file_snapshot(world.root)
    changed = {k for k in after if before.get(k) != after[k]}
    assert changed and all(k.startswith("data/smoke/") for k in changed)
    assert not (world.root / "results" / R.SUMMARY_NAME).exists() and not (world.root / "figures").exists()
    assert not (world.root / "logs").exists()
    s = summary(world, smoke=True)
    assert s["smoke"] is True and s["smoke_subset"] == info
    assert s["stage2"]["allowed"] is True and s["verdict_detail"]["a"] == "FAIL"
    assert s["measurements"]["full_unit_sizes"]["a_f0"]["n_test"] > 0
    assert s["measurements"]["full_stage2"]["n_windows"] == 146_622
    metrics = json.loads((world.root / "data" / "smoke" / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["smoke"] is True and metrics["tests"]["stage2"]["sheets"] == sheet


# ======================================================================================================================
# Real training code on tiny random patches
# ======================================================================================================================


@pytest.fixture(scope="module")
def fake_weights(tmp_path_factory):
    torch.manual_seed(1234)
    p = tmp_path_factory.mktemp("weights") / "fake_resnet18.pth"
    torch.save(torchvision.models.resnet18(weights=None).state_dict(), p)
    return p


class CountingDeadline(R.Deadline):
    """The cap is reached at the (allow + 1)-th check."""

    def __init__(self, allow: int):
        super().__init__(start=0.0, cap_s=6 * 3600.0, reserve_s=900.0, clock=lambda: 0.0)
        self.allow, self.n = allow, 0

    def over(self) -> bool:
        self.n += 1
        if self.n > self.allow:
            self.reached = True
        return self.reached


def test_cap_with_the_real_training_code(tmp_path, fake_weights):
    cfg = RS.make_cfg(px=32)
    meta = RS.full_meta()
    paths = RS.write_tree(tmp_path, cfg, meta)
    # checks: before a_f0_s0, train_unit's before epoch 1, after epoch 1 -> stop
    runner = R.Runner(cfg, paths, R.RunOptions(show_progress=False),
                      R.Backend(device=DEV, weights_path=fake_weights), deadline=CountingDeadline(2))
    assert runner.run() == R.EXIT_OK
    d = paths.run_dir("a_f0", 0)
    intr = json.loads((d / "interrupted.json").read_text(encoding="utf-8"))
    assert intr["epochs_completed"] == 1 and not (d / "done.json").exists() and not (d / "best.pt").exists()
    assert len(pd.read_csv(d / "history.csv")) == 1
    s = json.loads((paths.results / R.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert s["units"]["a_f0_s0"]["status"] == "interrupted"
    assert all(s["units"][u]["status"] == "not_started" for u in full_units(cfg, ("a", "b"))[1:])
    assert s["status"] == "stopped_at_cap" and s["verdict"] == "INCOMPLETE"
    m = json.loads((paths.results / "metrics.json").read_text(encoding="utf-8"))
    assert m["integrity"]["interrupted"] == ["a_f0_s0"] and m["verdict"] == "INCOMPLETE"


def test_smoke_end_to_end_with_the_real_code(tmp_path, fake_weights):
    """Real training (2 epochs, D24 subset), real stage 2 on a toy reserved sheet (baselines and CNN), real
    evaluation, through main(): every new or changed file is under data/smoke/."""
    from landslide import stage2 as S2

    cfg = RS.make_cfg(px=64, stage2_sheets={"flysch": ["90-00-01"], "nonflysch": ["90-00-02"]})
    cfg["stage2"]["stride_px"] = 16
    cfg["smoke"]["patch_frac"] = 0.5
    cfg["train"]["batch_size"] = 16
    meta = RS.full_meta()
    paths = RS.write_tree(tmp_path, cfg, meta, sheets_parquet=False, polygons=False, rasters=False,
                          stage2_outputs=False)
    RS.toy_stage2_world(paths, cfg)
    S2.run_baselines(cfg, paths, list(S2.BASELINE_KINDS), S2.load_inputs(cfg, paths))  # phase 1's outputs
    w = World(tmp_path, cfg, meta, paths)
    before = RS.file_snapshot(tmp_path)

    code = run_main(w, "--smoke", backend=R.Backend(device=DEV, weights_path=fake_weights))
    assert code == R.EXIT_OK, newest_log(w, smoke=True).read_text(encoding="utf-8")[-3000:]
    after = RS.file_snapshot(tmp_path)
    changed = {k for k in after if before.get(k) != after[k]}
    assert all(k.startswith("data/smoke/") for k in changed), sorted(changed)
    assert not (tmp_path / "figures").exists() and not (tmp_path / "logs").exists()
    assert {k for k in before if k.startswith("results/")} == {k for k in after if k.startswith("results/")}

    s = summary(w, smoke=True)
    assert s["status"] == "finished" and s["verdict"] in ("REPLICATED", "NOT REPLICATED", "INCOMPLETE")
    assert [s["units"][u]["status"] for u in ("a_f0_s0", "b_s0", "s2_s0")] == ["done"] * 3
    for u in ("a_f0_s0", "b_s0", "s2_s0"):
        m = s["measurements"]["units"][u]
        assert m["epochs"] == cfg["smoke"]["max_epochs"] and m["stop_reason"] == "epoch_limit"
        assert m["seconds_per_epoch_mean"] > 0 and m["n_train"] > 0
        if DEV == "cuda":
            assert m["peak_gpu_mem_gb"] > 0
    cnn = s["stage2"]["cnn"]
    assert cnn["status"] == "done" and list(cnn["sheets"]) == ["90-00-01"] and cnn["windows_per_s"] > 0
    assert s["stage2"]["baselines"]["sheets"] == ["90-00-01"]
    assert s["stage2"]["cnn_evaluation"]["n_sheets"] == 1
    assert s["smoke_subset"]["n_rows"] < len(meta)
    m = json.loads((tmp_path / "data" / "smoke" / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert m["smoke"] is True and m["tests"]["stage2"]["sheets"] == ["90-00-01"]
    assert m["integrity"]["n_done"] == 3
    det = pd.read_parquet(tmp_path / "data" / "smoke" / "stage2" / "detections.parquet")
    assert set(det["scorer"]) == {"cnn", *S2.BASELINE_KINDS} and set(det["sheet"]) == {"90-00-01"}
    s2_done = json.loads((Paths(smoke=True, root=tmp_path).run_dir("s2", 0) / "done.json").read_text("utf-8"))
    assert s2_done["smoke"] is True and s2_done["config_hash"] == config_hash(cfg)
