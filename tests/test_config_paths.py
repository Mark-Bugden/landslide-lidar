import json

import numpy as np
import pytest

from landslide import config
from landslide.paths import Paths, sheet_to_stem, stem_to_sheet


def test_hash_ignores_formatting_and_key_order(tmp_path):
    a = {"b": 1, "a": {"y": [1, 2], "x": "Zlínský"}}
    p1, p2 = tmp_path / "1.json", tmp_path / "2.json"
    p1.write_text(json.dumps(a, indent=4, ensure_ascii=False), encoding="utf-8")
    p2.write_text(json.dumps({"a": {"x": "Zlínský", "y": [1, 2]}, "b": 1}), encoding="utf-8")
    assert config.config_hash(config.load_config(p1)) == config.config_hash(config.load_config(p2))


def test_hash_changes_with_any_value():
    assert config.config_hash({"lr": 0.0003}) != config.config_hash({"lr": 0.0004})


def test_frozen_hash(tmp_path):
    assert config.frozen_hash(tmp_path / "missing.sha256") is None
    p = tmp_path / "x.sha256"
    p.write_text("abc123  configs/landslide.json\n", encoding="ascii")
    assert config.frozen_hash(p) == "abc123"


def test_rng_streams_are_deterministic_and_separate():
    cfg = {"replication": {"streams": {"one": 1, "two": 2}}}
    a1 = config.rng_for(cfg, 11, "one").random(5)
    a2 = config.rng_for(cfg, 11, "one").random(5)
    b = config.rng_for(cfg, 11, "two").random(5)
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b)
    np.testing.assert_array_equal(config.make_rng(11).random(3), np.random.default_rng(11).random(3))


def test_unregistered_stream():
    with pytest.raises(KeyError):
        config.stream_id({"replication": {"streams": {}}}, "nope")


def test_sheet_names():
    assert sheet_to_stem("14-34-07") == "14_34_07"
    assert stem_to_sheet("14_34_07") == "14-34-07"
    assert Paths().dtm("14-34-07").name == "14_34_07.tif"


def test_smoke_redirects_outputs_only():
    real, smoke = Paths(), Paths(smoke=True)
    for attr in ("runs", "stage2", "results", "figures", "logs"):
        assert (real.data / "smoke") in getattr(smoke, attr).parents
        assert (real.data / "smoke") not in getattr(real, attr).parents
    for attr in ("raw", "cache", "derived", "dataset", "patches_npy", "meta_csv", "baselines_dir"):
        assert getattr(smoke, attr) == getattr(real, attr)
    assert real.results == real.root / "results"


def test_raw_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("LANDSLIDE_DATA_DIR", raising=False)
    monkeypatch.setenv("LANDSLIDE_RAW_DIR", str(tmp_path))
    assert Paths().raw == tmp_path
    monkeypatch.delenv("LANDSLIDE_RAW_DIR")
    assert Paths().raw == Paths().root / "data" / "raw_data"


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("LANDSLIDE_RAW_DIR", raising=False)
    monkeypatch.setenv("LANDSLIDE_DATA_DIR", str(tmp_path))
    p = Paths()
    assert p.raw == tmp_path / "raw_data"
    assert p.patches_npy == tmp_path / "dataset" / "patches.npy"
    assert Paths(smoke=True).runs == tmp_path / "smoke" / "runs"
    assert p.results == p.root / "results"
