"""Load the frozen configuration and derive its hash and random streams.

Every value used by the pipeline comes from `configs/landslide.json`. The config hash is the sha256 of the
canonical JSON (sorted keys, no whitespace, UTF-8), so formatting changes to the file do not change it but any
value change does.

Random numbers (decision D02): always `numpy.random.default_rng`. A step that needs its own stream uses
`SeedSequence([seed, stream_id])`, with the stream ids listed in the config's `replication.streams` block.
The `random` module is never used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from landslide.paths import CONFIG_PATH, CONFIG_SHA_PATH


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the config JSON. Defaults to `configs/landslide.json`."""
    p = Path(path) if path is not None else CONFIG_PATH
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def canonical_json(cfg: dict[str, Any]) -> bytes:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def config_hash(cfg: dict[str, Any]) -> str:
    """sha256 hex digest of the canonical JSON of `cfg`."""
    return hashlib.sha256(canonical_json(cfg)).hexdigest()


def frozen_hash(path: Path | str | None = None) -> str | None:
    """The hash recorded at the freeze (`configs/landslide.sha256`), or None before the freeze."""
    p = Path(path) if path is not None else CONFIG_SHA_PATH
    if not p.exists():
        return None
    return p.read_text(encoding="ascii").split()[0].strip()


def stream_id(cfg: dict[str, Any], name: str) -> int:
    """The registered id of a named random stream (config `replication.streams`)."""
    try:
        return int(cfg["replication"]["streams"][name])
    except KeyError as e:
        raise KeyError(f"random stream {name!r} is not registered in config replication.streams") from e


def make_rng(seed: int, stream: int | None = None) -> np.random.Generator:
    """`default_rng(seed)`, or `default_rng(SeedSequence([seed, stream]))` for a separate stream."""
    if stream is None:
        return np.random.default_rng(seed)
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def rng_for(cfg: dict[str, Any], seed: int, name: str) -> np.random.Generator:
    """The generator for step `name` seeded with `seed`, using the stream id registered in the config."""
    return make_rng(seed, stream_id(cfg, name))
