"""Row 8: torchvision's ImageNet weights for ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1`).

The weights enum is the one `configs/landslide.json` `model.init` names. They are downloaded through torchvision's own
downloader (`WeightsEnum.get_state_dict`, which checks the sha256 prefix in the file name) with
`TORCH_HOME = data/cache/torch`, so the file lands where `landslide.model.imagenet_weights_path` expects it
(`data/cache/torch/hub/checkpoints/resnet18-f37072fd.pth`). This is the only download that does not go through
`landslide.net` (decision in docs/decisions/P1-01.md).

Then a fresh Python process with every socket connection disabled builds the model with
`landslide.model.build_model(cfg, path)` and runs a forward pass, which proves the weights load offline.

    uv run python scripts/fetch_weights.py            # download if missing, hash, offline check, sources.json
    uv run python scripts/fetch_weights.py --force    # download again
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from landslide.config import load_config
from landslide.paths import Paths

SECTION = "weights"
OFFLINE_FLAG = "--offline-check"
MARKER = "OFFLINE_CHECK_RESULT "


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class NetworkDisabled(OSError):
    """Raised by every socket connection attempt during the offline check."""


def disable_network() -> None:
    """Make every outgoing connection and name lookup raise `NetworkDisabled` for the rest of the process."""
    import socket

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise NetworkDisabled("network disabled for the offline check")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]


def network_is_disabled() -> bool:
    """True if a real request to the weights host fails with `NetworkDisabled`."""
    import urllib.request

    try:
        urllib.request.urlopen("https://download.pytorch.org/", timeout=5)
    except NetworkDisabled:
        return True
    except Exception as e:  # urllib may wrap the error
        return isinstance(getattr(e, "reason", None), NetworkDisabled)
    return False


def offline_check(paths: Paths) -> dict[str, Any]:
    """In this process: disable the network, then build the model from the cached file and run it."""
    disable_network()
    blocked = network_is_disabled()
    import torch

    from landslide.model import build_model, conv1_from_rgb, imagenet_weights_path, load_state_dict_file

    cfg = load_config()
    path = imagenet_weights_path(cfg, paths)
    model = build_model(cfg, path).eval()
    sd = load_state_dict_file(path)
    c = int(cfg["model"]["in_channels"])
    with torch.no_grad():
        out = model(torch.zeros(2, c, cfg["patch"]["px"], cfg["patch"]["px"]))
    backbone = model.state_dict()
    same_backbone = all(torch.equal(backbone[k], v) for k, v in sd.items() if k not in ("conv1.weight", "fc.weight",
                                                                                        "fc.bias"))
    return {"network_blocked": blocked, "weights_path": path.as_posix(), "loaded": True,
            "n_tensors_in_file": len(sd),
            "backbone_tensors_equal_file": bool(same_backbone),
            "conv1_shape": list(model.conv1.weight.shape),
            "conv1_from_rgb": bool(torch.equal(model.conv1.weight, conv1_from_rgb(sd["conv1.weight"], c))),
            "output_shape": list(out.shape), "output_finite": bool(torch.isfinite(out).all())}


def update_sources(path: Path, key: str, section: dict[str, Any]) -> None:
    """Replace one source's section of `results/sources.json`, keeping the others."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = section
    data["updated_utc"] = now_utc()
    data = {"task": "P1-07", **{k: v for k, v in data.items() if k != "task"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    paths = Paths()
    # Before torch is imported: torch.hub caches under $TORCH_HOME/hub/checkpoints.
    os.environ["TORCH_HOME"] = str(paths.torch_home)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download again even if the file is cached")
    ap.add_argument(OFFLINE_FLAG, action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.offline_check:
        print(MARKER + json.dumps(offline_check(paths)))
        return 0

    from landslide.logs import get_logger, setup_logging
    from landslide.sheets import sha256_file

    setup_logging(paths.logs / "fetch_weights.log")
    log = get_logger("fetch_weights")
    import torch
    import torchvision

    from landslide.model import _HUB_HASH, imagenet_weights, imagenet_weights_path

    cfg = load_config()
    weights = imagenet_weights(cfg)
    target = imagenet_weights_path(cfg, paths)
    hub_dir = Path(torch.hub.get_dir())
    if hub_dir.resolve() != (paths.torch_home / "hub").resolve():
        log.error("torch.hub dir is %s, expected %s", hub_dir, paths.torch_home / "hub")
        return 1
    downloaded = None
    if args.force or not target.is_file():
        if target.is_file():
            target.unlink()
        downloaded = now_utc()
        log.info("downloading %s (%s) through torchvision into %s", weights, weights.url, hub_dir)
        weights.get_state_dict(progress=False, check_hash=True)
    if not target.is_file():
        log.error("torchvision did not write %s", target)
        return 1
    sha = sha256_file(target)
    m = _HUB_HASH.search(target.name)
    prefix_ok = bool(m and sha.startswith(m.group(1)))
    log.info("%s: %d bytes, sha256 %s (file-name prefix %s)", target.name, target.stat().st_size, sha,
             "ok" if prefix_ok else "MISMATCH")

    env = {**os.environ, "TORCH_HOME": str(paths.torch_home), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), OFFLINE_FLAG], env=env,
                          capture_output=True, text=True, encoding="utf-8")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(MARKER)]
    if proc.returncode != 0 or not lines:
        log.error("offline check failed (exit %d):\n%s\n%s", proc.returncode, proc.stdout, proc.stderr)
        return 1
    check = json.loads(lines[-1][len(MARKER):])
    log.info("offline check: %s", check)

    prev = {}
    sj = paths.results / "sources.json"
    if sj.exists():
        prev = json.loads(sj.read_text(encoding="utf-8")).get(SECTION, {})
    meta = {k: v for k, v in weights.meta.items() if k != "categories"}
    section = {
        "row": 8,
        "source": "torchvision ResNet-18 ImageNet weights (torchvision's terms)",
        "weights": f"{type(weights).__name__}.{weights.name}",
        "url": weights.url,
        "downloader": "torchvision WeightsEnum.get_state_dict(check_hash=True) -> torch.hub.load_state_dict_from_url",
        "torch_home": "data/cache/torch (TORCH_HOME)",
        "file": target.relative_to(paths.data).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": sha,
        "sha256_prefix_in_name_ok": prefix_ok,
        "retrieved_utc": downloaded or prev.get("retrieved_utc"),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "meta": {**meta, "n_categories": len(weights.meta.get("categories", []))},
        "offline_check": {k: v for k, v in check.items() if k != "weights_path"},
        "n_requests": 1 if downloaded else 0,
    }
    update_sources(sj, SECTION, section)
    ok = prefix_ok and check["network_blocked"] and check["loaded"] and check["backbone_tensors_equal_file"] \
        and check["conv1_from_rgb"] and check["output_finite"]
    log.info("weights %s; sources.json updated", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
