"""Generate a deterministic, tiny clean-room fixture and two method outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FIXTURE_SCHEMA_VERSION = "1.0.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_fixture(out: Path, seed: int = 20260730) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    frames, height, width = 64, 48, 48
    yy, xx = np.mgrid[:height, :width]
    centers = [(12, 13), (15, 35), (31, 18), (35, 36)]
    traces = np.zeros((frames, len(centers)), dtype=np.float32)
    for index, onset in enumerate((8, 18, 28, 39)):
        time = np.arange(frames, dtype=np.float32)
        event = np.where(time >= onset, np.exp(-(time - onset) / (5.0 + index)), 0.0)
        event += 0.65 * np.where(
            time >= onset + 17,
            np.exp(-(time - onset - 17) / (4.0 + index)),
            0.0,
        )
        traces[:, index] = event

    clean = np.full((frames, height, width), 0.12, dtype=np.float32)
    for index, (cy, cx) in enumerate(centers):
        footprint = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 2.4**2))
        clean += traces[:, index, None, None] * footprint[None] * (0.18 + 0.03 * index)
    clean += 0.01 * np.sin(np.arange(frames, dtype=np.float32)[:, None, None] / 7.0)

    rng = np.random.default_rng(seed)
    raw = clean + rng.normal(0.0, 0.035, clean.shape).astype(np.float32)
    identity = raw.copy()
    padded = np.pad(raw, ((1, 1), (0, 0), (0, 0)), mode="edge")
    temporal_smooth = (
        0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    ).astype(np.float32)

    arrays = {
        "clean": clean,
        "raw": raw,
        "identity": identity,
        "temporal_smooth": temporal_smooth,
    }
    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name, array in arrays.items():
        path = out / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        files[name] = path.name
        hashes[name] = sha256(path)

    payload: dict[str, object] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "seed": seed,
        "shape": [frames, height, width],
        "methods": ["identity", "temporal_smooth"],
        "files": files,
        "sha256": hashes,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out / "fixture_manifest.json").write_bytes(serialized.encode("utf-8"))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args(argv)
    print(json.dumps(generate_fixture(args.out, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
