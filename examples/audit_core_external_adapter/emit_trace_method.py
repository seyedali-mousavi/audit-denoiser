"""Emit deterministic trace-only output through an external-style interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def emit(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    time = np.arange(64, dtype=np.float32)[:, None]
    rates = np.asarray([0.07, 0.11, 0.17, 0.23], dtype=np.float32)[None, :]
    traces = (0.5 + 0.25 * np.sin(time * rates)).astype(np.float32)
    path = out / "traces.npy"
    np.save(path, traces, allow_pickle=False)
    (out / "external_method_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "external-style-fixture-v1",
                "output": path.name,
                "shape": list(traces.shape),
                "dtype": str(traces.dtype),
                "scientific_evidence": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(emit(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
