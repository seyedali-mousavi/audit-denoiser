"""Versioned envelopes for the reusable denoiser-audit command surface.

The numerical tools keep their historical CSV/JSON layouts so existing evidence
and downstream readers remain byte-compatible.  Every command invocation also
writes one small, versioned manifest that identifies adapters, arguments, exit
status, and hashes of the produced artifacts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


RESULT_SCHEMA_ID = "org.calcium-denoiser-audit.command-manifest"
RESULT_SCHEMA_VERSION = "1.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_records(paths: Iterable[Path], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted({item.resolve() for item in paths if item.is_file()}):
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = path.name
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def command_manifest(
    *,
    command: str,
    name: str,
    arguments: dict[str, Any],
    dataset_adapter: str,
    method_adapter: str,
    exit_code: int,
    artifacts: Iterable[Path],
    output_root: Path,
    package_version: str,
) -> dict[str, Any]:
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in arguments.items()
        if key not in {"dataset_adapter", "method_adapter"}
    }
    return {
        "$schema": RESULT_SCHEMA_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "package_version": package_version,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "name": name,
        "dataset_adapter": dataset_adapter,
        "method_adapter": method_adapter,
        "arguments": serializable_args,
        "exit_code": int(exit_code),
        "status": "PASS" if exit_code == 0 else "FAIL",
        "artifacts": artifact_records(artifacts, output_root),
    }


def write_manifest(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_bytes(serialized.encode("utf-8"))
