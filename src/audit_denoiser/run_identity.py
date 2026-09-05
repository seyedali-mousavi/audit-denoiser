"""Immutable run identity and exact-output safety registry."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .contracts import ContractError, EVIDENCE_BOUNDARIES
from .schema import sha256


RUN_ID_SCHEMA_VERSION = "immutable-run-id-v1"


class DuplicateRunError(ContractError):
    def __init__(self, message: str):
        super().__init__("DUPLICATE_RUN_ID", message)


class RunCollisionError(ContractError):
    def __init__(self, message: str):
        super().__init__("RUN_ID_COLLISION", message)


class PartialRunError(ContractError):
    def __init__(self, message: str):
        super().__init__("PARTIAL_RUN", message)


class StaleOutputError(ContractError):
    def __init__(self, message: str):
        super().__init__("STALE_OUTPUT", message)


@dataclass(frozen=True)
class RunIdentityInputs:
    protocol_identity: str
    dataset_manifest_identity: str
    method_source_identity: str
    checkpoint_identity: str | None
    config: dict[str, Any]
    seed: int
    evidence_boundary: str
    code_identity: str
    parent_artifacts: dict[str, str] = field(default_factory=dict)
    schema_version: str = RUN_ID_SCHEMA_VERSION

    def canonical_payload(self) -> dict[str, Any]:
        if self.schema_version != RUN_ID_SCHEMA_VERSION:
            raise ContractError("SCHEMA_VERSION_MISMATCH", "unsupported run identity version")
        for name in ("protocol_identity", "dataset_manifest_identity", "method_source_identity", "code_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractError("MISSING_METADATA", f"{name} must be non-empty text")
        if self.checkpoint_identity is not None and (
            not isinstance(self.checkpoint_identity, str) or not self.checkpoint_identity.strip()
        ):
            raise ContractError("MISSING_METADATA", "checkpoint_identity must be text or null")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ContractError("INVALID_SEED", "seed must be an integer")
        if self.evidence_boundary not in EVIDENCE_BOUNDARIES:
            raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", self.evidence_boundary)
        if not isinstance(self.config, dict) or not isinstance(self.parent_artifacts, dict):
            raise ContractError("INVALID_RUN_IDENTITY", "config and parent_artifacts must be objects")
        # Match the JSON types stored on disk; tuples must not spuriously collide
        # with their list-valued representation when a run is inspected or resumed.
        return json.loads(canonical_json(asdict(self)))


def canonical_json(payload: dict[str, Any]) -> str:
    def check_keys(value: Any) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ContractError("INVALID_RUN_IDENTITY", "JSON object keys must be strings")
            for child in value.values():
                check_keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_keys(child)

    check_keys(payload)
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError("INVALID_RUN_IDENTITY", "identity must contain finite JSON-compatible values") from exc


def compute_run_id(inputs: RunIdentityInputs) -> str:
    digest = hashlib.sha256(canonical_json(inputs.canonical_payload()).encode("utf-8")).hexdigest()
    return f"run_{digest[:24]}"


def source_tree_identity(root: Path) -> str:
    """Hash every Python source path and byte stream below ``root`` deterministically."""
    root = Path(root).resolve()
    records = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
        for path in sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file()
    ]
    if not records:
        raise ContractError("EMPTY_CODE_IDENTITY", f"no Python sources found under {root}")
    digest = hashlib.sha256(canonical_json({"sources": records}).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_bytes(serialized.encode("utf-8"))
    os.replace(temporary, path)


class SafeRunRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _output_name(value: str) -> str:
        """One portable relative path, never a registry control file or traversal."""
        if not isinstance(value, str) or not value.strip():
            raise ContractError("INVALID_OUTPUT_PATH", "output path must be non-empty text")
        posix = value.replace("\\", "/")
        parts = posix.split("/")
        reserved = {"run_manifest.json", "output_manifest.json", "complete", "interrupted.json"}
        windows_devices = {"con", "prn", "aux", "nul"} | {
            f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
        }
        if (PurePosixPath(posix).is_absolute() or PureWindowsPath(value).drive
                or any(part in {"", ".", ".."} for part in parts)
                or any(part.endswith((" ", ".")) for part in parts)
                or any(part.split(".", 1)[0].casefold() in windows_devices for part in parts)
                or any(c in value for c in '<>:"|?*')
                or any(ord(c) < 32 for c in value) or parts[0].casefold() in reserved):
            raise ContractError("INVALID_OUTPUT_PATH", value)
        return posix

    @classmethod
    def _expected_names(cls, outputs: list[str]) -> list[str]:
        if not isinstance(outputs, list) or not outputs:
            raise ContractError("MISSING_EXPECTED_OUTPUT", "at least one expected output is required")
        names = [cls._output_name(item) for item in outputs]
        if len({name.casefold() for name in names}) != len(names):
            raise ContractError("DUPLICATE_EXPECTED_OUTPUT", "expected output paths must be unique")
        return sorted(names)

    @classmethod
    def run_id_for(cls, inputs: RunIdentityInputs, expected_outputs: list[str]) -> str:
        """Bind registry identity to both computational inputs and output contract."""
        names = cls._expected_names(expected_outputs)
        payload = {
            "identity_inputs": inputs.canonical_payload(),
            "expected_outputs": names,
        }
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        return f"run_{digest[:24]}"

    def _manifest(self, run_dir: Path) -> dict[str, Any]:
        if run_dir.resolve().parent != self.root.resolve():
            raise ContractError("INVALID_RUN_PATH", "run must be directly inside this registry")
        manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        inputs = RunIdentityInputs(**manifest["identity_inputs"])
        expected_id = self.run_id_for(inputs, manifest["expected_outputs"])
        if (manifest.get("schema_version") != RUN_ID_SCHEMA_VERSION
                or manifest.get("run_id") != expected_id or run_dir.name != expected_id
                or manifest.get("status") not in {"RUNNING", "COMPLETE"}):
            raise StaleOutputError("run manifest identity or state is inconsistent")
        expected_names = self._expected_names(manifest["expected_outputs"])
        if expected_names != manifest["expected_outputs"]:
            raise StaleOutputError("output path list is not canonical")
        return manifest

    @staticmethod
    def _contained_file(run_dir: Path, relative: str) -> Path:
        path = run_dir / relative
        if not path.resolve().is_relative_to(run_dir.resolve()):
            raise ContractError("INVALID_OUTPUT_PATH", "output resolves outside its run")
        return path

    def prepare(self, inputs: RunIdentityInputs, expected_outputs: list[str]) -> Path:
        payload = inputs.canonical_payload()
        clean_outputs = self._expected_names(expected_outputs)
        run_id = self.run_id_for(inputs, clean_outputs)
        run_dir = self.root / run_id
        if run_dir.exists():
            manifest_path = run_dir / "RUN_MANIFEST.json"
            if not manifest_path.is_file():
                raise RunCollisionError(f"{run_id} exists without an exact manifest")
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("identity_inputs") != payload:
                raise RunCollisionError(f"{run_id} manifest does not match requested identity")
            if (run_dir / "COMPLETE").is_file():
                raise DuplicateRunError(f"{run_id} already completed; overwrite is forbidden")
            raise PartialRunError(f"{run_id} exists in state {existing.get('status', 'UNKNOWN')}")
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(
            run_dir / "RUN_MANIFEST.json",
            {
                "schema_version": RUN_ID_SCHEMA_VERSION,
                "run_id": run_id,
                "status": "RUNNING",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "identity_inputs": payload,
                "expected_outputs": sorted(clean_outputs),
            },
        )
        return run_dir

    def mark_interrupted(self, run_dir: Path, reason: str) -> None:
        run_dir = Path(run_dir)
        manifest = self._manifest(run_dir)
        if manifest["status"] == "COMPLETE" or (run_dir / "COMPLETE").exists():
            raise DuplicateRunError("completed runs cannot be marked interrupted")
        atomic_json(
            Path(run_dir) / "INTERRUPTED.json",
            {"status": "INTERRUPTED", "reason": reason, "utc": datetime.now(timezone.utc).isoformat()},
        )

    def complete(self, run_dir: Path) -> dict[str, Any]:
        run_dir = Path(run_dir)
        manifest_path = run_dir / "RUN_MANIFEST.json"
        if not manifest_path.is_file():
            raise PartialRunError("RUN_MANIFEST.json is missing")
        manifest = self._manifest(run_dir)
        if manifest["status"] == "COMPLETE" or (run_dir / "COMPLETE").exists():
            raise DuplicateRunError("completion is immutable; overwrite is forbidden")
        if (run_dir / "OUTPUT_MANIFEST.json").exists():
            raise PartialRunError("an output manifest already exists; explicit recovery is required")
        records = []
        for rel in manifest["expected_outputs"]:
            path = self._contained_file(run_dir, rel)
            if not path.is_file():
                raise PartialRunError(f"expected output is missing: {rel}")
            records.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
        output_manifest = {
            "schema_version": "exact-output-manifest-v1",
            "run_id": manifest["run_id"],
            "outputs": records,
        }
        atomic_json(run_dir / "OUTPUT_MANIFEST.json", output_manifest)
        marker_tmp = run_dir / f".COMPLETE.{os.getpid()}.tmp"
        marker_tmp.write_bytes((manifest["run_id"] + "\n").encode("utf-8"))
        os.replace(marker_tmp, run_dir / "COMPLETE")
        manifest["status"] = "COMPLETE"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        return output_manifest

    def inspect(self, run_dir: Path) -> str:
        run_dir = Path(run_dir)
        manifest_path = run_dir / "RUN_MANIFEST.json"
        if not manifest_path.is_file():
            return "COLLISION"
        try:
            manifest = self._manifest(run_dir)
        except (ContractError, ValueError, TypeError, KeyError, OSError):
            return "STALE"
        if not (run_dir / "COMPLETE").is_file():
            return "INTERRUPTED" if (run_dir / "INTERRUPTED.json").is_file() else "PARTIAL"
        if manifest["status"] != "COMPLETE":
            return "PARTIAL"
        try:
            if (run_dir / "COMPLETE").read_text(encoding="utf-8") != manifest["run_id"] + "\n":
                return "STALE"
        except (OSError, UnicodeError):
            return "STALE"
        output_manifest_path = run_dir / "OUTPUT_MANIFEST.json"
        if not output_manifest_path.is_file():
            return "PARTIAL"
        try:
            output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return "STALE"
        if not isinstance(output_manifest, dict) or output_manifest.get("schema_version") != "exact-output-manifest-v1":
            return "STALE"
        if output_manifest.get("run_id") != manifest.get("run_id"):
            return "STALE"
        records = output_manifest.get("outputs")
        if not isinstance(records, list):
            return "STALE"
        for record in records:
            if not isinstance(record, dict) or not {"path", "bytes", "sha256"} <= set(record):
                return "STALE"
            try:
                relative = self._output_name(record["path"])
                path = self._contained_file(run_dir, relative)
            except (ContractError, OSError, ValueError):
                return "STALE"
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]
            ):
                return "STALE"
        if sorted(r["path"] for r in records) != sorted(manifest["expected_outputs"]):
            return "STALE"
        return "COMPLETE"

    def require_complete(self, run_dir: Path) -> None:
        status = self.inspect(run_dir)
        if status == "STALE":
            raise StaleOutputError(str(run_dir))
        if status != "COMPLETE":
            raise PartialRunError(f"expected COMPLETE, found {status}")
