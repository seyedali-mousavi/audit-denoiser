"""Build and verify a bounded, clean-source audit-core wheel.

The repository contains a historical ``build/`` tree and many scientific
modules that must not leak into the standalone CMPB formal-core distribution.
This builder stages only the declared package inputs in a fresh temporary tree,
sets a reproducible ZIP timestamp, builds without dependency/network access,
and refuses any wheel containing Python modules outside ``audit_denoiser``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "org.calcium-denoiser-audit.release-manifest"
MANIFEST_VERSION = "1.0.0"
REPRODUCIBLE_EPOCH = "315532800"  # 1980-01-01; earliest portable ZIP date.
REQUIRED_PACKAGE_FILES = {
    "__init__.py",
    "__main__.py",
    "adapters.py",
    "cli.py",
    "contracts.py",
    "fixture.py",
    "migration.py",
    "movie_io.py",
    "reference_free.py",
    "run_identity.py",
    "schema.py",
    "taxonomy.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_records(repo_root: Path) -> list[dict[str, Any]]:
    paths = [repo_root / "pyproject.toml", repo_root / "LICENSE", repo_root / "docs" / "AUDIT_DENOISER_CLI.md"]
    paths.extend(sorted((repo_root / "src" / "audit_denoiser").glob("*.py")))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing release inputs: " + ", ".join(str(path) for path in missing))
    return [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


def stage_source(repo_root: Path, stage: Path) -> None:
    (stage / "src").mkdir(parents=True)
    (stage / "docs").mkdir(parents=True)
    shutil.copy2(repo_root / "pyproject.toml", stage / "pyproject.toml")
    shutil.copy2(repo_root / "LICENSE", stage / "LICENSE")
    shutil.copy2(
        repo_root / "docs" / "AUDIT_DENOISER_CLI.md",
        stage / "docs" / "AUDIT_DENOISER_CLI.md",
    )
    shutil.copytree(repo_root / "src" / "audit_denoiser", stage / "src" / "audit_denoiser")


def inspect_wheel(wheel: Path) -> dict[str, Any]:
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        bad_python = [
            name for name in members if name.endswith(".py") and not name.startswith("audit_denoiser/")
        ]
        if bad_python:
            raise RuntimeError(f"wheel contains out-of-scope Python modules: {bad_python[:10]}")
        package_files = {
            Path(name).name
            for name in members
            if name.startswith("audit_denoiser/") and name.endswith(".py")
        }
        missing = sorted(REQUIRED_PACKAGE_FILES - package_files)
        if missing:
            raise RuntimeError(f"wheel is missing required package files: {missing}")
        metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
        entry_names = [name for name in members if name.endswith(".dist-info/entry_points.txt")]
        if len(metadata_names) != 1 or len(entry_names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA and entry_points.txt")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
        entry_points = archive.read(entry_names[0]).decode("utf-8")
        if "audit_denoiser = audit_denoiser.cli:main" not in entry_points:
            raise RuntimeError("audit_denoiser console entry point is missing")
        version_lines = [line for line in metadata.splitlines() if line.startswith("Version: ")]
        if len(version_lines) != 1:
            raise RuntimeError("wheel metadata has no unique Version field")
    return {
        "member_count": len(members),
        "python_file_count": sum(name.endswith(".py") for name in members),
        "out_of_scope_python_files": bad_python,
        "version": version_lines[0].split(": ", 1)[1],
        "members": members,
    }


def build_release(repo_root: Path, output_dir: Path) -> tuple[Path, Path]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"release output directory must be empty: {output_dir}")

    sources = source_records(repo_root)
    with tempfile.TemporaryDirectory(prefix="audit_core_clean_build_") as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / "source"
        wheel_dir = temporary_root / "wheel"
        wheel_dir.mkdir(parents=True)
        stage_source(repo_root, stage)
        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = REPRODUCIBLE_EPOCH
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=stage,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"clean wheel build failed ({completed.returncode}):\n{completed.stdout}")
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {len(wheels)}")
        built = wheels[0]
        inspection = inspect_wheel(built)
        with zipfile.ZipFile(built) as archive:
            licence_members = [name for name in archive.namelist()
                               if name.endswith(".dist-info/licenses/LICENSE")
                               or name.endswith(".dist-info/LICENSE")]
            if len(licence_members) != 1:
                raise RuntimeError("wheel must contain exactly one LICENSE file")
            licence_bytes = archive.read(licence_members[0])
            if licence_bytes != (repo_root / "LICENSE").read_bytes():
                raise RuntimeError("wheel LICENSE bytes differ from the canonical source")
        inspection["licence_member"] = licence_members[0]
        inspection["licence_sha256"] = hashlib.sha256(licence_bytes).hexdigest()
        inspection["licence_matches_source"] = True
        destination = output_dir / built.name
        temporary_destination = output_dir / f".{built.name}.{os.getpid()}.tmp"
        shutil.copy2(built, temporary_destination)
        os.replace(temporary_destination, destination)

    manifest_path = output_dir / "AUDIT_CORE_RELEASE_MANIFEST.json"
    payload = {
        "$schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "build_mode": "fresh-staged-source-no-deps-no-build-isolation",
        "source_date_epoch": REPRODUCIBLE_EPOCH,
        "python": sys.version,
        "sources": sources,
        "wheel": {
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            **inspection,
        },
    }
    atomic_json(manifest_path, payload)
    return destination, manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    wheel, manifest = build_release(args.repo_root, args.out)
    print(wheel)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
