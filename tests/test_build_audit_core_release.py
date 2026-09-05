from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools.build_audit_core_release import build_release, sha256


ROOT = Path(__file__).parents[1]


def test_clean_release_wheel_is_bounded_and_reproducible(tmp_path: Path) -> None:
    first_wheel, first_manifest_path = build_release(ROOT, tmp_path / "first")
    second_wheel, second_manifest_path = build_release(ROOT, tmp_path / "second")
    first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))

    assert sha256(first_wheel) == sha256(second_wheel)
    assert first_manifest["wheel"]["sha256"] == second_manifest["wheel"]["sha256"]
    assert first_manifest["wheel"]["out_of_scope_python_files"] == []
    assert first_manifest["wheel"]["python_file_count"] == 12

    with zipfile.ZipFile(first_wheel) as archive:
        names = archive.namelist()
    assert all(not name.endswith(".py") or name.startswith("audit_denoiser/") for name in names)
