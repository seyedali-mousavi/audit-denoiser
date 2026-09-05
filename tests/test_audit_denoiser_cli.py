from __future__ import annotations

from pathlib import Path

import json

import numpy as np

from audit_denoiser.adapters import validate_adapters
from audit_denoiser.cli import MODULES, main, parser
from audit_denoiser.fixture import generate_fixture
from audit_denoiser.schema import RESULT_SCHEMA_VERSION, command_manifest


def test_all_required_commands_are_registered() -> None:
    assert set(MODULES) == {"evaluate", "residuals", "traces", "amplitude", "seams", "verdict"}
    for command in MODULES:
        try:
            parser().parse_args([command, "--help"])
        except SystemExit as exc:
            assert exc.code == 0


def test_report_indexes_outputs_without_modifying_them(tmp_path: Path) -> None:
    source = tmp_path / "external_svd_verdict_demo.json"
    source.write_text('{"verdict":"WIN","frames":256}', encoding="utf-8")
    before = source.read_bytes()
    assert main(["report", "--name", "demo", "--out", str(tmp_path)]) == 0
    assert source.read_bytes() == before
    report = (tmp_path / "audit_report_demo.md").read_text(encoding="utf-8")
    assert "verdict: `WIN`" in report
    assert "frames: `256`" in report


def test_explicit_adapters_validate_fixture_inputs(tmp_path: Path) -> None:
    generate_fixture(tmp_path)
    validate_adapters(
        dataset_adapter="movie-io-v1",
        method_adapter="identity-fixture-v1",
        raw=tmp_path / "raw.npy",
        reference=tmp_path / "clean.npy",
        denoised=tmp_path / "identity.npy",
    )


def test_fixture_is_deterministic_and_contains_two_methods(tmp_path: Path) -> None:
    first = generate_fixture(tmp_path / "first")
    second = generate_fixture(tmp_path / "second")
    assert first["sha256"] == second["sha256"]
    assert first["methods"] == ["identity", "temporal_smooth"]
    for name in ("clean", "raw", "identity", "temporal_smooth"):
        array = np.load(tmp_path / "first" / f"{name}.npy")
        assert array.shape == (64, 48, 48)


def test_versioned_command_manifest_hashes_outputs(tmp_path: Path) -> None:
    output = tmp_path / "external_demo.csv"
    output.write_text("metric,value\npsnr,1.0\n", encoding="utf-8")
    payload = command_manifest(
        command="evaluate",
        name="demo",
        arguments={"name": "demo", "out": tmp_path},
        dataset_adapter="movie-io-v1",
        method_adapter="external-movie-v1",
        exit_code=0,
        artifacts=[output],
        output_root=tmp_path,
        package_version="1.0.0",
    )
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["status"] == "PASS"
    assert payload["artifacts"][0]["path"] == "external_demo.csv"
    assert len(payload["artifacts"][0]["sha256"]) == 64
    json.dumps(payload)
