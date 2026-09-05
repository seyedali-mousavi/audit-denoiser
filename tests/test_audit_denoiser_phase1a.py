from __future__ import annotations

import csv
import json
import re
import runpy
from pathlib import Path

import numpy as np
import pytest

from audit_denoiser import __version__
from audit_denoiser.adapters import validate_contract_method_output
from audit_denoiser.cli import main, project_bridge_available
from audit_denoiser.contracts import (
    CanonicalResultEnvelope,
    ContractError,
    DatasetContract,
    MethodAdapterContract,
    assert_metric_admissible,
    compare_dataset_rates,
    metric_admissibility,
    write_json,
)
from audit_denoiser.fixture import generate_fixture
from audit_denoiser.migration import migrate_legacy_verdict_csv
from audit_denoiser.reference_free import run_reference_free_audit
from audit_denoiser.run_identity import (
    DuplicateRunError,
    PartialRunError,
    RunCollisionError,
    RunIdentityInputs,
    SafeRunRegistry,
    StaleOutputError,
    compute_run_id,
    source_tree_identity,
)
from audit_denoiser.taxonomy import (
    CURRENT_TAXONOMY_VERSION,
    LEGACY_TAXONOMY_VERSION,
    migrate_label,
)


def dataset_contract(**overrides) -> DatasetContract:
    values = {
        "dataset_id": "fixture-reference-free",
        "modality": "calcium_movie",
        "axes": "THW",
        "shape": (64, 48, 48),
        "dtype": "float32",
        "units": "normalized_intensity",
        "frame_rate_hz": 30.0,
        "acquisition_id": "fixture-acq-1",
        "independent_unit": "acquisition",
        "nesting": ("acquisition", "frame", "pixel"),
        "reference_availability": "none",
        "evidence_boundary": "reference_free",
        "mask_availability": "none",
        "alignment_state": "not_applicable",
        "provenance": {"source": "deterministic fixture", "seed": 20260730},
    }
    values.update(overrides)
    return DatasetContract(**values)


def method_contract(output_class: str = "full_movie", domains=("reference_free_movie",)) -> MethodAdapterContract:
    return MethodAdapterContract(
        method_id=f"fixture-{output_class}",
        source_identity="src.audit_denoiser.fixture@1.0.0",
        checkpoint_identity=None,
        training_reference_access="none",
        output_class=output_class,
        supported_metric_domains=tuple(domains),
        required_inputs=("movie",),
        produced_outputs=("movie",) if output_class != "trace_only" else ("traces",),
    )


def run_inputs(seed: int = 7) -> RunIdentityInputs:
    return RunIdentityInputs(
        protocol_identity="fixture-protocol-v1",
        dataset_manifest_identity="dataset-sha256",
        method_source_identity="source@commit",
        checkpoint_identity=None,
        config={"frames": 64},
        seed=seed,
        evidence_boundary="reference_free",
        code_identity="code-sha256",
        parent_artifacts={"movie": "parent-sha256"},
    )


def test_c02_positive_dataset_contract_round_trip(tmp_path: Path) -> None:
    contract = dataset_contract().validate()
    path = tmp_path / "dataset.json"
    write_json(contract.to_dict(), path)
    assert DatasetContract.read(path) == contract


def test_c02_missing_metadata_is_explicit() -> None:
    payload = dataset_contract().to_dict()
    del payload["provenance"]
    with pytest.raises(ContractError, match="MISSING_METADATA"):
        DatasetContract.from_dict(payload)


def test_c02_wrong_axes_is_rejected() -> None:
    with pytest.raises(ContractError, match="WRONG_AXES"):
        dataset_contract(axes="TTW").validate()


def test_c02_frame_count_mismatch_is_rejected() -> None:
    with pytest.raises(ContractError, match="FRAME_COUNT_MISMATCH"):
        dataset_contract().validate_array(np.zeros((63, 48, 48), dtype=np.float32))


def test_c02_frame_rate_mismatch_is_rejected() -> None:
    with pytest.raises(ContractError, match="FRAME_RATE_MISMATCH"):
        compare_dataset_rates(dataset_contract(), dataset_contract(frame_rate_hz=20.0))


def test_c02_invalid_mask_shape_is_rejected() -> None:
    with pytest.raises(ContractError, match="INVALID_MASK_SHAPE"):
        dataset_contract().validate_mask(np.zeros((47, 48), dtype=bool))


def test_c02_schema_version_mismatch_is_rejected() -> None:
    payload = dataset_contract().to_dict()
    payload["schema_version"] = "999.0.0"
    with pytest.raises(ContractError, match="SCHEMA_VERSION_MISMATCH"):
        DatasetContract.from_dict(payload)


@pytest.mark.parametrize("output_class", ["full_movie", "component_reconstruction", "trace_only"])
def test_c03_all_method_output_classes_validate(output_class: str) -> None:
    assert method_contract(output_class).validate().output_class == output_class


def test_c03_unsupported_domain_is_withheld_not_failed() -> None:
    status = method_contract("trace_only", ("trace_fidelity",)).domain_status("movie_psnr")
    assert status["status"] == "WITHHELD"


def test_c03_full_movie_and_trace_outputs_validate(tmp_path: Path) -> None:
    generate_fixture(tmp_path)
    movie_status = validate_contract_method_output(method_contract(), tmp_path / "identity.npy", dataset_contract())
    assert movie_status["status"] == "SUPPORTED"
    traces = np.zeros((64, 4), dtype=np.float32)
    np.save(tmp_path / "traces.npy", traces)
    trace_status = validate_contract_method_output(method_contract("trace_only", ("trace_fidelity",)), tmp_path / "traces.npy", dataset_contract())
    assert trace_status["output_class"] == "trace_only"


def test_c03_movie_and_trace_frame_count_mismatch_is_rejected(tmp_path: Path) -> None:
    generate_fixture(tmp_path)
    short_movie = np.load(tmp_path / "identity.npy", allow_pickle=False)[:-1]
    np.save(tmp_path / "short_movie.npy", short_movie, allow_pickle=False)
    with pytest.raises(ContractError, match="FRAME_COUNT_MISMATCH"):
        validate_contract_method_output(method_contract(), tmp_path / "short_movie.npy", dataset_contract())
    np.save(tmp_path / "short_traces.npy", np.zeros((63, 4), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ContractError, match="FRAME_COUNT_MISMATCH"):
        validate_contract_method_output(
            method_contract("trace_only", ("trace_fidelity",)),
            tmp_path / "short_traces.npy",
            dataset_contract(),
        )


def test_c13_corrupt_tiff_is_rejected(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"not-a-tiff")
    with pytest.raises(Exception):
        validate_contract_method_output(method_contract(), corrupt, dataset_contract())


def test_c04_canonical_withheld_result_requires_reason() -> None:
    with pytest.raises(ContractError, match="MISSING_WITHHOLD_REASON"):
        CanonicalResultEnvelope(
            dataset_id="d", method_id="m", run_id="r", evidence_boundary="reference_free",
            independent_unit="acquisition", nesting=("acquisition",), metric_id="psnr_to_clean",
            metric_version="1", units="dB", point_estimate=None, uncertainty=None,
            admissibility="WITHHELD", warnings=(), taxonomy_version=CURRENT_TAXONOMY_VERSION,
            provenance_parents=("x",), hashes={"x": "y"},
        ).validate()


def test_c04_nonfinite_admissible_and_valued_withheld_results_are_rejected() -> None:
    common = dict(
        dataset_id="d", method_id="m", run_id="r", evidence_boundary="reference_free",
        independent_unit="acquisition", nesting=("acquisition",), metric_id="finite_fraction",
        metric_version="1", units="ratio", uncertainty=None,
        taxonomy_version=CURRENT_TAXONOMY_VERSION, provenance_parents=("x",), hashes={"x": "y"},
    )
    with pytest.raises(ContractError, match="NONFINITE_POINT_ESTIMATE"):
        CanonicalResultEnvelope(
            **common, point_estimate=float("nan"), admissibility="ADMISSIBLE", warnings=(),
        ).validate()
    with pytest.raises(ContractError, match="INVALID_WITHHELD_VALUE"):
        CanonicalResultEnvelope(
            **common, point_estimate=1.0, admissibility="WITHHELD", warnings=("not available",),
        ).validate()


def test_c04_legacy_migration_preserves_exact_numeric_lexemes(tmp_path: Path) -> None:
    source = tmp_path / "legacy.csv"
    source.write_text("delta_psnr,amp_pres,hf_ratio\n4.196471283555951,0.5813356041908264,0.3059423768949287\n", encoding="utf-8")
    envelopes, lexemes = migrate_legacy_verdict_csv(
        source,
        dataset_id="fov1",
        method_id="cnmf",
        run_id="legacy",
        dataset_contract_sha256="1" * 64,
        clean_reference_sha256="2" * 64,
    )
    round_trip = {item.metric_id: format(item.point_estimate, ".17g") for item in envelopes}
    assert float(round_trip["delta_psnr_vs_blind_svd"]) == float(lexemes["delta_psnr_vs_blind_svd"])
    assert float(round_trip["amp_pres_to_clean"]) == float(lexemes["amp_pres_to_clean"])
    assert all(item.validate() for item in envelopes)


def test_c06_incompatible_clean_metrics_are_withheld() -> None:
    assert metric_admissibility("psnr_to_clean", "reference_free")[0] == "WITHHELD"
    with pytest.raises(ContractError, match="INCOMPATIBLE_EVIDENCE_BOUNDARY"):
        assert_metric_admissible("amp_pres_to_clean", "reference_free")


def test_c06_reference_free_lane_never_requires_dummy_clean(tmp_path: Path) -> None:
    generate_fixture(tmp_path)
    results = run_reference_free_audit(
        movie_path=tmp_path / "identity.npy",
        dataset=dataset_contract(),
        method=method_contract(),
        run_id="fixture-run",
        max_frames=64,
    )
    admissible = {r.metric_id for r in results if r.admissibility == "ADMISSIBLE"}
    withheld = {r.metric_id for r in results if r.admissibility == "WITHHELD"}
    assert "split_half_mean_map_correlation" in admissible
    assert {"psnr_to_clean", "ssim_to_clean", "amp_pres_to_clean", "clean_derived_trace_fidelity"} <= withheld


def test_c06_undefined_reference_free_metrics_are_withheld_as_strict_json(tmp_path: Path) -> None:
    movie = np.zeros((64, 48, 48), dtype=np.float32)
    np.save(tmp_path / "constant.npy", movie, allow_pickle=False)
    results = run_reference_free_audit(
        movie_path=tmp_path / "constant.npy",
        dataset=dataset_contract(),
        method=method_contract(),
        run_id="constant-run",
        max_frames=64,
    )
    split_half = next(item for item in results if item.metric_id == "split_half_mean_map_correlation")
    assert split_half.admissibility == "WITHHELD" and split_half.point_estimate is None
    json.dumps([item.to_dict() for item in results], allow_nan=False)


def test_c06_reference_free_cli_creates_exact_safe_run(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    generate_fixture(fixture)
    dataset_path, method_path = tmp_path / "dataset.json", tmp_path / "method.json"
    write_json(dataset_contract().to_dict(), dataset_path)
    write_json(method_contract().to_dict(), method_path)
    out = tmp_path / "out"
    assert main([
        "reference-free", "--name", "fixture", "--movie", str(fixture / "identity.npy"),
        "--dataset-contract", str(dataset_path), "--method-contract", str(method_path),
        "--out", str(out), "--max-frames", "64",
    ]) == 0
    runs = list((out / "runs").iterdir())
    assert len(runs) == 1
    assert SafeRunRegistry(out / "runs").inspect(runs[0]) == "COMPLETE"
    payload = json.loads((runs[0] / "reference_free_results.json").read_text(encoding="utf-8"))
    assert all(row["metric_id"] != "dummy_reference" for row in payload)


def test_c06_external_style_trace_output_validates_without_core_registration(tmp_path: Path) -> None:
    example = Path(__file__).parents[1] / "examples" / "audit_core_external_adapter"
    namespace = runpy.run_path(str(example / "emit_trace_method.py"), run_name="external_style_fixture")
    output = namespace["emit"](tmp_path / "external")
    out = tmp_path / "audit"
    assert main([
        "validate-output", "--name", "external-trace-example", "--output", str(output),
        "--dataset-contract", str(example / "dataset_contract.json"),
        "--method-contract", str(example / "method_contract.json"),
        "--out", str(out),
    ]) == 0
    runs = list((out / "runs").iterdir())
    assert len(runs) == 1 and SafeRunRegistry(out / "runs").inspect(runs[0]) == "COMPLETE"
    payload = json.loads((runs[0] / "validation_result.json").read_text(encoding="utf-8"))
    assert payload["validation"]["output_class"] == "trace_only"
    assert payload["declared_metric_domains"]["trace_fidelity"]["status"] == "SUPPORTED"


def test_c07_run_id_is_deterministic_and_seed_sensitive() -> None:
    assert compute_run_id(run_inputs()) == compute_run_id(run_inputs())
    assert compute_run_id(run_inputs(7)) != compute_run_id(run_inputs(8))


def test_c07_code_identity_covers_every_python_source(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "b.py"
    second.write_text("VALUE = 2\n", encoding="utf-8")
    first_identity = source_tree_identity(tmp_path)
    second.write_text("VALUE = 3\n", encoding="utf-8")
    assert source_tree_identity(tmp_path) != first_identity


def test_c07_package_metadata_version_matches_runtime() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    project_block = pyproject.split("[project]", 1)[1].split("[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project_block, flags=re.MULTILINE)
    assert match is not None and match.group(1) == __version__


def test_c07_distribution_scope_is_bounded_to_audit_package() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'where = ["src"]' in pyproject
    assert 'include = ["audit_denoiser*"]' in pyproject
    assert 'audit_denoiser = "audit_denoiser.cli:main"' in pyproject


def test_c07_project_bridge_availability_is_explicit() -> None:
    assert not project_bridge_available("src.tools.run_external_baseline")
    assert not project_bridge_available("package_that_does_not_exist.module")


def test_c08_registry_nonoverwrite_duplicate_and_exact_manifest(tmp_path: Path) -> None:
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(run_inputs(), ["metrics.json"])
    (run / "metrics.json").write_text("{}\n", encoding="utf-8")
    output_manifest = registry.complete(run)
    assert output_manifest["outputs"][0]["path"] == "metrics.json"
    assert registry.inspect(run) == "COMPLETE"
    with pytest.raises(DuplicateRunError, match="DUPLICATE_RUN_ID"):
        registry.prepare(run_inputs(), ["metrics.json"])


def test_c08_partial_and_interrupted_runs_are_explicit(tmp_path: Path) -> None:
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(run_inputs(), ["metrics.json"])
    assert registry.inspect(run) == "PARTIAL"
    registry.mark_interrupted(run, "fixture interruption")
    assert registry.inspect(run) == "INTERRUPTED"
    with pytest.raises(PartialRunError, match="PARTIAL_RUN"):
        registry.require_complete(run)


def test_c08_stale_output_is_detected(tmp_path: Path) -> None:
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(run_inputs(), ["metrics.json"])
    target = run / "metrics.json"
    target.write_text("first\n", encoding="utf-8")
    registry.complete(run)
    target.write_text("changed\n", encoding="utf-8")
    assert registry.inspect(run) == "STALE"
    with pytest.raises(StaleOutputError, match="STALE_OUTPUT"):
        registry.require_complete(run)


def test_c08_tampered_output_manifest_identity_and_size_are_stale(tmp_path: Path) -> None:
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(run_inputs(), ["metrics.json"])
    target = run / "metrics.json"
    target.write_text("first\n", encoding="utf-8")
    registry.complete(run)
    output_manifest_path = run / "OUTPUT_MANIFEST.json"
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    output_manifest["run_id"] = "run_wrong"
    output_manifest_path.write_text(json.dumps(output_manifest), encoding="utf-8")
    assert registry.inspect(run) == "STALE"

    output_manifest["run_id"] = run.name
    output_manifest["outputs"][0]["bytes"] += 1
    output_manifest_path.write_text(json.dumps(output_manifest), encoding="utf-8")
    assert registry.inspect(run) == "STALE"


def test_c08_collision_is_rejected(tmp_path: Path) -> None:
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(run_inputs(), ["metrics.json"])
    manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["identity_inputs"]["seed"] = 999
    (run / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunCollisionError, match="RUN_ID_COLLISION"):
        registry.prepare(run_inputs(), ["metrics.json"])


def test_c16_identity_migration_is_exact() -> None:
    result = migrate_label("AMPLITUDE_COMPRESSION", CURRENT_TAXONOMY_VERSION)
    assert result.status == "EXACT" and result.target_label == "AMPLITUDE_COMPRESSION"


def test_c16_legacy_labels_are_never_silently_renamed() -> None:
    assert migrate_label("OVERSMOOTH", LEGACY_TAXONOMY_VERSION).status == "AMBIGUOUS"
    assert migrate_label("LOSE", LEGACY_TAXONOMY_VERSION).status == "REQUIRES_REVIEW"


def test_c13_unsupported_domain_request_is_deterministically_withheld() -> None:
    result = method_contract("trace_only", ("trace_fidelity",)).domain_status("reference_free_movie")
    assert result == {
        "status": "WITHHELD",
        "domain": "reference_free_movie",
        "reason": "trace_only output does not declare this metric domain",
    }
