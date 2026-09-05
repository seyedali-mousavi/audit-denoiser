"""Focused regressions for non-equivalent survivors in the first B4 campaign."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from audit_denoiser.contracts import (
    CanonicalResultEnvelope, ContractError, DatasetContract, MethodAdapterContract,
    write_json,
)
from audit_denoiser.cli import write_report
from audit_denoiser.fixture import generate_fixture
from audit_denoiser.reference_free import write_reference_free_results
from audit_denoiser.run_identity import (
    RUN_ID_SCHEMA_VERSION, RunIdentityInputs, SafeRunRegistry, atomic_json,
    canonical_json, compute_run_id,
)
from audit_denoiser.schema import write_manifest
from audit_denoiser.taxonomy import CURRENT_TAXONOMY_VERSION


def dataset():
    return DatasetContract("fixture", "trace", "TC", (8, 2), "float32", "uV", 160.0,
        "acq", "subject", ("subject", "channel"), "none", "reference_free",
        "none", "not_applicable", {"source": "generated fixture"})


def method():
    return MethodAdapterContract("identity", "source", None, "none", "trace_only",
        ("finite_fraction",), (), ("trace",))


def envelope():
    return CanonicalResultEnvelope("fixture", "identity", "run_fixture", "reference_free",
        "subject", ("subject",), "finite_fraction", "1", "fraction", 1.0, None,
        "ADMISSIBLE", (), CURRENT_TAXONOMY_VERSION, ("input",), {"input": "hash"})


def identity():
    return RunIdentityInputs("protocol", "dataset", "method", None, {"alpha": 1}, 7,
        "reference_free", "code", {"input": "hash"})


@pytest.mark.parametrize("factory", [dataset, method, envelope])
@pytest.mark.parametrize("version", ["0.0.0", "z.0.0"])
def test_versions_reject_both_lexical_directions(factory, version):
    with pytest.raises(ContractError, match="SCHEMA_VERSION_MISMATCH"):
        replace(factory(), schema_version=version).validate()


@pytest.mark.parametrize("factory", [dataset, method, envelope])
@pytest.mark.parametrize("schema", ["aaa", "zzz"])
def test_schema_ids_reject_both_lexical_directions(factory, schema):
    with pytest.raises(ContractError, match="SCHEMA_ID_MISMATCH"):
        replace(factory(), schema_id=schema).validate()


@pytest.mark.parametrize("field", [
    "dataset_id", "modality", "axes", "dtype", "units", "acquisition_id",
    "independent_unit", "mask_availability",
])
@pytest.mark.parametrize("bad", ["", 3])
def test_every_dataset_text_field_is_validated(field, bad):
    with pytest.raises(ContractError):
        replace(dataset(), **{field: bad}).validate()


@pytest.mark.parametrize("shape", [(), (0, 2), (-1, 2), "8x2"])
def test_shape_strictly_positive_nonempty_sequence(shape):
    with pytest.raises(ContractError):
        replace(dataset(), shape=shape).validate()


@pytest.mark.parametrize("nesting", [(), ("subject", ""), ("subject", 3), ("z",)])
def test_nesting_nonempty_typed_and_begins_with_unit(nesting):
    with pytest.raises(ContractError):
        replace(dataset(), nesting=nesting).validate()


@pytest.mark.parametrize("factory", [dataset, method, envelope])
def test_contract_objects_are_immutable(factory):
    value = factory()
    with pytest.raises(FrozenInstanceError):
        value.schema_version = "changed"


@pytest.mark.parametrize("field", ["supported_metric_domains", "produced_outputs"])
def test_required_method_sequences_cannot_be_empty(field):
    with pytest.raises(ContractError):
        replace(method(), **{field: ()}).validate()


def test_required_inputs_may_be_empty_by_design():
    assert method().validate().required_inputs == ()


@pytest.mark.parametrize("field", [
    "protocol_identity", "dataset_manifest_identity", "method_source_identity", "code_identity",
])
@pytest.mark.parametrize("bad", ["", 3])
def test_every_run_identity_text_field_is_validated(field, bad):
    with pytest.raises(ContractError, match="MISSING_METADATA"):
        replace(identity(), **{field: bad}).canonical_payload()


@pytest.mark.parametrize("version", ["aaa", "zzz"])
def test_run_identity_version_rejects_both_lexical_directions(version):
    with pytest.raises(ContractError, match="SCHEMA_VERSION_MISMATCH"):
        replace(identity(), schema_version=version).canonical_payload()


@pytest.mark.parametrize("checkpoint", ["", 3, True])
def test_checkpoint_identity_is_null_or_nonblank_text(checkpoint):
    with pytest.raises(ContractError, match="MISSING_METADATA"):
        replace(identity(), checkpoint_identity=checkpoint).canonical_payload()


def test_run_id_has_exact_prefix_and_24_hex_digits():
    run_id = compute_run_id(identity())
    assert run_id.startswith("run_") and len(run_id) == 28
    assert int(run_id[4:], 16) >= 0


def test_canonical_identity_json_has_stable_order_and_spacing():
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_atomic_json_has_stable_bytes(tmp_path):
    output = tmp_path / "value.json"
    atomic_json(output, {"b": 2, "a": 1})
    assert output.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'


def test_contract_json_has_stable_bytes(tmp_path):
    output = tmp_path / "value.json"
    write_json({"b": 2, "a": 1}, output)
    assert output.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'


def test_command_manifest_has_stable_bytes(tmp_path):
    output = tmp_path / "manifest.json"
    write_manifest({"b": 2, "a": 1}, output)
    assert output.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'


def test_fixture_manifest_uses_only_lf(tmp_path):
    generate_fixture(tmp_path)
    payload = (tmp_path / "fixture_manifest.json").read_bytes()
    assert payload.endswith(b"\n") and b"\r" not in payload


def test_reference_free_results_use_only_lf(tmp_path):
    output = tmp_path / "results.json"
    write_reference_free_results([envelope()], output)
    payload = output.read_bytes()
    assert payload.endswith(b"\n") and b"\r" not in payload


def test_text_report_uses_only_lf(tmp_path):
    (tmp_path / "demo.json").write_bytes(b'{"status":"PASS"}\n')
    assert write_report("demo", tmp_path) == 0
    payload = (tmp_path / "audit_report_demo.md").read_bytes()
    assert payload.endswith(b"\n") and b"\r" not in payload


def test_complete_marker_has_stable_bytes(tmp_path):
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(identity(), ["value.bin"])
    (run / "value.bin").write_bytes(b"value")
    registry.complete(run)
    assert (run / "COMPLETE").read_bytes() == (run.name + "\n").encode("ascii")


@pytest.mark.parametrize("path", [
    "/absolute", "C:relative", "C:/absolute", "../escape", "nested/../escape",
    "bad\x00name", "RUN_MANIFEST.json", "OUTPUT_MANIFEST.json", "COMPLETE",
])
def test_portable_output_paths_reject_each_safety_class(tmp_path, path):
    with pytest.raises(ContractError, match="INVALID_OUTPUT_PATH"):
        SafeRunRegistry(tmp_path).prepare(identity(), [path])


def test_space_is_valid_in_portable_output_filename(tmp_path):
    run = SafeRunRegistry(tmp_path).prepare(identity(), ["metric values.json"])
    assert json.loads((run / "RUN_MANIFEST.json").read_text())["expected_outputs"] == ["metric values.json"]
