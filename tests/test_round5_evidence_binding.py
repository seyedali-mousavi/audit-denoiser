"""Round-5 regressions for evidence binding and survivor-derived safety gaps."""

from dataclasses import replace
import inspect
import json

import pytest

from audit_denoiser.contracts import (
    CanonicalResultEnvelope,
    ContractError,
    DatasetContract,
    MethodAdapterContract,
    _text_sequence,
    compare_dataset_rates,
    validate_contract_collection,
    write_json,
)
from audit_denoiser.run_identity import RunIdentityInputs, SafeRunRegistry, canonical_json
from audit_denoiser.schema import sha256
from audit_denoiser.taxonomy import CURRENT_TAXONOMY_VERSION


def clean_dataset(clean_reference_sha256: str = "2" * 64) -> DatasetContract:
    return DatasetContract(
        dataset_id="clean-fixture",
        modality="movie",
        axes="THW",
        shape=(2, 2, 2),
        dtype="float32",
        units="arbitrary",
        frame_rate_hz=30.0,
        acquisition_id="fixture-acquisition",
        independent_unit="scene",
        nesting=("scene", "frame", "pixel"),
        reference_availability="exact_clean",
        evidence_boundary="clean_reference",
        mask_availability="none",
        alignment_state="spatiotemporal",
        provenance={"source": "generated software fixture", "clean_reference_sha256": clean_reference_sha256},
    ).validate()


def clean_envelope(hashes: dict[str, str]) -> CanonicalResultEnvelope:
    return CanonicalResultEnvelope(
        dataset_id="clean-fixture",
        method_id="identity",
        run_id="run_fixture",
        evidence_boundary="clean_reference",
        independent_unit="scene",
        nesting=("scene", "roi"),
        metric_id="psnr_to_clean",
        metric_version="fixture-v1",
        units="dB",
        point_estimate=30.0,
        uncertainty=None,
        admissibility="ADMISSIBLE",
        warnings=(),
        taxonomy_version=CURRENT_TAXONOMY_VERSION,
        provenance_parents=("clean-fixture",),
        hashes=hashes,
    )


def identity() -> RunIdentityInputs:
    return RunIdentityInputs(
        "protocol", "dataset", "method", None, {"alpha": 1}, 7,
        "reference_free", "code", {"input": "hash"},
    )


def method() -> MethodAdapterContract:
    return MethodAdapterContract(
        "identity", "fixture-source", None, "none", "trace_only",
        ("finite_fraction",), (), ("trace",),
    ).validate()


@pytest.mark.parametrize(
    "hashes",
    [
        {},
        {"dataset_contract_sha256": "1" * 64},
        {"dataset_contract_sha256": "not-a-digest", "clean_reference_sha256": "2" * 64},
        {"dataset_contract_sha256": "1" * 64, "clean_reference_sha256": "2" * 63},
    ],
)
def test_direct_clean_metric_cannot_be_admissible_without_explicit_evidence_digests(hashes):
    with pytest.raises(ContractError, match="MISSING_EVIDENCE_BINDING"):
        clean_envelope(hashes).validate()


def test_clean_metric_binds_to_specific_dataset_contract(tmp_path):
    dataset = clean_dataset()
    contract_path = tmp_path / "dataset-contract.json"
    write_json(dataset.to_dict(), contract_path)
    contract_digest = sha256(contract_path)
    envelope = clean_envelope(
        {
            "dataset_contract_sha256": contract_digest,
            "clean_reference_sha256": dataset.provenance["clean_reference_sha256"],
        }
    )
    assert envelope.validate_against_dataset(
        dataset, dataset_contract_sha256=contract_digest
    ) is envelope


@pytest.mark.parametrize("mutated", ["dataset", "contract_hash", "reference_hash"])
def test_clean_metric_binding_rejects_each_mismatch(tmp_path, mutated):
    dataset = clean_dataset()
    contract_path = tmp_path / "dataset-contract.json"
    write_json(dataset.to_dict(), contract_path)
    contract_digest = sha256(contract_path)
    envelope = clean_envelope(
        {
            "dataset_contract_sha256": contract_digest,
            "clean_reference_sha256": dataset.provenance["clean_reference_sha256"],
        }
    )
    if mutated == "dataset":
        dataset = replace(dataset, dataset_id="other-fixture")
    elif mutated == "contract_hash":
        contract_digest = "3" * 64
    else:
        dataset = replace(
            dataset,
            provenance={"source": "generated software fixture", "clean_reference_sha256": "4" * 64},
        )
    with pytest.raises(ContractError, match="MISMATCH"):
        envelope.validate_against_dataset(dataset, dataset_contract_sha256=contract_digest)


@pytest.mark.parametrize("bad_hashes", [None, [], {"": "x"}, {"x": ""}, {"x": 1}])
def test_result_hash_mapping_is_typed_and_nonblank(bad_hashes):
    base = clean_envelope(
        {"dataset_contract_sha256": "1" * 64, "clean_reference_sha256": "2" * 64}
    )
    with pytest.raises(ContractError, match="MISSING_EVIDENCE_BINDING"):
        replace(base, metric_id="finite_fraction", evidence_boundary="reference_free", hashes=bad_hashes).validate()


def test_rate_comparison_rejects_missing_either_side_and_both_orderings():
    base = clean_dataset()
    for left, right in (
        (replace(base, axes="HW", shape=(2, 2), frame_rate_hz=None), base),
        (base, replace(base, axes="HW", shape=(2, 2), frame_rate_hz=None)),
        (replace(base, frame_rate_hz=29.0), base),
        (replace(base, frame_rate_hz=31.0), base),
    ):
        with pytest.raises(ContractError):
            compare_dataset_rates(left, right)


@pytest.mark.parametrize(
    "names",
    [
        ["A.json", "a.JSON"],
        ["con"],
        ["nul.txt"],
        ["nested\\..\\escape"],
        ["bad\x1fname"],
    ],
)
def test_registry_rejects_case_collisions_reserved_names_and_control_chars(tmp_path, names):
    with pytest.raises(ContractError, match="INVALID_OUTPUT_PATH|DUPLICATE_EXPECTED_OUTPUT"):
        SafeRunRegistry(tmp_path).prepare(identity(), names)


def test_registry_rejects_manifest_schema_and_expected_output_tampering(tmp_path):
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(identity(), ["value.bin"])
    manifest_path = run / "RUN_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, value in (("schema_version", "zzz"), ("expected_outputs", ["other.bin"])):
        changed = dict(payload)
        changed[field] = value
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        assert registry.inspect(run) == "STALE"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_json_keeps_ascii_and_exact_compact_layout():
    assert canonical_json({"omega": "Ω", "b": 2, "a": 1}) == '{"a":1,"b":2,"omega":"\\u03a9"}'


def test_direct_contract_payloads_expose_json_array_fields_as_lists():
    method_payload = method().to_dict()
    assert all(
        isinstance(method_payload[field], list)
        for field in ("supported_metric_domains", "required_inputs", "produced_outputs")
    )
    envelope_payload = clean_envelope(
        {"dataset_contract_sha256": "1" * 64, "clean_reference_sha256": "2" * 64}
    ).validate().to_dict()
    assert all(
        isinstance(envelope_payload[field], list)
        for field in ("nesting", "warnings", "provenance_parents")
    )


def test_contract_collection_actually_validates_every_member():
    invalid = replace(method(), method_id="")
    with pytest.raises(ContractError, match="MISSING_METADATA"):
        validate_contract_collection([clean_dataset(), invalid])


def test_text_sequence_allow_empty_remains_keyword_only():
    parameter = inspect.signature(_text_sequence).parameters["allow_empty"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
