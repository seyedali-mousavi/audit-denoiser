"""Generated contract fixtures, independent of the scientific research data."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from audit_denoiser.adapters import validate_contract_method_output
from audit_denoiser.contracts import (
    CanonicalResultEnvelope, CLEAN_REFERENCE_METRICS, ContractError,
    DatasetContract, MethodAdapterContract, metric_admissibility,
)
from audit_denoiser.run_identity import RunIdentityInputs, SafeRunRegistry, compute_run_id
from audit_denoiser.taxonomy import CURRENT_TAXONOMY_VERSION, LEGACY_TAXONOMY_VERSION, TAXONOMIES, migrate_label

settings.register_profile("b4_frozen", max_examples=100, derandomize=True, database=None, deadline=None)
settings.load_profile("b4_frozen")


def dataset():
    return DatasetContract("fixture", "synthetic-software-fixture", "THW", (8, 4, 4), "float32", "arbitrary",
                           30., "acquisition", "acquisition", ("acquisition", "frame", "pixel"),
                           "none", "reference_free", "none", "not_applicable", {"source": "generated test"})


def method():
    return MethodAdapterContract("external-test", "fixture-source", None, "none", "trace_only",
                                 ("trace_fidelity",), ("movie",), ("traces",))


def envelope():
    return CanonicalResultEnvelope("fixture", "external-test", "run-fixture", "reference_free", "acquisition",
        ("acquisition",), "finite_fraction", "1", "ratio", .5, None, "ADMISSIBLE", (),
        CURRENT_TAXONOMY_VERSION, ("fixture-parent",), {"fixture": "fixture-hash"})


def identity():
    return RunIdentityInputs("protocol", "dataset", "method", None, {"alpha": 1, "beta": 2}, 7,
                             "reference_free", "source", {"fixture": "hash"})


@given(st.integers(1, 12), st.integers(1, 12), st.integers(1, 12), st.floats(1e-6, 1e6))
def test_i1_valid_dataset_round_trip(t, h, w, fps):
    x = replace(dataset(), shape=(t, h, w), frame_rate_hz=fps).validate()
    assert DatasetContract.from_dict(json.loads(json.dumps(x.to_dict(), allow_nan=False))) == x


@pytest.mark.parametrize("rate", [None, 0, -1, float("nan"), float("inf"), -float("inf"), True, "30"])
def test_i1_temporal_rate_is_finite_number(rate):
    with pytest.raises(ContractError):
        replace(dataset(), frame_rate_hz=rate).validate()


@given(st.one_of(st.booleans(), st.text(max_size=8), st.floats(min_value=.1, max_value=.9)))
def test_i1_shape_never_silently_coerces(value):
    payload = dataset().to_dict()
    payload["shape"] = [value, 4, 4]
    with pytest.raises(ContractError):
        DatasetContract.from_dict(payload)


@pytest.mark.parametrize("factory", [dataset, method, envelope])
def test_i1_direct_constructor_checks_schema(factory):
    with pytest.raises(ContractError, match="SCHEMA_ID_MISMATCH"):
        replace(factory(), schema_id="wrong-schema").validate()


@given(st.sampled_from(["", " ", "\n", "\t", 5, None, True]))
def test_i1_nesting_members_are_nonblank_text(value):
    payload = dataset().to_dict()
    payload["nesting"] = ["acquisition", value]
    with pytest.raises(ContractError):
        DatasetContract.from_dict(payload)


@given(st.sampled_from(sorted(CLEAN_REFERENCE_METRICS)),
       st.sampled_from(["reference_free", "pseudo_reference", "trace_event_truth", "locked_historical_final_test"]))
def test_i2_envelope_cannot_bypass_metric_boundary(metric, boundary):
    assert metric_admissibility(metric, boundary)[0] == "WITHHELD"
    with pytest.raises(ContractError, match="INCOMPATIBLE_EVIDENCE_BOUNDARY"):
        replace(envelope(), metric_id=metric, evidence_boundary=boundary).validate()


@given(st.sampled_from([(), ("",), (" ",), ("\n",)]))
def test_i2_withheld_reason_has_content(reasons):
    with pytest.raises(ContractError):
        replace(envelope(), admissibility="WITHHELD", point_estimate=None, warnings=reasons).validate()


@given(st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
def test_i2_withheld_never_has_value(value):
    with pytest.raises(ContractError, match="INVALID_WITHHELD_VALUE"):
        replace(envelope(), point_estimate=value, admissibility="WITHHELD", warnings=("unavailable",)).validate()


@given(st.integers(1, 12), st.integers(1, 6))
def test_i3_array_shape_dtype(n, channels):
    x = replace(dataset(), axes="TR", shape=(n, channels)).validate()
    x.validate_array(np.zeros(x.shape, dtype="float32"))
    with pytest.raises(ContractError, match="DTYPE_MISMATCH"):
        x.validate_array(np.zeros(x.shape, dtype="int16"))
    with pytest.raises(ContractError, match="FRAME_COUNT_MISMATCH"):
        x.validate_array(np.zeros((n + 1, channels), dtype="float32"))


@pytest.mark.parametrize("value", ["0.5", "nan", True, False, float("nan"), float("inf"), -float("inf")])
def test_i4_estimate_is_numeric_finite_not_boolean(value):
    with pytest.raises(ContractError):
        replace(envelope(), point_estimate=value).validate()


@given(st.floats(-1e100, 1e100, allow_nan=False, allow_infinity=False))
def test_i4_valid_finite_result_round_trip(value):
    x = replace(envelope(), point_estimate=value).validate()
    assert CanonicalResultEnvelope.from_dict(json.loads(json.dumps(x.to_dict(), allow_nan=False))) == x


@given(st.sampled_from(["supported_metric_domains", "required_inputs", "produced_outputs"]),
       st.sampled_from(["", " ", None, 3, True]))
def test_i1_method_lists_are_typed_nonblank(field, bad):
    payload = method().to_dict()
    payload[field] = [bad]
    with pytest.raises(ContractError):
        MethodAdapterContract.from_dict(payload)


@given(st.integers(-2**31, 2**31 - 1))
def test_i5_order_invariance_and_seed_sensitivity(seed):
    x = replace(identity(), seed=seed)
    y = replace(x, config={"beta": 2, "alpha": 1})
    assert compute_run_id(x) == compute_run_id(y)
    assert compute_run_id(x) != compute_run_id(replace(y, seed=seed + 1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_i5_nonfinite_nested_identity_is_rejected(bad):
    with pytest.raises(ContractError):
        compute_run_id(replace(identity(), config={"nested": [0, {"bad": bad}]}))


@pytest.mark.parametrize("path", ["", ".", "..", "../escape", "/absolute", "C:relative",
                                  "C:/absolute", "nested/../../escape", "nested\\..\\escape",
                                  "RUN_MANIFEST.json", "OUTPUT_MANIFEST.json", "COMPLETE", "INTERRUPTED.json"])
def test_i6_invalid_and_control_paths_rejected(path):
    with tempfile.TemporaryDirectory() as temp:
        with pytest.raises(ContractError):
            SafeRunRegistry(Path(temp)).prepare(identity(), [path])


@given(st.binary(min_size=0, max_size=100))
def test_i6_completion_and_stale_detection(content):
    with tempfile.TemporaryDirectory() as temp:
        registry = SafeRunRegistry(Path(temp))
        run = registry.prepare(identity(), ["metrics.json"])
        assert registry.inspect(run) == "PARTIAL"
        (run / "metrics.json").write_bytes(content)
        registry.complete(run)
        assert registry.inspect(run) == "COMPLETE"
        (run / "metrics.json").write_bytes(content + b"x")
        assert registry.inspect(run) == "STALE"


@pytest.mark.parametrize("tamper", ["marker", "identity", "repeat_complete", "repeat_complete_changed"])
def test_i6_completed_state_cannot_be_rewritten_or_forged(tamper):
    with tempfile.TemporaryDirectory() as temp:
        registry = SafeRunRegistry(Path(temp))
        run = registry.prepare(identity(), ["metrics.json"])
        (run / "metrics.json").write_text("{}")
        registry.complete(run)
        if tamper == "marker":
            (run / "COMPLETE").write_text("other-run\n")
            assert registry.inspect(run) == "STALE"
        elif tamper == "identity":
            path = run / "RUN_MANIFEST.json"
            payload = json.loads(path.read_text())
            payload["identity_inputs"]["seed"] += 1
            path.write_text(json.dumps(payload))
            assert registry.inspect(run) == "STALE"
        else:
            if tamper == "repeat_complete_changed":
                (run / "metrics.json").write_text("changed")
            old = (run / "OUTPUT_MANIFEST.json").read_bytes()
            with pytest.raises(ContractError, match="DUPLICATE_RUN_ID"):
                registry.complete(run)
            assert (run / "OUTPUT_MANIFEST.json").read_bytes() == old


@given(st.sampled_from(sorted(TAXONOMIES[CURRENT_TAXONOMY_VERSION])))
def test_i7_current_taxonomy_identity(label):
    result = migrate_label(label, CURRENT_TAXONOMY_VERSION)
    assert result.status == "EXACT" and result.target_label == label


@given(st.sampled_from(sorted(TAXONOMIES[LEGACY_TAXONOMY_VERSION])))
def test_i7_legacy_taxonomy_not_relabelled(label):
    result = migrate_label(label, LEGACY_TAXONOMY_VERSION)
    assert result.status in {"AMBIGUOUS", "REQUIRES_REVIEW"} and result.target_label is None


@given(st.integers(2, 12), st.integers(1, 5))
def test_i8_external_producer_needs_no_core_registration(n, channels):
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "external.npy"
        np.save(path, np.zeros((n, channels), dtype="float32"), allow_pickle=False)
        x = replace(dataset(), shape=(n, 4, 4))
        result = validate_contract_method_output(method(), path, x)
        assert result["status"] == "SUPPORTED" and result["output_class"] == "trace_only"
        with pytest.raises(ContractError, match="FRAME_COUNT_MISMATCH"):
            validate_contract_method_output(method(), path, replace(x, shape=(n + 1, 4, 4)))
