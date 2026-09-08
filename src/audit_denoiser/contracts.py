"""Versioned CMPB core contracts for datasets, methods, and results.

The contracts make evidence boundaries and unsupported metric domains explicit.
They deliberately avoid framework-specific model assumptions.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


DATASET_SCHEMA_ID = "org.calcium-denoiser-audit.dataset-contract"
DATASET_SCHEMA_VERSION = "1.0.0"
METHOD_SCHEMA_ID = "org.calcium-denoiser-audit.method-contract"
METHOD_SCHEMA_VERSION = "1.0.0"
RESULT_ENVELOPE_SCHEMA_ID = "org.calcium-denoiser-audit.result-envelope"
RESULT_ENVELOPE_SCHEMA_VERSION = "1.2.0"

OUTPUT_CLASSES = {"full_movie", "component_reconstruction", "trace_only"}
EVIDENCE_BOUNDARIES = {
    "clean_reference",
    "pseudo_reference",
    "reference_free",
    "trace_event_truth",
    "locked_historical_final_test",
}
REFERENCE_AVAILABILITY = {"exact_clean", "pseudo_reference", "none", "event_truth_only"}
ALIGNMENT_STATES = {"not_applicable", "unknown", "unaligned", "spatial_only", "spatiotemporal"}
ADMISSIBILITY = {"ADMISSIBLE", "WITHHELD", "INVALID"}

CLEAN_REFERENCE_METRICS = {
    "psnr_to_clean",
    "ssim_to_clean",
    "amp_pres_to_clean",
    "clean_derived_trace_fidelity",
}
REFERENCE_FREE_METRICS = {
    "finite_fraction",
    "robust_dynamic_range",
    "split_half_mean_map_correlation",
    "global_trace_lag1_autocorrelation",
    "temporal_difference_energy_ratio",
    "seam_energy_ratio",
}
WAVEFORM_CLEAN_REFERENCE_METRICS = {
    "waveform_pearson_to_clean", "waveform_nrmse_to_clean", "waveform_snr_improvement_db",
}


class ContractError(ValueError):
    """Deterministic validation error carrying a stable machine code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("MISSING_METADATA", f"{field_name} must be non-empty")


def _require_version(actual: str, expected: str, schema: str) -> None:
    if actual != expected:
        raise ContractError(
            "SCHEMA_VERSION_MISMATCH",
            f"{schema} requires {expected}, found {actual!r}",
        )


def _require_schema(actual: str, expected: str) -> None:
    if actual != expected:
        raise ContractError("SCHEMA_ID_MISMATCH", f"expected {expected}, found {actual!r}")


def _text_sequence(value: Any, field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or (not value and not allow_empty):
        raise ContractError("MISSING_METADATA", f"{field_name} must be a sequence of text values")
    for item in value:
        _require_text(item, field_name)
    return tuple(value)


def _finite_number(value: Any) -> bool:
    # bool is an int subclass, but is not a JSON numeric measurement.
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and math.isfinite(value)
    )


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise ContractError(
            "MISSING_EVIDENCE_BINDING",
            f"{field_name} must be a 64-character SHA-256 digest",
        )
    return value.lower()


def validate_numeric_array(array: np.ndarray) -> None:
    """Accept real integer/float measurements, allowing legitimate dtype conversion.

    Finiteness is an endpoint policy: finite_fraction must be able to inspect a
    numerical array containing NaNs. Strings, booleans, objects, and complex arrays
    are not a real-valued movie/trace output and must never be certified as one.
    """
    if np.asarray(array).dtype.kind not in "iuf":
        raise ContractError("NONNUMERIC_DTYPE", f"expected real numeric array, found {np.asarray(array).dtype}")


@dataclass(frozen=True)
class MetricContract:
    """Explicit endpoint requirements; external definitions need no core registration.

    These declarations are trusted scientific policy, not proof that a chosen
    endpoint or reference is biologically appropriate. Known built-ins cannot be
    weakened by supplying another definition with the same identifier.
    """
    metric_id: str
    domain: str
    evidence_boundaries: tuple[str, ...]
    output_classes: tuple[str, ...]
    requires_clean_reference: bool = False
    version: str = "1.0.0"

    def validate(self) -> "MetricContract":
        for name in ("metric_id", "domain", "version"):
            _require_text(getattr(self, name), name)
        _text_sequence(self.evidence_boundaries, "evidence_boundaries")
        _text_sequence(self.output_classes, "output_classes")
        if not set(self.evidence_boundaries) <= EVIDENCE_BOUNDARIES:
            raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", "metric declares an unknown boundary")
        if not set(self.output_classes) <= OUTPUT_CLASSES:
            raise ContractError("UNSUPPORTED_OUTPUT_CLASS", "metric declares an unknown output class")
        if not isinstance(self.requires_clean_reference, bool):
            raise ContractError("INVALID_METRIC_CONTRACT", "requires_clean_reference must be boolean")
        if self.requires_clean_reference and set(self.evidence_boundaries) != {"clean_reference"}:
            raise ContractError("INVALID_METRIC_CONTRACT", "clean-dependent metrics require only clean_reference")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_boundaries"] = list(self.evidence_boundaries)
        payload["output_classes"] = list(self.output_classes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetricContract":
        try:
            data = dict(payload)
            for key in ("evidence_boundaries", "output_classes"):
                data[key] = _text_sequence(data[key], key)
            return cls(**data).validate()
        except (TypeError, KeyError) as exc:
            raise ContractError("INVALID_METRIC_CONTRACT", "incomplete or malformed metric contract") from exc


def resolve_metric_contract(metric_id: str, specification: MetricContract | None = None) -> MetricContract | None:
    """Resolve explicit requirements, never infer admissibility from an unknown ID."""
    movie = ("full_movie", "component_reconstruction")
    builtin = None
    if metric_id in CLEAN_REFERENCE_METRICS:
        builtin = MetricContract(metric_id, metric_id, ("clean_reference",), movie, True)
    elif metric_id in WAVEFORM_CLEAN_REFERENCE_METRICS:
        builtin = MetricContract(metric_id, "generic_waveform_fidelity", ("clean_reference",), ("trace_only",), True)
    elif metric_id in REFERENCE_FREE_METRICS:
        builtin = MetricContract(metric_id, "reference_free_movie", tuple(sorted(EVIDENCE_BOUNDARIES)), movie)
    if specification is not None:
        specification.validate()
        if specification.metric_id != metric_id:
            raise ContractError("METRIC_ID_MISMATCH", "request and metric contract identifiers differ")
        if builtin is not None and specification != builtin:
            raise ContractError("METRIC_CONTRACT_CONFLICT", "a built-in metric definition cannot be overridden")
        return specification
    return builtin


def contract_digest(contract: "DatasetContract | MethodAdapterContract | MetricContract") -> str:
    """Hash canonical contract semantics; raw file hashes are recorded separately."""
    serialized = json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetContract:
    dataset_id: str
    modality: str
    axes: str
    shape: tuple[int, ...]
    dtype: str
    units: str
    frame_rate_hz: float | None
    acquisition_id: str
    independent_unit: str
    nesting: tuple[str, ...]
    reference_availability: str
    evidence_boundary: str
    mask_availability: str
    alignment_state: str
    provenance: dict[str, Any]
    schema_version: str = DATASET_SCHEMA_VERSION
    schema_id: str = DATASET_SCHEMA_ID

    def validate(self) -> "DatasetContract":
        _require_schema(self.schema_id, DATASET_SCHEMA_ID)
        _require_version(self.schema_version, DATASET_SCHEMA_VERSION, DATASET_SCHEMA_ID)
        for value, name in (
            (self.dataset_id, "dataset_id"),
            (self.modality, "modality"),
            (self.axes, "axes"),
            (self.dtype, "dtype"),
            (self.units, "units"),
            (self.acquisition_id, "acquisition_id"),
            (self.independent_unit, "independent_unit"),
            (self.mask_availability, "mask_availability"),
        ):
            _require_text(value, name)
        if not isinstance(self.shape, (tuple, list)) or not self.shape:
            raise ContractError("INVALID_SHAPE", "shape must be a non-empty sequence")
        if len(self.axes) != len(self.shape) or len(set(self.axes)) != len(self.axes):
            raise ContractError("WRONG_AXES", f"axes {self.axes!r} do not uniquely describe shape {self.shape}")
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in self.shape):
            raise ContractError("INVALID_SHAPE", f"shape must contain positive integers: {self.shape}")
        try:
            np.dtype(self.dtype)
        except (TypeError, ValueError) as exc:
            raise ContractError("INVALID_DTYPE", f"unsupported dtype {self.dtype!r}") from exc
        if self.frame_rate_hz is not None and (not _finite_number(self.frame_rate_hz) or self.frame_rate_hz <= 0):
            raise ContractError("MISSING_METADATA", "frame_rate_hz must be a finite positive number or null")
        if "T" in self.axes and self.frame_rate_hz is None:
            raise ContractError("MISSING_METADATA", "positive frame_rate_hz is required for temporal data")
        if self.reference_availability not in REFERENCE_AVAILABILITY:
            raise ContractError("INVALID_REFERENCE_AVAILABILITY", self.reference_availability)
        if self.evidence_boundary not in EVIDENCE_BOUNDARIES:
            raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", self.evidence_boundary)
        if self.alignment_state not in ALIGNMENT_STATES:
            raise ContractError("INVALID_ALIGNMENT_STATE", self.alignment_state)
        _text_sequence(self.nesting, "nesting")
        if self.nesting[0] != self.independent_unit:
            raise ContractError("INVALID_NESTING", "nesting must begin with independent_unit")
        if not isinstance(self.provenance, dict) or not self.provenance.get("source"):
            raise ContractError("MISSING_METADATA", "provenance.source is required")
        _require_text(self.provenance["source"], "provenance.source")
        if self.evidence_boundary == "reference_free" and self.reference_availability != "none":
            raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", "reference_free requires reference_availability=none")
        if self.evidence_boundary == "clean_reference" and self.reference_availability != "exact_clean":
            raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", "clean_reference requires exact_clean")
        return self

    def validate_array(self, array: np.ndarray) -> None:
        actual = np.asarray(array)
        if actual.shape != self.shape:
            raise ContractError("FRAME_COUNT_MISMATCH", f"expected {self.shape}, found {actual.shape}")
        if np.dtype(actual.dtype) != np.dtype(self.dtype):
            raise ContractError("DTYPE_MISMATCH", f"expected {self.dtype}, found {actual.dtype}")

    def validate_mask(self, mask: np.ndarray) -> None:
        spatial = tuple(self.shape[self.axes.index(axis)] for axis in self.axes if axis in {"H", "W", "Z"})
        if tuple(np.asarray(mask).shape) != spatial:
            raise ContractError("INVALID_MASK_SHAPE", f"expected spatial mask {spatial}, found {np.asarray(mask).shape}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["nesting"] = list(self.nesting)
        payload["$schema"] = payload.pop("schema_id")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetContract":
        data = dict(payload)
        schema_id = data.pop("$schema", data.pop("schema_id", DATASET_SCHEMA_ID))
        if schema_id != DATASET_SCHEMA_ID:
            raise ContractError("SCHEMA_ID_MISMATCH", f"expected {DATASET_SCHEMA_ID}, found {schema_id}")
        required = {
            "dataset_id", "modality", "axes", "shape", "dtype", "units", "frame_rate_hz",
            "acquisition_id", "independent_unit", "nesting", "reference_availability",
            "evidence_boundary", "mask_availability", "alignment_state", "provenance",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ContractError("MISSING_METADATA", f"missing fields: {', '.join(missing)}")
        if not isinstance(data["shape"], (list, tuple)):
            raise ContractError("INVALID_SHAPE", "shape must be a sequence")
        data["shape"] = tuple(data["shape"])
        data["nesting"] = _text_sequence(data["nesting"], "nesting")
        return cls(schema_id=schema_id, **data).validate()

    @classmethod
    def read(cls, path: Path) -> "DatasetContract":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class MethodAdapterContract:
    method_id: str
    source_identity: str
    checkpoint_identity: str | None
    training_reference_access: str
    output_class: str
    supported_metric_domains: tuple[str, ...]
    required_inputs: tuple[str, ...]
    produced_outputs: tuple[str, ...]
    schema_version: str = METHOD_SCHEMA_VERSION
    schema_id: str = METHOD_SCHEMA_ID

    def validate(self) -> "MethodAdapterContract":
        _require_schema(self.schema_id, METHOD_SCHEMA_ID)
        _require_version(self.schema_version, METHOD_SCHEMA_VERSION, METHOD_SCHEMA_ID)
        for value, name in (
            (self.method_id, "method_id"),
            (self.source_identity, "source_identity"),
            (self.training_reference_access, "training_reference_access"),
        ):
            _require_text(value, name)
        if self.output_class not in OUTPUT_CLASSES:
            raise ContractError("UNSUPPORTED_OUTPUT_CLASS", self.output_class)
        _text_sequence(self.supported_metric_domains, "supported_metric_domains")
        _text_sequence(self.required_inputs, "required_inputs", allow_empty=True)
        _text_sequence(self.produced_outputs, "produced_outputs")
        if self.checkpoint_identity is not None:
            _require_text(self.checkpoint_identity, "checkpoint_identity")
        return self

    def domain_status(self, domain: str) -> dict[str, str]:
        if domain in self.supported_metric_domains:
            return {"status": "SUPPORTED", "domain": domain, "reason": "declared by method contract"}
        return {
            "status": "WITHHELD",
            "domain": domain,
            "reason": f"{self.output_class} output does not declare this metric domain",
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("supported_metric_domains", "required_inputs", "produced_outputs"):
            payload[key] = list(payload[key])
        payload["$schema"] = payload.pop("schema_id")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MethodAdapterContract":
        data = dict(payload)
        schema_id = data.pop("$schema", data.pop("schema_id", METHOD_SCHEMA_ID))
        if schema_id != METHOD_SCHEMA_ID:
            raise ContractError("SCHEMA_ID_MISMATCH", f"expected {METHOD_SCHEMA_ID}, found {schema_id}")
        for key in ("supported_metric_domains", "required_inputs", "produced_outputs"):
            if key not in data:
                raise ContractError("MISSING_METADATA", f"missing field: {key}")
            data[key] = _text_sequence(data[key], key, allow_empty=key == "required_inputs")
        return cls(schema_id=schema_id, **data).validate()

    @classmethod
    def read(cls, path: Path) -> "MethodAdapterContract":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CanonicalResultEnvelope:
    dataset_id: str
    method_id: str
    run_id: str
    evidence_boundary: str
    independent_unit: str
    nesting: tuple[str, ...]
    metric_id: str
    metric_version: str
    units: str
    point_estimate: float | int | None
    uncertainty: dict[str, Any] | None
    admissibility: str
    warnings: tuple[str, ...]
    taxonomy_version: str
    provenance_parents: tuple[str, ...]
    hashes: dict[str, str]
    schema_version: str = RESULT_ENVELOPE_SCHEMA_VERSION
    schema_id: str = RESULT_ENVELOPE_SCHEMA_ID
    metric_contract: dict[str, Any] | None = None

    def validate(self) -> "CanonicalResultEnvelope":
        """Validate record structure and declared metric policy, not execution context.

        Runtime producers must use evaluate_metric, whose preflight and emission
        checks bind the concrete dataset/method/reference and gate the callback.
        """
        _require_schema(self.schema_id, RESULT_ENVELOPE_SCHEMA_ID)
        _require_version(self.schema_version, RESULT_ENVELOPE_SCHEMA_VERSION, RESULT_ENVELOPE_SCHEMA_ID)
        for value, name in (
            (self.dataset_id, "dataset_id"), (self.method_id, "method_id"),
            (self.run_id, "run_id"), (self.independent_unit, "independent_unit"),
            (self.metric_id, "metric_id"), (self.metric_version, "metric_version"),
            (self.units, "units"), (self.taxonomy_version, "taxonomy_version"),
        ):
            _require_text(value, name)
        if self.evidence_boundary not in EVIDENCE_BOUNDARIES:
            raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", self.evidence_boundary)
        if self.admissibility not in ADMISSIBILITY:
            raise ContractError("INVALID_ADMISSIBILITY", self.admissibility)
        if self.admissibility == "ADMISSIBLE" and self.point_estimate is None:
            raise ContractError("MISSING_POINT_ESTIMATE", "admissible result requires a point estimate")
        if self.admissibility == "ADMISSIBLE" and not _finite_number(self.point_estimate):
            raise ContractError("NONFINITE_POINT_ESTIMATE", "admissible result requires a finite point estimate")
        specification = resolve_metric_contract(
            self.metric_id, MetricContract.from_dict(self.metric_contract) if self.metric_contract is not None else None,
        )
        if self.admissibility == "ADMISSIBLE":
            assert_metric_admissible(self.metric_id, self.evidence_boundary, metric_contract=specification)
        if not isinstance(self.hashes, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in self.hashes.items()
        ):
            raise ContractError("MISSING_EVIDENCE_BINDING", "hashes must be a text-to-text mapping")
        if self.admissibility == "ADMISSIBLE" and specification is not None and specification.requires_clean_reference:
            if self.evidence_boundary != "clean_reference":
                raise ContractError(
                    "INCOMPATIBLE_EVIDENCE_BOUNDARY",
                    f"{self.metric_id} requires clean_reference evidence",
                )
            for key in ("dataset_contract_sha256", "clean_reference_sha256"):
                if key not in self.hashes:
                    raise ContractError(
                        "MISSING_EVIDENCE_BINDING",
                        f"admissible clean-reference result requires hashes.{key}",
                    )
                _require_sha256(self.hashes[key], f"hashes.{key}")
        if self.admissibility != "ADMISSIBLE" and self.point_estimate is not None:
            raise ContractError("INVALID_WITHHELD_VALUE", "withheld or invalid result cannot carry a point estimate")
        if self.admissibility == "WITHHELD" and (
            not isinstance(self.warnings, (list, tuple))
            or not any(isinstance(w, str) and w.strip() for w in self.warnings)
        ):
            raise ContractError("MISSING_WITHHOLD_REASON", "withheld result requires a warning")
        _text_sequence(self.warnings, "warnings", allow_empty=True)
        _text_sequence(self.provenance_parents, "provenance_parents", allow_empty=True)
        _text_sequence(self.nesting, "nesting")
        if not self.nesting or self.nesting[0] != self.independent_unit:
            raise ContractError("INVALID_NESTING", "nesting must start with independent_unit")
        if self.uncertainty is not None and not isinstance(self.uncertainty, dict):
            raise ContractError("INVALID_UNCERTAINTY", "uncertainty must be an object or null")
        try:
            json.dumps(self.uncertainty, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ContractError("INVALID_UNCERTAINTY", "uncertainty must contain finite JSON-compatible values") from exc
        return self

    def validate_against_dataset(
        self,
        dataset: "DatasetContract",
        *,
        dataset_contract_sha256: str,
    ) -> "CanonicalResultEnvelope":
        """Validate declared evidence against one concrete dataset contract.

        ``validate()`` checks the envelope and requires explicit digests for clean-only
        metrics. This method additionally binds those digests to the supplied dataset
        contract. It does not open or score scientific arrays.
        """
        self.validate()
        dataset.validate()
        contract_digest = _require_sha256(dataset_contract_sha256, "dataset_contract_sha256")
        if self.dataset_id != dataset.dataset_id:
            raise ContractError("DATASET_ID_MISMATCH", f"{self.dataset_id} != {dataset.dataset_id}")
        if self.evidence_boundary != dataset.evidence_boundary:
            raise ContractError(
                "INCOMPATIBLE_EVIDENCE_BOUNDARY",
                f"envelope {self.evidence_boundary} != dataset {dataset.evidence_boundary}",
            )
        if self.hashes.get("dataset_contract_sha256", "").lower() != contract_digest:
            raise ContractError("EVIDENCE_HASH_MISMATCH", "dataset-contract digest does not match envelope")
        specification = resolve_metric_contract(
            self.metric_id, MetricContract.from_dict(self.metric_contract) if self.metric_contract is not None else None,
        )
        if self.admissibility == "ADMISSIBLE" and specification is not None and specification.requires_clean_reference:
            if dataset.reference_availability != "exact_clean":
                raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", "dataset lacks exact clean reference")
            reference_digest = _require_sha256(
                dataset.provenance.get("clean_reference_sha256"),
                "dataset.provenance.clean_reference_sha256",
            )
            if self.hashes["clean_reference_sha256"].lower() != reference_digest:
                raise ContractError("EVIDENCE_HASH_MISMATCH", "clean-reference digest does not match dataset")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("nesting", "warnings", "provenance_parents"):
            payload[key] = list(payload[key])
        payload["$schema"] = payload.pop("schema_id")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalResultEnvelope":
        data = dict(payload)
        schema_id = data.pop("$schema", data.pop("schema_id", RESULT_ENVELOPE_SCHEMA_ID))
        if schema_id != RESULT_ENVELOPE_SCHEMA_ID:
            raise ContractError("SCHEMA_ID_MISMATCH", f"expected {RESULT_ENVELOPE_SCHEMA_ID}, found {schema_id}")
        for key in ("nesting", "warnings", "provenance_parents"):
            data[key] = tuple(data.get(key, ()))
        return cls(schema_id=schema_id, **data).validate()


def metric_admissibility(
    metric_id: str, evidence_boundary: str, *, metric_contract: MetricContract | None = None,
) -> tuple[str, str]:
    if evidence_boundary not in EVIDENCE_BOUNDARIES:
        raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", evidence_boundary)
    specification = resolve_metric_contract(metric_id, metric_contract)
    if specification is None:
        return "WITHHELD", f"{metric_id} has no declared metric requirements"
    if evidence_boundary not in specification.evidence_boundaries:
        return "WITHHELD", f"{metric_id} requires evidence in {specification.evidence_boundaries}"
    return "ADMISSIBLE", "metric is compatible with the declared evidence boundary"


def assert_metric_admissible(
    metric_id: str, evidence_boundary: str, *, metric_contract: MetricContract | None = None,
) -> None:
    status, reason = metric_admissibility(metric_id, evidence_boundary, metric_contract=metric_contract)
    if status != "ADMISSIBLE":
        raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", reason)


def contextual_admissibility(
    metric_id: str, dataset: DatasetContract, method: MethodAdapterContract,
    *, hashes: dict[str, str] | None = None, metric_contract: MetricContract | None = None,
) -> tuple[str, str, dict[str, str]]:
    """Compose structural, evidence, method-domain, and concrete binding checks.

    Returns status/reason plus bound semantic digests. Invalid declarations or
    conflicting bindings raise ContractError. Unsupported requests are value-free
    abstentions. Files/arrays must be loaded and validated by the caller; a supplied
    clean-reference digest identifies those observed bytes, not biological truth.
    """
    dataset.validate()
    method.validate()
    specification = resolve_metric_contract(metric_id, metric_contract)
    bound = dict(hashes or {})
    for key, contract in (("dataset_contract_sha256", dataset), ("method_contract_sha256", method)):
        expected = contract_digest(contract)
        if key in bound and _require_sha256(bound[key], key) != expected:
            raise ContractError("EVIDENCE_HASH_MISMATCH", f"{key} does not identify the supplied contract")
        bound[key] = expected
    if specification is not None:
        expected = contract_digest(specification)
        if "metric_contract_sha256" in bound and _require_sha256(bound["metric_contract_sha256"], "metric_contract_sha256") != expected:
            raise ContractError("EVIDENCE_HASH_MISMATCH", "metric requirements changed")
        bound["metric_contract_sha256"] = expected
    status, reason = metric_admissibility(metric_id, dataset.evidence_boundary, metric_contract=specification)
    if status != "ADMISSIBLE":
        return status, reason, bound
    assert specification is not None
    if method.output_class not in specification.output_classes:
        return "WITHHELD", f"{metric_id} does not support output class {method.output_class}", bound
    domain = method.domain_status(specification.domain)
    if domain["status"] != "SUPPORTED":
        return "WITHHELD", domain["reason"], bound
    if specification.requires_clean_reference:
        if dataset.reference_availability != "exact_clean":
            raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", "dataset lacks exact clean reference")
        declared = _require_sha256(dataset.provenance.get("clean_reference_sha256"), "dataset.provenance.clean_reference_sha256")
        observed = _require_sha256(bound.get("clean_reference_sha256"), "hashes.clean_reference_sha256")
        if declared != observed:
            raise ContractError("EVIDENCE_HASH_MISMATCH", "observed clean reference differs from dataset declaration")
    return "ADMISSIBLE", "evidence, output class, domain, and contract bindings agree", bound


def validate_result_against_context(
    result: CanonicalResultEnvelope, dataset: DatasetContract, method: MethodAdapterContract,
) -> CanonicalResultEnvelope:
    """Recheck context on emission/import; structural validate() alone is insufficient."""
    result.validate()
    specification = MetricContract.from_dict(result.metric_contract) if result.metric_contract is not None else None
    required = ["dataset_contract_sha256", "method_contract_sha256"]
    if resolve_metric_contract(result.metric_id, specification) is not None:
        required.append("metric_contract_sha256")
    for key in required:
        _require_sha256(result.hashes.get(key), f"result.hashes.{key}")
    status, reason, _ = contextual_admissibility(
        result.metric_id, dataset, method, hashes=result.hashes, metric_contract=specification,
    )
    result.validate_against_dataset(dataset, dataset_contract_sha256=contract_digest(dataset))
    if result.method_id != method.method_id:
        raise ContractError("METHOD_ID_MISMATCH", "result identifies another method")
    if result.independent_unit != dataset.independent_unit or tuple(result.nesting) != tuple(dataset.nesting):
        raise ContractError("INVALID_NESTING", "result hierarchy differs from the declared dataset")
    if result.admissibility == "ADMISSIBLE" and status != "ADMISSIBLE":
        raise ContractError("INCOMPATIBLE_METRIC_DOMAIN", reason)
    return result


def evaluate_metric(
    metric_id: str, dataset: DatasetContract, method: MethodAdapterContract, *,
    run_id: str, metric_version: str, units: str, analysis: Callable[[], Any],
    taxonomy_version: str, hashes: dict[str, str] | None = None,
    provenance_parents: tuple[str, ...] = (), warnings: tuple[str, ...] = (),
    metric_contract: MetricContract | None = None,
) -> CanonicalResultEnvelope:
    """The public runtime gate: check before calling analysis and again on emission.

    Analysis returns a scalar or (scalar, uncertainty). Nonfinite/undefined scalars
    become reason-bearing WITHHELD records. Other invalid values raise a typed
    error. No callback is invoked for unknown or incompatible metric requests.
    """
    specification = resolve_metric_contract(metric_id, metric_contract)
    status, reason, bound = contextual_admissibility(
        metric_id, dataset, method, hashes=hashes, metric_contract=specification,
    )
    value, uncertainty = None, None
    emitted_warnings = tuple(warnings)
    if status == "ADMISSIBLE":
        computed = analysis()
        if isinstance(computed, tuple) and len(computed) == 2:
            value, uncertainty = computed
        else:
            value = computed
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            status, value, uncertainty = "WITHHELD", None, None
            emitted_warnings += ("metric is undefined or non-finite for this input",)
        elif not _finite_number(value):
            raise ContractError("NONFINITE_POINT_ESTIMATE", "analysis must return a real finite numeric estimate")
    else:
        emitted_warnings += (reason,)
    result = CanonicalResultEnvelope(
        dataset_id=dataset.dataset_id, method_id=method.method_id, run_id=run_id,
        evidence_boundary=dataset.evidence_boundary, independent_unit=dataset.independent_unit,
        nesting=dataset.nesting, metric_id=metric_id, metric_version=metric_version,
        units=units, point_estimate=value, uncertainty=uncertainty, admissibility=status,
        warnings=emitted_warnings, taxonomy_version=taxonomy_version,
        provenance_parents=provenance_parents, hashes=bound,
        metric_contract=specification.to_dict() if specification is not None else None,
    )
    return validate_result_against_context(result, dataset, method)


def compare_dataset_rates(left: DatasetContract, right: DatasetContract) -> None:
    if left.frame_rate_hz is None or right.frame_rate_hz is None:
        raise ContractError("MISSING_METADATA", "both frame rates are required")
    if left.frame_rate_hz != right.frame_rate_hz:
        raise ContractError("FRAME_RATE_MISMATCH", f"{left.frame_rate_hz} != {right.frame_rate_hz}")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(serialized.encode("utf-8"))


def validate_contract_collection(contracts: Iterable[DatasetContract | MethodAdapterContract]) -> None:
    for contract in contracts:
        contract.validate()
