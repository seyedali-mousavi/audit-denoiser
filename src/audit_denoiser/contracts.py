"""Versioned CMPB core contracts for datasets, methods, and results.

The contracts make evidence boundaries and unsupported metric domains explicit.
They deliberately avoid framework-specific model assumptions.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASET_SCHEMA_ID = "org.calcium-denoiser-audit.dataset-contract"
DATASET_SCHEMA_VERSION = "1.0.0"
METHOD_SCHEMA_ID = "org.calcium-denoiser-audit.method-contract"
METHOD_SCHEMA_VERSION = "1.0.0"
RESULT_ENVELOPE_SCHEMA_ID = "org.calcium-denoiser-audit.result-envelope"
RESULT_ENVELOPE_SCHEMA_VERSION = "1.1.0"

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

    def validate(self) -> "CanonicalResultEnvelope":
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
        if self.admissibility == "ADMISSIBLE":
            assert_metric_admissible(self.metric_id, self.evidence_boundary)
        if not isinstance(self.hashes, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in self.hashes.items()
        ):
            raise ContractError("MISSING_EVIDENCE_BINDING", "hashes must be a text-to-text mapping")
        if self.admissibility == "ADMISSIBLE" and self.metric_id in CLEAN_REFERENCE_METRICS:
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
        if self.admissibility == "ADMISSIBLE" and self.metric_id in CLEAN_REFERENCE_METRICS:
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


def metric_admissibility(metric_id: str, evidence_boundary: str) -> tuple[str, str]:
    if evidence_boundary not in EVIDENCE_BOUNDARIES:
        raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", evidence_boundary)
    if metric_id in CLEAN_REFERENCE_METRICS and evidence_boundary != "clean_reference":
        return "WITHHELD", f"{metric_id} requires an exact clean-reference evidence boundary"
    return "ADMISSIBLE", "metric is compatible with the declared evidence boundary"


def assert_metric_admissible(metric_id: str, evidence_boundary: str) -> None:
    status, reason = metric_admissibility(metric_id, evidence_boundary)
    if status != "ADMISSIBLE":
        raise ContractError("INCOMPATIBLE_EVIDENCE_BOUNDARY", reason)


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
