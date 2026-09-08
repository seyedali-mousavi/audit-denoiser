"""Explicit input adapters for reusable denoiser auditing.

Adapters validate and identify inputs only.  They never rescale, denoise, or
change the frozen numerical audit implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .contracts import ContractError, DatasetContract, MethodAdapterContract, validate_numeric_array
from .movie_io import load_movie, supported_extensions


class DatasetAdapter(Protocol):
    adapter_id: str

    def validate(self, raw: Path | None, reference: Path) -> None: ...


class MethodAdapter(Protocol):
    adapter_id: str

    def validate(self, denoised: Path) -> None: ...


def _require_movie(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{role} movie not found: {path}")
    if path.suffix.lower() not in set(supported_extensions()):
        raise ValueError(
            f"{role} movie uses unsupported extension {path.suffix!r}; "
            f"supported: {', '.join(supported_extensions())}"
        )


@dataclass(frozen=True)
class MovieIODatasetAdapter:
    """Raw/reference paths loaded by the production ``movie_io`` reader."""

    adapter_id: str = "movie-io-v1"

    def validate(self, raw: Path | None, reference: Path) -> None:
        if raw is not None:
            _require_movie(raw, "raw")
        _require_movie(reference, "reference")


@dataclass(frozen=True)
class ExternalMovieMethodAdapter:
    """A precomputed denoised movie aligned frame-for-frame to the input."""

    adapter_id: str = "external-movie-v1"

    def validate(self, denoised: Path) -> None:
        _require_movie(denoised, "denoised")


@dataclass(frozen=True)
class IdentityFixtureMethodAdapter(ExternalMovieMethodAdapter):
    """Named adapter for deterministic fixture identity outputs."""

    adapter_id: str = "identity-fixture-v1"


DATASET_ADAPTERS: dict[str, DatasetAdapter] = {
    "movie-io-v1": MovieIODatasetAdapter(),
}
METHOD_ADAPTERS: dict[str, MethodAdapter] = {
    "external-movie-v1": ExternalMovieMethodAdapter(),
    "identity-fixture-v1": IdentityFixtureMethodAdapter(),
}


def validate_adapters(
    *,
    dataset_adapter: str,
    method_adapter: str,
    raw: str | Path | None,
    reference: str | Path,
    denoised: str | Path,
) -> None:
    try:
        dataset = DATASET_ADAPTERS[dataset_adapter]
    except KeyError as exc:
        raise ValueError(
            f"unknown dataset adapter {dataset_adapter!r}; "
            f"available: {', '.join(sorted(DATASET_ADAPTERS))}"
        ) from exc
    try:
        method = METHOD_ADAPTERS[method_adapter]
    except KeyError as exc:
        raise ValueError(
            f"unknown method adapter {method_adapter!r}; "
            f"available: {', '.join(sorted(METHOD_ADAPTERS))}"
        ) from exc
    dataset.validate(Path(raw) if raw is not None else None, Path(reference))
    method.validate(Path(denoised))


def validate_contract_method_output(
    contract: MethodAdapterContract,
    output: str | Path,
    dataset: DatasetContract,
) -> dict[str, str]:
    """Validate an output by declared class; unsupported domains are withheld."""
    contract.validate()
    dataset.validate()
    path = Path(output)
    if not path.is_file():
        raise FileNotFoundError(f"method output not found: {path}")
    if contract.output_class in {"full_movie", "component_reconstruction"}:
        if dataset.axes != "THW":
            raise ContractError("WRONG_AXES", f"movie output requires THW dataset, found {dataset.axes}")
        movie, meta = load_movie(path, stack_order=dataset.axes, max_frames=1)
        if tuple(meta["original_shape"]) != tuple(dataset.shape) or tuple(meta["canonical_shape"]) != tuple(dataset.shape):
            raise ContractError(
                "FRAME_COUNT_MISMATCH",
                f"movie shape {tuple(meta['original_shape'])} does not match dataset {dataset.shape}",
            )
        if movie.shape != (1, *dataset.shape[1:]):
            raise ContractError("FRAME_COUNT_MISMATCH", "loaded movie does not match declared prefix shape")
        return {"status": "SUPPORTED", "output_class": contract.output_class, "source_format": meta["source_format"],
                "dtype": meta["original_dtype"], "dtype_policy": "real_numeric", "axes": "THW"}
    if contract.output_class == "trace_only":
        if path.suffix.lower() == ".npy":
            traces = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".csv":
            traces = np.loadtxt(path, delimiter=",", skiprows=1)
        else:
            raise ContractError("UNSUPPORTED_TRACE_FORMAT", path.suffix)
        validate_numeric_array(traces)
        if traces.ndim != 2:
            raise ContractError("WRONG_AXES", f"trace-only output must be [T,R], found {traces.shape}")
        if "T" not in dataset.axes:
            raise ContractError("WRONG_AXES", f"trace-only output requires a temporal dataset, found {dataset.axes}")
        expected_frames = dataset.shape[dataset.axes.index("T")]
        if traces.shape[0] != expected_frames:
            raise ContractError(
                "FRAME_COUNT_MISMATCH",
                f"trace output has {traces.shape[0]} frames; dataset declares {expected_frames}",
            )
        if traces.shape[1] <= 0:
            raise ContractError("INVALID_SHAPE", "trace output requires at least one trace")
        return {"status": "SUPPORTED", "output_class": "trace_only", "shape": str(tuple(traces.shape)),
                "dtype": str(traces.dtype), "dtype_policy": "real_numeric", "axes": "TR"}
    raise ContractError("UNSUPPORTED_OUTPUT_CLASS", contract.output_class)
