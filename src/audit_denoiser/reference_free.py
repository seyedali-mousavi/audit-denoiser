"""Genuine reference-free audit lane: no dummy clean input is accepted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    CLEAN_REFERENCE_METRICS,
    CanonicalResultEnvelope,
    DatasetContract,
    MethodAdapterContract,
    ContractError,
    REFERENCE_FREE_METRICS,
    evaluate_metric,
)
from .movie_io import load_movie
from .schema import sha256
from .taxonomy import CURRENT_TAXONOMY_VERSION


REFERENCE_FREE_METRIC_VERSION = "reference-free-core-v1"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    x, y = x - np.mean(x), y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else float("nan")


def _seam_ratio(movie: np.ndarray, step: int = 16) -> float:
    gradients = np.abs(np.diff(movie.astype(np.float64), axis=2))
    boundaries = np.arange(step - 1, gradients.shape[2], step)
    if boundaries.size == 0:
        return float("nan")
    mask = np.ones(gradients.shape[2], dtype=bool)
    mask[boundaries] = False
    seam = float(np.mean(gradients[:, :, boundaries]))
    background = float(np.mean(gradients[:, :, mask])) if mask.any() else float("nan")
    return seam / max(background, 1e-12)


def reference_free_metrics(movie: np.ndarray) -> dict[str, float]:
    array = np.asarray(movie, dtype=np.float32)
    odd, even = array[0::2], array[1::2]
    pairs = min(len(odd), len(even))
    global_trace = np.mean(array, axis=(1, 2), dtype=np.float64)
    centered = global_trace - np.mean(global_trace)
    lag1 = _pearson(centered[1:], centered[:-1]) if len(centered) > 2 else float("nan")
    total_energy = float(np.mean((array - np.mean(array, axis=0, keepdims=True)) ** 2))
    diff_energy = float(np.mean(np.diff(array, axis=0) ** 2)) if len(array) > 1 else float("nan")
    return {
        "finite_fraction": float(np.isfinite(array).mean()),
        "robust_dynamic_range": float(np.percentile(array, 99) - np.percentile(array, 1)),
        "split_half_mean_map_correlation": _pearson(np.mean(odd[:pairs], axis=0), np.mean(even[:pairs], axis=0)) if pairs else float("nan"),
        "global_trace_lag1_autocorrelation": lag1,
        "temporal_difference_energy_ratio": diff_energy / max(total_energy, 1e-12),
        "seam_energy_ratio": _seam_ratio(array),
    }


def run_reference_free_audit(
    *,
    movie_path: Path,
    dataset: DatasetContract,
    method: MethodAdapterContract,
    run_id: str,
    max_frames: int = 256,
) -> list[CanonicalResultEnvelope]:
    dataset.validate()
    method.validate()
    if dataset.evidence_boundary not in {"reference_free", "pseudo_reference"}:
        raise ValueError("reference-free lane requires reference_free or pseudo_reference evidence boundary")
    if method.output_class not in {"full_movie", "component_reconstruction"}:
        raise ContractError("UNSUPPORTED_OUTPUT_CLASS", "reference-free movie audit requires movie/component output")
    if dataset.axes != "THW":
        raise ContractError("WRONG_AXES", f"reference-free movie lane requires THW axes, found {dataset.axes}")
    parent_hash = sha256(movie_path)
    envelopes: list[CanonicalResultEnvelope] = []
    movie, meta = load_movie(movie_path, stack_order=dataset.axes, max_frames=max_frames)
    if tuple(meta["original_shape"]) != tuple(dataset.shape) or tuple(meta["canonical_shape"]) != tuple(dataset.shape):
        raise ContractError("FRAME_COUNT_MISMATCH", f"dataset contract shape {dataset.shape} does not match source {tuple(meta['original_shape'])}")
    expected = (min(dataset.shape[0], max_frames), *dataset.shape[1:]) if max_frames is not None else dataset.shape
    if movie.shape != tuple(expected):
        raise ContractError("FRAME_COUNT_MISMATCH", f"loaded prefix {movie.shape} != declared prefix {expected}")
    if np.dtype(meta["original_dtype"]) != np.dtype(dataset.dtype):
        raise ContractError("DTYPE_MISMATCH", f"dataset contract dtype {dataset.dtype} does not match source {meta['original_dtype']}")
    cached: dict[str, float] = {}

    def analyze(metric_id: str) -> float:
        # This cache is populated only from an admitted request. All six built-in
        # reference-free movie endpoints share the same evidence/domain policy.
        if not cached:
            cached.update(reference_free_metrics(movie))
        return cached[metric_id]

    for metric_id in sorted(REFERENCE_FREE_METRICS | CLEAN_REFERENCE_METRICS):
        envelopes.append(evaluate_metric(
            metric_id, dataset, method, run_id=run_id, metric_version=REFERENCE_FREE_METRIC_VERSION,
            units=dataset.units if metric_id == "robust_dynamic_range" else "ratio",
            analysis=lambda metric_id=metric_id: analyze(metric_id),
            taxonomy_version=CURRENT_TAXONOMY_VERSION, provenance_parents=(str(movie_path),),
            hashes={"movie_sha256": parent_hash},
        ))
    return envelopes


def write_reference_free_results(results: list[CanonicalResultEnvelope], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps([r.validate().to_dict() for r in results], indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(serialized.encode("utf-8"))
