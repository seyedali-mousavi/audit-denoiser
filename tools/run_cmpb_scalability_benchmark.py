"""Run the frozen JR12 canonical-result-envelope scalability benchmark.

This is a synthetic software benchmark. It does not load biomedical arrays or
compute scientific metrics. The confirmatory CLI deliberately exposes no knobs
for changing the frozen counts, repetition count, payload mix, or seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterable

from audit_denoiser.contracts import (
    CanonicalResultEnvelope,
    ContractError,
    DatasetContract,
    MethodAdapterContract,
    RESULT_ENVELOPE_SCHEMA_VERSION,
    MetricContract,
)


SCHEMA_VERSION = "jr12-cmpb-scalability-v1"
FROZEN_COUNTS = (100, 1_000, 10_000, 50_000)
FROZEN_REPEATS = 5
FROZEN_SEED = 20260822
MAX_TOTAL_SECONDS = 60.0
MAX_PEAK_MIB = 1_024.0
MAX_TIME_EXPONENT = 1.20
RAW_FIELDS = (
    "schema_version",
    "count",
    "repetition",
    "status",
    "generation_seconds",
    "validation_seconds",
    "serialization_seconds",
    "total_seconds",
    "microseconds_per_envelope",
    "peak_traced_mib",
    "serialized_bytes",
    "serialized_sha256",
    "admissible_count",
    "withheld_count",
    "semantic_validation_failures",
    "error",
)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _contracts() -> tuple[DatasetContract, MethodAdapterContract]:
    dataset = DatasetContract(
        dataset_id="jr12_synthetic_trace_contract",
        modality="generic_biomedical_trace",
        axes="TC",
        shape=(128, 2),
        dtype="float64",
        units="arbitrary_unit",
        frame_rate_hz=100.0,
        acquisition_id="jr12_synthetic_acquisition",
        independent_unit="source_recording",
        nesting=("source_recording", "lead", "condition"),
        reference_availability="exact_clean",
        evidence_boundary="clean_reference",
        mask_availability="none",
        alignment_state="spatiotemporal",
        provenance={"source": "deterministic JR12 synthetic contract workload"},
    ).validate()
    method = MethodAdapterContract(
        method_id="jr12_identity_trace_adapter",
        source_identity="sha256:" + "1" * 64,
        checkpoint_identity=None,
        training_reference_access="none",
        output_class="trace_only",
        supported_metric_domains=("generic_waveform_fidelity",),
        required_inputs=("trace",),
        produced_outputs=("trace",),
    ).validate()
    return dataset, method


def make_payload(index: int, seed: int = FROZEN_SEED) -> dict[str, Any]:
    """Create one deterministic mixed-semantics canonical envelope payload."""
    admissible = index % 2 == 0
    return {
        "$schema": "org.calcium-denoiser-audit.result-envelope",
        "schema_version": RESULT_ENVELOPE_SCHEMA_VERSION,
        "dataset_id": "jr12_synthetic_trace_contract",
        "method_id": "jr12_identity_trace_adapter",
        "run_id": f"run_jr12_{seed}_{index:08d}",
        "evidence_boundary": "clean_reference",
        "independent_unit": "source_recording",
        "nesting": ["source_recording", "lead", "condition"],
        "metric_id": f"generic_waveform_fidelity_{index:08d}",
        # Explicit synthetic record policy, not biomedical fidelity validation.
        # This benchmark measures structural envelope processing only.
        "metric_contract": MetricContract(
            f"generic_waveform_fidelity_{index:08d}", "synthetic_envelope_workload",
            ("clean_reference",), ("trace_only",), False,
        ).to_dict(),
        "metric_version": "1.0.0",
        "units": "dimensionless",
        "point_estimate": ((index % 1_000) + 1) / 1_001.0 if admissible else None,
        "uncertainty": None,
        "admissibility": "ADMISSIBLE" if admissible else "WITHHELD",
        "warnings": [] if admissible else ["synthetic unsupported-domain benchmark row"],
        "taxonomy_version": "1.0.0",
        "provenance_parents": ["sha256:" + "2" * 64],
        "hashes": {"synthetic_parent_sha256": "3" * 64},
    }


def run_repetition(count: int, repetition: int, seed: int = FROZEN_SEED) -> dict[str, Any]:
    """Execute one independent benchmark unit and retain failures as rows."""
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "count": count,
        "repetition": repetition,
        "status": "FAILED",
        "generation_seconds": "",
        "validation_seconds": "",
        "serialization_seconds": "",
        "total_seconds": "",
        "microseconds_per_envelope": "",
        "peak_traced_mib": "",
        "serialized_bytes": "",
        "serialized_sha256": "",
        "admissible_count": 0,
        "withheld_count": 0,
        "semantic_validation_failures": 0,
        "error": "",
    }
    tracemalloc.start()
    total_start = time.perf_counter()
    try:
        _contracts()
        generation_start = time.perf_counter()
        payloads = [make_payload(index, seed) for index in range(count)]
        generation_seconds = time.perf_counter() - generation_start

        validation_start = time.perf_counter()
        canonical: list[dict[str, Any]] = []
        failures = 0
        for payload in payloads:
            try:
                canonical.append(CanonicalResultEnvelope.from_dict(payload).to_dict())
            except ContractError:
                failures += 1
        validation_seconds = time.perf_counter() - validation_start

        serialization_start = time.perf_counter()
        serialized = json.dumps(
            {"schema_version": SCHEMA_VERSION, "count": count, "results": canonical},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        serialized_sha256 = hashlib.sha256(serialized).hexdigest()
        serialization_seconds = time.perf_counter() - serialization_start
        total_seconds = time.perf_counter() - total_start
        _, peak_bytes = tracemalloc.get_traced_memory()

        admissible_count = sum(item["admissibility"] == "ADMISSIBLE" for item in canonical)
        withheld_count = sum(item["admissibility"] == "WITHHELD" for item in canonical)
        if failures or len(canonical) != count or admissible_count + withheld_count != count:
            raise RuntimeError(
                f"semantic count mismatch: valid={len(canonical)} failures={failures} "
                f"admissible={admissible_count} withheld={withheld_count} expected={count}"
            )
        row.update(
            {
                "status": "PASS",
                "generation_seconds": generation_seconds,
                "validation_seconds": validation_seconds,
                "serialization_seconds": serialization_seconds,
                "total_seconds": total_seconds,
                "microseconds_per_envelope": total_seconds * 1_000_000.0 / count,
                "peak_traced_mib": peak_bytes / (1024.0 * 1024.0),
                "serialized_bytes": len(serialized),
                "serialized_sha256": serialized_sha256,
                "admissible_count": admissible_count,
                "withheld_count": withheld_count,
                "semantic_validation_failures": failures,
            }
        )
    except Exception as exc:  # retain failed benchmark units without hiding them
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["semantic_validation_failures"] = max(1, int(row["semantic_validation_failures"]))
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    return row


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("scaling fit requires distinct counts")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sxx
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x for x in xs]
    sst = sum((y - y_mean) ** 2 for y in ys)
    sse = sum((y - fit) ** 2 for y, fit in zip(ys, fitted))
    r_squared = 1.0 if sst == 0 else 1.0 - sse / sst
    return slope, r_squared


def summarize(rows: list[dict[str, Any]], counts: tuple[int, ...], repeats: int) -> dict[str, Any]:
    expected = {(count, rep) for count in counts for rep in range(1, repeats + 1)}
    observed = {(int(row["count"]), int(row["repetition"])) for row in rows}
    complete_matrix = observed == expected
    summary_rows: list[dict[str, Any]] = []
    for count in counts:
        selected = [row for row in rows if int(row["count"]) == count]
        passed = [row for row in selected if row["status"] == "PASS"]
        item: dict[str, Any] = {
            "count": count,
            "expected_repetitions": repeats,
            "observed_repetitions": len(selected),
            "passed_repetitions": len(passed),
            "failed_repetitions": len(selected) - len(passed),
        }
        for field in (
            "generation_seconds",
            "validation_seconds",
            "serialization_seconds",
            "total_seconds",
            "microseconds_per_envelope",
            "peak_traced_mib",
            "serialized_bytes",
        ):
            values = [float(row[field]) for row in passed]
            item[f"median_{field}"] = statistics.median(values) if values else None
            item[f"min_{field}"] = min(values) if values else None
            item[f"max_{field}"] = max(values) if values else None
        summary_rows.append(item)

    fit_rows = [item for item in summary_rows if item["count"] >= 1_000 and item["median_total_seconds"]]
    time_slope, time_r2 = _linear_fit(
        [math.log(item["count"]) for item in fit_rows],
        [math.log(item["median_total_seconds"]) for item in fit_rows],
    )
    memory_slope, memory_r2 = _linear_fit(
        [math.log(item["count"]) for item in fit_rows],
        [math.log(item["median_peak_traced_mib"]) for item in fit_rows],
    )
    largest = next(item for item in summary_rows if item["count"] == max(counts))
    zero_failures = all(
        row["status"] == "PASS" and int(row["semantic_validation_failures"]) == 0 for row in rows
    )
    criteria = {
        "complete_matrix": complete_matrix,
        "all_repetitions_pass_with_zero_semantic_failures": zero_failures,
        "largest_median_total_seconds_at_most_60": largest["median_total_seconds"] <= MAX_TOTAL_SECONDS,
        "largest_median_peak_mib_at_most_1024": largest["median_peak_traced_mib"] <= MAX_PEAK_MIB,
        "time_scaling_exponent_at_most_1p20": time_slope <= MAX_TIME_EXPONENT,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "classification": (
            "PRACTICALLY_SCALABLE_TO_50000_ENVELOPES_ON_TESTED_WINDOWS_PATH"
            if all(criteria.values())
            else "BOUNDED_SCALABILITY_LIMITATION"
        ),
        "counts": list(counts),
        "repeats": repeats,
        "seed": FROZEN_SEED,
        "criteria": criteria,
        "summary_rows": summary_rows,
        "scaling": {
            "fit_counts": [item["count"] for item in fit_rows],
            "time_loglog_slope": time_slope,
            "time_loglog_r_squared": time_r2,
            "memory_loglog_slope": memory_slope,
            "memory_loglog_r_squared": memory_r2,
        },
    }


def _environment() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "implementation": platform.python_implementation(),
        "tracemalloc_note": "Python allocator-traced peak; excludes interpreter and non-Python native allocations",
    }


def run_matrix(
    out: Path,
    counts: tuple[int, ...] = FROZEN_COUNTS,
    repeats: int = FROZEN_REPEATS,
    seed: int = FROZEN_SEED,
) -> dict[str, Any]:
    if not counts or repeats <= 0 or any(count <= 0 for count in counts):
        raise ValueError("counts and repeats must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "JR12_RAW_REPETITIONS.csv"
    rows: list[dict[str, Any]] = []
    if raw_path.is_file():
        with raw_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    existing = {(int(row["count"]), int(row["repetition"])) for row in rows}

    for count in counts:
        # Excluded warm-up. It never enters the raw or summary files.
        run_repetition(min(count, 1_000), 0, seed)
        for repetition in range(1, repeats + 1):
            key = (count, repetition)
            if key in existing:
                continue
            row = run_repetition(count, repetition, seed)
            rows.append(row)
            rows.sort(key=lambda item: (int(item["count"]), int(item["repetition"])))
            _write_csv(raw_path, rows, RAW_FIELDS)
            _atomic_json(out / "JR12_PROGRESS.json", {
                "schema_version": SCHEMA_VERSION,
                "completed_units": len(rows),
                "expected_units": len(counts) * repeats,
                "last_unit": {"count": count, "repetition": repetition, "status": row["status"]},
            })

    result = summarize(rows, counts, repeats)
    _atomic_json(out / "JR12_SUMMARY.json", result)
    summary_fields = list(result["summary_rows"][0])
    _write_csv(out / "JR12_SUMMARY.csv", result["summary_rows"], summary_fields)
    _atomic_json(out / "JR12_ENVIRONMENT.json", _environment())
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("journal_scientific_upgrade/JR12_CMPB_SCALABILITY"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_matrix(args.out)
    print(json.dumps({
        "status": result["status"],
        "classification": result["classification"],
        "summary": str(args.out / "JR12_SUMMARY.json"),
    }, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
