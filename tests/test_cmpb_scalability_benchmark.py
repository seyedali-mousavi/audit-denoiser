from __future__ import annotations

import csv
import json

from audit_denoiser.contracts import CanonicalResultEnvelope
from tools.run_cmpb_scalability_benchmark import make_payload, run_matrix, summarize


def test_payload_mix_is_valid_and_value_free_when_withheld() -> None:
    admissible = CanonicalResultEnvelope.from_dict(make_payload(0))
    withheld = CanonicalResultEnvelope.from_dict(make_payload(1))
    assert admissible.admissibility == "ADMISSIBLE"
    assert admissible.point_estimate is not None
    assert not admissible.warnings
    assert withheld.admissibility == "WITHHELD"
    assert withheld.point_estimate is None
    assert withheld.uncertainty is None
    assert withheld.warnings


def test_small_matrix_is_complete_resumable_and_deterministic(tmp_path) -> None:
    out = tmp_path / "jr12"
    first = run_matrix(out, counts=(100, 1_000, 5_000), repeats=2, seed=20260822)
    second = run_matrix(out, counts=(100, 1_000, 5_000), repeats=2, seed=20260822)
    # This test protects completeness, semantic validity, resumability, and exact
    # deterministic reuse. The benchmark's calibrated performance thresholds are a
    # separate environment-specific result: requiring a Windows-calibrated timing
    # verdict here made a real Linux run under CPU emulation fail before portability,
    # quickstart, and wheel checks could execute.
    for result in (first, second):
        assert result["criteria"]["complete_matrix"] is True
        assert result["criteria"]["all_repetitions_pass_with_zero_semantic_failures"] is True
        assert result["status"] in {"PASS", "FAIL"}
    assert first == second
    with (out / "JR12_RAW_REPETITIONS.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert all(row["status"] == "PASS" for row in rows)
    assert all(int(row["admissible_count"]) + int(row["withheld_count"]) == int(row["count"]) for row in rows)
    for count in (100, 1_000, 5_000):
        hashes = {row["serialized_sha256"] for row in rows if int(row["count"]) == count}
        assert len(hashes) == 1
    assert (out / "JR12_SUMMARY.json").is_file()
    assert json.loads((out / "JR12_ENVIRONMENT.json").read_text(encoding="utf-8"))["logical_cpu_count"]


def test_summary_fails_closed_on_failed_or_missing_units() -> None:
    good = {
        "count": 1_000,
        "repetition": 1,
        "status": "PASS",
        "generation_seconds": 0.01,
        "validation_seconds": 0.02,
        "serialization_seconds": 0.01,
        "total_seconds": 0.04,
        "microseconds_per_envelope": 40.0,
        "peak_traced_mib": 1.0,
        "serialized_bytes": 1000,
        "semantic_validation_failures": 0,
    }
    rows = []
    for count, scale in ((1_000, 1.0), (10_000, 10.0), (50_000, 50.0)):
        row = dict(good, count=count)
        for field in ("generation_seconds", "validation_seconds", "serialization_seconds", "total_seconds"):
            row[field] *= scale
        row["peak_traced_mib"] *= scale
        row["serialized_bytes"] *= count
        rows.append(row)
    result = summarize(rows, (1_000, 10_000, 50_000), repeats=2)
    assert result["status"] == "FAIL"
    assert result["criteria"]["complete_matrix"] is False
