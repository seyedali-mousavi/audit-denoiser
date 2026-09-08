"""Execute the frozen JR10 ECG second-modality contract case.

WFDB is an adapter-only runtime dependency and is intentionally not added to
the standalone audit core.  The core registry is unchanged: contracts,
admissibility, run identity, and exact-output state are consumed through their
public interfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import audit_denoiser
from scipy.signal import butter, sosfiltfilt

from audit_denoiser.adapters import validate_contract_method_output
from audit_denoiser.contracts import (
    CanonicalResultEnvelope,
    ContractError,
    DatasetContract,
    MethodAdapterContract,
    contract_digest,
    evaluate_metric,
    validate_numeric_array,
)
from audit_denoiser.run_identity import (
    RunIdentityInputs,
    SafeRunRegistry,
    atomic_json,
    compute_run_id,
    source_tree_identity,
)
from audit_denoiser.schema import sha256
from audit_denoiser.taxonomy import CURRENT_TAXONOMY_VERSION


SCHEMA = "jr10-physionet-ecg-contract-case-v1"
START_SAMPLE = 108000
STOP_SAMPLE = 151200
EXPECTED_FS = 360.0
SNR_SUFFIXES = (("24", "e24"), ("18", "e18"), ("12", "e12"), ("6", "e06"), ("0", "e00"), ("-6", "e_6"))
WITHHELD_DOMAINS = (
    "psnr_to_clean",
    "ssim_to_clean",
    "amp_pres_to_clean",
    "clean_derived_trace_fidelity",
    "roi_trace_fidelity",
    "fluorescence_event_kinetics",
    "seam_energy_ratio",
    "residual_spatial_decomposition",
)


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def filter_waveform(noisy: np.ndarray, fs: float) -> np.ndarray:
    noisy = np.asarray(noisy, dtype=np.float64)
    if noisy.ndim != 2:
        raise ContractError("WRONG_AXES", f"expected [T,R], found {noisy.shape}")
    if not np.isfinite(noisy).all():
        raise ContractError("NONFINITE_OUTPUT", "input waveform contains non-finite values")
    if float(fs) != EXPECTED_FS:
        raise ContractError("FRAME_RATE_MISMATCH", f"expected {EXPECTED_FS}, found {fs}")
    sos = butter(4, (0.5, 40.0), btype="bandpass", fs=fs, output="sos")
    output = sosfiltfilt(sos, noisy, axis=0).astype(np.float64, copy=False)
    if not np.isfinite(output).all():
        raise ContractError("NONFINITE_OUTPUT", "filter output contains non-finite values")
    return output


def evaluate_waveforms(
    noisy: np.ndarray,
    estimate: np.ndarray,
    clean: np.ndarray | None,
    fs: float,
) -> list[dict[str, float | int]]:
    if clean is None:
        raise ContractError("MISSING_REFERENCE", "generic waveform fidelity requires exact clean data")
    noisy = np.asarray(noisy, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if noisy.shape != clean.shape or estimate.shape != clean.shape:
        raise ContractError(
            "FRAME_COUNT_MISMATCH",
            f"noisy={noisy.shape}, estimate={estimate.shape}, clean={clean.shape}",
        )
    if clean.ndim != 2:
        raise ContractError("WRONG_AXES", f"expected [T,R], found {clean.shape}")
    if float(fs) != EXPECTED_FS:
        raise ContractError("FRAME_RATE_MISMATCH", f"expected {EXPECTED_FS}, found {fs}")
    if not np.isfinite(noisy).all() or not np.isfinite(estimate).all() or not np.isfinite(clean).all():
        raise ContractError("NONFINITE_OUTPUT", "waveform arrays must be finite")

    noisy_c = noisy - noisy.mean(axis=0, keepdims=True)
    estimate_c = estimate - estimate.mean(axis=0, keepdims=True)
    clean_c = clean - clean.mean(axis=0, keepdims=True)
    rows: list[dict[str, float | int]] = []
    for lead in range(clean.shape[1]):
        reference = clean_c[:, lead]
        candidate = estimate_c[:, lead]
        raw = noisy_c[:, lead]
        robust_range = float(np.percentile(reference, 95) - np.percentile(reference, 5))
        if robust_range == 0.0:
            raise ContractError("ZERO_REFERENCE_RANGE", f"lead {lead}")
        denom = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
        if denom == 0.0:
            raise ContractError("ZERO_REFERENCE_RANGE", f"lead {lead} correlation denominator")
        signal_power = float(np.sum(reference**2))
        pre_error = float(np.sum((raw - reference) ** 2))
        post_error = float(np.sum((candidate - reference) ** 2))
        if signal_power == 0.0 or pre_error == 0.0 or post_error == 0.0:
            raise ContractError("UNDEFINED_SNR", f"lead {lead} has zero signal or error power")
        rows.append(
            {
                "lead_index": lead,
                "pearson_to_clean": float(np.dot(reference, candidate) / denom),
                "nrmse_to_clean": float(np.sqrt(np.mean((candidate - reference) ** 2)) / robust_range),
                "snr_improvement_db": float(10.0 * math.log10(signal_power / post_error) - 10.0 * math.log10(signal_power / pre_error)),
            }
        )
    return rows


def _digital_to_physical(digital: np.ndarray, adc_zero: list[float], adc_gain: list[float]) -> np.ndarray:
    digital = np.asarray(digital, dtype=np.float64)
    zero = np.asarray(adc_zero, dtype=np.float64)[None, :]
    gain = np.asarray(adc_gain, dtype=np.float64)[None, :]
    if digital.ndim != 2 or zero.shape[1] != digital.shape[1] or gain.shape[1] != digital.shape[1]:
        raise ContractError("WRONG_AXES", "digital signal and ADC metadata disagree")
    if np.any(gain == 0) or not np.isfinite(gain).all():
        raise ContractError("MISSING_METADATA", "finite nonzero ADC gain is required")
    return (digital - zero) / gain


def _load_record(base: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - exercised in the isolated runtime
        raise RuntimeError("JR10 requires the adapter-only 'wfdb' package") from exc
    record = wfdb.rdrecord(str(base), sampfrom=START_SAMPLE, sampto=STOP_SAMPLE, physical=False)
    values = _digital_to_physical(record.d_signal, record.adc_zero, record.adc_gain)
    metadata = {
        "fs": float(record.fs),
        "sig_name": list(record.sig_name or []),
        "units": list(record.units or []),
        "adc_gain": [float(value) for value in record.adc_gain],
        "adc_zero": [int(value) for value in record.adc_zero],
        "physical_conversion": "(d_signal - adc_zero) / adc_gain",
        "shape": list(values.shape),
    }
    return values, metadata


def _alignment_evidence(clean_base: Path, noisy_base: Path) -> dict[str, Any]:
    import wfdb

    clean = wfdb.rdann(str(clean_base), "atr")
    noisy = wfdb.rdann(str(noisy_base), "atr")
    count_equal = len(clean.sample) == len(noisy.sample)
    symbols_equal = clean.symbol == noisy.symbol
    if count_equal:
        shifts = np.asarray(noisy.sample, dtype=np.int64) - np.asarray(clean.sample, dtype=np.int64)
        shifted = shifts != 0
        shift_count = int(np.count_nonzero(shifted))
        max_shift = int(np.max(np.abs(shifts[shifted]))) if shifted.any() else 0
    else:
        shift_count, max_shift = -1, -1

    clean_record = wfdb.rdrecord(str(clean_base), sampfrom=0, sampto=START_SAMPLE, physical=False)
    noisy_record = wfdb.rdrecord(str(noisy_base), sampfrom=0, sampto=START_SAMPLE, physical=False)
    clean_zeroed = np.asarray(clean_record.d_signal, dtype=np.int64) - np.asarray(clean_record.adc_zero, dtype=np.int64)[None, :]
    noisy_zeroed = np.asarray(noisy_record.d_signal, dtype=np.int64) - np.asarray(noisy_record.adc_zero, dtype=np.int64)[None, :]
    pre_noise_equal = bool(np.array_equal(clean_zeroed, noisy_zeroed))
    return {
        "annotation_count_equal": count_equal,
        "annotation_symbols_equal": symbols_equal,
        "annotation_sample_arrays_equal": bool(np.array_equal(clean.sample, noisy.sample)),
        "annotation_shifted_sample_count": shift_count,
        "annotation_max_abs_shift_samples": max_shift,
        "pre_noise_digital_signal_equal_after_adc_zero": pre_noise_equal,
        "alignment_eligible": bool(count_equal and symbols_equal and pre_noise_equal),
    }


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _contract_for(record: str, snr: str, clean_meta: dict[str, Any], provenance: dict[str, Any]) -> DatasetContract:
    return DatasetContract(
        dataset_id=f"physionet-nstdb-{record}-snr{snr}",
        modality="electrocardiography",
        axes="TR",
        shape=(STOP_SAMPLE - START_SAMPLE, 2),
        dtype="float64",
        units="millivolt",
        frame_rate_hz=float(clean_meta["fs"]),
        acquisition_id=f"mitdb-source-record-{record}",
        independent_unit="source_recording",
        nesting=("source_recording", "lead", "sample"),
        reference_availability="exact_clean",
        evidence_boundary="clean_reference",
        mask_availability="none",
        alignment_state="spatiotemporal",
        provenance=provenance,
    ).validate()


def _method_contract(script_path: Path) -> MethodAdapterContract:
    return MethodAdapterContract(
        method_id="external-ecg-butterworth-bandpass-0p5-40hz-order4",
        source_identity=f"sha256:{sha256(script_path)}",
        checkpoint_identity=None,
        training_reference_access="none",
        output_class="trace_only",
        supported_metric_domains=("generic_waveform_fidelity",),
        required_inputs=("finite_time_by_lead_waveform", "sampling_rate_hz"),
        produced_outputs=("filtered_time_by_lead_waveform",),
    ).validate()


def _envelopes(
    dataset: DatasetContract,
    method: MethodAdapterContract,
    run_id: str,
    record_metrics: dict[str, float] | None,
    hashes: dict[str, str],
    *, analysis: Callable[[], dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    units = {
        "waveform_pearson_to_clean": "correlation",
        "waveform_nrmse_to_clean": "normalized_error",
        "waveform_snr_improvement_db": "dB",
    }
    fields = {
        "waveform_pearson_to_clean": "pearson_to_clean",
        "waveform_nrmse_to_clean": "nrmse_to_clean",
        "waveform_snr_improvement_db": "snr_improvement_db",
    }
    cache = record_metrics

    def value_for(field: str) -> float:
        nonlocal cache
        if cache is None:
            if analysis is None:
                raise ContractError("MISSING_ANALYSIS", "admitted ECG request requires an analysis callback")
            cache = analysis()
        return cache[field]

    output = []
    for metric_id, field in fields.items():
        output.append(evaluate_metric(
            metric_id, dataset, method, run_id=run_id, metric_version="jr10-v2-contextual-gate",
            units=units[metric_id], analysis=lambda field=field: value_for(field),
            warnings=("descriptive two-record cross-modality case; leads are nested",),
            taxonomy_version=CURRENT_TAXONOMY_VERSION,
            provenance_parents=tuple(sorted(hashes.values())), hashes=hashes,
        ).to_dict())
    for domain in WITHHELD_DOMAINS:
        def unavailable() -> float:
            raise ContractError("MISSING_ANALYSIS", "this adapter implements only generic waveform endpoints")
        output.append(evaluate_metric(
            domain, dataset, method, run_id=run_id, metric_version="jr10-domain-admissibility-v2",
            units="not_applicable", analysis=unavailable, taxonomy_version=CURRENT_TAXONOMY_VERSION,
            provenance_parents=tuple(sorted(hashes.values())), hashes=hashes,
        ).to_dict())
    return output


def evaluate_ecg_waveforms(
    noisy: np.ndarray, estimate: np.ndarray, clean: np.ndarray | None, fs: float,
    dataset: DatasetContract, method: MethodAdapterContract, run_id: str,
    hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, float | int]], dict[str, float]]:
    """Actual ECG producer: gate the shared analysis before any waveform metric runs."""
    dataset.validate()
    if dataset.axes != "TR":
        raise ContractError("WRONG_AXES", "ECG producer requires declared time-by-lead TR axes")
    dataset.validate_array(noisy)
    validate_numeric_array(noisy)
    validate_numeric_array(estimate)
    if np.asarray(estimate).shape != dataset.shape:
        raise ContractError("FRAME_COUNT_MISMATCH", "estimated waveform differs from declared dataset shape")
    if float(fs) != dataset.frame_rate_hz or float(fs) != EXPECTED_FS:
        raise ContractError("FRAME_RATE_MISMATCH", "observed ECG rate differs from declared/fixed rate")
    lead_metrics: list[dict[str, float | int]] = []
    record_metrics: dict[str, float] = {}
    observed = dict(hashes)
    if clean is not None:
        validate_numeric_array(clean)
        if np.asarray(clean).shape != dataset.shape:
            raise ContractError("FRAME_COUNT_MISMATCH", "clean waveform differs from declared dataset shape")
        digest = _array_sha256(clean)
        if "clean_reference_sha256" in observed and observed["clean_reference_sha256"] != digest:
            raise ContractError("EVIDENCE_HASH_MISMATCH", "provided digest differs from actual clean array")
        observed["clean_reference_sha256"] = digest

    def analysis() -> dict[str, float]:
        lead_metrics.extend(evaluate_waveforms(noisy, estimate, clean, fs))
        record_metrics.update({
            key: float(np.mean([float(row[key]) for row in lead_metrics]))
            for key in ("pearson_to_clean", "nrmse_to_clean", "snr_improvement_db")
        })
        return record_metrics

    envelopes = _envelopes(dataset, method, run_id, None, observed, analysis=analysis)
    return envelopes, lead_metrics, record_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    acquisition = json.loads((args.data_root / "acquisition_manifest.json").read_text(encoding="utf-8"))
    if acquisition.get("status") != "COMPLETE" or acquisition.get("full_database_mirrored") is not False:
        raise RuntimeError("JR10 acquisition manifest is incomplete or unbounded")
    script_path = Path(__file__).resolve()
    method = _method_contract(script_path)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.evidence_root / "method_contract.json", method.to_dict())
    method_contract_hash = sha256(args.evidence_root / "method_contract.json")
    code_identity = source_tree_identity(Path(audit_denoiser.__file__).parent)
    registry = SafeRunRegistry(args.evidence_root / "runs")
    lead_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    deterministic_check: dict[str, Any] | None = None

    for record in ("118", "119"):
        clean_base = args.data_root / "mitdb" / record
        clean, clean_meta = _load_record(clean_base)
        if clean_meta["fs"] != EXPECTED_FS or clean.shape != (STOP_SAMPLE - START_SAMPLE, 2):
            raise ContractError("FRAME_RATE_MISMATCH", f"clean {record}: {clean_meta}")
        for snr, suffix in SNR_SUFFIXES:
            noisy_id = f"{record}{suffix}"
            noisy_base = args.data_root / "nstdb" / noisy_id
            noisy, noisy_meta = _load_record(noisy_base)
            if noisy_meta["fs"] != EXPECTED_FS or noisy.shape != clean.shape:
                raise ContractError("FRAME_RATE_MISMATCH", f"noisy {noisy_id}: {noisy_meta}")
            alignment = _alignment_evidence(clean_base, noisy_base)
            if not alignment["alignment_eligible"]:
                raise ContractError("ALIGNMENT_MISMATCH", f"{record} vs {noisy_id}: {alignment}")
            estimate = filter_waveform(noisy, noisy_meta["fs"])
            cell_id = f"{record}_snr_{snr.replace('-', 'm')}"
            output_path = args.output_root / cell_id / "filtered.npy"
            _atomic_npy(output_path, estimate)
            provenance = {
                "source": "PhysioNet MIT-BIH Noise Stress Test Database and MIT-BIH Arrhythmia Database",
                "clean_record": record,
                "noisy_record": noisy_id,
                "segment_samples": [START_SAMPLE, STOP_SAMPLE],
                "segment_seconds": [300, 420],
                "nominal_snr_db": int(snr),
                "alignment_evidence": alignment,
                "clean_metadata": clean_meta,
                "noisy_metadata": noisy_meta,
                "clean_reference_sha256": _array_sha256(clean),
            }
            dataset = _contract_for(record, snr, clean_meta, provenance)
            contract_path = args.evidence_root / "contracts" / f"{cell_id}_dataset.json"
            atomic_json(contract_path, dataset.to_dict())
            validation = validate_contract_method_output(method, output_path, dataset)
            if validation.get("status") != "SUPPORTED":
                raise RuntimeError(f"output contract failed for {cell_id}: {validation}")
            hashes = {
                "clean_dat_sha256": sha256(args.data_root / "mitdb" / f"{record}.dat"),
                "noisy_dat_sha256": sha256(args.data_root / "nstdb" / f"{noisy_id}.dat"),
                "output_sha256": sha256(output_path),
                "dataset_contract_file_sha256": sha256(contract_path),
                "method_contract_file_sha256": method_contract_hash,
                "dataset_contract_sha256": contract_digest(dataset),
                "method_contract_sha256": contract_digest(method),
                "clean_reference_sha256": _array_sha256(clean),
            }
            identity = RunIdentityInputs(
                protocol_identity=SCHEMA,
                dataset_manifest_identity=hashes["dataset_contract_sha256"],
                method_source_identity=method.source_identity,
                checkpoint_identity=None,
                config={"filter": "butterworth", "order": 4, "band_hz": [0.5, 40.0], "segment": [START_SAMPLE, STOP_SAMPLE]},
                seed=0,
                evidence_boundary=dataset.evidence_boundary,
                code_identity=code_identity,
                parent_artifacts=hashes,
            )
            run_id = registry.run_id_for(identity, ["cell_result.json"])
            run_dir = registry.root / run_id
            result_envelopes, metrics, record_metrics = evaluate_ecg_waveforms(
                noisy, estimate, clean, noisy_meta["fs"], dataset, method, run_id, hashes,
            )
            for row in metrics:
                lead_rows.append({"record": record, "snr_db": int(snr), **row})
            cell_payload = {
                "schema_version": SCHEMA,
                "status": "PASS",
                "cell_id": cell_id,
                "record": record,
                "snr_db": int(snr),
                "validation": validation,
                "lead_metrics": metrics,
                "record_metrics": record_metrics,
                "withheld_domains": list(WITHHELD_DOMAINS),
                "result_envelopes": result_envelopes,
                "hashes": hashes,
            }
            if run_dir.exists():
                registry.require_complete(run_dir)
                existing = json.loads((run_dir / "cell_result.json").read_text(encoding="utf-8"))
                if existing != cell_payload:
                    raise RuntimeError(f"stale deterministic result for {cell_id}")
            else:
                run_dir = registry.prepare(identity, ["cell_result.json"])
                atomic_json(run_dir / "cell_result.json", cell_payload)
                registry.complete(run_dir)
            record_rows.append({"record": record, "snr_db": int(snr), "run_id": run_id, **record_metrics})
            if record == "118" and snr == "0":
                repeat = filter_waveform(noisy, noisy_meta["fs"])
                deterministic_check = {
                    "cell_id": cell_id,
                    "array_equal": bool(np.array_equal(estimate, repeat)),
                    "first_array_sha256": _array_sha256(estimate),
                    "second_array_sha256": _array_sha256(repeat),
                    "same_array_sha256": _array_sha256(estimate) == _array_sha256(repeat),
                    "run_id_first": run_id,
                    "run_id_second": registry.run_id_for(identity, ["cell_result.json"]),
                    "same_run_id": run_id == registry.run_id_for(identity, ["cell_result.json"]),
                }
            print(f"PASS {cell_id} r={record_metrics['pearson_to_clean']:.6f} nrmse={record_metrics['nrmse_to_clean']:.6f} dSNR={record_metrics['snr_improvement_db']:+.6f}", flush=True)

    if deterministic_check is None or not all(
        deterministic_check[key] for key in ("array_equal", "same_array_sha256", "same_run_id")
    ):
        raise RuntimeError("118e00 deterministic rerun failed")
    lead_frame = pd.DataFrame(lead_rows).sort_values(["record", "snr_db", "lead_index"], ascending=[True, False, True])
    record_frame = pd.DataFrame(record_rows).sort_values(["record", "snr_db"], ascending=[True, False])
    lead_frame.to_csv(args.evidence_root / "JR10_LEAD_LEVEL.csv", index=False)
    record_frame.to_csv(args.evidence_root / "JR10_RECORD_LEVEL.csv", index=False)
    by_snr = []
    for snr, group in record_frame.groupby("snr_db", sort=True):
        item: dict[str, Any] = {"snr_db": int(snr), "n_source_recordings": int(group.record.nunique())}
        for metric in ("pearson_to_clean", "nrmse_to_clean", "snr_improvement_db"):
            values = group[metric].to_numpy(float)
            item[f"{metric}_record_mean"] = float(np.mean(values))
            item[f"{metric}_record_min"] = float(np.min(values))
            item[f"{metric}_record_max"] = float(np.max(values))
        by_snr.append(item)
    summary = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "classification": "BOUNDED_CROSS_MODALITY_CONTRACT_VALIDATION",
        "independent_unit": "source ECG recording",
        "n_source_recordings": 2,
        "n_cells": 12,
        "n_leads_per_record": 2,
        "primary_contract_cells_passed": 12,
        "image_calcium_domains_withheld_per_cell": len(WITHHELD_DOMAINS),
        "deterministic_rerun": deterministic_check,
        "by_snr": by_snr,
        "limitations": [
            "two source recordings only; descriptive, not population inference",
            "fixed Butterworth adapter is an integration case, not an optimal ECG denoiser",
            "no beat-detection, arrhythmia, or clinical-performance endpoint",
            "does not resolve Linux or independent-adopter validation",
        ],
    }
    atomic_json(args.evidence_root / "JR10_SUMMARY.json", summary)
    report = f"""# JR10 — PhysioNet ECG second-modality contract case

**Status:** PASS  
**Classification:** bounded cross-modality contract validation

All 12 predeclared record/SNR cells produced finite, shape-preserving trace outputs and canonical result envelopes.
Generic waveform fidelity was admitted only under exact clean-reference evidence; {len(WITHHELD_DOMAINS)}
calcium/image-specific domains per cell were value-free and explicitly withheld. The 118e00 deterministic rerun
preserved both array identity and content-derived run identity.

The two source recordings are the independent units. Leads and SNR cells are nested. The waveform values in the
CSV are descriptive adapter diagnostics, not clinical validation or a denoiser leaderboard.
"""
    (args.evidence_root / "JR10_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
