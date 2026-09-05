from __future__ import annotations

import numpy as np
import pytest

from audit_denoiser.contracts import ContractError
from tools.run_physionet_jr10 import (
    EXPECTED_FS,
    _digital_to_physical,
    evaluate_waveforms,
    filter_waveform,
)


def _signals() -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(3600, dtype=np.float64) / EXPECTED_FS
    clean = np.stack((np.sin(2 * np.pi * 1.2 * time), np.cos(2 * np.pi * 1.0 * time)), axis=1)
    noisy = clean + 0.1 * np.sin(2 * np.pi * 60.0 * time)[:, None]
    return clean, noisy


def test_filter_and_generic_metrics_are_finite() -> None:
    clean, noisy = _signals()
    estimate = filter_waveform(noisy, EXPECTED_FS)
    rows = evaluate_waveforms(noisy, estimate, clean, EXPECTED_FS)
    assert len(rows) == 2
    assert all(np.isfinite(row["pearson_to_clean"]) for row in rows)
    assert all(np.isfinite(row["nrmse_to_clean"]) for row in rows)
    assert all(np.isfinite(row["snr_improvement_db"]) for row in rows)


def test_missing_clean_fails_closed() -> None:
    _, noisy = _signals()
    with pytest.raises(ContractError, match="MISSING_REFERENCE"):
        evaluate_waveforms(noisy, noisy, None, EXPECTED_FS)


def test_shape_nonfinite_and_rate_mismatches_fail_closed() -> None:
    clean, noisy = _signals()
    with pytest.raises(ContractError, match="FRAME_COUNT_MISMATCH"):
        evaluate_waveforms(noisy[:-1], noisy, clean, EXPECTED_FS)
    broken = noisy.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ContractError, match="NONFINITE_OUTPUT"):
        evaluate_waveforms(broken, noisy, clean, EXPECTED_FS)
    with pytest.raises(ContractError, match="FRAME_RATE_MISMATCH"):
        filter_waveform(noisy, 250.0)


def test_adc_zero_conversion_reconciles_clean_and_nstdb_encodings() -> None:
    clean_digital = np.asarray([[825, 930], [821, 933]], dtype=np.int16)
    noisy_digital = clean_digital - 1024
    clean = _digital_to_physical(clean_digital, [1024, 1024], [200, 200])
    noisy = _digital_to_physical(noisy_digital, [0, 0], [200, 200])
    assert np.array_equal(clean, noisy)
