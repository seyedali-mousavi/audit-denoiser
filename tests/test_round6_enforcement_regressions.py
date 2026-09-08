"""Independent pre-submission probes, frozen before the v9 repairs.

These tests reproduce public CLI and actual producer boundary failures in v1.2.0.
They intentionally assert scientific semantics, not a particular repair API.
"""
from dataclasses import replace
import json

import numpy as np
import pytest

from audit_denoiser.cli import main
from audit_denoiser.contracts import (
    ContractError, DatasetContract, MethodAdapterContract, metric_admissibility, write_json,
)
from audit_denoiser.reference_free import run_reference_free_audit
from tools.run_physionet_jr10 import _envelopes


def dataset(shape=(8, 4, 1024)):
    return DatasetContract(
        'review-fixture', 'movie', 'THW', shape, 'float32', 'arbitrary', 30.0,
        'fixture-acquisition', 'acquisition', ('acquisition', 'frame', 'pixel'),
        'none', 'reference_free', 'none', 'not_applicable',
        {'source': 'independent review fixture'},
    ).validate()


def method():
    return MethodAdapterContract(
        'review-method', 'review-source', None, 'none', 'full_movie',
        ('reference_free_movie',), (), ('movie',),
    ).validate()


def cli_paths(tmp_path, data, adapter, array):
    movie, dp, mp = tmp_path / 'output.npy', tmp_path / 'dataset.json', tmp_path / 'method.json'
    np.save(movie, array, allow_pickle=False)
    write_json(data.to_dict(), dp)
    write_json(adapter.to_dict(), mp)
    return movie, dp, mp


def test_declared_thw_survives_wide_short_movie_cli(tmp_path):
    data = dataset()
    array = np.random.default_rng(2026).normal(size=data.shape).astype(np.float32)
    array += np.arange(data.shape[0], dtype=np.float32)[:, None, None] * 10
    movie, dp, mp = cli_paths(tmp_path, data, method(), array)
    assert main([
        'reference-free', '--name', 'wide-short', '--movie', str(movie),
        '--dataset-contract', str(dp), '--method-contract', str(mp),
        '--out', str(tmp_path / 'audit'), '--max-frames', '8',
    ]) == 0
    output = next((tmp_path / 'audit' / 'runs').glob('*/reference_free_results.json'))
    rows = {row['metric_id']: row for row in json.loads(output.read_text())}
    # Compute independently from declared temporal axes, without the package loader.
    trace = np.mean(array, axis=(1, 2), dtype=np.float64)
    expected = float(np.corrcoef(trace[1:], trace[:-1])[0, 1])
    assert rows['global_trace_lag1_autocorrelation']['point_estimate'] == pytest.approx(expected, abs=1e-10)
    assert rows['seam_energy_ratio']['admissibility'] == 'ADMISSIBLE'


def test_string_trace_is_not_certified_by_validate_output_cli(tmp_path):
    data = replace(dataset(), axes='TR', shape=(8, 4))
    adapter = replace(method(), output_class='trace_only', supported_metric_domains=('trace_fidelity',))
    movie, dp, mp = cli_paths(tmp_path, data, adapter, np.full((8, 4), 'x', dtype='<U1'))
    with pytest.raises(ContractError, match='DTYPE|NUMERIC'):
        main(['validate-output', '--name', 'string-trace', '--output', str(movie),
              '--dataset-contract', str(dp), '--method-contract', str(mp), '--out', str(tmp_path / 'audit')])
    assert not list((tmp_path / 'audit').glob('runs/*/COMPLETE'))


@pytest.mark.parametrize('metric_id', [
    'waveform_pearson_to_clean', 'waveform_nrmse_to_clean', 'waveform_snr_improvement_db',
])
def test_actual_ecg_clean_metric_ids_require_clean_evidence(metric_id):
    assert metric_admissibility(metric_id, 'reference_free')[0] == 'WITHHELD'


def test_unknown_metric_does_not_default_to_admissible():
    assert metric_admissibility('unregistered_clinical_score', 'reference_free')[0] == 'WITHHELD'


def test_ecg_producer_cannot_emit_values_outside_evidence_and_method_domain():
    data = dataset()
    adapter = replace(method(), output_class='trace_only', supported_metric_domains=('unrelated_domain',))
    rows = _envelopes(data, adapter, 'review-run',
                      {'pearson_to_clean': .9, 'nrmse_to_clean': .1, 'snr_improvement_db': 1.0}, {})
    assert all(row['admissibility'] != 'ADMISSIBLE' and row['point_estimate'] is None for row in rows)


def test_reference_free_producer_rejects_inconsistent_trace_output_class(tmp_path):
    data = dataset((8, 4, 32))
    movie = tmp_path / 'movie.npy'
    np.save(movie, np.ones(data.shape, dtype=np.float32))
    adapter = replace(method(), output_class='trace_only')
    with pytest.raises(ContractError, match='OUTPUT_CLASS|WRONG_AXES'):
        run_reference_free_audit(movie_path=movie, dataset=data, method=adapter, run_id='review-run')


def test_ecg_producer_rejects_mismatched_concrete_clean_reference_binding():
    data = replace(dataset(), evidence_boundary='clean_reference', reference_availability='exact_clean',
                   provenance={'source': 'review fixture', 'clean_reference_sha256': '2' * 64})
    adapter = replace(method(), output_class='trace_only', supported_metric_domains=('generic_waveform_fidelity',))
    with pytest.raises(ContractError, match='EVIDENCE_HASH_MISMATCH'):
        _envelopes(data, adapter, 'review-run',
                   {'pearson_to_clean': .9, 'nrmse_to_clean': .1, 'snr_improvement_db': 1.0},
                   {'clean_reference_sha256': '3' * 64})


def test_legitimate_numeric_output_conversion_remains_supported(tmp_path):
    data = dataset((8, 4, 32))
    data = replace(data, dtype='uint16')
    movie, dp, mp = cli_paths(tmp_path, data, method(), np.ones(data.shape, dtype=np.float32))
    assert main(['validate-output', '--name', 'numeric-conversion', '--output', str(movie),
                 '--dataset-contract', str(dp), '--method-contract', str(mp), '--out', str(tmp_path / 'audit')]) == 0
