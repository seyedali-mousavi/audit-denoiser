"""Context-bound integration tests for v1.3.0; synthetic software fixtures only."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from audit_denoiser.adapters import validate_contract_method_output
from audit_denoiser.contracts import (
    CanonicalResultEnvelope, ContractError, DatasetContract, MethodAdapterContract, MetricContract,
    contract_digest, contextual_admissibility, evaluate_metric, validate_result_against_context,
)
from audit_denoiser.run_identity import SafeRunRegistry
from tools import run_physionet_jr10 as ecg


def waveform_context(clean=None):
    data = DatasetContract('waveform', 'ecg', 'TR', (3600, 2), 'float64', 'mV', 360.0,
        'record', 'record', ('record', 'lead', 'sample'), 'none', 'reference_free', 'none',
        'not_applicable', {'source': 'generated software fixture'}).validate()
    method = MethodAdapterContract('trace-method', 'fixture-code', None, 'none', 'trace_only',
        ('external_integrity', 'generic_waveform_fidelity'), (), ('traces',)).validate()
    if clean is not None:
        data = replace(data, reference_availability='exact_clean', evidence_boundary='clean_reference',
            provenance={'source': 'generated software fixture', 'clean_reference_sha256': ecg._array_sha256(clean)})
    return data, method


def external_policy():
    return MetricContract('external_trace_integrity', 'external_integrity', ('reference_free',), ('trace_only',))


def evaluate(data, method, **kwargs):
    return evaluate_metric('external_trace_integrity', data, method, run_id='fixture-run', metric_version='fixture-v1',
        units='ratio', taxonomy_version='fixture-v1', metric_contract=external_policy(), **kwargs)


def test_external_metric_roundtrips_without_core_registration():
    data, method = waveform_context()
    calls = []
    result = evaluate(data, method, analysis=lambda: calls.append('called') or .5)
    assert calls == ['called']
    assert result.admissibility == 'ADMISSIBLE'
    assert result.hashes['dataset_contract_sha256'] == contract_digest(data)
    assert result.hashes['method_contract_sha256'] == contract_digest(method)
    restored = CanonicalResultEnvelope.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    assert validate_result_against_context(restored, data, method) == result


@pytest.mark.parametrize('incompatibility', ['evidence', 'domain', 'class'])
def test_incompatible_context_never_calls_analysis(incompatibility):
    data, method = waveform_context()
    if incompatibility == 'evidence':
        data = replace(data, evidence_boundary='pseudo_reference', reference_availability='pseudo_reference')
    elif incompatibility == 'domain':
        method = replace(method, supported_metric_domains=('other',))
    else:
        method = replace(method, output_class='full_movie')
    def forbidden():
        pytest.fail('inadmissible analysis was invoked')
    result = evaluate(data, method, analysis=forbidden)
    assert result.admissibility == 'WITHHELD' and result.point_estimate is None and result.warnings


def test_unknown_metric_never_calls_analysis():
    data, method = waveform_context()
    def forbidden():
        pytest.fail('unknown metric was executed')
    result = evaluate_metric('unknown', data, method, run_id='run', metric_version='v1', units='ratio',
        taxonomy_version='v1', analysis=forbidden)
    assert result.admissibility == 'WITHHELD' and result.point_estimate is None


@pytest.mark.parametrize('field', ['dataset_contract_sha256', 'method_contract_sha256', 'metric_contract_sha256'])
def test_conflicting_contract_digests_stop_before_analysis(field):
    data, method = waveform_context()
    with pytest.raises(ContractError, match='EVIDENCE_HASH_MISMATCH'):
        evaluate(data, method, hashes={field: '0'*64}, analysis=lambda: pytest.fail('callback executed'))


@pytest.mark.parametrize('mutation', ['dataset_shape', 'method_domain', 'method_id', 'hierarchy'])
def test_emission_context_cannot_be_replaced(mutation):
    data, method = waveform_context()
    result = evaluate(data, method, analysis=lambda: .5)
    if mutation == 'dataset_shape':
        data = replace(data, shape=(3600, 3))
    elif mutation == 'method_domain':
        method = replace(method, supported_metric_domains=('other',))
    elif mutation == 'method_id':
        result = replace(result, method_id='another-method')
    else:
        result = replace(result, independent_unit='lead', nesting=('lead', 'sample'))
    with pytest.raises(ContractError):
        validate_result_against_context(result, data, method)


def test_explicit_policy_cannot_weaken_a_builtin_metric():
    data, method = waveform_context()
    weak = MetricContract('waveform_pearson_to_clean', 'external_integrity', ('reference_free',), ('trace_only',))
    with pytest.raises(ContractError, match='METRIC_CONTRACT_CONFLICT'):
        contextual_admissibility(weak.metric_id, data, method, metric_contract=weak)


def test_undefined_estimate_is_value_free_and_strict_json():
    data, method = waveform_context()
    result = evaluate(data, method, analysis=lambda: float('nan'))
    assert result.admissibility == 'WITHHELD' and result.point_estimate is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_nonfinite_uncertainty_cannot_be_emitted():
    data, method = waveform_context()
    with pytest.raises(ContractError, match='INVALID_UNCERTAINTY'):
        evaluate(data, method, analysis=lambda: (.5, {'upper': float('nan')}))


@pytest.mark.parametrize('dtype', ['bool', 'complex128', '<U1'])
@pytest.mark.parametrize('output_class', ['trace_only', 'full_movie'])
def test_nonnumeric_output_classes_are_rejected(tmp_path, dtype, output_class):
    data, method = waveform_context()
    if output_class == 'full_movie':
        data = replace(data, axes='THW', shape=(8, 4, 32))
        method = replace(method, output_class='full_movie')
    array = np.ones(data.shape).astype(dtype)
    path = tmp_path / 'output.npy'
    np.save(path, array)
    with pytest.raises(ContractError, match='NONNUMERIC_DTYPE'):
        validate_contract_method_output(method, path, data)


def signals():
    t = np.arange(3600) / 360.0
    clean = np.stack((np.sin(2*np.pi*1.2*t), np.cos(2*np.pi*t)), axis=1)
    noisy = clean + .1*np.sin(2*np.pi*60*t)[:, None]
    return noisy, ecg.filter_waveform(noisy, 360.0), clean


def test_actual_ecg_producer_binds_clean_array_and_emits_three_values_eight_withholds():
    noisy, estimate, clean = signals()
    data, method = waveform_context(clean)
    rows, lead_metrics, record_metrics = ecg.evaluate_ecg_waveforms(noisy, estimate, clean, 360.0, data, method, 'run', {})
    assert len(lead_metrics) == 2 and len(record_metrics) == 3
    assert sum(r['admissibility']=='ADMISSIBLE' for r in rows) == 3
    assert sum(r['admissibility']=='WITHHELD' and r['point_estimate'] is None for r in rows) == 8
    for row in rows:
        validate_result_against_context(CanonicalResultEnvelope.from_dict(row), data, method)


def test_actual_ecg_producer_does_not_score_reference_free_data(monkeypatch):
    noisy, estimate, clean = signals()
    data, method = waveform_context()
    monkeypatch.setattr(ecg, 'evaluate_waveforms', lambda *args: pytest.fail('waveform evaluator invoked'))
    rows, leads, record = ecg.evaluate_ecg_waveforms(noisy, estimate, None, 360.0, data, method, 'run', {})
    assert not leads and not record
    assert all(r['admissibility']=='WITHHELD' and r['point_estimate'] is None for r in rows)


def test_actual_ecg_producer_rejects_changed_clean_array(monkeypatch):
    noisy, estimate, clean = signals()
    data, method = waveform_context(clean)
    monkeypatch.setattr(ecg, 'evaluate_waveforms', lambda *args: pytest.fail('waveform evaluator invoked'))
    with pytest.raises(ContractError, match='EVIDENCE_HASH_MISMATCH'):
        ecg.evaluate_ecg_waveforms(noisy, estimate, clean + 1, 360.0, data, method, 'run', {})


def test_full_ecg_main_executes_on_synthetic_sources_and_replays(tmp_path, monkeypatch):
    """Exercise public runner glue and exact-output state, without downloading recordings."""
    data_root, out, evidence = tmp_path/'sources', tmp_path/'outputs', tmp_path/'evidence'
    (data_root/'mitdb').mkdir(parents=True)
    (data_root/'nstdb').mkdir()
    (data_root/'acquisition_manifest.json').write_text(json.dumps({'status':'COMPLETE','full_database_mirrored':False}))
    for record in ('118','119'):
        (data_root/'mitdb'/f'{record}.dat').write_bytes(b'generated fixture source identity')
        for _, suffix in ecg.SNR_SUFFIXES:
            (data_root/'nstdb'/f'{record}{suffix}.dat').write_bytes(b'generated fixture noisy identity')
    t=np.arange(ecg.STOP_SAMPLE-ecg.START_SAMPLE)/ecg.EXPECTED_FS
    clean=np.stack((np.sin(2*np.pi*1.2*t),np.cos(2*np.pi*t)),axis=1)
    noisy=clean+.1*np.sin(2*np.pi*60*t)[:,None]
    monkeypatch.setattr(ecg,'_load_record',lambda path:(clean.copy() if path.parent.name=='mitdb' else noisy.copy(), {'fs':360.0}))
    monkeypatch.setattr(ecg,'_alignment_evidence',lambda *args:{'alignment_eligible':True,'fixture_only':True})
    monkeypatch.setattr(sys,'argv',['run_physionet_jr10','--data-root',str(data_root),'--output-root',str(out),'--evidence-root',str(evidence)])
    ecg.main()
    initial=(evidence/'JR10_SUMMARY.json').read_bytes()
    ecg.main()
    assert (evidence/'JR10_SUMMARY.json').read_bytes()==initial
    runs=list((evidence/'runs').iterdir())
    assert len(runs)==12
    assert all(SafeRunRegistry(evidence/'runs').inspect(run)=='COMPLETE' for run in runs)
    for run in runs:
        rows=json.loads((run/'cell_result.json').read_text())['result_envelopes']
        assert sum(r['admissibility']=='ADMISSIBLE' for r in rows)==3
        assert sum(r['admissibility']=='WITHHELD' for r in rows)==8
