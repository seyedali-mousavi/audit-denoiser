"""Final review regression: contextual imports must carry their own bindings."""
from dataclasses import replace
import pytest

from audit_denoiser.contracts import ContractError, validate_result_against_context
from test_contextual_evaluation import evaluate, waveform_context


@pytest.mark.parametrize('missing', ['method_contract_sha256', 'metric_contract_sha256'])
def test_contextual_import_rejects_missing_result_binding(missing):
    data, method = waveform_context()
    result = evaluate(data, method, analysis=lambda: .5)
    hashes = dict(result.hashes)
    del hashes[missing]
    unbound = replace(result, hashes=hashes)
    with pytest.raises(ContractError, match='MISSING_EVIDENCE_BINDING'):
        validate_result_against_context(unbound, data, method)
