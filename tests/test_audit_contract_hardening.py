"""Additional deterministic B4 implementation-review regressions.

Separate from the pre-frozen 52-case property suite; no scientific input is used.
"""
from dataclasses import replace
import json

import pytest

from audit_denoiser.contracts import ContractError
from audit_denoiser.run_identity import DuplicateRunError, RunIdentityInputs, SafeRunRegistry, compute_run_id


def inputs():
    return RunIdentityInputs("p", "d", "m", None, {"range": (1, 2)}, 7, "reference_free", "code")


def test_tuple_identity_roundtrip_and_duplicate(tmp_path):
    x = inputs()
    equivalent = replace(x, config={"range": [1, 2]})
    assert compute_run_id(x) == compute_run_id(equivalent)
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(x, ["metric.json"])
    (run / "metric.json").write_text("{}")
    registry.complete(run)
    with pytest.raises(DuplicateRunError):
        registry.prepare(x, ["metric.json"])


@pytest.mark.parametrize("bad", [{1: "not-text"}, {"nested": [{False: "not-text"}]}])
def test_nonstring_identity_keys_rejected(bad):
    with pytest.raises(ContractError, match="INVALID_RUN_IDENTITY"):
        compute_run_id(replace(inputs(), config=bad))


def test_unfinished_manifest_not_reported_complete(tmp_path):
    registry = SafeRunRegistry(tmp_path)
    run = registry.prepare(inputs(), ["metric.json"])
    (run / "metric.json").write_text("{}")
    registry.complete(run)
    manifest_path = run / "RUN_MANIFEST.json"
    payload = json.loads(manifest_path.read_text())
    payload["status"] = "RUNNING"
    manifest_path.write_text(json.dumps(payload))
    assert registry.inspect(run) == "PARTIAL"
