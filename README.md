# calcium-denoiser-audit

Executable evidence contracts for fail-closed evaluation of heterogeneous biomedical
processing pipelines. Package version: **1.3.0**.

- Repository: https://github.com/seyedali-mousavi/audit-denoiser
- Previous public version 1.2.0: https://doi.org/10.5281/zenodo.22336513
- Version 1.3.0 release: https://github.com/seyedali-mousavi/audit-denoiser/releases/tag/v1.3.0
- The previous version DOI does not identify this repaired source. Archival version
  metadata is supplied by the repository's existing Zenodo release integration.
- Package index: this software is not published to PyPI; install from source.

## What this is

Evaluation software can run reproducibly while using an inadmissible reference,
accepting an output with the wrong shape, serializing an undefined statistic as a
number, or associating old outputs with changed configuration. This package makes
those evidence and output checks explicit.

Dataset evidence, method output class, metric admissibility, independent-unit
metadata, result semantics, run identity, and output state are declared as
contracts. A request can produce an admissible finite value, a reason-bearing
`WITHHELD` envelope without a value, or a typed structural error. A withhold is
distinguished from a poor score and an execution failure.

## Scope and boundaries

The standalone surface includes `DatasetContract`, `MethodAdapterContract`,
`MetricContract`, `CanonicalResultEnvelope`, the contextual `evaluate_metric` gate,
clean-reference digest binding,
content-derived run identity, and a non-overwrite result registry. The standalone
commands are `reference-free`, `validate-output`, and `report`.

Scientific bridge commands are visible in `--help` but require the separate
research checkout and fail closed when it is unavailable. This package does not
provide a denoiser, workflow engine, security sandbox, clinical system, or regulatory
validation. It does not replace W3C PROV, RO-Crate, BioCompute Objects, or CWL.

The application evidence concerns calcium imaging and bounded ECG examples.
Recorded-trace electrical references do not constitute clean-movie ground truth.
No dataset, trained model, checkpoint, or third-party scientific source is included.

## Verification provenance

Historical public CI applies to its recorded release commit. The v1.3.0 repairs are driven by frozen negative tests from
the independent review. New contextual integration tests also exercise explicit
external metric definitions and the complete ECG runner using generated fixtures.
See `CHANGELOG.md` for the defects, API boundaries, and version transition.

Historical v7/v8 platform, mutation, and performance receipts retain their original
version scope. The new release notes identify the v1.3.0 suite, wheel hashes and
commit-specific CI run. No unaffiliated adoption is claimed.

## Install from this local source

Requires Python 3.10 or later. From this directory:

```text
python -m venv .venv-release
.venv-release/Scripts/python -m pip install -e ".[test]"
.venv-release/Scripts/python -m pytest -q -p no:cacheprovider
```

On POSIX systems the corresponding executable is `.venv-release/bin/python`;
the commit-specific CI run records which platforms and interpreters were exercised.

Core dependencies are NumPy, pandas, SciPy, and tifffile. Optional extras are
`hdf5` (h5py), `movie` (OpenCV video I/O), `ecg` (WFDB for the separate
`tools/run_physionet_jr10.py` tool), and `test`.

## Commands and integration

```text
audit_denoiser --help
audit_denoiser reference-free --help
audit_denoiser validate-output --help
audit_denoiser report --help
```

`validate-output` checks output class, declared axes, real numerical dtype, and
class-relative shape, then records declared method domains. It creates a manifest without calculating a scientific
performance metric. See `docs/API.md`, `docs/AUDIT_DENOISER_CLI.md`, and
`schemas/` for the supported interfaces.

## Deterministic fixture example

This recipe uses generated software-test data, not a research dataset:

```text
audit_denoiser_fixture --out fixture
audit_denoiser reference-free --name demo --movie fixture/identity.npy --dataset-contract examples/reference_free/dataset_contract.json --method-contract examples/reference_free/method_contract.json --out demo_audit --max-frames 64
```

Use previously nonexistent output directories: completed runs cannot be overwritten.
The reference-free command calculates fixture measurements through the contextual
gate. The example is a software fixture, not a biomedical efficacy experiment.

## Contracts and identity

A dataset contract declares acquisition identity, axes, complete shape, dtype,
sampling, units, independent unit and nesting, evidence boundary, alignment, and
provenance. A method contract declares source/checkpoint/training identity,
output class, supported metric domains, required inputs, and produced outputs.
Output requirements are derived from the class and dataset: THW movie/component
shape must match the dataset; TR trace output must preserve T and contain at least
one trace. Real integer and floating dtypes are accepted, so output dtype need not
equal input dtype. No separate declared output-shape or method-provenance field is implied.

Metric owners supply explicit evidence, domain, and output-class requirements.
Unknown metrics produce value-free WITHHELD results. `evaluate_metric` composes
the requirements with concrete dataset/method/reference digests before invoking
analysis and checks the resulting record again on emission. A standalone envelope
`validate()` call checks record structure and declared policy; it is not a runtime
execution certificate. External integrations should use `evaluate_metric`.

In contextual v1.3.0 results, `dataset_contract_sha256` and
`method_contract_sha256` hash canonical contract semantics. Optional corresponding
`*_file_sha256` hashes preserve exact input-file identity. Result schema 1.2.0 carries
explicit metric requirements. Historical result schemas/bytes are not silently upgraded.

Run identity binds canonical protocol, dataset, method, configuration, seed,
evidence boundary, bounded-core source bytes, and declared parent hashes. Completed
outputs remain bound to their recorded identity.

## Building and release checks

The documented staging tool is:

```text
python tools/build_audit_core_release.py --repo-root . --out build-check-1
python tools/build_audit_core_release.py --repo-root . --out build-check-2
```

Each output directory must be new or empty. Review the wheel member lists and
licence inclusion as well as the hashes; a successful build alone does not establish
cross-platform execution or release publication. Preserve the v1.2.0 release and
publish v1.3.0 as a new version with its own verification and archival identity.

## Citation, licence, and third-party components

Author metadata is supplied in `CITATION.cff`. The authors confirmed the spelling
Majid Badieirostami in WO-09; current publication metadata uses that spelling.

The package's selected licence is MIT; see `LICENSE` for the canonical OSI text
with the author-supplied copyright line. The public v1.2.0 release remains available;
the repaired version has a separate tag and release record.

See `THIRD_PARTY_NOTICES.md` for verified dependency licence metadata and its
limits. Dependency distributions retain their own bundled notices. No research
dataset is redistributed; access to CRCNS data remains subject to CRCNS terms.
