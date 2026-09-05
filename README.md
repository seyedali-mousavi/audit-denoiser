# calcium-denoiser-audit

Executable evidence contracts for fail-closed evaluation of heterogeneous biomedical
processing pipelines. Package version: **1.2.0**.

- Repository: https://github.com/seyedali-mousavi/audit-denoiser
- Archival DOI: pending — a Zenodo DOI is minted on the first tagged release.
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
`CanonicalResultEnvelope`, admissibility checks, clean-reference digest binding,
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

The retained candidate-v8 report records **202 local Windows source tests passed**.
That report predates the author-metadata preparation and is a historical test
receipt, not a hash manifest for every current metadata file. The separate
candidate-v7 Windows/Ubuntu/container results must not be attributed to v8.

The prepared CI workflow has not been executed on a public host for this candidate.
Public release, archival identity, Linux execution of v8, and unaffiliated adoption
are not claimed. See `RELEASE_CHECKLIST.md` for the current preparation checks and
remaining author actions.

## Install from this local source

Requires Python 3.10 or later. From this directory:

```text
python -m venv .venv-release
.venv-release/Scripts/python -m pip install -e ".[test]"
.venv-release/Scripts/python -m pytest -q -p no:cacheprovider
```

On POSIX systems the corresponding executable is `.venv-release/bin/python`;
providing that path is not evidence of a v8 Linux test run.

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

`validate-output` checks output class, axes, dtype, complete shape, and declared
metric domains and creates an output manifest without calculating a scientific
performance metric. See `docs/API.md`, `docs/AUDIT_DENOISER_CLI.md`, and
`schemas/` for the supported interfaces.

## Deterministic fixture example

This recipe uses generated software-test data, not a research dataset:

```text
audit_denoiser_fixture --out fixture
audit_denoiser reference-free --name demo --movie fixture/identity.npy --dataset-contract examples/reference_free/dataset_contract.json --method-contract examples/reference_free/method_contract.json --out demo_audit --max-frames 64
```

Use previously nonexistent output directories: completed runs cannot be overwritten.
The reference-free command calculates fixture measurements. WO-08 metadata
preparation does not re-execute this measurement-producing example.

## Contracts and identity

A dataset contract declares acquisition identity, axes, complete shape, dtype,
sampling, units, independent unit and nesting, evidence boundary, alignment, and
provenance. A method contract declares its identity, output class and shape,
supported metric domains, inputs, and provenance.

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
licence inclusion as well as the hashes; a successful build alone does not complete
the release checklist. Historical `RELEASE_EXECUTION.md`,
`RELEASE_BLOCKERS.md`, and `RELEASE_CANDIDATE_MANIFEST.json` retain their original
preparation state. The dated `RELEASE_CHECKLIST.md` records this metadata update.

## Citation, licence, and third-party components

Author metadata is supplied in `CITATION.cff`. The authors confirmed the spelling
Majid Badieirostami in WO-09; current publication metadata uses that spelling.

The package's selected licence is MIT; see `LICENSE` for the canonical OSI text
with the author-supplied copyright line. Publication remains an author action.

See `THIRD_PARTY_NOTICES.md` for verified dependency licence metadata and its
limits. Dependency distributions retain their own bundled notices. No research
dataset is redistributed; access to CRCNS data remains subject to CRCNS terms.
