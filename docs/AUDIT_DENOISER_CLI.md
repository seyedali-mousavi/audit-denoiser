# Reusable `audit_denoiser` command surface

`calcium-denoiser-audit` exposes a standalone formal audit core and, when run
inside the full research checkout, an optional bridge to the frozen scientific
audit instruments. The denoiser itself never receives the clean reference.

## Install

From a source checkout:

```text
python -m pip install .
audit_denoiser --help
```

The package requires Python 3.10 or newer. Core dependencies are declared in
`pyproject.toml`; HDF5 input is optional via `.[hdf5]`.

## Stable interface

The independently installable core supports:

```text
audit_denoiser reference-free --name NAME --movie MOVIE \
  --dataset-contract DATASET.json --method-contract METHOD.json --out OUT
audit_denoiser validate-output --name NAME --output METHOD_OUTPUT \
  --dataset-contract DATASET.json --method-contract METHOD.json --out OUT
audit_denoiser report --name NAME --out OUT
```

The following project-bridge commands are available only from the full research
source checkout, where their frozen numerical implementations and dependencies
exist:

```text
audit_denoiser evaluate  --name NAME --raw RAW --reference CLEAN --denoised EST --out OUT
audit_denoiser residuals --name NAME           --reference CLEAN --denoised EST --out OUT
audit_denoiser traces    --name NAME --raw RAW --reference CLEAN --denoised EST --out OUT
audit_denoiser amplitude --name NAME --raw RAW --reference CLEAN --denoised EST --out OUT
audit_denoiser seams     --name NAME           --reference CLEAN --denoised EST --out OUT
audit_denoiser verdict   --name NAME --raw RAW --reference CLEAN --denoised EST --out OUT
audit_denoiser report    --name NAME --out OUT
```

`evaluate`, `residuals`, `traces`, `amplitude`, `seams`, and `verdict`
delegate to the already audited production implementations. `traces` and
`amplitude` intentionally invoke the same functional audit so their outputs
remain jointly defined rather than silently diverging.

If a bridge command is requested from the standalone wheel, the CLI fails
closed with an explicit message rather than silently substituting an
implementation.

## Adapters and schemas

- Dataset adapter `movie-io-v1` validates formats supported by the production
  movie reader (`.tif/.tiff/.npy/.npz/.h5/.hdf5` and optional preview video formats).
- Method adapter `external-movie-v1` accepts an aligned, precomputed denoised
  movie. `identity-fixture-v1` is reserved for the bundled deterministic
  clean-room example.
- Each numerical command writes
  `audit_command_<name>_<command>.json`, schema
  `org.calcium-denoiser-audit.command-manifest`, version `1.0.0`. The manifest
  records adapter identities, arguments, exit status, and SHA-256 hashes of
  generated outputs. Historical metric CSV/JSON layouts remain unchanged.

The template at `configs/audit_denoiser/external_movie_example.json` maps
directly to command flags and documents the stable input contract.

The versioned dataset/method contracts and the standalone `reference-free`
lane never accept a dummy clean reference. Clean-reference-only metric domains
are emitted as `WITHHELD` with a machine-readable reason.

`validate-output` accepts `full_movie`, `component_reconstruction`, and
`trace_only` method contracts, verifies real numerical output type and class-relative shape, and emits a
content-addressed exact-output manifest without computing a scientific metric.
The external-style example in `examples/audit_core_external_adapter/` shows a
trace-producing integration that requires no change to the core registry.

The contract-driven movie commands use the declared THW order and verify the
actual loaded prefix. Output dtype may differ from input dtype, but strings,
booleans, objects and complex values are not certified as real biomedical arrays.
Movie/component shapes must match the dataset; trace outputs preserve T and contain
at least one trace. Domain declarations in an output-validation manifest do not
certify that a metric is admissible.

Runtime metrics in the standalone reference-free producer and ECG tool pass through
`evaluate_metric`: explicit endpoint requirements, evidence, output class, method
domain and concrete reference/contract bindings are checked before analysis and
again on emission. Unknown metrics are withheld. `CanonicalResultEnvelope.validate`
alone checks structure and declared policy, not execution context. Optional project
bridges retain their historical interfaces and are not covered by this new gate.

## Deterministic fixture

```text
audit_denoiser_fixture --out fixture
audit_denoiser reference-free --name identity --movie fixture/identity.npy \
  --dataset-contract examples/reference_free/dataset_contract.json \
  --method-contract examples/reference_free/method_contract.json --out audit --max-frames 64
```

The fixture contains four deterministic spatial sources, transient traces,
noise, and two method outputs (`identity`, `temporal_smooth`). It is for
software reproduction only, not scientific inference.

## Interpretation boundaries

Clean-derived masks and transient windows are audit instruments, not
independent physiological spike truth. Scene-level replication, not ROI count,
is the inferential unit for multi-scene scientific claims. The CLI does not
train or select a denoiser and does not alter the frozen decision thresholds.
