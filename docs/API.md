# Public API

## `DatasetContract`

Declares the dataset identity, modality, axis order, complete shape and dtype, units,
sampling rate, acquisition identity, independent unit and nesting, reference
availability, evidence boundary, mask availability, alignment state, and provenance.
`validate()` enforces structural metadata invariants. `validate_array()` and
`validate_mask()` fail closed on incompatible shape or type.

## `MethodAdapterContract`

Declares method/source/checkpoint identity, training-reference access, output class,
supported metric domains, required inputs, and produced outputs. `domain_status()`
returns a supported status or an explicit reason-bearing withhold; there is no global
method registry that must be edited for each external producer. It does not contain
an independent output axes/shape or provenance field. The output policy is derived
from class and dataset: movie/component outputs require the declared THW shape;
TR traces preserve dataset T and require at least one trace. Real integer/floating
dtypes are accepted, including conversions from the input dtype. Nonnumeric,
boolean, and complex arrays are rejected. A numeric nonfinite sample is inspectable
by integrity metrics; it is not automatically a valid finite endpoint value.

## `MetricContract`

Declares `metric_id`, `domain`, accepted `evidence_boundaries`, accepted
`output_classes`, `requires_clean_reference`, and policy `version`. Built-in
requirements include the three generic ECG clean-reference endpoints. An external
metric supplies this object explicitly and needs no core registry edit. Unknown
metric IDs without requirements are withheld; explicit definitions cannot override
a built-in metric's requirements. These are trusted endpoint requirements, not
proof of biological appropriateness. See `schemas/metric-contract.schema.json`.

## `CanonicalResultEnvelope`

Carries one metric identity, units, finite point estimate when admissible, optional
uncertainty, evidence boundary, independent-unit nesting, taxonomy version, warnings,
provenance parents, and hashes. `WITHHELD`/`INVALID` envelopes cannot carry a value;
`ADMISSIBLE` envelopes require a finite value. An admissible clean-reference metric also
requires `dataset_contract_sha256` and `clean_reference_sha256`; a boundary label alone is
not evidence. Result schema 1.2.0 also carries `metric_contract` when requirements
are known. Unknown admissible metrics require an explicit contract. Nonfinite
uncertainty is rejected.

`validate()` verifies record structure and declared policy only. The legacy
`validate_against_dataset(dataset, dataset_contract_sha256=...)` additionally checks
the supplied dataset identity/boundary/reference declaration against the supplied
digest. Neither call independently loads scientific arrays or proves execution.
Runtime producers use the composed gate below; imported records can be checked
with `validate_result_against_context(result, dataset, method)`.

## Admissibility

`metric_admissibility(metric_id, evidence_boundary, metric_contract=...)` checks
only declared evidence compatibility and returns ADMISSIBLE or WITHHELD. An
unknown metric is WITHHELD. `assert_metric_admissible` raises `ContractError` on
an incompatible request.

`contextual_admissibility(metric_id, dataset, method, hashes=..., metric_contract=...)`
validates both contracts, resolves requirements, checks evidence, output class and
method domain, and binds canonical dataset/method/metric contract digests. Clean
requests additionally require the observed reference digest to match
`dataset.provenance.clean_reference_sha256`. Conflicting or absent required
bindings are INVALID typed exceptions; supported structure with unsupported
evidence/domain is a WITHHELD verdict.

`evaluate_metric(..., analysis=callback)` calls that composed gate **before**
invoking analysis, emits a finite value or value-free reason-bearing abstention,
and calls `validate_result_against_context` on the completed record. A callback
returns a scalar or `(scalar, uncertainty)`. It is not called for unknown or
incompatible requests. Undefined/nonfinite scalars are withheld, while malformed
values and uncertainty raise typed errors. Actual array validation belongs to the
producer before this gate; no array-free API can establish reference identity
unless the producer hashes the array it actually uses.

In runtime records, `contract_digest()` defines the SHA-256 over canonical JSON
contract semantics, including all declared fields. `dataset_contract_sha256`,
`method_contract_sha256` and `metric_contract_sha256` refer to those semantics.
CLI file-byte identities use separate `dataset_contract_file_sha256` and
`method_contract_file_sha256` keys. A caller that supplies a conflicting semantic
digest receives EVIDENCE_HASH_MISMATCH. Historical results are kept under their
original schema/version; they are not silently read as new contextual results.

## Run identity and registry

`RunIdentityInputs` binds protocol, dataset manifest, method/checkpoint, configuration,
seed, evidence boundary, bounded source identity, and parent artifacts.
`compute_run_id` produces a content-derived computational identifier. `SafeRunRegistry`
additionally binds the canonical expected-output list into its run-directory identity,
then provides atomic preparation/completion and distinguishes partial, interrupted,
duplicate, colliding, stale, and complete output state.

## Integration pattern

An external producer writes its output and a method contract. `validate-output`
checks output class, numerical dtype, class-relative shape, and records declared domains, then creates an
exact-output manifest without computing a scientific metric. See
`examples/audit_core_external_adapter/`. This operation does not issue a scientific
metric admissibility certificate. `examples/contextual_trace_metric.py` additionally
demonstrates an external trace endpoint through `evaluate_metric`, including
unknown and incompatible requests without invoking their numerical callbacks.

`DatasetContract.validate_array` checks the input dtype. The external movie adapter
does not require its output dtype to equal the input dtype: integer input and
floating-point reconstructed output are legitimate. No unimplemented output-dtype
contract is claimed. Contract-driven movie readers use the declared THW order;
they never guess temporal axes from aspect ratio. The generic non-contract loader's
optional `stack_order="auto"` behavior is outside that guarantee.
