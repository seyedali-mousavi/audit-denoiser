# Public API

## `DatasetContract`

Declares the dataset identity, modality, axis order, complete shape and dtype, units,
sampling rate, acquisition identity, independent unit and nesting, reference
availability, evidence boundary, mask availability, alignment state, and provenance.
`validate()` enforces structural invariants and requires explicit evidence digests for
clean-reference-only metrics; `validate_against_dataset()` binds those declarations to a
specific dataset contract. `validate_array()` and
`validate_mask()` fail closed on incompatible shape or type.

## `MethodAdapterContract`

Declares method/source/checkpoint identity, training-reference access, output class,
supported metric domains, required inputs, and produced outputs. `domain_status()`
returns a supported status or an explicit reason-bearing withhold; there is no global
method registry that must be edited for each external producer.

## `CanonicalResultEnvelope`

Carries one metric identity, units, finite point estimate when admissible, optional
uncertainty, evidence boundary, independent-unit nesting, taxonomy version, warnings,
provenance parents, and hashes. `WITHHELD`/`INVALID` envelopes cannot carry a value;
`ADMISSIBLE` envelopes require a finite value. An admissible clean-reference metric also
requires `dataset_contract_sha256` and `clean_reference_sha256`; a boundary label alone is
not evidence.

## Admissibility

`metric_admissibility(metric_id, evidence_boundary)` deterministically returns
`ADMISSIBLE` or `WITHHELD`. `assert_metric_admissible` raises a stable `ContractError`
when a caller attempts an unsupported comparison.

## Run identity and registry

`RunIdentityInputs` binds protocol, dataset manifest, method/checkpoint, configuration,
seed, evidence boundary, bounded source identity, and parent artifacts.
`compute_run_id` produces a content-derived computational identifier. `SafeRunRegistry`
additionally binds the canonical expected-output list into its run-directory identity,
then provides atomic preparation/completion and distinguishes partial, interrupted,
duplicate, colliding, stale, and complete output state.

## Integration pattern

An external producer writes its output and a method contract. `validate-output`
checks output class, complete shape, and declared domains, then creates an
exact-output manifest without computing a scientific metric. See
`examples/audit_core_external_adapter/`.

`DatasetContract.validate_array` checks the input dtype. The external movie adapter
does not require its output dtype to equal the input dtype: integer input and
floating-point reconstructed output are legitimate. No unimplemented output-dtype
contract is claimed.
