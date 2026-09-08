# Version 1.3.0 (2026-09-08)

This version repairs three counterexamples to the v1.2.0 enforcement claims.
The previous public release and its evidence remain unchanged.

- Contract-driven movie loading now treats declared THW axes as authoritative,
  checks both source and canonical shape, and verifies the actual loaded prefix.
  The legacy general-purpose loader can still infer axes when explicitly used
  without a contract; contract-driven commands never use that inference.
- Output validation accepts real integer/floating arrays and rejects strings,
  booleans, complex values and objects. Integer input to floating output remains
  valid. Movie/component outputs preserve complete THW shape; trace outputs
  preserve T and require at least one trace. Numeric nonfinite samples are allowed
  for integrity inspection; individual endpoints must handle them explicitly.
- `MetricContract`, `contextual_admissibility`, `evaluate_metric`, and
  `validate_result_against_context` compose evidence, output-class, method-domain,
  and contract/reference bindings. Unknown metrics are withheld. External metric
  requirements are explicit and cannot weaken a built-in rule.
- The reference-free and ECG producers call the gate before evaluating metrics
  and validate again on emission. ECG clean-reference digests are computed from
  the actual decoded clean array, including its dtype and shape.
- The ECG runner locates source bytes through the imported package rather than
  an absent source-checkout directory. A generated-fixture test executes its full
  matrix and deterministic replay; this is not a new PhysioNet performance result.
- Result schema is 1.2.0. Runtime contract digests use canonical semantic JSON;
  raw contract-file hashes use distinct `*_file_sha256` keys. Old records are not
  silently upgraded and must remain associated with their original software/schema.
- Nonfinite uncertainty is rejected. Historical scalar migration and the synthetic
  envelope benchmark now declare external metric requirements explicitly. Neither
  operation constitutes fresh scientific scoring or contextual execution validation.

The frozen regression suite was executed against unchanged v1.2.0 source before
repair. Detailed red/green logs and source hashes accompany the manuscript repair
record. The test count and release/CI/archival status should be taken from receipts
tied to the final release commit, not inherited from v1.2.0.
