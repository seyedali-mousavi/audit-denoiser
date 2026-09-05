# External-style trace adapter example

This example demonstrates the public-contract integration path without adding a
method to the audit core's source registry. It is a deterministic software
fixture, not a scientific denoiser or result.

From the repository checkout:

```text
python examples/audit_core_external_adapter/emit_trace_method.py --out example_output
python -m src.audit_denoiser validate-output \
  --name external-trace-example \
  --output example_output/traces.npy \
  --dataset-contract examples/audit_core_external_adapter/dataset_contract.json \
  --method-contract examples/audit_core_external_adapter/method_contract.json \
  --out example_audit
```

The same commands work with the installed `audit_denoiser` console entry point.
The output is validated as `[T,R]`; movie-only domains remain undeclared rather
than being fabricated from trace output.
