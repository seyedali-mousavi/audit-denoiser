"""Unified CLI that preserves the existing audited numerical implementations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .adapters import (
    DATASET_ADAPTERS,
    METHOD_ADAPTERS,
    validate_adapters,
    validate_contract_method_output,
)
from .contracts import DatasetContract, MethodAdapterContract
from .reference_free import run_reference_free_audit, write_reference_free_results
from .run_identity import RunIdentityInputs, SafeRunRegistry, atomic_json, source_tree_identity
from .schema import command_manifest, sha256, write_manifest


MODULES = {
    "evaluate": "src.tools.run_external_baseline",
    "residuals": "src.tools.run_external_residual_decomp",
    "traces": "src.tools.run_functional_audit",
    "amplitude": "src.tools.run_functional_audit",
    "seams": "src.tools.run_external_residual_decomp",
    "verdict": "src.tools.run_external_svd_verdict",
}


def common_io(parser: argparse.ArgumentParser, include_raw: bool = True) -> None:
    parser.add_argument("--name", required=True)
    if include_raw:
        parser.add_argument("--raw", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--denoised", required=True)
    parser.add_argument("--out", default="results/audit_denoiser")
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument(
        "--dataset-adapter",
        choices=sorted(DATASET_ADAPTERS),
        default="movie-io-v1",
        help="input adapter identity recorded in the versioned command manifest",
    )
    parser.add_argument(
        "--method-adapter",
        choices=sorted(METHOD_ADAPTERS),
        default="external-movie-v1",
        help="method-output adapter identity recorded in the versioned command manifest",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="audit_denoiser", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "verdict"):
        child = sub.add_parser(command)
        common_io(child)
        child.add_argument("--est-in-raw-norm", action="store_true")
    for command in ("traces", "amplitude"):
        child = sub.add_parser(command)
        common_io(child)
        child.add_argument("--fps", type=float, default=30.0)
        child.add_argument("--est-in-raw-norm", action="store_true")
    for command in ("residuals", "seams"):
        child = sub.add_parser(command)
        common_io(child, include_raw=False)
        child.add_argument("--sigma", type=float, default=16.0)
        child.add_argument("--canonical-step", type=int)
        child.add_argument("--seam-steps", type=int, nargs="+")
    report = sub.add_parser("report")
    report.add_argument("--name", required=True)
    report.add_argument("--out", type=Path, default=Path("results/audit_denoiser"))
    reference_free = sub.add_parser(
        "reference-free",
        help="run admissible no-reference metrics without a dummy clean movie",
    )
    reference_free.add_argument("--name", required=True)
    reference_free.add_argument("--movie", type=Path, required=True)
    reference_free.add_argument("--dataset-contract", type=Path, required=True)
    reference_free.add_argument("--method-contract", type=Path, required=True)
    reference_free.add_argument("--out", type=Path, default=Path("results/audit_denoiser/reference_free"))
    reference_free.add_argument("--max-frames", type=int, default=256)
    reference_free.add_argument("--seed", type=int, default=0)
    validate_output = sub.add_parser(
        "validate-output",
        help="validate an external output and declared metric domains without computing metrics",
    )
    validate_output.add_argument("--name", required=True)
    validate_output.add_argument("--output", type=Path, required=True)
    validate_output.add_argument("--dataset-contract", type=Path, required=True)
    validate_output.add_argument("--method-contract", type=Path, required=True)
    validate_output.add_argument("--out", type=Path, default=Path("results/audit_denoiser/validation"))
    validate_output.add_argument("--seed", type=int, default=0)
    return root


def forwarded_args(namespace: argparse.Namespace) -> list[str]:
    args: list[str] = []
    for key, value in vars(namespace).items():
        if key in {"command", "dataset_adapter", "method_adapter"} or value is None or value is False:
            continue
        option = "--" + key.replace("_", "-")
        if value is True:
            args.append(option)
        elif isinstance(value, list):
            args.append(option)
            args.extend(str(item) for item in value)
        else:
            args.extend([option, str(value)])
    return args


def matching_artifacts(out: Path, name: str, before: set[Path]) -> list[Path]:
    """Return files created or replaced by one command, excluding its own manifest."""
    candidates = {
        path.resolve()
        for path in out.rglob("*")
        if path.is_file()
        and name in path.name
        and not path.name.startswith("audit_command_")
    }
    # Existing outputs are legitimate command products when a deterministic rerun
    # replaces them. Include all name-matched files, not only newly created paths.
    return sorted(candidates | {path for path in before if path in candidates})


def project_bridge_available(module: str) -> bool:
    """Return whether a source-checkout-only scientific command is importable."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def write_report(name: str, out: Path) -> int:
    candidates = sorted(path for path in out.glob(f"*{name}*") if path.is_file())
    lines = [f"# Audit report: {name}", "", "This report indexes machine-readable outputs; it does not alter their values.", ""]
    if not candidates:
        lines.append("No matching audit outputs were found.")
    for path in candidates:
        lines.append(f"- `{path.name}` ({path.stat().st_size} bytes)")
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                for key in ("verdict", "status", "frames", "n_rois", "delta_psnr", "amp_pres", "hf_ratio"):
                    if key in data:
                        lines.append(f"  - {key}: `{data[key]}`")
            except (json.JSONDecodeError, OSError):
                lines.append("  - JSON summary unavailable (file retained unchanged).")
    report = out / f"audit_report_{name}.md"
    out.mkdir(parents=True, exist_ok=True)
    report.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "report":
        return write_report(args.name, args.out)
    if args.command == "validate-output":
        dataset = DatasetContract.read(args.dataset_contract)
        method = MethodAdapterContract.read(args.method_contract)
        parents = {
            "output_sha256": sha256(args.output),
            "dataset_contract_sha256": sha256(args.dataset_contract),
            "method_contract_sha256": sha256(args.method_contract),
        }
        identity = RunIdentityInputs(
            protocol_identity="output-validation-v1",
            dataset_manifest_identity=parents["dataset_contract_sha256"],
            method_source_identity=method.source_identity,
            checkpoint_identity=method.checkpoint_identity,
            config={"name": args.name, "output_class": method.output_class},
            seed=args.seed,
            evidence_boundary=dataset.evidence_boundary,
            code_identity=source_tree_identity(Path(__file__).parent),
            parent_artifacts=parents,
        )
        registry = SafeRunRegistry(args.out / "runs")
        run_dir = registry.prepare(identity, ["validation_result.json"])
        validation = validate_contract_method_output(method, args.output, dataset)
        payload = {
            "$schema": "org.calcium-denoiser-audit.output-validation",
            "schema_version": "1.0.0",
            "run_id": run_dir.name,
            "dataset_id": dataset.dataset_id,
            "method_id": method.method_id,
            "evidence_boundary": dataset.evidence_boundary,
            "validation": validation,
            "declared_metric_domains": {
                domain: method.domain_status(domain) for domain in method.supported_metric_domains
            },
            "hashes": parents,
        }
        atomic_json(run_dir / "validation_result.json", payload)
        registry.complete(run_dir)
        print(run_dir)
        return 0
    if args.command == "reference-free":
        dataset = DatasetContract.read(args.dataset_contract)
        method = MethodAdapterContract.read(args.method_contract)
        parents = {
            "movie_sha256": sha256(args.movie),
            "dataset_contract_sha256": sha256(args.dataset_contract),
            "method_contract_sha256": sha256(args.method_contract),
        }
        identity = RunIdentityInputs(
            protocol_identity="reference-free-core-v1",
            dataset_manifest_identity=parents["dataset_contract_sha256"],
            method_source_identity=method.source_identity,
            checkpoint_identity=method.checkpoint_identity,
            config={"name": args.name, "max_frames": args.max_frames},
            seed=args.seed,
            evidence_boundary=dataset.evidence_boundary,
            code_identity=source_tree_identity(Path(__file__).parent),
            parent_artifacts=parents,
        )
        registry = SafeRunRegistry(args.out / "runs")
        run_dir = registry.prepare(identity, ["reference_free_results.json"])
        results = run_reference_free_audit(
            movie_path=args.movie,
            dataset=dataset,
            method=method,
            run_id=run_dir.name,
            max_frames=args.max_frames,
        )
        write_reference_free_results(results, run_dir / "reference_free_results.json")
        registry.complete(run_dir)
        print(run_dir)
        return 0
    module = MODULES[args.command]
    if not project_bridge_available(module):
        print(
            f"command {args.command!r} requires the optional project scientific-command bridge "
            f"({module}); the standalone audit core supports 'reference-free' and 'report'",
            file=sys.stderr,
        )
        return 2
    validate_adapters(
        dataset_adapter=args.dataset_adapter,
        method_adapter=args.method_adapter,
        raw=getattr(args, "raw", None),
        reference=args.reference,
        denoised=args.denoised,
    )
    out = Path(args.out)
    before = {path.resolve() for path in out.rglob("*") if path.is_file()} if out.exists() else set()
    command = [sys.executable, "-m", module, *forwarded_args(args)]
    exit_code = subprocess.run(command, check=False).returncode
    artifacts = matching_artifacts(out, args.name, before)
    payload = command_manifest(
        command=args.command,
        name=args.name,
        arguments=vars(args),
        dataset_adapter=args.dataset_adapter,
        method_adapter=args.method_adapter,
        exit_code=exit_code,
        artifacts=artifacts,
        output_root=out,
        package_version=__version__,
    )
    write_manifest(payload, out / f"audit_command_{args.name}_{args.command}.json")
    return exit_code
