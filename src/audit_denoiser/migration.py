"""Exact-value migration of representative compact legacy evidence."""

from __future__ import annotations

import csv
from pathlib import Path

from .contracts import CanonicalResultEnvelope
from .schema import sha256
from .taxonomy import LEGACY_TAXONOMY_VERSION


LEGACY_FIELDS = {
    "delta_psnr": ("delta_psnr_vs_blind_svd", "dB"),
    "amp_pres": ("amp_pres_to_clean", "ratio"),
    "hf_ratio": ("high_frequency_residual_ratio_vs_blind_svd", "ratio"),
}


def migrate_legacy_verdict_csv(
    path: Path,
    *,
    dataset_id: str,
    method_id: str,
    run_id: str,
    evidence_boundary: str = "clean_reference",
    dataset_contract_sha256: str | None = None,
    clean_reference_sha256: str | None = None,
) -> tuple[list[CanonicalResultEnvelope], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one legacy row, found {len(rows)}")
    row = rows[0]
    source_hash = sha256(path)
    envelopes: list[CanonicalResultEnvelope] = []
    lexemes: dict[str, str] = {}
    for legacy_field, (metric_id, units) in LEGACY_FIELDS.items():
        lexeme = row[legacy_field]
        lexemes[metric_id] = lexeme
        hashes = {"source_sha256": source_hash}
        if metric_id == "amp_pres_to_clean" and evidence_boundary == "clean_reference":
            if dataset_contract_sha256 is None or clean_reference_sha256 is None:
                raise ValueError(
                    "clean-reference legacy migration requires dataset_contract_sha256 "
                    "and clean_reference_sha256"
                )
            hashes.update(
                dataset_contract_sha256=dataset_contract_sha256,
                clean_reference_sha256=clean_reference_sha256,
            )
        envelopes.append(
            CanonicalResultEnvelope(
                dataset_id=dataset_id,
                method_id=method_id,
                run_id=run_id,
                evidence_boundary=evidence_boundary,
                independent_unit="scene",
                nesting=("scene", "roi"),
                metric_id=metric_id,
                metric_version="legacy-compact-v1-exact-value-migration",
                units=units,
                point_estimate=float(lexeme),
                uncertainty=None,
                admissibility="ADMISSIBLE",
                warnings=(f"legacy source value lexeme: {lexeme}",),
                taxonomy_version=LEGACY_TAXONOMY_VERSION,
                provenance_parents=(path.as_posix(),),
                hashes=hashes,
            ).validate()
        )
    return envelopes, lexemes
