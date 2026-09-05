"""Versioned failure taxonomy with explicit, non-silent migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .contracts import ContractError


LEGACY_TAXONOMY_VERSION = "failure-taxonomy-legacy-v1"
CURRENT_TAXONOMY_VERSION = "failure-taxonomy-domain-v2"

TAXONOMIES = {
    LEGACY_TAXONOMY_VERSION: {"WIN", "TIE", "LOSE", "OVERSMOOTH"},
    CURRENT_TAXONOMY_VERSION: {
        "PASS_ALL_GATES",
        "AMPLITUDE_COMPRESSION",
        "AMPLITUDE_INFLATION",
        "HIGH_FREQUENCY_DISTORTION",
        "GLOBAL_LOSS",
        "WITHHELD",
    },
}


@dataclass(frozen=True)
class TaxonomyMigration:
    source_version: str
    source_label: str
    target_version: str
    status: str
    target_label: str | None
    reason: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def migrate_label(label: str, source_version: str, target_version: str = CURRENT_TAXONOMY_VERSION) -> TaxonomyMigration:
    if source_version not in TAXONOMIES or target_version not in TAXONOMIES:
        raise ContractError("UNKNOWN_TAXONOMY_VERSION", f"{source_version} -> {target_version}")
    if label not in TAXONOMIES[source_version]:
        raise ContractError("UNKNOWN_FAILURE_LABEL", f"{label!r} under {source_version}")
    if source_version == target_version:
        return TaxonomyMigration(source_version, label, target_version, "EXACT", label, "identity migration")
    if source_version == LEGACY_TAXONOMY_VERSION and target_version == CURRENT_TAXONOMY_VERSION:
        if label == "OVERSMOOTH":
            return TaxonomyMigration(
                source_version, label, target_version, "AMBIGUOUS", None,
                "legacy OVERSMOOTH can reflect amplitude or high-frequency vetoes; continuous metrics are required",
            )
        if label == "LOSE":
            return TaxonomyMigration(
                source_version, label, target_version, "REQUIRES_REVIEW", None,
                "legacy LOSE does not encode whether loss was global, amplitude-specific, or domain-withheld",
            )
        if label in {"WIN", "TIE"}:
            return TaxonomyMigration(
                source_version, label, target_version, "REQUIRES_REVIEW", None,
                "legacy decision labels do not imply that every current domain-specific gate was measured",
            )
    return TaxonomyMigration(source_version, label, target_version, "REQUIRES_REVIEW", None, "no exact semantic mapping is registered")
