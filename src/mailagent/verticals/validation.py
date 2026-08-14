"""Static validation for installed vertical plugins and external profiles."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from mailagent.classification.rule_classifier import RuleClassifier
from mailagent.classification.taxonomy import TaxonomyLoader, TaxonomyTree, load_taxonomy
from mailagent.core.target_profile import TargetProfile
from mailagent.preprocessing.retrieval_models import load_retrieval_cleaning_policy

from .selection import SelectedVertical, load_selected_vertical


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    component: str
    passed: bool
    detail: str
    path: str | None = None


@dataclass(slots=True)
class VerticalValidationReport:
    vertical_id: str
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vertical_id": self.vertical_id,
            "valid": self.valid,
            "checks": [
                {
                    "component": check.component,
                    "status": "passed" if check.passed else "failed",
                    "detail": check.detail,
                    **({"path": check.path} if check.path else {}),
                }
                for check in self.checks
            ],
        }


def _passed(component: str, detail: str, path: Path | str | None = None) -> ValidationCheck:
    return ValidationCheck(component, True, detail, str(path) if path else None)


def _failed(component: str, exc: Exception, path: Path | str | None = None) -> ValidationCheck:
    return ValidationCheck(component, False, str(exc), str(path) if path else None)


def _validate_target_profiles(path: Path, taxonomy: TaxonomyTree) -> int:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse target profiles: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("target profiles must be a YAML mapping")
    entries = raw.get("targets", [])
    if not isinstance(entries, list):
        raise ValueError("target profiles 'targets' must be a list")
    valid_codes = taxonomy.all_codes()
    labels: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"targets[{index}] must be a mapping")
        profile = TargetProfile.model_validate(entry)
        labels.append(profile.label)
        unknown = sorted(({profile.label} | set(profile.vector_scope)) - valid_codes)
        if unknown:
            raise ValueError(
                f"targets[{index}] references unknown taxonomy labels: {', '.join(unknown)}"
            )
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"duplicate target profile labels: {', '.join(duplicates)}")
    return len(entries)


def _validate_rag_declaration(path: Path) -> int:
    sources_path = path / "sources.yaml" if path.is_dir() else path
    try:
        raw = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse RAG sources: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("RAG sources must be a YAML mapping")
    if not isinstance(raw.get("version"), str) or not raw["version"].strip():
        raise ValueError("RAG sources require a non-blank string version")
    sources = raw.get("sources")
    if not isinstance(sources, list):
        raise ValueError("RAG sources require a sources list")
    if any(not isinstance(source, dict) for source in sources):
        raise ValueError("every RAG source must be a mapping")
    return len(sources)


def _validate_structural_conditions(conditions: list[str]) -> None:
    allowed_variables = {"has_attachments", "body_length", "has_recipients"}
    atom_pattern = re.compile(
        r"^(has_attachments|body_length|has_recipients)"
        r"(?:\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?|"
        r"has_attachments|body_length|has_recipients))?$"
    )
    for condition in conditions:
        atoms = re.split(r"\s+(?:and|or)\s+", condition, flags=re.IGNORECASE)
        if not atoms or any(not atom_pattern.fullmatch(atom.strip()) for atom in atoms):
            raise ValueError(f"unsupported structural condition: {condition!r}")
        bare = [atom.strip() for atom in atoms if not re.search(r">=|<=|==|!=|>|<", atom)]
        if any(atom not in allowed_variables for atom in bare):
            raise ValueError(f"unsupported structural condition: {condition!r}")


def validate_vertical_profile(settings: Any) -> VerticalValidationReport:
    """Validate one selected profile without constructing runtime services."""

    report = VerticalValidationReport(vertical_id=settings.id)
    try:
        selected: SelectedVertical = load_selected_vertical(settings)
    except Exception as exc:
        report.checks.append(_failed("profile", exc))
        return report

    loaded = selected.assets
    profile_root = loaded.taxonomy_path.parent
    report.checks.append(
        _passed(
            "profile",
            f"installed plugin {selected.plugin.id!r} matches namespace "
            f"{selected.plugin.namespace!r}; all declared assets exist",
            profile_root / "manifest.yaml",
        )
    )

    taxonomy: TaxonomyTree | None = None
    try:
        taxonomy = load_taxonomy(loaded.taxonomy_path)
        if taxonomy.node_count() == 0:
            raise ValueError("taxonomy must declare at least one label")
        display_labels = [node.label.strip() for node in taxonomy.nodes]
        if any(not label for label in display_labels):
            raise ValueError("taxonomy labels must not be blank")
        duplicate_labels = sorted(
            {label for label in display_labels if display_labels.count(label) > 1}
        )
        if duplicate_labels:
            raise ValueError(
                f"duplicate taxonomy display labels: {', '.join(duplicate_labels)}"
            )
        report.checks.append(
            _passed(
                "taxonomy",
                f"{taxonomy.node_count()} unique labels",
                loaded.taxonomy_path,
            )
        )
    except Exception as exc:
        report.checks.append(_failed("taxonomy", exc, loaded.taxonomy_path))

    try:
        Draft202012Validator.check_schema(loaded.data_schema)
        report.checks.append(
            _passed("data_schema", "valid JSON Schema", loaded.data_schema_path)
        )
    except Exception as exc:
        report.checks.append(_failed("data_schema", exc, loaded.data_schema_path))

    if loaded.rules is not None:
        try:
            if taxonomy is None:
                raise ValueError("rules cannot be checked until taxonomy is valid")
            snapshot = RuleClassifier(
                loaded.rules.path,
                TaxonomyLoader(loaded.taxonomy_path, poll_interval=0),
            ).get_snapshot()
            rules = snapshot.value
            _validate_structural_conditions(
                [rule.condition for rule in rules.structural]
            )
            count = sum(
                len(group)
                for group in (
                    rules.sender_domains,
                    rules.subject_patterns,
                    rules.body_keywords,
                    rules.structural,
                )
            )
            report.checks.append(
                _passed("rules", f"{count} rules; all labels resolve", loaded.rules.path)
            )
        except Exception as exc:
            report.checks.append(_failed("rules", exc, loaded.rules.path))

    if loaded.target_profiles is not None:
        try:
            if taxonomy is None:
                raise ValueError("target profiles cannot be checked until taxonomy is valid")
            count = _validate_target_profiles(loaded.target_profiles.path, taxonomy)
            report.checks.append(
                _passed(
                    "target_profiles",
                    f"{count} target profiles; all labels resolve",
                    loaded.target_profiles.path,
                )
            )
        except Exception as exc:
            report.checks.append(
                _failed("target_profiles", exc, loaded.target_profiles.path)
            )

    if loaded.rag is not None:
        try:
            count = _validate_rag_declaration(loaded.rag.path)
            report.checks.append(
                _passed("rag", f"valid declaration with {count} sources", loaded.rag.path)
            )
        except Exception as exc:
            report.checks.append(_failed("rag", exc, loaded.rag.path))

    if loaded.retrieval_cleaning is not None:
        try:
            policy = load_retrieval_cleaning_policy(loaded.retrieval_cleaning.path)
            report.checks.append(
                _passed(
                    "retrieval_cleaning",
                    f"valid policy version {policy.version!r}",
                    loaded.retrieval_cleaning.path,
                )
            )
        except Exception as exc:
            report.checks.append(
                _failed("retrieval_cleaning", exc, loaded.retrieval_cleaning.path)
            )

    if selected.plugin.validate_profile is not None:
        try:
            for result in selected.plugin.validate_profile(loaded):
                report.checks.append(
                    ValidationCheck(
                        result.component,
                        result.passed,
                        result.detail,
                        result.path,
                    )
                )
        except Exception as exc:
            report.checks.append(_failed("plugin_assets", exc, profile_root))

    return report
