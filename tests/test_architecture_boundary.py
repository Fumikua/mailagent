"""Regression tests for the public framework/private vertical boundary."""
from __future__ import annotations

import ast
import base64
from pathlib import Path
import tomllib

from mailagent.domain.models import Decision, SampleRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    PROJECT_ROOT / "src" / "mailagent",
    PROJECT_ROOT / "migrations",
)


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def test_core_and_classification_import_no_vertical_implementation() -> None:
    violations: list[str] = []
    for package in ("core", "classification"):
        root = PROJECT_ROOT / "src" / "mailagent" / package
        for path in _python_files(root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any("mailagent.verticals." in name for name in names):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert violations == []


def test_compatibility_decision_does_not_enumerate_business_categories() -> None:
    assert Decision.model_fields["category"].annotation is str


def test_sample_contract_has_only_generic_fields() -> None:
    assert set(SampleRecord.model_fields) == {
        "id",
        "mail_hash",
        "subject_raw",
        "subject_clean",
        "sender",
        "sender_domain",
        "body",
        "label_l1",
        "label_l2",
        "label_l3",
        "confidence",
        "source",
        "reviewed",
        "thread_parsed",
        "created_at",
        "batch_confirmed_at",
        "taxonomy_schema_version",
        "retrieval_document",
        "retrieval_fingerprint",
        "retrieval_policy_version",
        "quality",
        "review_override_reason",
    }


def test_public_runtime_contains_no_private_vertical_vocabulary() -> None:
    # Encoded so the public framework does not itself publish the protected
    # vocabulary that this regression gate rejects.
    encoded_needles = (
        "dW5pc2Nv",
        "bWFlcnNr",
        "c2hpcC1hZ2VuY3k=",
        "c2hpcF9hZ2VuY3k=",
        "dmVzc2VsX25hbWU=",
        "dmVzc2VsX2V4dHJhY3Rpb24=",
        "cHJlLWFycml2YWw=",
    )
    needles = [base64.b64decode(value).decode("ascii") for value in encoded_needles]
    violations: list[str] = []
    for root in RUNTIME_ROOTS:
        for path in sorted(
            file
            for file in root.rglob("*")
            if file.is_file() and file.suffix in {".py", ".yml", ".yaml", ".toml"}
        ):
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(needle in text for needle in needles):
                violations.append(str(path.relative_to(PROJECT_ROOT)))
    assert violations == []


def test_core_wheel_excludes_plugin_examples() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/mailagent"
    ]
    workspace_members = (
        project.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    )
    assert "examples/vertical-plugin-template" not in workspace_members
    assert (PROJECT_ROOT / "examples" / "vertical-plugin-template").is_dir()
