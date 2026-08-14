from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml

from mailagent.verticals.validation import validate_vertical_profile


PROJECT_ROOT = Path(__file__).parents[3]


def _copy_profile(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "verticals"
    shutil.copytree(PROJECT_ROOT / "verticals" / "example_triage", root / "example_triage")
    return SimpleNamespace(id="example-triage", verticals_path=str(root))


def test_example_triage_profile_passes_static_validation() -> None:
    report = validate_vertical_profile(
        SimpleNamespace(
            id="example-triage",
            verticals_path=str(PROJECT_ROOT / "verticals"),
        )
    )

    assert report.valid is True
    assert {check.component for check in report.checks} >= {"profile", "taxonomy", "data_schema", "rules"}


def test_unknown_rule_label_is_reported(tmp_path: Path) -> None:
    settings = _copy_profile(tmp_path)
    rules_path = Path(settings.verticals_path) / "example_triage/rules/body_keywords.yaml"
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    rules.append({"keywords": ["example"], "label": "missing_label"})
    rules_path.write_text(yaml.safe_dump(rules, allow_unicode=True), encoding="utf-8")

    report = validate_vertical_profile(settings)
    check = next(check for check in report.checks if check.component == "rules")

    assert report.valid is False
    assert check.passed is False
    assert "missing_label" in check.detail
