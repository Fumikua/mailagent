from pathlib import Path

import pytest
import yaml

from mailagent.classification.rule_classifier import RuleClassifier
from mailagent.core.target_profile import TargetProfileLoader
from mailagent.core.versioning import ClassificationVersionProvider, _digest
from mailagent.classification.taxonomy import TaxonomyLoader
from mailagent.preprocessing.retrieval_models import load_retrieval_cleaning_policy


def _write_taxonomy(path: Path, code: str = "schedule") -> None:
    path.write_text(
        yaml.safe_dump({"nodes": [{"code": code, "label": code}]}),
        encoding="utf-8",
    )


def _write_rules(rules_dir: Path, subject_label: str | None = None) -> None:
    rules_dir.mkdir()
    for filename in (
        "sender_domains.yaml",
        "subject_patterns.yaml",
        "body_keywords.yaml",
        "structural.yaml",
    ):
        rules = (
            [{"pattern": "STATUS", "label": subject_label}]
            if filename == "subject_patterns.yaml" and subject_label is not None
            else []
        )
        (rules_dir / filename).write_text(
            yaml.safe_dump(rules),
            encoding="utf-8",
        )


def _provider(
    taxonomy_loader: TaxonomyLoader,
    *,
    rule_classifier: RuleClassifier | None = None,
    preprocessing_extension: object | None = None,
    retrieval_cleaning_policy: object | None = None,
    target_profile_loader: TargetProfileLoader | None = None,
) -> ClassificationVersionProvider:
    return ClassificationVersionProvider(
        taxonomy_loader=taxonomy_loader,
        rule_classifier=rule_classifier,
        preprocessing_extension=preprocessing_extension,
        retrieval_cleaning_policy=retrieval_cleaning_policy,
        target_profile_loader=target_profile_loader,
        prompt_version="llm-classifier-v1",
        model_version="test-model",
        embedding_version=None,
    )


def test_version_provider_binds_target_profiles_snapshot_and_digest(
    tmp_path: Path,
) -> None:
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text(
        yaml.safe_dump(
            {
                "nodes": [
                    {"code": "schedule", "label": "schedule"},
                    {"code": "notification", "label": "notification"},
                ]
            }
        ),
        encoding="utf-8",
    )
    profiles = tmp_path / "target_profiles.yaml"
    profiles.write_text(
        yaml.safe_dump(
            {
                "targets": [
                    {"label": "schedule", "vector_scope": ["schedule"]}
                ]
            }
        ),
        encoding="utf-8",
    )
    taxonomy_loader = TaxonomyLoader(taxonomy, poll_interval=0)
    profile_loader = TargetProfileLoader(profiles, taxonomy_loader, poll_interval=0)
    provider = _provider(
        taxonomy_loader,
        target_profile_loader=profile_loader,
    )

    first = provider.bind()
    profiles.write_text(
        yaml.safe_dump(
            {
                "targets": [
                    {
                        "label": "notification",
                        "vector_scope": ["notification"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    second = provider.bind()

    assert first.versions.target_profiles is not None
    assert first.versions.target_profiles != second.versions.target_profiles
    assert first.asset_snapshots["target_profiles"].version == (
        first.versions.target_profiles
    )
    assert second.asset_snapshots["target_profiles"].version == (
        second.versions.target_profiles
    )


def test_version_provider_changes_only_after_validated_taxonomy_reload(
    tmp_path: Path,
) -> None:
    taxonomy = tmp_path / "taxonomy.yaml"
    _write_taxonomy(taxonomy)
    loader = TaxonomyLoader(taxonomy, poll_interval=0)
    provider = _provider(loader)

    first = provider.snapshot()
    _write_taxonomy(taxonomy, code="notification")
    second = provider.snapshot()

    assert first.taxonomy != second.taxonomy
    assert first.prompt == second.prompt == "llm-classifier-v1"


def test_version_provider_uses_active_multi_file_rule_snapshot(tmp_path: Path) -> None:
    taxonomy = tmp_path / "taxonomy.yaml"
    _write_taxonomy(taxonomy)
    rules_dir = tmp_path / "rules"
    _write_rules(rules_dir, subject_label="schedule")
    loader = TaxonomyLoader(taxonomy, poll_interval=0)
    classifier = RuleClassifier(rules_dir, loader)
    provider = _provider(loader, rule_classifier=classifier)

    first = provider.snapshot()
    (rules_dir / "body_keywords.yaml").write_text(
        yaml.safe_dump(
            [{"keywords": ["DG"], "label": "schedule", "confidence": 0.8}]
        ),
        encoding="utf-8",
    )
    classifier._last_check = 0
    second = provider.snapshot()

    assert first.rules is not None
    assert first.rules != second.rules


def test_version_provider_versions_are_independent_of_installation_root(
    tmp_path: Path,
) -> None:
    versions = []
    for root_name in ("first", "second"):
        root = tmp_path / root_name
        root.mkdir()
        taxonomy = root / "taxonomy.yaml"
        _write_taxonomy(taxonomy)
        rules_dir = root / "rules"
        _write_rules(rules_dir, subject_label="schedule")
        cleaning = root / "retrieval_cleaning.yaml"
        cleaning.write_text(
            yaml.safe_dump({"version": "clean-v1", "min_meaningful_chars": 4}),
            encoding="utf-8",
        )
        loader = TaxonomyLoader(taxonomy, poll_interval=0)
        versions.append(
            _provider(
                loader,
                rule_classifier=RuleClassifier(rules_dir, loader),
                retrieval_cleaning_policy=load_retrieval_cleaning_policy(cleaning),
            ).snapshot()
        )

    assert versions[0] == versions[1]


def test_version_provider_represents_unused_optional_components(
    tmp_path: Path,
) -> None:
    taxonomy = tmp_path / "taxonomy.yaml"
    _write_taxonomy(taxonomy)

    versions = _provider(TaxonomyLoader(taxonomy)).snapshot()

    assert versions.rules is None
    assert versions.embedding is None
    assert versions.preprocessing == "none"


def test_digest_is_deterministic_for_explicit_logical_names() -> None:
    forward = _digest(
        [
            ("rules:sender_domains", b"sender"),
            ("rules:subject_patterns", b"subject"),
        ]
    )
    reverse = _digest(
        [
            ("rules:subject_patterns", b"subject"),
            ("rules:sender_domains", b"sender"),
        ]
    )

    assert forward == reverse


def test_digest_rejects_duplicate_logical_names() -> None:
    with pytest.raises(ValueError, match="duplicate logical asset name"):
        _digest(
            [
                ("preprocessing:patterns", b"first"),
                ("preprocessing:patterns", b"second"),
            ]
        )
