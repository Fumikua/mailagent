from pathlib import Path

import pytest

from mailagent.verticals.loader import (
    VerticalConfigurationError,
    load_runtime_factory,
    load_vertical,
)


PROJECT_ROOT = Path(__file__).parents[3]


def test_example_triage_manifest_resolves_its_own_assets() -> None:
    vertical = load_vertical(PROJECT_ROOT / "verticals/example_triage/manifest.yaml")

    assert vertical.manifest.id == "example-triage"
    assert vertical.manifest.namespace == "example_triage"
    assert vertical.manifest.data_schema_version == "1"
    assert vertical.taxonomy_path == (PROJECT_ROOT / "verticals/example_triage/taxonomy.yaml").resolve()
    assert vertical.data_schema_path == (PROJECT_ROOT / "verticals/example_triage/data-schema.json").resolve()
    assert vertical.rules is not None
    assert vertical.rules.path == (PROJECT_ROOT / "verticals/example_triage/rules").resolve()
    assert vertical.rules.version == "1"
    assert vertical.rag is None
    assert vertical.preprocessing is None
    assert vertical.retrieval_cleaning is None
    assert vertical.manifest.llm is None
    assert vertical.manifest.runtime_factory is None
    with pytest.raises(VerticalConfigurationError, match="legacy runtime_factory"):
        load_runtime_factory(vertical)


def test_legacy_manifest_runtime_factory_remains_loadable(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
id: legacy
namespace: legacy
data_schema_version: "1"
taxonomy: taxonomy.yaml
data_schema: data-schema.json
runtime_factory: mailagent.verticals.runtime:build_empty_runtime
enrichers: []
""",
        encoding="utf-8",
    )
    (tmp_path / "taxonomy.yaml").write_text("version: 1\nnodes: []\n", encoding="utf-8")
    (tmp_path / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")

    assert callable(load_runtime_factory(load_vertical(manifest)))


def test_classification_only_fixture_loads_without_extra_assets() -> None:
    vertical = load_vertical(
        PROJECT_ROOT / "tests/fixtures/verticals/classification_only/manifest.yaml"
    )

    assert vertical.manifest.id == "classification-only"
    assert vertical.manifest.enrichers == []
    assert vertical.manifest.namespace == "classification_only"


def test_manifest_with_missing_taxonomy_fails_before_mail_is_processed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
id: test-vertical
namespace: test_vertical
data_schema_version: "1"
taxonomy: missing-taxonomy.yaml
data_schema: data-schema.json
runtime_factory: mailagent.verticals.runtime:build_empty_runtime
enrichers: []
""",
        encoding="utf-8",
    )
    (tmp_path / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")

    with pytest.raises(VerticalConfigurationError, match="taxonomy"):
        load_vertical(manifest)


def test_target_profiles_asset_loaded_when_declared(tmp_path: Path) -> None:
    """5.7: manifest declares target_profiles → load_vertical resolves the asset."""
    target_profiles = tmp_path / "target_profiles.yaml"
    target_profiles.write_text("targets: []\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
id: test-vertical
namespace: test_vertical
data_schema_version: "1"
taxonomy: taxonomy.yaml
data_schema: data-schema.json
runtime_factory: mailagent.verticals.runtime:build_empty_runtime
target_profiles:
  path: target_profiles.yaml
  version: "1"
enrichers: []
""",
        encoding="utf-8",
    )
    (tmp_path / "taxonomy.yaml").write_text(
        "nodes:\n  - code: entity\n    label: x\n    description: x\n", encoding="utf-8"
    )
    (tmp_path / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")

    vertical = load_vertical(manifest)
    assert vertical.target_profiles is not None
    assert vertical.target_profiles.path == target_profiles.resolve()
    assert vertical.target_profiles.version == "1"


def test_target_profiles_missing_path_raises_clear_error(tmp_path: Path) -> None:
    """5.7: declared target_profiles path does not exist → VerticalConfigurationError."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
id: test-vertical
namespace: test_vertical
data_schema_version: "1"
taxonomy: taxonomy.yaml
data_schema: data-schema.json
runtime_factory: mailagent.verticals.runtime:build_empty_runtime
target_profiles:
  path: nonexistent_profiles.yaml
  version: "1"
enrichers: []
""",
        encoding="utf-8",
    )
    (tmp_path / "taxonomy.yaml").write_text(
        "nodes:\n  - code: entity\n    label: x\n    description: x\n", encoding="utf-8"
    )
    (tmp_path / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")

    with pytest.raises(VerticalConfigurationError, match="target_profiles"):
        load_vertical(manifest)


def test_target_profiles_optional_defaults_to_none(tmp_path: Path) -> None:
    """5.7: manifest without target_profiles → loaded.target_profiles is None (backward compat)."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
id: test-vertical
namespace: test_vertical
data_schema_version: "1"
taxonomy: taxonomy.yaml
data_schema: data-schema.json
runtime_factory: mailagent.verticals.runtime:build_empty_runtime
enrichers: []
""",
        encoding="utf-8",
    )
    (tmp_path / "taxonomy.yaml").write_text(
        "nodes:\n  - code: entity\n    label: x\n    description: x\n", encoding="utf-8"
    )
    (tmp_path / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")

    vertical = load_vertical(manifest)
    assert vertical.target_profiles is None
