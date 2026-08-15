from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mailagent.verticals import (
    VerticalConfigurationError,
    VerticalPlugin,
    VerticalPluginRegistry,
    load_selected_vertical,
)
from mailagent.verticals.runtime import build_empty_runtime
from mailagent.verticals import plugin as plugin_module


PROJECT_ROOT = Path(__file__).parents[3]


def test_builtin_example_triage_plugin_matches_external_profile() -> None:
    from mailagent.verticals.example_triage.plugin import (
        plugin as example_triage_plugin,
    )

    settings = SimpleNamespace(
        id="example-triage",
        verticals_path=str(PROJECT_ROOT / "verticals"),
    )

    selected = load_selected_vertical(
        settings,
        registry=VerticalPluginRegistry([example_triage_plugin]),
    )

    assert selected.plugin is example_triage_plugin
    assert (
        selected.assets.taxonomy_path
        == (PROJECT_ROOT / "verticals/example_triage/taxonomy.yaml").resolve()
    )
    assert selected.assets.rules is not None


def test_external_profile_cannot_select_uninstalled_code(tmp_path: Path) -> None:
    profile = tmp_path / "example"
    profile.mkdir()
    (profile / "manifest.yaml").write_text(
        """
id: example
namespace: example
data_schema_version: "1"
taxonomy: taxonomy.yaml
data_schema: data-schema.json
runtime_factory: arbitrary.module:factory
enrichers: []
""",
        encoding="utf-8",
    )
    (profile / "taxonomy.yaml").write_text("version: 1\nnodes: []\n", encoding="utf-8")
    (profile / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")

    with pytest.raises(VerticalConfigurationError, match="is not installed"):
        load_selected_vertical(
            SimpleNamespace(id="example", verticals_path=str(tmp_path)),
            registry=VerticalPluginRegistry(),
        )


def test_plugin_and_profile_namespace_must_match(tmp_path: Path) -> None:
    profile = tmp_path / "example"
    profile.mkdir()
    (profile / "manifest.yaml").write_text(
        """
id: example
namespace: wrong_namespace
data_schema_version: "1"
taxonomy: taxonomy.yaml
data_schema: data-schema.json
enrichers: []
""",
        encoding="utf-8",
    )
    (profile / "taxonomy.yaml").write_text("version: 1\nnodes: []\n", encoding="utf-8")
    (profile / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")
    registry = VerticalPluginRegistry(
        [VerticalPlugin("example", "example", build_empty_runtime)]
    )

    with pytest.raises(
        VerticalConfigurationError, match="does not match profile namespace"
    ):
        load_selected_vertical(
            SimpleNamespace(id="example", verticals_path=str(tmp_path)),
            registry=registry,
        )


def test_duplicate_plugin_ids_fail_fast() -> None:
    registry = VerticalPluginRegistry(
        [VerticalPlugin("example", "example", build_empty_runtime)]
    )

    with pytest.raises(VerticalConfigurationError, match="duplicate"):
        registry.register(VerticalPlugin("example", "other", build_empty_runtime))


def test_third_party_entry_point_is_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    third_party = VerticalPlugin(
        "customer-service", "customer_service", build_empty_runtime
    )

    class EntryPoints:
        def select(self, *, group: str):
            assert group == "mailagent.verticals"
            return [SimpleNamespace(name="customer-service", load=lambda: third_party)]

    monkeypatch.setattr(plugin_module.metadata, "entry_points", lambda: EntryPoints())

    assert VerticalPluginRegistry.discover().resolve("customer-service") == third_party


def test_discovery_only_loads_selected_third_party_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = VerticalPlugin(
        "customer-service", "customer_service", build_empty_runtime
    )
    def unrelated_load():
        raise RuntimeError("must not load")

    class EntryPoints:
        def select(self, *, group: str):
            assert group == "mailagent.verticals"
            return [
                SimpleNamespace(name="unrelated", load=unrelated_load),
                SimpleNamespace(name="customer-service", load=lambda: selected),
            ]

    monkeypatch.setattr(plugin_module.metadata, "entry_points", lambda: EntryPoints())

    assert (
        VerticalPluginRegistry.discover("customer-service").resolve("customer-service")
        is selected
    )


def test_incompatible_plugin_api_version_fails_fast() -> None:
    with pytest.raises(ValueError, match="Core supports"):
        VerticalPlugin(
            "customer-service",
            "customer_service",
            build_empty_runtime,
            api_version="999",
        )
