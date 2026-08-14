from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mailagent.infra.config import Settings, VerticalSettings
from mailagent.infra.queue import build_mail_understanding_pipeline, worker_on_startup
from mailagent.verticals.runtime import VerticalRuntime, build_empty_runtime
from mailagent.verticals import VerticalPlugin, load_vertical


def test_vertical_settings_have_no_business_default() -> None:
    settings = VerticalSettings()

    assert settings.id == ""
    assert settings.verticals_path == "./verticals"


def test_settings_load_selected_vertical_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    config.write_text(
        """
app:
  name: test
  environment: test
vertical:
  id: customer-service
  verticals_path: ./business-verticals
""",
        encoding="utf-8",
    )

    settings = Settings.from_yaml(config)

    assert settings.vertical.id == "customer-service"
    assert settings.vertical.verticals_path == "./business-verticals"


def test_selected_vertical_rejects_unknown_enricher_before_worker_starts(tmp_path: Path) -> None:
    vertical_dir = tmp_path / "example_triage"
    vertical_dir.mkdir()
    (vertical_dir / "manifest.yaml").write_text(
        """
id: example-triage
namespace: example_triage
data_schema_version: "1"
taxonomy: taxonomy.yaml
data_schema: data-schema.json
runtime_factory: mailagent.verticals.runtime:build_empty_runtime
enrichers: [unknown]
""",
        encoding="utf-8",
    )
    (vertical_dir / "taxonomy.yaml").write_text("version: 1\ntaxonomy: []\n", encoding="utf-8")
    (vertical_dir / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")
    settings = Settings(vertical=VerticalSettings(id="example-triage", verticals_path=str(tmp_path)))

    with pytest.raises(ValueError, match="do not match"):
        build_mail_understanding_pipeline(settings, MagicMock(), VerticalRuntime())


async def test_non_ship_vertical_does_not_initialize_entity_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vertical_dir = tmp_path / "customer_service"
    vertical_dir.mkdir()
    (vertical_dir / "manifest.yaml").write_text(
        """
id: customer-service
namespace: customer_service
data_schema_version: "1"
taxonomy: taxonomy.yaml
data_schema: data-schema.json
runtime_factory: mailagent.verticals.runtime:build_empty_runtime
enrichers: []
""",
        encoding="utf-8",
    )
    (vertical_dir / "taxonomy.yaml").write_text("version: 1\nnodes: []\n", encoding="utf-8")
    (vertical_dir / "data-schema.json").write_text('{"type": "object"}', encoding="utf-8")
    settings = Settings(
        vertical=VerticalSettings(id="customer-service", verticals_path=str(tmp_path))
    )
    monkeypatch.setattr(Settings, "from_yaml", classmethod(lambda cls: settings))
    monkeypatch.setattr("mailagent.infra.migrations.upgrade_database", AsyncMock())
    monkeypatch.setattr("mailagent.infra.store.SqlStore", MagicMock())
    monkeypatch.setattr("mailagent.llm.client.LLMClient", MagicMock())
    selected = SimpleNamespace(
        plugin=VerticalPlugin("customer-service", "customer_service", build_empty_runtime),
        assets=load_vertical(vertical_dir / "manifest.yaml"),
    )
    monkeypatch.setattr("mailagent.verticals.load_selected_vertical", lambda _: selected)
    ctx: dict[str, object] = {}
    await worker_on_startup(ctx)

    assert "entity_registry" not in ctx
    assert ctx["vertical_runtime"].context == {}
    assert "mail_understanding_pipeline" in ctx
