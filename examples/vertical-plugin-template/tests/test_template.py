from __future__ import annotations

import asyncio
from pathlib import Path

from mailagent.verticals.loader import load_vertical
from mailagent.verticals.runtime import VerticalRuntime
from mailagent_example_plugin.plugin import plugin


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_plugin_descriptor_matches_external_profile() -> None:
    loaded = load_vertical(
        PROJECT_ROOT / "verticals" / "example_plugin" / "manifest.yaml"
    )

    assert plugin.id == loaded.manifest.id == "example-plugin"
    assert plugin.namespace == loaded.manifest.namespace == "example_plugin"


def test_template_builds_a_valid_classification_only_runtime() -> None:
    runtime = asyncio.run(plugin.build_runtime(None))

    assert isinstance(runtime, VerticalRuntime)
    assert runtime.enrichers == []
