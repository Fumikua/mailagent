"""Installed plugin descriptor for the example vertical."""
from __future__ import annotations

from mailagent.verticals.plugin import VerticalPlugin
from mailagent.verticals.runtime import build_empty_runtime


plugin = VerticalPlugin(
    id="example-plugin",
    namespace="example_plugin",
    build_runtime=build_empty_runtime,
)
