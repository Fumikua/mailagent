"""Resolve one installed vertical plugin and its external business profile."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import LoadedVertical, VerticalConfigurationError, load_vertical
from .plugin import VerticalPlugin, VerticalPluginRegistry


@dataclass(frozen=True, slots=True)
class SelectedVertical:
    plugin: VerticalPlugin
    assets: LoadedVertical


def load_selected_vertical(
    settings: Any,
    *,
    registry: VerticalPluginRegistry | None = None,
) -> SelectedVertical:
    """Load an external business profile and match it to installed code."""

    vertical_dir = settings.id.replace("-", "_")
    assets = load_vertical(Path(settings.verticals_path) / vertical_dir / "manifest.yaml")
    if assets.manifest.id != settings.id:
        raise VerticalConfigurationError(
            f"selected vertical id {settings.id!r} does not match profile "
            f"id {assets.manifest.id!r}"
        )

    plugin = (registry or VerticalPluginRegistry.discover()).resolve(settings.id)
    if plugin.namespace != assets.manifest.namespace:
        raise VerticalConfigurationError(
            f"vertical plugin {plugin.id!r} namespace {plugin.namespace!r} does not "
            f"match profile namespace {assets.manifest.namespace!r}"
        )
    return SelectedVertical(plugin=plugin, assets=assets)
