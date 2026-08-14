"""Installed vertical plugin descriptors and discovery."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from .loader import VerticalConfigurationError


VERTICAL_ENTRY_POINT_GROUP = "mailagent.verticals"


@dataclass(frozen=True, slots=True)
class PluginValidationResult:
    """One plugin-specific profile validation result."""

    component: str
    passed: bool
    detail: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class VerticalPlugin:
    """Executable part of a vertical, owned by an installable Python package."""

    id: str
    namespace: str
    build_runtime: Callable[..., Any]
    validate_profile: Callable[[Any], Iterable[PluginValidationResult]] | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.namespace or not callable(self.build_runtime):
            raise ValueError("vertical plugin requires id, namespace, and a runtime builder")


class VerticalPluginRegistry:
    """Resolve built-in and installed vertical plugins by stable ID."""

    def __init__(self, plugins: Iterable[VerticalPlugin] = ()) -> None:
        self._plugins: dict[str, VerticalPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: VerticalPlugin) -> None:
        existing = self._plugins.get(plugin.id)
        if existing is not None:
            if existing == plugin:
                return
            raise VerticalConfigurationError(
                f"duplicate installed vertical plugin id: {plugin.id}"
            )
        self._plugins[plugin.id] = plugin

    def resolve(self, plugin_id: str) -> VerticalPlugin:
        try:
            return self._plugins[plugin_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._plugins)) or "none"
            raise VerticalConfigurationError(
                f"vertical plugin {plugin_id!r} is not installed; available: {available}"
            ) from exc

    @classmethod
    def discover(cls) -> "VerticalPluginRegistry":
        registry = cls(_builtin_plugins())
        entry_points = metadata.entry_points()
        selected = (
            entry_points.select(group=VERTICAL_ENTRY_POINT_GROUP)
            if hasattr(entry_points, "select")
            else entry_points.get(VERTICAL_ENTRY_POINT_GROUP, ())
        )
        for entry_point in selected:
            try:
                candidate = entry_point.load()
            except Exception as exc:
                raise VerticalConfigurationError(
                    f"cannot load vertical plugin entry point {entry_point.name!r}"
                ) from exc
            if not isinstance(candidate, VerticalPlugin):
                raise VerticalConfigurationError(
                    f"vertical plugin entry point {entry_point.name!r} "
                    "must expose a VerticalPlugin descriptor"
                )
            if entry_point.name != candidate.id:
                raise VerticalConfigurationError(
                    f"vertical plugin entry point {entry_point.name!r} exposes "
                    f"mismatched id {candidate.id!r}"
                )
            registry.register(candidate)
        return registry


def _builtin_plugins() -> tuple[VerticalPlugin, ...]:
    # Core ships one built-in vertical: example-triage, a minimal
    # classification-only template for contributors to fork. Business verticals
    # are NOT built-in — they install as separate pip packages and are
    # discovered via the ``mailagent.verticals`` entry point.
    from .example_triage.plugin import plugin as example_triage_plugin

    return (example_triage_plugin,)
