"""Vertical plugin loading and selected-vertical runtime contracts."""

from .loader import (  # noqa: F401
    LoadedVertical,
    VerticalAsset,
    VerticalConfigurationError,
    VerticalManifest,
    load_runtime_factory,
    load_vertical,
)
from .runtime import VerticalRuntime, VerticalRuntimeDependencies, build_vertical_runtime  # noqa: F401
from .plugin import (  # noqa: F401
    VERTICAL_ENTRY_POINT_GROUP,
    PluginValidationResult,
    VerticalPlugin,
    VerticalPluginRegistry,
)
from .selection import SelectedVertical, load_selected_vertical  # noqa: F401
