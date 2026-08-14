"""Generic selected-vertical runtime factory contracts."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mailagent.classification.contracts import Enricher
from mailagent.domain.models import ClassificationResponse
from mailagent.preprocessing.contracts import MailPreprocessingExtension
from mailagent.preprocessing.retrieval_models import RetrievalCleaningPolicy

if TYPE_CHECKING:
    from .loader import LoadedVertical


CompatibilityProjector = Callable[[ClassificationResponse], ClassificationResponse]


@dataclass(slots=True)
class VerticalRuntimeDependencies:
    settings: Any
    llm_client: Any
    loaded_vertical: LoadedVertical


@dataclass(slots=True)
class VerticalRuntime:
    enrichers: list[Enricher] = field(default_factory=list)
    preprocessing_extension: MailPreprocessingExtension | None = None
    retrieval_cleaning_policy: RetrievalCleaningPolicy | None = None
    compatibility_projector: CompatibilityProjector | None = None
    context: dict[str, Any] = field(default_factory=dict)

    async def close(self) -> None:
        """Release vertical-owned resources when the worker exits."""


async def build_vertical_runtime(factory: Callable[..., Any], deps: VerticalRuntimeDependencies) -> VerticalRuntime:
    runtime = factory(deps)
    if inspect.isawaitable(runtime):
        runtime = await runtime
    if not isinstance(runtime, VerticalRuntime):
        raise TypeError("vertical runtime factory must return VerticalRuntime")
    return runtime


async def build_empty_runtime(_: VerticalRuntimeDependencies) -> VerticalRuntime:
    """Provide an explicit no-enrichment factory for classification-only verticals."""

    return VerticalRuntime()
