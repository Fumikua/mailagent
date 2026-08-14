"""Vertical-neutral classification foundation.

Classifier contracts, taxonomy configuration, and the Rules/Vector/LLM
implementations live together here. The package root exposes only lightweight
contracts; implementations use explicit submodule imports so importing a
contract does not initialize provider or preprocessing dependencies. Business
knowledge remains in the active vertical's declared assets.
"""

from .contracts import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationCoreResult,
    ClassificationOrchestrator,
    ClassificationRequest,
    Classifier,
    Enricher,
    EnrichmentPatch,
)
__all__ = [
    "AttemptStatus",
    "ClassificationAttempt",
    "ClassificationCoreResult",
    "ClassificationOrchestrator",
    "ClassificationRequest",
    "Classifier",
    "Enricher",
    "EnrichmentPatch",
]
