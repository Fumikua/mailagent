"""Deprecated compatibility facade for classification contracts.

New code must import from :mod:`mailagent.classification` or
:mod:`mailagent.classification.contracts`.
"""

from mailagent.classification.contracts import (
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
