"""Vertical-neutral contracts for mail classification and enrichment."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from mailagent.domain.models import CalibrationLog, ClassificationMeta, MailEvent, TaxonomyLabel


class AttemptStatus(StrEnum):
    """Execution state reported by one classifier implementation."""

    SUCCESS = "success"
    NO_MATCH = "no_match"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class ClassificationRequest(BaseModel):
    """Common input supplied to every classifier for the selected vertical."""

    mail: MailEvent
    context: dict[str, Any] = Field(default_factory=dict)
    asset_snapshots: dict[str, Any] = Field(default_factory=dict, exclude=True)


class ClassificationAttempt(BaseModel):
    """A candidate result from exactly one classifier, before orchestration."""

    source: str
    status: AttemptStatus
    labels: list[TaxonomyLabel] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.0)
    meta: ClassificationMeta = Field(default_factory=ClassificationMeta)
    calibration_log: CalibrationLog | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ClassificationCoreResult(BaseModel):
    """The final generic classification result selected by an orchestrator."""

    labels: list[TaxonomyLabel] = Field(default_factory=list)
    meta: ClassificationMeta = Field(default_factory=ClassificationMeta)
    calibration_log: CalibrationLog | None = None
    selected_source: str | None = None
    attempts: list[ClassificationAttempt] = Field(default_factory=list)
    audit: dict[str, Any] = Field(default_factory=dict)


class EnrichmentPatch(BaseModel):
    """One namespace-scoped business-data patch returned by an enricher."""

    namespace: str
    data: dict[str, Any] = Field(default_factory=dict)


class Classifier(Protocol):
    """One independently replaceable classification mechanism."""

    source: str

    async def classify(self, request: ClassificationRequest) -> ClassificationAttempt: ...


class ClassificationOrchestrator(Protocol):
    """Selects classifier attempts and owns fallback and review semantics."""

    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult: ...


class Enricher(Protocol):
    """Adds data for one vertical-owned namespace after classification."""

    id: str
    namespace: str

    async def enrich(
        self,
        request: ClassificationRequest,
        classification: ClassificationCoreResult,
    ) -> EnrichmentPatch: ...
