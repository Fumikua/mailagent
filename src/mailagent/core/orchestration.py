"""Default safe orchestration for independent classifier implementations."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from mailagent.classification.contracts import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationCoreResult,
    ClassificationRequest,
    Classifier,
)
from mailagent.domain.models import ClassificationMeta


class CascadeClassificationOrchestrator:
    """Runs configured classifiers in order and selects the first safe result."""

    def __init__(
        self,
        classifiers: Iterable[Classifier],
        acceptance_thresholds: Mapping[str, float],
        order: tuple[str, ...] = ("rules", "vector", "llm"),
    ) -> None:
        classifier_list = list(classifiers)
        self._classifiers = {classifier.source: classifier for classifier in classifier_list}
        if len(self._classifiers) != len(classifier_list):
            raise ValueError("classifier sources must be unique")
        self._thresholds = dict(acceptance_thresholds)
        self._order = order

    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        attempts: list[ClassificationAttempt] = []
        suggestions: list[ClassificationAttempt] = []

        for source in self._order:
            classifier = self._classifiers.get(source)
            if classifier is None:
                attempts.append(
                    ClassificationAttempt(
                        source=source,
                        status=AttemptStatus.UNAVAILABLE,
                        error="classifier is not configured",
                    )
                )
                continue

            attempt = await classifier.classify(request)
            attempts.append(attempt)
            if attempt.status == AttemptStatus.SUCCESS and attempt.labels:
                suggestions.append(attempt)
            if self._is_accepted(attempt):
                return self._selected(
                    attempt,
                    attempts,
                    fallback=any(previous.status == AttemptStatus.SUCCESS for previous in attempts[:-1]),
                )

        if suggestions:
            suggestion = max(suggestions, key=lambda item: item.confidence)
            return ClassificationCoreResult(
                labels=suggestion.labels,
                meta=suggestion.meta.model_copy(
                    update={
                        "overall_confidence": suggestion.confidence,
                        "needs_human_review": True,
                        "fallback": True,
                    }
                ),
                calibration_log=suggestion.calibration_log,
                selected_source=suggestion.source,
                attempts=attempts,
                audit={
                    "accepted": False,
                    "reason": "below_acceptance_threshold",
                },
            )

        return ClassificationCoreResult(
            attempts=attempts,
            meta=ClassificationMeta(needs_human_review=True, fallback=True),
            audit={"reason": "no classifier produced an accepted result"},
        )

    def _is_accepted(self, attempt: ClassificationAttempt) -> bool:
        return (
            attempt.status == AttemptStatus.SUCCESS
            and bool(attempt.labels)
            and not attempt.meta.needs_human_review
            and attempt.confidence >= self._thresholds.get(attempt.source, 1.0)
        )

    @staticmethod
    def _selected(
        attempt: ClassificationAttempt,
        attempts: list[ClassificationAttempt],
        *,
        fallback: bool,
    ) -> ClassificationCoreResult:
        return ClassificationCoreResult(
            labels=attempt.labels,
            meta=attempt.meta.model_copy(
                update={
                    "overall_confidence": attempt.confidence,
                    "fallback": fallback,
                }
            ),
            calibration_log=attempt.calibration_log,
            selected_source=attempt.source,
            attempts=attempts,
        )
