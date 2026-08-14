from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationRequest,
)
from mailagent.core.orchestration import CascadeClassificationOrchestrator
from mailagent.domain.models import MailEvent, TaxonomyLabel


class StubClassifier:
    def __init__(self, source: str, attempt: ClassificationAttempt) -> None:
        self.source = source
        self.attempt = attempt
        self.calls = 0

    async def classify(self, request: ClassificationRequest) -> ClassificationAttempt:
        self.calls += 1
        return self.attempt


def _request() -> ClassificationRequest:
    return ClassificationRequest(
        mail=MailEvent(
            message_id="cascade-1",
            sender="ops@example.com",
            subject="STATUS update",
            body="STATUS changed to 18:00",
        )
    )


def _attempt(source: str, confidence: float) -> ClassificationAttempt:
    return ClassificationAttempt(
        source=source,
        status=AttemptStatus.SUCCESS,
        confidence=confidence,
        labels=[
            TaxonomyLabel(
                l1_code="entity_report",
                l1_label="实体报告",
                confidence=confidence,
                reasoning=f"{source} result",
            )
        ],
    )


async def test_accepted_rule_result_stops_the_cascade() -> None:
    rules = StubClassifier("rules", _attempt("rules", 0.96))
    vector = StubClassifier("vector", _attempt("vector", 0.95))
    llm = StubClassifier("llm", _attempt("llm", 0.94))
    orchestrator = CascadeClassificationOrchestrator(
        classifiers=[rules, vector, llm],
        acceptance_thresholds={"rules": 0.85, "vector": 0.85, "llm": 0.85},
    )

    result = await orchestrator.classify(_request())

    assert result.selected_source == "rules"
    assert result.labels[0].confidence == 0.96
    assert result.meta.needs_human_review is False
    assert rules.calls == 1
    assert vector.calls == 0
    assert llm.calls == 0


async def test_cascade_accepts_a_classifier_iterator() -> None:
    rules = StubClassifier("rules", _attempt("rules", 0.96))
    orchestrator = CascadeClassificationOrchestrator(
        classifiers=iter([rules]),
        acceptance_thresholds={"rules": 0.85},
    )

    result = await orchestrator.classify(_request())

    assert result.selected_source == "rules"


async def test_low_confidence_results_fall_through_to_llm() -> None:
    rules = StubClassifier("rules", _attempt("rules", 0.70))
    vector = StubClassifier("vector", _attempt("vector", 0.75))
    llm = StubClassifier("llm", _attempt("llm", 0.91))
    orchestrator = CascadeClassificationOrchestrator(
        classifiers=[rules, vector, llm],
        acceptance_thresholds={"rules": 0.85, "vector": 0.85, "llm": 0.85},
    )

    result = await orchestrator.classify(_request())

    assert result.selected_source == "llm"
    assert result.meta.fallback is True
    assert [attempt.source for attempt in result.attempts] == ["rules", "vector", "llm"]


async def test_unconfigured_earlier_stages_do_not_mark_default_llm_as_fallback() -> (
    None
):
    llm = StubClassifier("llm", _attempt("llm", 0.91))
    orchestrator = CascadeClassificationOrchestrator(
        classifiers=[llm],
        acceptance_thresholds={"llm": 0.85},
    )

    result = await orchestrator.classify(_request())

    assert result.selected_source == "llm"
    assert result.meta.fallback is False


async def test_below_threshold_success_is_preserved_for_review() -> None:
    orchestrator = CascadeClassificationOrchestrator(
        classifiers=[StubClassifier("llm", _attempt("llm", 0.60))],
        acceptance_thresholds={"llm": 0.80},
    )

    result = await orchestrator.classify(_request())

    assert result.labels[0].l1_code == "entity_report"
    assert result.labels[0].l2_code is None
    assert result.labels[0].l3_code is None
    assert result.selected_source == "llm"
    assert result.meta.needs_human_review is True
    assert result.meta.fallback is True
    assert result.audit["accepted"] is False
    assert result.audit["reason"] == "below_acceptance_threshold"
