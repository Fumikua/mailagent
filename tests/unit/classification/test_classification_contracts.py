from mailagent.classification import AttemptStatus, ClassificationAttempt
from mailagent.domain.models import ClassificationResponse, TaxonomyLabel


def test_classification_attempt_keeps_one_classifier_result_without_review_decision() -> None:
    attempt = ClassificationAttempt(
        source="rules",
        status=AttemptStatus.SUCCESS,
        labels=[
            TaxonomyLabel(
                l1_code="entity",
                l1_label="实体相关",
                confidence=0.96,
                reasoning="matched an explicit rule",
            )
        ],
        confidence=0.96,
        evidence={"rule_id": "eta-change-subject"},
    )

    assert attempt.source == "rules"
    assert attempt.confidence == 0.96
    assert attempt.labels[0].l1_code == "entity"
    assert not hasattr(attempt, "needs_human_review")


def test_classification_response_has_default_versioned_data_envelope() -> None:
    response = ClassificationResponse()

    assert response.vertical_id == ""
    assert response.data_schema_version == "1"
    assert response.data == {}
