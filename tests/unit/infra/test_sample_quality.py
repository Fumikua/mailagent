"""Unit tests for flat-taxonomy sample admission contracts."""
from __future__ import annotations

from mailagent.preprocessing.retrieval_models import RetrievalDocument


def _document(
    *,
    eligible: bool = True,
    reason: str | None = None,
) -> RetrievalDocument:
    return RetrievalDocument(
        text="Subject: STATUS update\nLatest message:\nETA revised to 14:00.",
        primary_text="STATUS revised to 14:00.",
        context_text="",
        policy_version="example-triage-v1",
        eligible=eligible,
        ineligible_reason=reason,
        latest_char_count=21,
        context_char_count=0,
    )


def test_accepts_eligible_document_with_active_flat_category() -> None:
    from mailagent.infra.sample_quality import assess_sample_quality

    assessment = assess_sample_quality(
        _document(), label_l1="schedule", valid_labels={"schedule", "operation"}
    )

    assert assessment.disposition == "accepted"
    assert assessment.reasons == []
    assert assessment.taxonomy_schema_version == "flat-v1"


def test_rejects_attachment_dependent_document() -> None:
    from mailagent.infra.sample_quality import assess_sample_quality

    assessment = assess_sample_quality(
        _document(eligible=False, reason="attachment_dependent"),
        label_l1="document",
        valid_labels={"document"},
    )

    assert assessment.disposition == "rejected"
    assert assessment.reasons == ["attachment_dependent"]


def test_rejects_unknown_flat_category() -> None:
    from mailagent.infra.sample_quality import assess_sample_quality

    assessment = assess_sample_quality(
        _document(), label_l1="obsolete_category", valid_labels={"schedule"}
    )

    assert assessment.disposition == "rejected"
    assert assessment.reasons == ["unknown_taxonomy_category"]


def test_fingerprint_is_stable_for_equivalent_retrieval_text() -> None:
    from mailagent.infra.sample_quality import assess_sample_quality

    first = assess_sample_quality(_document(), label_l1="schedule", valid_labels={"schedule"})
    second = assess_sample_quality(_document(), label_l1="schedule", valid_labels={"schedule"})

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
