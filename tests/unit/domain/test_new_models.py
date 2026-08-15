"""Unit tests for vector-similarity-path-b domain models."""

from __future__ import annotations


import pytest
from pydantic import ValidationError

from mailagent.domain.models import (
    ClassificationFeedbackRequest,
    ClassificationResponse,
    FusionMeta,
    NormalizedSubject,
    PathBCandidate,
    PathBResult,
    RuleMatch,
    RuleResult,
    SampleRecord,
    TaxonomyLabel,
    ThreadSegment,
)


@pytest.mark.parametrize(
    "payload",
    [
        {"final_labels": [], "error_reasons": ["wrong_label"]},
        {
            "final_labels": ["schedule", "schedule"],
            "error_reasons": ["wrong_label"],
        },
        {"final_labels": [" schedule"], "error_reasons": ["wrong_label"]},
        {"final_labels": ["schedule"], "error_reasons": []},
        {
            "final_labels": ["schedule"],
            "error_reasons": ["wrong_label", "wrong_label"],
        },
        {
            "final_labels": ["schedule"],
            "error_reasons": ["wrong_label"],
            "reviewer_id": "body-reviewer",
        },
        {
            "final_labels": ["schedule"],
            "error_reasons": ["wrong_label"],
            "eligible_for_sample_proposal": False,
        },
    ],
)
def test_classification_feedback_request_rejects_invalid_domain_input(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ClassificationFeedbackRequest.model_validate(payload)


class TestNormalizedSubject:
    def test_field_population(self):
        ns = NormalizedSubject(
            raw="Re: Re: Example Berlin STATUS", clean="Example Berlin STATUS"
        )
        assert ns.raw == "Re: Re: Example Berlin STATUS"
        assert ns.clean == "Example Berlin STATUS"
        assert ns.model_dump() == {
            "raw": "Re: Re: Example Berlin STATUS",
            "clean": "Example Berlin STATUS",
        }

    def test_empty_optional_fields(self):
        ns = NormalizedSubject(raw="Hello", clean="Hello")
        assert ns.model_dump() == {"raw": "Hello", "clean": "Hello"}


class TestThreadSegment:
    def test_position_and_is_latest(self):
        seg0 = ThreadSegment(text="latest content", position=0, is_latest=True)
        seg1 = ThreadSegment(text="quoted content", position=1, is_latest=False)
        assert seg0.is_latest is True
        assert seg1.is_latest is False
        assert seg0.position < seg1.position


class TestRuleResult:
    def test_conflict_resolution_selected_single(self):
        matches = [
            RuleMatch(
                rule_type="sender_domains",
                label="locationing",
                confidence=0.95,
                matched_pattern="@example.com",
            ),
            RuleMatch(
                rule_type="subject_patterns",
                label="arrival",
                confidence=0.80,
                matched_pattern="STATUS.*",
            ),
        ]
        result = RuleResult(matches=matches, selected=matches[0], conflict_logged=True)
        assert result.selected is not None
        assert result.selected.label == "locationing"
        assert result.conflict_logged is True

    def test_empty_matches(self):
        result = RuleResult()
        assert result.matches == []
        assert result.selected is None
        assert result.conflict_logged is False


class TestPathBResult:
    def test_empty_candidates(self):
        result = PathBResult()
        assert result.candidates == []
        assert result.top1_label is None
        assert result.confidence == 0.0

    def test_with_candidates(self):
        candidates = [
            PathBCandidate(
                label="locationing",
                max_similarity=0.92,
                count=3,
                mean_similarity=0.88,
                confidence=0.90,
            ),
            PathBCandidate(
                label="arrival",
                max_similarity=0.85,
                count=2,
                mean_similarity=0.82,
                confidence=0.83,
            ),
        ]
        result = PathBResult(
            candidates=candidates,
            top1_label="locationing",
            top1_similarity=0.92,
            confidence=0.90,
        )
        assert len(result.candidates) == 2
        assert result.top1_label == "locationing"


class TestFusionMeta:
    @pytest.mark.parametrize(
        "strategy",
        [
            "rule_only",
            "rule_vector_confirmed",
            "vector_only",
            "llm_fallback",
            "all_low_review",
        ],
    )
    def test_all_strategies(self, strategy: str):
        fm = FusionMeta(
            fusion_strategy=strategy,
            source="rule"
            if "rule" in strategy
            else "vector"
            if "vector" in strategy
            else "llm",
            confidence=0.85,
        )
        assert fm.fusion_strategy == strategy

    def test_with_rule_result(self):
        rr = RuleResult(
            matches=[
                RuleMatch(
                    rule_type="sender_domains",
                    label="locationing",
                    confidence=0.95,
                    matched_pattern="@x.com",
                )
            ],
            selected=RuleMatch(
                rule_type="sender_domains",
                label="locationing",
                confidence=0.95,
                matched_pattern="@x.com",
            ),
        )
        fm = FusionMeta(
            fusion_strategy="rule_only", source="rule", confidence=0.95, rule_result=rr
        )
        assert fm.rule_result is not None
        assert fm.rule_result.selected is not None
        assert fm.vector_confirmed is False

    def test_target_profile_default_none(self):
        """target_profile defaults to None (feature off / no match)."""
        fm = FusionMeta(fusion_strategy="rule_only", source="rule", confidence=0.95)
        assert fm.target_profile is None

    def test_target_profile_round_trip(self):
        """target_profile survives model_dump / model_validate round-trip."""
        fm = FusionMeta(
            fusion_strategy="rule_vector_confirmed",
            source="rule",
            confidence=0.82,
            vector_confirmed=True,
            target_profile="entity_report",
        )
        data = fm.model_dump()
        assert data["target_profile"] == "entity_report"
        fm2 = FusionMeta.model_validate(data)
        assert fm2.target_profile == "entity_report"
        assert fm2.fusion_strategy == "rule_vector_confirmed"


class TestSampleRecord:
    def test_new_flat_sample_defaults_deprecated_levels_to_none(self):
        sr = SampleRecord(
            mail_hash="flat-1",
            subject_raw="STATUS Update",
            subject_clean="STATUS Update",
            sender="ops@example.com",
            sender_domain="example.com",
            body="STATUS revised to 14:00.",
            label_l1="schedule",
            confidence=0.95,
            source="seed",
        )

        assert sr.label_l2 is None
        assert sr.label_l3 is None
        assert sr.taxonomy_schema_version == "flat-v1"

    def test_serialization(self):
        sr = SampleRecord(
            mail_hash="abc123",
            subject_raw="Re: Test",
            subject_clean="Test",
            sender="user@example.com",
            sender_domain="example.com",
            body="Hello world",
            label_l1="operations",
            label_l2="locationing",
            label_l3="arrival_notice",
            confidence=0.95,
            source="seed",
        )
        data = sr.model_dump()
        assert data["mail_hash"] == "abc123"
        assert data["source"] == "seed"
        # Round-trip
        sr2 = SampleRecord(**data)
        assert sr2.mail_hash == sr.mail_hash

    def test_source_enum_validation(self):
        sr = SampleRecord(
            mail_hash="abc",
            subject_raw="T",
            subject_clean="T",
            sender="s@x.com",
            sender_domain="x.com",
            body="B",
            label_l1="a",
            label_l2="b",
            label_l3="c",
            confidence=0.5,
            source="human_fix",
        )
        assert sr.source == "human_fix"

    def test_invalid_source_rejected(self):
        with pytest.raises(ValueError):
            SampleRecord(
                mail_hash="abc",
                subject_raw="T",
                subject_clean="T",
                sender="s@x.com",
                sender_domain="x.com",
                body="B",
                label_l1="a",
                label_l2="b",
                label_l3="c",
                confidence=0.5,
                source="invalid_source",
            )


class TestClassificationResponseFusionMeta:
    def test_fusion_meta_none_backward_compatible(self):
        resp = ClassificationResponse(
            labels=[
                TaxonomyLabel(l1_code="ops", l1_label="Operations", confidence=0.9)
            ],
        )
        assert resp.fusion_meta is None

    def test_fusion_meta_populated(self):
        fm = FusionMeta(fusion_strategy="rule_only", source="rule", confidence=0.95)
        resp = ClassificationResponse(
            labels=[
                TaxonomyLabel(l1_code="ops", l1_label="Operations", confidence=0.95)
            ],
            fusion_meta=fm,
        )
        assert resp.fusion_meta is not None
        assert resp.fusion_meta.fusion_strategy == "rule_only"
