"""End-to-end integration tests for FusionOrchestrator (Section 18.4).

Exercises all five fusion strategies against a real ``FusionOrchestrator``
instance with mock rule / vector / LLM classifiers. Verifies ``FusionMeta``
audit completeness for each strategy and that strategy boundaries match the
configured ``FusionSettings`` thresholds.
"""
from __future__ import annotations

from typing import Any

import pytest

from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationCoreResult,
    ClassificationRequest,
)
from mailagent.core.fusion_orchestrator import FusionOrchestrator
from mailagent.domain.models import (
    FusionMeta,
    MailEvent,
    PathBCandidate,
    PathBResult,
    RuleResult,
    RuleMatch,
    TaxonomyLabel,
)
from mailagent.infra.config import FusionSettings


# ---------------------------------------------------------------------------
# Stub classifier — records calls and returns a canned attempt
# ---------------------------------------------------------------------------


class StubClassifier:
    """A test double implementing the Classifier Protocol."""

    def __init__(self, source: str, attempt: ClassificationAttempt) -> None:
        self.source = source
        self._attempt = attempt
        self.calls = 0

    async def classify(self, request: ClassificationRequest) -> ClassificationAttempt:
        self.calls += 1
        return self._attempt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mail() -> MailEvent:
    return MailEvent(
        message_id="fusion-int-1",
        sender="ops@example.com",
        subject="STATUS update",
        body="Entity STATUS changed to 18:00",
    )


def _request() -> ClassificationRequest:
    return ClassificationRequest(mail=_mail())


def _label(code: str, confidence: float) -> TaxonomyLabel:
    return TaxonomyLabel(l1_code=code, l1_label=code, confidence=confidence)


def _success(
    source: str,
    code: str,
    confidence: float,
    evidence: dict[str, Any] | None = None,
) -> ClassificationAttempt:
    return ClassificationAttempt(
        source=source,
        status=AttemptStatus.SUCCESS,
        confidence=confidence,
        labels=[_label(code, confidence)],
        evidence=evidence or {},
    )


def _no_match(source: str) -> ClassificationAttempt:
    return ClassificationAttempt(
        source=source,
        status=AttemptStatus.NO_MATCH,
        confidence=0.0,
    )


def _unavailable(source: str) -> ClassificationAttempt:
    return ClassificationAttempt(
        source=source,
        status=AttemptStatus.UNAVAILABLE,
        confidence=0.0,
        error="classifier is unavailable",
    )


def _rule_success(code: str, confidence: float) -> ClassificationAttempt:
    return _success(
        "rules",
        code,
        confidence,
        evidence={"rule_type": "subject_patterns", "matched_pattern": "STATUS.*"},
    )


def _vector_success(code: str, confidence: float) -> ClassificationAttempt:
    path_b = PathBResult(
        candidates=[
            PathBCandidate(
                label=code,
                max_similarity=confidence,
                count=2,
                mean_similarity=confidence,
                confidence=confidence,
            )
        ],
        top1_label=code,
        top1_similarity=confidence,
        confidence=confidence,
        reason="ok",
    )
    return _success(
        "vector",
        code,
        confidence,
        evidence={"path_b_result": path_b.model_dump()},
    )


def _settings(**kwargs: Any) -> FusionSettings:
    defaults: dict[str, Any] = {
        "rule_confidence_threshold": 0.9,
        "vector_confidence_threshold": 0.85,
        "llm_fallback_threshold": 0.7,
    }
    defaults.update(kwargs)
    return FusionSettings(**defaults)


# ---------------------------------------------------------------------------
# Strategy: rule_only
# ---------------------------------------------------------------------------


class TestRuleOnlyIntegration:
    """rule_only strategy: rule confidence ≥ rule_confidence_threshold."""

    async def test_rule_only_strategy_selected(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.95))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.90))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.92))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert isinstance(result, ClassificationCoreResult)
        assert result.selected_source == "rules"
        assert result.labels[0].l1_code == "eta_update"
        # rule_only short-circuits; vector / llm not called.
        assert vector.calls == 0
        assert llm.calls == 0
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_only"
        assert meta.source == "rule"
        assert meta.confidence == pytest.approx(0.95)
        assert meta.vector_confirmed is False
        assert meta.rule_result is not None
        assert meta.vector_result is None
        assert meta.llm_result is None


# ---------------------------------------------------------------------------
# Strategy: rule_vector_confirmed
# ---------------------------------------------------------------------------


class TestRuleVectorConfirmedIntegration:
    """rule_vector_confirmed: rule in [0.7, 0.9) and vector confirms same label."""

    async def test_rule_vector_confirmed_strategy(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.80))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.88))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "rules"
        assert result.labels[0].l1_code == "eta_update"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.vector_confirmed is True
        assert meta.rule_result is not None
        assert isinstance(meta.rule_result, RuleResult)
        assert meta.rule_result.selected is not None
        assert isinstance(meta.rule_result.selected, RuleMatch)
        assert meta.vector_result is not None
        assert meta.vector_result.top1_label == "eta_update"
        assert meta.llm_result is None


# ---------------------------------------------------------------------------
# Strategy: vector_only (no conflicting medium-confidence rule)
# ---------------------------------------------------------------------------


class TestVectorOnlyIntegration:
    """vector_only: no eligible rule and vector ≥ 0.85."""

    @pytest.mark.parametrize(
        ("llm_attempt", "expected_source", "expected_label"),
        [
            (_success("llm", "eta_update", 0.90), "llm", "eta_update"),
            (_success("llm", "location_plan", 0.90), "llm", "location_plan"),
            (_success("llm", "arrival_notice", 0.90), "llm", "arrival_notice"),
            (_unavailable("llm"), "vector", "location_plan"),
        ],
        ids=("llm_agrees_with_rule", "llm_agrees_with_vector", "llm_third_label", "llm_unavailable"),
    )
    async def test_rule_vector_conflict_requires_review_even_when_vector_is_high(
        self,
        llm_attempt: ClassificationAttempt,
        expected_source: str,
        expected_label: str,
    ) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.80))
        vector = StubClassifier("vector", _vector_success("location_plan", 0.92))
        llm = StubClassifier("llm", llm_attempt)
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.meta.needs_human_review is True
        assert result.selected_source == expected_source
        assert result.labels[0].l1_code == expected_label
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "all_low_review"
        assert meta.vector_confirmed is False
        assert meta.conflict is not None
        assert meta.conflict.sources == ["rule", "vector"]
        assert meta.conflict.labels == ["eta_update", "location_plan"]

    async def test_no_rule_vector_high_confidence(self) -> None:
        vector = StubClassifier("vector", _vector_success("eta_update", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[vector], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "vector"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "vector_only"


# ---------------------------------------------------------------------------
# Strategy: llm_fallback
# ---------------------------------------------------------------------------


class TestLlmFallbackIntegration:
    """llm_fallback: rule + vector both low, LLM produces a SUCCESS attempt."""

    async def test_llm_fallback_strategy(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.50))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.60))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.75))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "llm"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "llm_fallback"
        assert meta.source == "llm"
        assert meta.llm_result is not None
        assert meta.llm_result["confidence"] == pytest.approx(0.75)
        # All three path results preserved in audit.
        assert meta.rule_result is not None
        assert meta.vector_result is not None


# ---------------------------------------------------------------------------
# Strategy: all_low_review
# ---------------------------------------------------------------------------


class TestAllLowReviewIntegration:
    """all_low_review: no path produces sufficient confidence → needs review."""

    async def test_all_low_review_triggers_human_review(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.50))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.50))
        llm = StubClassifier("llm", _no_match("llm"))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.meta.needs_human_review is True
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "all_low_review"


# ---------------------------------------------------------------------------
# FusionMeta completeness for each strategy
# ---------------------------------------------------------------------------


class TestFusionMetaCompletenessIntegration:
    """FusionMeta audit dict carries all expected fields for each strategy."""

    @pytest.mark.parametrize(
        "strategy,rule_conf,vector_conf,llm_conf,expected_source",
        [
            ("rule_only", 0.95, None, None, "rule"),
            ("rule_vector_confirmed", 0.80, 0.88, None, "rule"),
            ("vector_only", None, 0.90, None, "vector"),
            ("llm_fallback", 0.50, 0.60, 0.75, "llm"),
        ],
    )
    async def test_meta_fields_for_each_strategy(
        self,
        strategy: str,
        rule_conf: float | None,
        vector_conf: float | None,
        llm_conf: float | None,
        expected_source: str,
    ) -> None:
        classifiers: list[StubClassifier] = []
        if rule_conf is not None:
            classifiers.append(StubClassifier("rules", _rule_success("eta_update", rule_conf)))
        if vector_conf is not None:
            classifiers.append(StubClassifier("vector", _vector_success("eta_update", vector_conf)))
        if llm_conf is not None:
            classifiers.append(StubClassifier("llm", _success("llm", "eta_update", llm_conf)))

        orchestrator = FusionOrchestrator(
            classifiers=classifiers, settings=_settings()
        )
        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == strategy
        assert meta.source == expected_source
        assert 0.0 <= meta.confidence <= 1.0
        # vector_confirmed is True only for rule_vector_confirmed
        assert meta.vector_confirmed == (strategy == "rule_vector_confirmed")
