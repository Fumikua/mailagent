"""Unit tests for FusionOrchestrator: three-path fusion cascade.

Covers the five fusion strategies and their boundary conditions:
    - ``rule_only`` — rule confidence ≥ rule_confidence_threshold (0.9).
    - ``rule_vector_confirmed`` — rule in [0.7, 0.9) and vector confirms.
    - rule/vector conflict is fail-closed and requires human review.
    - ``vector_only`` — no rule, or rule too low, vector ≥ 0.85.
    - ``llm_fallback`` — all prior paths fail, LLM produces a result.
    - ``all_low_review`` — no path succeeds; sets needs_human_review=True.
    - FusionMeta field completeness in audit.
    - Custom threshold configuration.
    - Missing classifier graceful degradation (e.g., no rules classifier).
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
    CalibrationLog,
    FusionMeta,
    MailEvent,
    PathBCandidate,
    PathBResult,
    RuleMatch,
    RuleResult,
    TaxonomyLabel,
)
from mailagent.infra.config import FusionSettings


# ---------------------------------------------------------------------------
# Stub classifier — records classify() calls, returns a canned attempt.
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
# Attempt builders
# ---------------------------------------------------------------------------


def _mail() -> MailEvent:
    return MailEvent(
        message_id="fusion-1",
        sender="ops@example.com",
        subject="STATUS update",
        body="Entity STATUS changed to 18:00",
    )


def _request() -> ClassificationRequest:
    return ClassificationRequest(mail=_mail())


def _label(code: str, confidence: float) -> TaxonomyLabel:
    return TaxonomyLabel(
        l1_code=code,
        l1_label=code,
        confidence=confidence,
    )


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


def _rule_success(code: str, confidence: float, pattern: str = "STATUS.*") -> ClassificationAttempt:
    """Rule attempt with evidence keys rule_classifier writes."""
    return _success(
        "rules",
        code,
        confidence,
        evidence={"rule_type": "subject_patterns", "matched_pattern": pattern},
    )


def _vector_success(code: str, confidence: float) -> ClassificationAttempt:
    """Vector attempt with a PathBResult in evidence."""
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


def _vector_no_match(reason: str) -> ClassificationAttempt:
    return ClassificationAttempt(
        source="vector",
        status=AttemptStatus.NO_MATCH,
        confidence=0.0,
        evidence={
            "path_b_result": PathBResult(reason=reason).model_dump(),
        },
    )


def _settings(**kwargs: Any) -> FusionSettings:
    """FusionSettings with default thresholds unless overridden."""
    defaults: dict[str, Any] = {
        "rule_confidence_threshold": 0.9,
        "vector_confidence_threshold": 0.85,
        "llm_fallback_threshold": 0.7,
    }
    defaults.update(kwargs)
    return FusionSettings(**defaults)


async def test_fusion_preserves_selected_attempt_calibration_log() -> None:
    """The final result retains the calibration audit of Fusion's selected attempt."""
    log = CalibrationLog(raw=0.82, calibrated=0.87, anchor="fairly certain")
    llm_attempt = _success("llm", "schedule", 0.82).model_copy(
        update={"calibration_log": log}
    )
    orchestrator = FusionOrchestrator(
        classifiers=[
            StubClassifier("rules", _no_match("rules")),
            StubClassifier("vector", _no_match("vector")),
            StubClassifier("llm", llm_attempt),
        ],
        settings=_settings(),
    )

    result = await orchestrator.classify(_request())

    assert result.calibration_log == log


# ---------------------------------------------------------------------------
# Strategy: rule_only
# ---------------------------------------------------------------------------


class TestRuleOnly:
    """rule_only: rule confidence ≥ rule_confidence_threshold → use rule."""

    async def test_rule_high_confidence_uses_rule_only(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.95))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.95))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.92))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "rules"
        assert result.labels[0].l1_code == "eta_update"
        assert result.meta.needs_human_review is False
        assert rule.calls == 1
        # rule_only short-circuits → vector and llm never called
        assert vector.calls == 0
        assert llm.calls == 0
        # audit carries FusionMeta
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_only"
        assert meta.source == "rule"
        assert meta.vector_confirmed is False

    async def test_rule_at_exact_threshold_uses_rule_only(self) -> None:
        """Boundary: confidence == rule_confidence_threshold (0.9)."""
        rule = StubClassifier("rules", _rule_success("eta_update", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "rules"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_only"


# ---------------------------------------------------------------------------
# Strategy: rule_vector_confirmed
# ---------------------------------------------------------------------------


class TestRuleVectorConfirmed:
    """rule_vector_confirmed: rule in [0.7, 0.9) + vector confirms same label."""

    async def test_rule_medium_confidence_vector_confirms(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.80))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.88))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        # rule_vector_confirmed uses rule's label
        assert result.selected_source == "rules"
        assert result.labels[0].l1_code == "eta_update"
        # rule was called, vector was called, llm was NOT called
        assert rule.calls == 1
        assert vector.calls == 1
        assert llm.calls == 0
        # audit: vector_confirmed=True
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.source == "rule"
        assert meta.vector_confirmed is True
        assert meta.rule_result is not None
        assert meta.rule_result.selected is not None
        assert meta.rule_result.selected.label == "eta_update"
        assert meta.vector_result is not None
        assert meta.vector_result.top1_label == "eta_update"

    async def test_rule_at_llm_fallback_threshold_triggers_confirmation(self) -> None:
        """Boundary: rule confidence == llm_fallback_threshold (0.7)."""
        rule = StubClassifier("rules", _rule_success("eta_update", 0.70))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.88))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "rules"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"

    async def test_same_label_vector_below_confirmation_threshold_falls_back_to_llm(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.80))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.80))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "llm"
        assert FusionMeta.model_validate(result.audit).fusion_strategy == "llm_fallback"


# ---------------------------------------------------------------------------
# Rule/vector conflict → fail closed with a reviewer suggestion
# ---------------------------------------------------------------------------


class TestRuleVectorConflict:
    """Rule and vector disagreement always requires human review."""

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
        """A disagreement cannot bypass review, even with high vector confidence."""
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
        assert rule.calls == 1
        assert vector.calls == 1
        assert llm.calls == 1
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "all_low_review"
        assert meta.vector_confirmed is False
        assert meta.conflict is not None
        assert meta.conflict.sources == ["rule", "vector"]
        assert meta.conflict.labels == ["eta_update", "location_plan"]


# ---------------------------------------------------------------------------
# Strategy: vector_only
# ---------------------------------------------------------------------------


class TestVectorOnly:
    """vector_only: no rule (or rule too low) + vector ≥ vector_confidence_threshold."""

    async def test_no_rule_vector_high_confidence_uses_vector_only(self) -> None:
        vector = StubClassifier("vector", _vector_success("eta_update", 0.90))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.92))
        orchestrator = FusionOrchestrator(
            classifiers=[vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "vector"
        assert result.labels[0].l1_code == "eta_update"
        assert vector.calls == 1
        assert llm.calls == 0
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "vector_only"
        assert meta.source == "vector"

    async def test_rule_no_match_vector_high_confidence_uses_vector_only(self) -> None:
        rule = StubClassifier("rules", _no_match("rules"))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.88))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "vector"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "vector_only"

    async def test_rule_low_confidence_vector_high_uses_vector_only(self) -> None:
        """Rule confidence below llm_fallback_threshold skips confirmation."""
        rule = StubClassifier("rules", _rule_success("eta_update", 0.55))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "vector"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "vector_only"

    async def test_vector_at_exact_threshold_uses_vector_only(self) -> None:
        """Boundary: vector confidence == vector_confidence_threshold (0.85)."""
        vector = StubClassifier("vector", _vector_success("eta_update", 0.85))
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


class TestLlmFallback:
    """llm_fallback: rule + vector both low → LLM produces a result."""

    async def test_all_paths_low_llm_succeeds(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.50))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.60))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.75))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "llm"
        assert result.labels[0].l1_code == "eta_update"
        assert rule.calls == 1
        assert vector.calls == 1
        assert llm.calls == 1
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "llm_fallback"
        assert meta.source == "llm"
        assert meta.llm_result is not None
        assert meta.llm_result["confidence"] == pytest.approx(0.75)

    async def test_no_rule_no_vector_llm_succeeds(self) -> None:
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.80))
        orchestrator = FusionOrchestrator(
            classifiers=[llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "llm"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "llm_fallback"


# ---------------------------------------------------------------------------
# Strategy: all_low_review
# ---------------------------------------------------------------------------


class TestAllLowReview:
    """all_low_review: no path produces sufficient confidence → needs review."""

    async def test_all_paths_low_confidence_triggers_review(self) -> None:
        """When LLM also fails (NO_MATCH), all_low_review picks the best attempt."""
        rule = StubClassifier("rules", _rule_success("eta_update", 0.50))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.50))
        llm = StubClassifier("llm", _no_match("llm"))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        # all_low_review picks the best attempt (rule with 0.50, first tie)
        assert result.selected_source == "rules"
        assert result.meta.needs_human_review is True
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "all_low_review"

    async def test_llm_low_confidence_success_requires_review(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.50))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.50))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.40))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "llm"
        assert result.meta.needs_human_review is True
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "all_low_review"

    async def test_llm_no_match_with_low_rule_and_vector_triggers_review(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.50))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.60))
        llm = StubClassifier("llm", _no_match("llm"))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        # best attempt is vector with 0.60
        assert result.selected_source == "vector"
        assert result.meta.needs_human_review is True
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "all_low_review"

    async def test_all_attempts_no_match_returns_empty_result_with_review(self) -> None:
        rule = StubClassifier("rules", _no_match("rules"))
        vector = StubClassifier("vector", _no_match("vector"))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source is None
        assert result.labels == []
        assert result.meta.needs_human_review is True
        assert result.meta.fallback is True
        assert result.audit["fusion_strategy"] == "all_low_review"


# ---------------------------------------------------------------------------
# FusionMeta field completeness
# ---------------------------------------------------------------------------


class TestFusionMetaCompleteness:
    """FusionMeta audit dict carries all expected fields."""

    async def test_rule_vector_confirmed_audit_includes_all_paths(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.80))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.88))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.source == "rule"
        assert meta.confidence == pytest.approx(0.80)
        assert meta.rule_result is not None
        assert isinstance(meta.rule_result, RuleResult)
        assert meta.rule_result.selected is not None
        assert isinstance(meta.rule_result.selected, RuleMatch)
        assert meta.rule_result.selected.rule_type == "subject_patterns"
        assert meta.rule_result.selected.matched_pattern == "STATUS.*"
        assert meta.vector_result is not None
        assert isinstance(meta.vector_result, PathBResult)
        assert meta.vector_result.top1_label == "eta_update"
        assert meta.llm_result is None  # LLM never called
        assert meta.vector_confirmed is True

    async def test_llm_fallback_audit_includes_rule_and_vector(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.50))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.60))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.75))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "llm_fallback"
        assert meta.source == "llm"
        # all three path results are preserved in audit
        assert meta.rule_result is not None
        assert meta.vector_result is not None
        assert meta.llm_result is not None
        assert meta.vector_confirmed is False


class TestVectorEvidenceFallback:
    async def test_ambiguous_vector_result_preserves_llm_fallback(self) -> None:
        rule = StubClassifier("rules", _rule_success("eta_update", 0.80))
        vector = StubClassifier("vector", _vector_no_match("ambiguous_candidates"))
        llm = StubClassifier("llm", _success("llm", "schedule", 0.85))

        result = await FusionOrchestrator(
            classifiers=[rule, vector, llm], settings=_settings()
        ).classify(_request())

        assert result.selected_source == "llm"
        assert result.labels[0].l1_code == "schedule"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "llm_fallback"
        assert meta.vector_result is not None
        assert meta.vector_result.reason == "ambiguous_candidates"

    async def test_stale_vector_category_preserves_human_review_when_llm_fails(self) -> None:
        vector = StubClassifier("vector", _vector_no_match("stale_category"))
        llm = StubClassifier("llm", _no_match("llm"))

        result = await FusionOrchestrator(
            classifiers=[vector, llm], settings=_settings()
        ).classify(_request())

        assert result.meta.needs_human_review is True
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "all_low_review"
        assert meta.vector_result is not None
        assert meta.vector_result.reason == "stale_category"


# ---------------------------------------------------------------------------
# Custom threshold configuration
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    """Custom FusionSettings thresholds affect strategy boundaries."""

    async def test_lowered_rule_threshold_promotes_medium_rule_to_rule_only(self) -> None:
        """rule_confidence_threshold=0.75 → a 0.80 rule becomes rule_only."""
        rule = StubClassifier("rules", _rule_success("eta_update", 0.80))
        orchestrator = FusionOrchestrator(
            classifiers=[rule],
            settings=_settings(rule_confidence_threshold=0.75),
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "rules"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_only"

    async def test_lowered_vector_threshold_promotes_low_vector_to_vector_only(self) -> None:
        """vector_confidence_threshold=0.70 → a 0.75 vector becomes vector_only."""
        vector = StubClassifier("vector", _vector_success("eta_update", 0.75))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.80))
        orchestrator = FusionOrchestrator(
            classifiers=[vector, llm],
            settings=_settings(vector_confidence_threshold=0.70),
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "vector"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "vector_only"

    async def test_raised_llm_fallback_threshold_skips_vector_confirmation(self) -> None:
        """llm_fallback_threshold=0.85 → a 0.80 rule no longer triggers
        vector confirmation; falls through to vector_only if vector is high."""
        rule = StubClassifier("rules", _rule_success("eta_update", 0.80))
        vector = StubClassifier("vector", _vector_success("eta_update", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector],
            settings=_settings(llm_fallback_threshold=0.85),
        )

        result = await orchestrator.classify(_request())

        # rule 0.80 < 0.85 threshold → no confirmation → vector_only
        assert result.selected_source == "vector"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "vector_only"


# ---------------------------------------------------------------------------
# Missing classifier graceful degradation
# ---------------------------------------------------------------------------


class TestMissingClassifierDegradation:
    """Missing classifiers are silently skipped — orchestrator degrades
    gracefully to whatever paths are available."""

    async def test_no_rules_classifier_vector_only_path(self) -> None:
        vector = StubClassifier("vector", _vector_success("eta_update", 0.90))
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.80))
        orchestrator = FusionOrchestrator(
            classifiers=[vector, llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "vector"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "vector_only"
        # rule_result is None because rule classifier was never registered
        assert meta.rule_result is None

    async def test_only_llm_classifier(self) -> None:
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.85))
        orchestrator = FusionOrchestrator(
            classifiers=[llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source == "llm"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "llm_fallback"

    async def test_only_llm_records_unavailable_rules_and_vector_attempts(self) -> None:
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.85))
        result = await FusionOrchestrator(
            classifiers=[llm], settings=_settings()
        ).classify(_request())

        assert [(attempt.source, attempt.status) for attempt in result.attempts] == [
            ("rules", AttemptStatus.UNAVAILABLE),
            ("vector", AttemptStatus.UNAVAILABLE),
            ("llm", AttemptStatus.SUCCESS),
        ]

    async def test_only_vector_classifier_low_confidence_review(self) -> None:
        vector = StubClassifier("vector", _vector_success("eta_update", 0.50))
        orchestrator = FusionOrchestrator(
            classifiers=[vector], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        # vector low + no LLM → all_low_review
        assert result.meta.needs_human_review is True
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "all_low_review"

    async def test_no_classifiers_returns_empty_review(self) -> None:
        orchestrator = FusionOrchestrator(
            classifiers=[], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert result.selected_source is None
        assert result.labels == []
        assert result.meta.needs_human_review is True
        assert result.audit["fusion_strategy"] == "all_low_review"


# ---------------------------------------------------------------------------
# ClassificationOrchestrator Protocol contract
# ---------------------------------------------------------------------------


class TestOrchestratorProtocolContract:
    """FusionOrchestrator structurally satisfies ClassificationOrchestrator."""

    def test_protocol_structural_contract(self) -> None:
        """Structural subtyping: FusionOrchestrator exposes ``classify``.

        Note: the ClassificationOrchestrator Protocol is not
        ``@runtime_checkable``, so we verify the structural contract via
        attribute presence rather than isinstance.
        """
        orchestrator = FusionOrchestrator(
            classifiers=[], settings=_settings()
        )
        assert hasattr(orchestrator, "classify")
        assert callable(orchestrator.classify)

    async def test_classify_returns_classification_core_result(self) -> None:
        llm = StubClassifier("llm", _success("llm", "eta_update", 0.85))
        orchestrator = FusionOrchestrator(
            classifiers=[llm], settings=_settings()
        )

        result = await orchestrator.classify(_request())

        assert isinstance(result, ClassificationCoreResult)

    def test_classifiers_indexed_by_source(self) -> None:
        """Duplicate sources overwrite; the dict is keyed by source string."""
        rule1 = StubClassifier("rules", _rule_success("a", 0.95))
        rule2 = StubClassifier("rules", _rule_success("b", 0.95))
        vector = StubClassifier("vector", _vector_success("c", 0.95))
        orchestrator = FusionOrchestrator(
            classifiers=[rule1, rule2, vector], settings=_settings()
        )

        assert orchestrator._classifiers["rules"] is rule2  # last one wins
        assert orchestrator._classifiers["vector"] is vector
