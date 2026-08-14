"""Unit tests for FusionOrchestrator target scoped retrieval (P0).

Validates that when a ``target_profile_loader`` is configured and the rule
candidate's label path matches a target profile, Step 2 sets
``request.context["vector_scope"]`` so the upcoming VectorClassifier call
runs a scoped knn_search, and that ``FusionMeta.target_profile`` is recorded
for audit. Also verifies backward compatibility when no loader is configured.
"""
from __future__ import annotations

from typing import Any

from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationRequest,
)
from mailagent.core.fusion_orchestrator import FusionOrchestrator
from mailagent.core.target_profile import TargetProfileLoader
from mailagent.domain.models import (
    FusionMeta,
    MailEvent,
    PathBCandidate,
    PathBResult,
    TaxonomyLabel,
)
from mailagent.infra.config import FusionSettings
from mailagent.classification.taxonomy import TaxonomyLoader


# ---------------------------------------------------------------------------
# Stubs & builders
# ---------------------------------------------------------------------------


class StubClassifier:
    """A test double implementing the Classifier Protocol.

    Captures the request so tests can inspect ``context["vector_scope"]``.
    """

    def __init__(self, source: str, attempt: ClassificationAttempt) -> None:
        self.source = source
        self._attempt = attempt
        self.calls = 0
        self.last_request: ClassificationRequest | None = None

    async def classify(self, request: ClassificationRequest) -> ClassificationAttempt:
        self.calls += 1
        self.last_request = request
        return self._attempt


def _mail() -> MailEvent:
    return MailEvent(
        message_id="target-1",
        sender="ops@example.com",
        subject="Status Report",
        body="Entity status report",
    )


def _request() -> ClassificationRequest:
    return ClassificationRequest(mail=_mail())


def _full_label(
    l1: str, l2: str, l3: str, confidence: float, l1_label: str = "", l2_label: str = "", l3_label: str = ""
) -> TaxonomyLabel:
    return TaxonomyLabel(
        l1_code=l1,
        l1_label=l1_label or l1,
        l2_code=l2,
        l2_label=l2_label or l2,
        l3_code=l3,
        l3_label=l3_label or l3,
        confidence=confidence,
    )


def _leaf_only_label(code: str, confidence: float) -> TaxonomyLabel:
    return TaxonomyLabel(
        l1_code=code,
        l1_label=code,
        confidence=confidence,
    )


def _rule_success(label: TaxonomyLabel, confidence: float) -> ClassificationAttempt:
    return ClassificationAttempt(
        source="rules",
        status=AttemptStatus.SUCCESS,
        confidence=confidence,
        labels=[label],
        evidence={"rule_type": "subject_patterns", "matched_pattern": "Status Report"},
    )


def _vector_success(label_code: str, confidence: float) -> ClassificationAttempt:
    path_b = PathBResult(
        candidates=[
            PathBCandidate(
                label=label_code,
                max_similarity=confidence,
                count=2,
                mean_similarity=confidence,
                confidence=confidence,
            )
        ],
        top1_label=label_code,
        top1_similarity=confidence,
        confidence=confidence,
        reason="ok",
    )
    return ClassificationAttempt(
        source="vector",
        status=AttemptStatus.SUCCESS,
        confidence=confidence,
        labels=[TaxonomyLabel(l1_code=label_code, l1_label=label_code, confidence=confidence)],
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


def _taxonomy_file(tmp_path) -> Any:
    """Write a small flat taxonomy used by target-scope tests."""

    path = tmp_path / "taxonomy.yaml"
    path.write_text(
        """
nodes:
  - code: entity_report
    label: 实体报告
    description: 报告
    keywords: [status report]
  - code: schedule
    label: 船期
    description: 船期
    keywords: [eta]
  - code: internal_coordination
    label: 内部协调
    description: 协调
    keywords: [coordination]
""",
        encoding="utf-8",
    )
    return path


def _target_profiles_file(tmp_path, targets: list[dict]) -> Any:
    import yaml

    path = tmp_path / "target_profiles.yaml"
    path.write_text(yaml.safe_dump({"targets": targets}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Target scoped retrieval tests
# ---------------------------------------------------------------------------


class TestTargetScopedRetrieval:
    """When the rule candidate matches a target profile, Step 2 sets
    ``context["vector_scope"]`` and records ``target_profile`` in FusionMeta."""

    async def test_bound_target_snapshot_is_used_after_live_file_reload(
        self, tmp_path
    ) -> None:
        taxonomy_loader = TaxonomyLoader(_taxonomy_file(tmp_path), poll_interval=0)
        _target_profiles_file(
            tmp_path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(
            tmp_path / "target_profiles.yaml", taxonomy_loader, poll_interval=0
        )
        bound = loader.get_snapshot()
        _target_profiles_file(
            tmp_path,
            [{"label": "entity_report", "vector_scope": ["schedule"]}],
        )
        assert loader.get_snapshot().version != bound.version
        rule = StubClassifier(
            "rules", _rule_success(_leaf_only_label("entity_report", 0.75), 0.75)
        )
        vector = StubClassifier("vector", _vector_success("entity_report", 0.9))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector],
            settings=_settings(),
            target_profile_loader=loader,
        )
        request = _request().model_copy(
            update={"asset_snapshots": {"target_profiles": bound}}
        )

        await orchestrator.classify(request)

        assert vector.last_request is not None
        assert vector.last_request.context["vector_scope"] == ["entity_report"]

    async def test_rule_matches_target_sets_vector_scope(
        self, tmp_path
    ) -> None:
        taxonomy_loader = TaxonomyLoader(_taxonomy_file(tmp_path), poll_interval=0.05)
        _target_profiles_file(
            tmp_path,
            [
                {
                    "label": "entity_report",
                    "vector_scope": ["entity_report", "schedule"],
                }
            ],
        )
        loader = TargetProfileLoader(
            tmp_path / "target_profiles.yaml", taxonomy_loader
        )

        # Rule matches entity_report with confidence 0.75 (medium-confidence band).
        # RuleClassifier populates only l1_code with the leaf code.
        rule_label = _leaf_only_label("entity_report", 0.75)
        rule = StubClassifier("rules", _rule_success(rule_label, 0.75))
        vector = StubClassifier("vector", _vector_success("entity_report", 0.9))
        llm = StubClassifier("llm", _vector_success("operation", 0.8))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=loader,
        )

        result = await orchestrator.classify(_request())

        # Vector was called in Step 2
        assert vector.calls == 1
        # context["vector_scope"] was set before the vector call
        assert vector.last_request is not None
        assert vector.last_request.context.get("vector_scope") == ["entity_report", "schedule"]
        # rule_vector_confirmed because vector agrees on label
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.target_profile == "entity_report"
        assert meta.vector_confirmed is True

    async def test_scoped_vector_confirms_same_label_rule_vector_confirmed(
        self, tmp_path
    ) -> None:
        """Rule matches target + confidence 0.75 → scoped vector confirms →
        fusion_strategy=rule_vector_confirmed, target_profile=label."""

        taxonomy_loader = TaxonomyLoader(_taxonomy_file(tmp_path), poll_interval=0.05)
        _target_profiles_file(
            tmp_path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(
            tmp_path / "target_profiles.yaml", taxonomy_loader
        )

        rule_label = _leaf_only_label("entity_report", 0.75)
        rule = StubClassifier("rules", _rule_success(rule_label, 0.75))
        vector = StubClassifier("vector", _vector_success("entity_report", 0.92))
        llm = StubClassifier("llm", _vector_success("x", 0.8))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=loader,
        )

        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.target_profile == "entity_report"

    async def test_scoped_vector_does_not_confirm_falls_through(
        self, tmp_path
    ) -> None:
        """Rule matches target label + confidence 0.75 → scoped vector does NOT
        confirm (different top1) → require review;
        ``target_profile`` records the target label (audit: scoped attempted)."""

        taxonomy_loader = TaxonomyLoader(_taxonomy_file(tmp_path), poll_interval=0.05)
        _target_profiles_file(
            tmp_path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(
            tmp_path / "target_profiles.yaml", taxonomy_loader
        )

        rule_label = _leaf_only_label("entity_report", 0.75)
        rule = StubClassifier("rules", _rule_success(rule_label, 0.75))
        # Vector disagrees (returns operation) and requires review.
        vector = StubClassifier("vector", _vector_success("operation", 0.6))
        llm = StubClassifier("llm", _vector_success("operation", 0.8))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=loader,
        )

        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert result.meta.needs_human_review is True
        assert meta.fusion_strategy == "all_low_review"
        assert meta.conflict is not None
        assert meta.conflict.labels == ["entity_report", "operation"]
        # target_profile still recorded (scoped was attempted)
        assert meta.target_profile == "entity_report"
        # vector_scope WAS set on the request even though it didn't confirm
        assert vector.last_request is not None
        assert vector.last_request.context.get("vector_scope") == ["entity_report"]

    async def test_rule_does_not_match_target_no_scope_set(
        self, tmp_path
    ) -> None:
        """Rule matches a non-target label → no vector_scope set → global vector."""

        taxonomy_loader = TaxonomyLoader(_taxonomy_file(tmp_path), poll_interval=0.05)
        _target_profiles_file(
            tmp_path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(
            tmp_path / "target_profiles.yaml", taxonomy_loader
        )

        # Rule matches internal_coordination — not a target leaf code
        rule_label = _leaf_only_label("internal_coordination", 0.75)
        rule = StubClassifier("rules", _rule_success(rule_label, 0.75))
        vector = StubClassifier(
            "vector",
            _vector_success("internal_coordination", 0.9),
        )
        llm = StubClassifier("llm", _vector_success("x", 0.8))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=loader,
        )

        result = await orchestrator.classify(_request())

        # No vector_scope was set
        assert vector.last_request is not None
        assert "vector_scope" not in vector.last_request.context
        meta = FusionMeta.model_validate(result.audit)
        assert meta.target_profile is None
        assert meta.fusion_strategy == "rule_vector_confirmed"

    async def test_rule_confidence_above_threshold_no_target_branch(
        self, tmp_path
    ) -> None:
        """Rule confidence ≥ 0.9 → rule_only, target branch not triggered."""

        taxonomy_loader = TaxonomyLoader(_taxonomy_file(tmp_path), poll_interval=0.05)
        _target_profiles_file(
            tmp_path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(
            tmp_path / "target_profiles.yaml", taxonomy_loader
        )

        rule_label = _leaf_only_label("entity_report", 0.92)
        rule = StubClassifier("rules", _rule_success(rule_label, 0.92))
        vector = StubClassifier("vector", _vector_success("entity_report", 0.9))
        llm = StubClassifier("llm", _vector_success("x", 0.8))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=loader,
        )

        result = await orchestrator.classify(_request())

        # rule_only short-circuits — vector never called
        assert vector.calls == 0
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_only"
        assert meta.target_profile is None

    async def test_rule_confidence_below_fallback_threshold_no_target_branch(
        self, tmp_path
    ) -> None:
        """Rule confidence < 0.7 → target branch skipped, falls to fallback."""

        taxonomy_loader = TaxonomyLoader(_taxonomy_file(tmp_path), poll_interval=0.05)
        _target_profiles_file(
            tmp_path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(
            tmp_path / "target_profiles.yaml", taxonomy_loader
        )

        rule_label = _leaf_only_label("entity_report", 0.65)
        rule = StubClassifier("rules", _rule_success(rule_label, 0.65))
        vector = StubClassifier("vector", _vector_success("entity_report", 0.9))
        llm = StubClassifier("llm", _vector_success("entity_report", 0.8))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=loader,
        )

        result = await orchestrator.classify(_request())

        # Rule below 0.7 → Step 2 not entered → vector_scope NOT set
        # (Step 3 calls vector, but request.context has no vector_scope)
        assert vector.last_request is not None
        assert "vector_scope" not in vector.last_request.context
        meta = FusionMeta.model_validate(result.audit)
        # vector_only since rule too low and vector ≥ 0.85
        assert meta.fusion_strategy == "vector_only"
        # target_profile None because target branch never entered
        assert meta.target_profile is None

    async def test_no_loader_backward_compatible(self) -> None:
        """target_profile_loader=None → target branch never triggered."""

        rule_label = _leaf_only_label("entity_report", 0.75)
        rule = StubClassifier("rules", _rule_success(rule_label, 0.75))
        vector = StubClassifier("vector", _vector_success("entity_report", 0.9))
        llm = StubClassifier("llm", _vector_success("x", 0.8))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            # no target_profile_loader
        )

        result = await orchestrator.classify(_request())

        # No vector_scope set
        assert vector.last_request is not None
        assert "vector_scope" not in vector.last_request.context
        meta = FusionMeta.model_validate(result.audit)
        assert meta.target_profile is None
        assert meta.fusion_strategy == "rule_vector_confirmed"
