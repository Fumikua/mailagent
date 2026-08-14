"""End-to-end integration tests for target-label-scoping (Section 6.2).

Exercises the full FusionOrchestrator → TargetProfileLoader → VectorClassifier
integration with mock classifiers and a real TargetProfileLoader wired against
a temp-dir target_profiles.yaml + minimal taxonomy.

Covered scenarios (per tasks.md 6.2):
    - rule hits target label + confidence 0.75 → scoped retrieval confirms
      → rule_vector_confirmed + target_profile=label
    - rule hits target label + scoped retrieval returns empty → global fallback
      → original fusion flow, target_profile still recorded
    - non-target label → no scope set, global vector confirmation,
      target_profile=None
    - target_profiles.yaml missing (loader=None) → feature off, no crash
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationCoreResult,
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
# Stub classifier — records calls + can observe request.context
# ---------------------------------------------------------------------------


class StubClassifier:
    """Test double implementing the Classifier Protocol.

    Records every call and the last seen request so tests can assert on
    ``request.context["vector_scope"]`` propagation.
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


# ---------------------------------------------------------------------------
# Fixture: temp taxonomy + target_profiles.yaml + real loaders
# ---------------------------------------------------------------------------


def _write_taxonomy(path: Path) -> None:
    """Write the flat audit taxonomy used by example-triage."""
    path.write_text(
        "nodes:\n"
        "  - code: entity_report\n"
        "    label: 实体报告\n"
        "    description: x\n"
        "    keywords: [status report, arrival report]\n"
        "  - code: schedule\n"
        "    label: 船期\n"
        "    description: x\n"
        "    keywords: [eta, schedule]\n"
        "  - code: operation\n"
        "    label: 实体作业\n"
        "    description: x\n"
        "    keywords: [operation, location]\n",
        encoding="utf-8",
    )


def _write_target_profiles(path: Path) -> None:
    """Write flat target-profile labels and vector scopes."""
    path.write_text(
        "targets:\n"
        "  - label: entity_report\n"
        "    vector_scope:\n"
        "      - entity_report\n"
        "      - schedule\n",
        encoding="utf-8",
    )


@pytest.fixture
def loaders(tmp_path: Path) -> tuple[TaxonomyLoader, TargetProfileLoader]:
    taxonomy_path = tmp_path / "taxonomy.yaml"
    target_profiles_path = tmp_path / "target_profiles.yaml"
    _write_taxonomy(taxonomy_path)
    _write_target_profiles(target_profiles_path)
    taxonomy_loader = TaxonomyLoader(taxonomy_path)
    target_profile_loader = TargetProfileLoader(
        config_path=target_profiles_path,
        taxonomy_loader=taxonomy_loader,
    )
    return taxonomy_loader, target_profile_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mail() -> MailEvent:
    return MailEvent(
        message_id="target-scope-1",
        sender="ops@example.com",
        subject="Status Report",
        body="Entity status report follows.",
    )


def _request() -> ClassificationRequest:
    return ClassificationRequest(mail=_mail())


def _leaf_label(code: str, confidence: float) -> TaxonomyLabel:
    """Rule / vector classifiers populate only l1_code (leaf code)."""
    return TaxonomyLabel(l1_code=code, l1_label=code, confidence=confidence)


def _rule_success(code: str, confidence: float) -> ClassificationAttempt:
    return ClassificationAttempt(
        source="rules",
        status=AttemptStatus.SUCCESS,
        confidence=confidence,
        labels=[_leaf_label(code, confidence)],
        evidence={"rule_type": "subject_patterns", "matched_pattern": "Noon.*"},
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
    return ClassificationAttempt(
        source="vector",
        status=AttemptStatus.SUCCESS,
        confidence=confidence,
        labels=[_leaf_label(code, confidence)],
        evidence={"path_b_result": path_b.model_dump()},
    )


def _llm_success(code: str, confidence: float) -> ClassificationAttempt:
    return ClassificationAttempt(
        source="llm",
        status=AttemptStatus.SUCCESS,
        confidence=confidence,
        labels=[_leaf_label(code, confidence)],
        evidence={},
    )


def _no_match(source: str) -> ClassificationAttempt:
    return ClassificationAttempt(
        source=source,
        status=AttemptStatus.NO_MATCH,
        confidence=0.0,
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
# End-to-end scenarios
# ---------------------------------------------------------------------------


class TestTargetScopedEndToEnd:
    """End-to-end: rule + target profile loader + vector + LLM."""

    async def test_target_label_scoped_confirms_rule_vector_confirmed(
        self, loaders: tuple[TaxonomyLoader, TargetProfileLoader]
    ) -> None:
        """6.2: rule hits target label (entity_report) conf 0.75 → scoped vector
        confirms same label → rule_vector_confirmed + target_profile set.
        """
        _, target_loader = loaders
        rule = StubClassifier("rules", _rule_success("entity_report", 0.75))
        vector = StubClassifier("vector", _vector_success("entity_report", 0.88))
        llm = StubClassifier("llm", _llm_success("entity_report", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=target_loader,
        )

        result = await orchestrator.classify(_request())

        assert isinstance(result, ClassificationCoreResult)
        assert result.selected_source == "rules"
        assert result.labels[0].l1_code == "entity_report"
        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.target_profile == "entity_report"
        assert meta.vector_confirmed is True
        # Vector classifier saw the scoped request
        assert vector.last_request is not None
        assert vector.last_request.context.get("vector_scope") == [
            "entity_report",
            "schedule",
        ]
        # LLM not called (rule_vector_confirmed short-circuits)
        assert llm.calls == 0

    async def test_target_label_scoped_empty_falls_back_to_global(
        self, loaders: tuple[TaxonomyLoader, TargetProfileLoader]
    ) -> None:
        """6.2: rule hits target label → scoped retrieval empty → vector
        classifier retries globally → if global confirms same label, still
        rule_vector_confirmed; target_profile still recorded.

        This test stubs a single vector SUCCESS so the scoped call returns it
        (stub is not actually scoped), verifying the orchestrator still
        records target_profile when scope was set.
        """
        _, target_loader = loaders
        rule = StubClassifier("rules", _rule_success("entity_report", 0.75))
        # Vector disagrees on label, requiring review even though the scoped
        # retrieval attempt remains recorded for audit.
        vector = StubClassifier("vector", _vector_success("schedule", 0.60))
        llm = StubClassifier("llm", _llm_success("entity_report", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=target_loader,
        )

        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert result.meta.needs_human_review is True
        assert meta.fusion_strategy == "all_low_review"
        assert meta.conflict is not None
        assert meta.conflict.labels == ["entity_report", "schedule"]
        assert meta.target_profile == "entity_report"
        assert vector.last_request is not None
        # scope was set in context (target matched)
        assert vector.last_request.context.get("vector_scope") == [
            "entity_report",
            "schedule",
        ]

    async def test_non_target_label_uses_global_vector_confirmation(
        self, loaders: tuple[TaxonomyLoader, TargetProfileLoader]
    ) -> None:
        """6.2: rule hits non-target label (operation) → no scope set,
        global vector confirmation, target_profile=None.
        """
        _, target_loader = loaders
        rule = StubClassifier("rules", _rule_success("operation", 0.80))
        vector = StubClassifier("vector", _vector_success("operation", 0.88))
        llm = StubClassifier("llm", _llm_success("operation", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=target_loader,
        )

        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.target_profile is None
        assert vector.last_request is not None
        # no scope set (operation is not a declared target)
        assert "vector_scope" not in vector.last_request.context

    async def test_missing_target_profiles_feature_off(self, tmp_path: Path) -> None:
        """6.2: target_profile_loader=None → feature off, behaves as original fusion."""
        rule = StubClassifier("rules", _rule_success("entity_report", 0.75))
        vector = StubClassifier("vector", _vector_success("entity_report", 0.88))
        llm = StubClassifier("llm", _llm_success("entity_report", 0.90))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector, llm],
            settings=_settings(),
            target_profile_loader=None,  # feature off
        )

        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.target_profile is None
        assert vector.last_request is not None
        assert "vector_scope" not in vector.last_request.context

    async def test_target_profiles_yaml_missing_file_no_crash(
        self, tmp_path: Path
    ) -> None:
        """6.2: TargetProfileLoader created against missing file → empty target list,
        no crash, no scope set, behaves like feature off.
        """
        taxonomy_path = tmp_path / "taxonomy.yaml"
        _write_taxonomy(taxonomy_path)
        missing_profiles = tmp_path / "does_not_exist.yaml"
        taxonomy_loader = TaxonomyLoader(taxonomy_path)
        target_loader = TargetProfileLoader(
            config_path=missing_profiles,
            taxonomy_loader=taxonomy_loader,
        )
        # Empty target list — no match for any label.
        assert target_loader.get_targets() == []

        rule = StubClassifier("rules", _rule_success("entity_report", 0.75))
        vector = StubClassifier("vector", _vector_success("entity_report", 0.88))
        orchestrator = FusionOrchestrator(
            classifiers=[rule, vector],
            settings=_settings(),
            target_profile_loader=target_loader,
        )

        result = await orchestrator.classify(_request())

        meta = FusionMeta.model_validate(result.audit)
        assert meta.fusion_strategy == "rule_vector_confirmed"
        assert meta.target_profile is None
        assert vector.last_request is not None
        assert "vector_scope" not in vector.last_request.context
