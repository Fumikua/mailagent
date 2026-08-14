import pytest

from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationCoreResult,
    ClassificationRequest,
    EnrichmentPatch,
)
from mailagent.core.pipeline import MailUnderstandingPipeline
from mailagent.core.versioning import ClassificationAssetBinding
from mailagent.domain.models import (
    AttachmentMeta,
    ClassificationMeta,
    ClassificationVersions,
    FusionMeta,
    MailEvent,
    TaxonomyLabel,
)


class StubOrchestrator:
    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        return ClassificationCoreResult(
            labels=[TaxonomyLabel(l1_code="notice", l1_label="通知", confidence=0.95)],
            meta=ClassificationMeta(overall_confidence=0.95),
            selected_source="rules",
        )


class SuccessfulOrchestrator:
    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        return ClassificationCoreResult(
            labels=[TaxonomyLabel(l1_code="schedule", l1_label="船期", confidence=0.95)],
            meta=ClassificationMeta(overall_confidence=0.95),
            selected_source="llm",
        )


class RecordingOrchestrator(StubOrchestrator):
    def __init__(self) -> None:
        self.request: ClassificationRequest | None = None

    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        self.request = request
        return await super().classify(request)


class StaticVersionProvider:
    def __init__(self, binding: ClassificationAssetBinding) -> None:
        self._binding = binding

    def bind(self) -> ClassificationAssetBinding:
        return self._binding


class FusionAuditOrchestrator(StubOrchestrator):
    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        result = await super().classify(request)
        return result.model_copy(
            update={
                "audit": FusionMeta(
                    fusion_strategy="rule_only",
                    source="rule",
                    confidence=0.95,
                ).model_dump()
            }
        )


class NonFusionAuditOrchestrator(StubOrchestrator):
    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        result = await super().classify(request)
        return result.model_copy(update={"audit": {"reason": "legacy cascade"}})


class CascadeAuditOrchestrator(StubOrchestrator):
    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        result = await super().classify(request)
        return result.model_copy(
            update={
                "attempts": [
                    ClassificationAttempt(source="rules", status=AttemptStatus.NO_MATCH),
                    ClassificationAttempt(
                        source="llm",
                        status=AttemptStatus.SUCCESS,
                        labels=result.labels,
                        confidence=0.95,
                    ),
                ],
                "audit": {"reason": "rules unavailable"},
            }
        )


class StubEnricher:
    id = "notice-details"
    namespace = "notice"

    async def enrich(
        self,
        request: ClassificationRequest,
        classification: ClassificationCoreResult,
    ) -> EnrichmentPatch:
        return EnrichmentPatch(namespace=self.namespace, data={"requires_reply": False})


class SecondNoticeEnricher(StubEnricher):
    id = "second-notice-details"

    async def enrich(
        self,
        request: ClassificationRequest,
        classification: ClassificationCoreResult,
    ) -> EnrichmentPatch:
        return EnrichmentPatch(namespace=self.namespace, data={"priority": "high"})


class MutatingEnricher(StubEnricher):
    id = "mutating-details"

    async def enrich(
        self,
        request: ClassificationRequest,
        classification: ClassificationCoreResult,
    ) -> EnrichmentPatch:
        classification.meta = classification.meta.model_copy(
            update={"needs_human_review": False}
        )
        classification.audit["acceptance"] = {
            "status": "accepted",
            "reason": "mutated_by_enricher",
        }
        return EnrichmentPatch(namespace=self.namespace, data={"requires_reply": False})


class WrongNamespaceEnricher(StubEnricher):
    id = "wrong-namespace-details"
    namespace = "wrong"


def _mail() -> MailEvent:
    return MailEvent(message_id="pipeline-1", sender="ops@example.com", subject="Notice", body="FYI")


def _pipeline(
    orchestrator: StubOrchestrator | SuccessfulOrchestrator,
    *,
    auto_accept_enabled: bool = False,
) -> MailUnderstandingPipeline:
    return MailUnderstandingPipeline(
        orchestrator=orchestrator,
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice",
        enrichers=[],
        auto_accept_enabled=auto_accept_enabled,
    )


class TestAuthoritativeRequestFacts:
    @pytest.mark.parametrize(
        ("mail", "expected"),
        [
            (_mail().model_copy(update={"attachments": ["crew-list.pdf"]}), True),
            (
                _mail().model_copy(
                    update={"attachment_meta": [AttachmentMeta(filename="crew-list.pdf")]}
                ),
                True,
            ),
            (_mail(), False),
        ],
    )
    async def test_pipeline_populates_authoritative_attachment_context(
        self, mail: MailEvent, expected: bool
    ) -> None:
        """Mail attachment fields, rather than model output, drive rule context."""
        orchestrator = RecordingOrchestrator()

        await _pipeline(orchestrator).process(mail)

        assert orchestrator.request is not None
        assert orchestrator.request.context["has_attachments"] is expected

    async def test_pipeline_binds_versions_and_assets_from_one_snapshot(
        self,
    ) -> None:
        taxonomy_snapshot = object()
        versions = ClassificationVersions(
            taxonomy="sha256:taxonomy",
            preprocessing="none",
        )
        binding = ClassificationAssetBinding(
            versions=versions,
            asset_snapshots={"taxonomy": taxonomy_snapshot},
        )
        orchestrator = RecordingOrchestrator()
        pipeline = MailUnderstandingPipeline(
            orchestrator=orchestrator,
            vertical_id="notice-management",
            data_schema_version="2",
            vertical_namespace="notice",
            enrichers=[],
            version_provider=StaticVersionProvider(binding),  # type: ignore[arg-type]
        )

        response = await pipeline.process(_mail())

        assert response.versions == versions
        assert orchestrator.request is not None
        assert orchestrator.request.asset_snapshots["taxonomy"] is taxonomy_snapshot


class AuditStubOrchestrator(StubOrchestrator):
    def __init__(self, audit: dict[str, object]) -> None:
        self._audit = audit

    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        result = await super().classify(request)
        return result.model_copy(update={"audit": self._audit})


async def test_p0_gate_preserves_labels_but_requires_review() -> None:
    pipeline = _pipeline(
        orchestrator=SuccessfulOrchestrator(),
        auto_accept_enabled=False,
    )

    response = await pipeline.process(_mail())

    assert response.labels[0].l1_code == "schedule"
    assert response.meta.needs_human_review is True
    assert response.orchestration_audit is not None
    assert response.orchestration_audit.details["acceptance"] == {
        "status": "review",
        "reason": "p0_auto_accept_disabled",
    }


async def test_p0_gate_cannot_be_reversed_by_mutating_enricher() -> None:
    pipeline = MailUnderstandingPipeline(
        orchestrator=SuccessfulOrchestrator(),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice",
        enrichers=[MutatingEnricher()],
        auto_accept_enabled=False,
    )

    response = await pipeline.process(_mail())

    assert response.meta.needs_human_review is True
    assert response.orchestration_audit is not None
    assert response.orchestration_audit.details["acceptance"] == {
        "status": "review",
        "reason": "p0_auto_accept_disabled",
    }


async def test_auto_accept_enabled_preserves_orchestrator_behavior() -> None:
    pipeline = _pipeline(
        orchestrator=SuccessfulOrchestrator(),
        auto_accept_enabled=True,
    )

    response = await pipeline.process(_mail())

    assert response.labels[0].l1_code == "schedule"
    assert response.meta.needs_human_review is False
    assert response.orchestration_audit is None


async def test_pipeline_returns_orchestrated_labels_and_namespaced_vertical_data() -> None:
    pipeline = MailUnderstandingPipeline(
        orchestrator=StubOrchestrator(),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice",
        enrichers=[StubEnricher()],
    )

    result = await pipeline.process(_mail())

    assert result.labels[0].l1_code == "notice"
    assert result.vertical_id == "notice-management"
    assert result.data_schema_version == "2"
    assert result.data == {"notice": {"requires_reply": False}}


async def test_pipeline_ignores_non_fusion_audit() -> None:
    pipeline = MailUnderstandingPipeline(
        orchestrator=NonFusionAuditOrchestrator(),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice",
        enrichers=[],
    )

    result = await pipeline.process(_mail())

    assert result.fusion_meta is None
    assert result.orchestration_audit is not None
    assert result.orchestration_audit.selected_source == "rules"
    assert result.orchestration_audit.attempts == []
    assert result.orchestration_audit.details == {
        "reason": "legacy cascade",
        "acceptance": {
            "status": "review",
            "reason": "p0_auto_accept_disabled",
        },
    }


async def test_pipeline_retains_cascade_attempts_for_non_fusion_orchestrator() -> None:
    pipeline = MailUnderstandingPipeline(
        orchestrator=CascadeAuditOrchestrator(),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice",
        enrichers=[],
    )

    result = await pipeline.process(_mail())

    assert result.orchestration_audit is not None
    assert result.orchestration_audit.selected_source == "rules"
    assert [attempt["source"] for attempt in result.orchestration_audit.attempts] == [
        "rules",
        "llm",
    ]


async def test_pipeline_merges_enrichers_in_same_namespace() -> None:
    """Multiple enrichers writing the same namespace merge their data dicts."""
    pipeline = MailUnderstandingPipeline(
        orchestrator=StubOrchestrator(),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice",
        enrichers=[StubEnricher(), SecondNoticeEnricher()],
    )

    result = await pipeline.process(_mail())

    notice_data = result.data["notice"]
    assert notice_data["requires_reply"] is False
    assert notice_data["priority"] == "high"


def test_pipeline_rejects_enricher_namespace_other_than_selected_vertical() -> None:
    with pytest.raises(ValueError, match="must write selected namespace notice"):
        MailUnderstandingPipeline(
            orchestrator=StubOrchestrator(),
            vertical_id="notice-management",
            data_schema_version="2",
            vertical_namespace="notice",
            enrichers=[WrongNamespaceEnricher()],
        )


async def test_pipeline_keeps_classification_when_an_enricher_fails() -> None:
    class FailingEnricher(StubEnricher):
        id = "failing-details"

        async def enrich(
            self,
            request: ClassificationRequest,
            classification: ClassificationCoreResult,
        ) -> EnrichmentPatch:
            raise RuntimeError("enrichment service unavailable")

    pipeline = MailUnderstandingPipeline(
        orchestrator=StubOrchestrator(),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice",
        enrichers=[FailingEnricher()],
    )

    result = await pipeline.process(_mail())

    assert result.labels[0].l1_code == "notice"
    assert result.data == {}
    assert result.meta.needs_human_review is True
    assert result.enrichment_errors[0].enricher_id == "failing-details"
    assert result.enrichment_errors[0].message == "enrichment service unavailable"


async def test_pipeline_rejects_enrichment_data_that_breaks_vertical_schema() -> None:
    """业务包数据不符合声明的 schema 时，不能写入通用 data。"""

    class InvalidEnricher(StubEnricher):
        async def enrich(
            self,
            request: ClassificationRequest,
            classification: ClassificationCoreResult,
        ) -> EnrichmentPatch:
            return EnrichmentPatch(namespace=self.namespace, data={"requires_reply": "later"})

    pipeline = MailUnderstandingPipeline(
        orchestrator=StubOrchestrator(),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice",
        data_schema={
            "type": "object",
            "properties": {"requires_reply": {"type": "boolean"}},
            "required": ["requires_reply"],
            "additionalProperties": False,
        },
        enrichers=[InvalidEnricher()],
    )

    result = await pipeline.process(_mail())

    assert result.data == {}
    assert result.meta.needs_human_review is True
    assert result.enrichment_errors[0].namespace == "notice"
    assert "boolean" in result.enrichment_errors[0].message


async def test_pipeline_copies_valid_fusion_audit_to_response() -> None:
    audit = FusionMeta(
        fusion_strategy="rule_only",
        source="rule",
        confidence=0.95,
    )
    pipeline = MailUnderstandingPipeline(
        orchestrator=AuditStubOrchestrator(audit.model_dump()),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice_management",
        enrichers=[],
    )

    result = await pipeline.process(_mail())

    assert result.fusion_meta == audit


async def test_pipeline_ignores_invalid_fusion_audit() -> None:
    pipeline = MailUnderstandingPipeline(
        orchestrator=AuditStubOrchestrator({"fusion_strategy": "invalid"}),
        vertical_id="notice-management",
        data_schema_version="2",
        vertical_namespace="notice_management",
        enrichers=[],
    )

    result = await pipeline.process(_mail())

    assert result.fusion_meta is None
