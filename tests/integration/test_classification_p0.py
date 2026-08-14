"""P0 trusted-baseline integration across classification and evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml
from fastapi.testclient import TestClient

from mailagent.api.main import app
from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationCoreResult,
    ClassificationRequest,
    EnrichmentPatch,
)
from mailagent.core.pipeline import MailUnderstandingPipeline
from mailagent.core.versioning import ClassificationVersionProvider
from mailagent.domain.models import ClassificationMeta, RunStatus, TaxonomyLabel
from mailagent.evaluation import (
    ReleaseGate,
    evaluate_predictions,
    load_gold_manifest,
    load_prediction_snapshot,
)
from mailagent.infra.queue import classify_job


class _SyntheticOrchestrator:
    async def classify(
        self, request: ClassificationRequest
    ) -> ClassificationCoreResult:
        label = TaxonomyLabel(
            l1_code="notification",
            l1_label="Notification",
            confidence=0.94,
            reasoning="Synthetic STATUS update",
        )
        attempt = ClassificationAttempt(
            source="synthetic",
            status=AttemptStatus.SUCCESS,
            labels=[label],
            confidence=0.94,
            meta=ClassificationMeta(overall_confidence=0.94),
        )
        return ClassificationCoreResult(
            labels=[label],
            meta=attempt.meta,
            selected_source=attempt.source,
            attempts=[attempt],
            audit={"strategy": "synthetic"},
        )


class _MutatingEnricher:
    id = "malicious-mutator"
    namespace = "example_triage"

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
        return EnrichmentPatch(namespace=self.namespace, data={})


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'p0-integration.db'}"
    monkeypatch.setenv("MAILAGENT_DATABASE_URL", database_url)
    monkeypatch.setenv(
        "MAILAGENT_CLASSIFICATION_FEEDBACK__MODE", "trusted_internal"
    )

    async def unavailable_redis_pool(_redis_url: str) -> None:
        raise ConnectionError("Redis unavailable in P0 integration test")

    monkeypatch.setattr(
        "mailagent.api.main.create_redis_pool",
        unavailable_redis_pool,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_worker_persists_review_after_mutating_enricher(client: TestClient) -> None:
    created = client.post(
        "/api/v1/runs",
        json={
            "email": {
                "message_id": uuid4().hex,
                "sender": "synthetic-ops@example.invalid",
                "subject": "Synthetic STATUS update",
                "body": "Synthetic entity STATUS changed to 08:00 UTC.",
            }
        },
    )
    assert created.status_code == 202
    run_id = UUID(created.json()["id"])
    assert client.portal is not None
    pipeline = MailUnderstandingPipeline(
        orchestrator=_SyntheticOrchestrator(),
        vertical_id="example-triage",
        data_schema_version="synthetic-v1",
        vertical_namespace="example_triage",
        enrichers=[_MutatingEnricher()],
        auto_accept_enabled=False,
    )

    result = client.portal.call(
        classify_job,
        {
            "store": client.app.state.store,
            "mail_understanding_pipeline": pipeline,
            "service": client.app.state.service,
        },
        str(run_id),
    )
    saved = client.portal.call(client.app.state.store.get_run, run_id)

    assert result["status"] == RunStatus.WAITING_APPROVAL.value
    assert saved is not None
    assert saved.status == RunStatus.WAITING_APPROVAL
    assert saved.classification is not None
    assert saved.classification.meta.needs_human_review is True
    assert saved.classification.orchestration_audit is not None
    assert saved.classification.orchestration_audit.details["acceptance"] == {
        "status": "review",
        "reason": "p0_auto_accept_disabled",
    }


def test_p0_review_feedback_and_evaluation_are_fail_closed_end_to_end(
    client: TestClient,
    tmp_path: Path,
) -> None:
    created = client.post(
        "/api/v1/runs",
        json={
            "email": {
                "message_id": uuid4().hex,
                "sender": "synthetic-ops@example.invalid",
                "subject": "Synthetic STATUS update",
                "body": "Synthetic entity STATUS changed to 08:00 UTC.",
            }
        },
    )
    assert created.status_code == 202
    run_id = UUID(created.json()["id"])
    assert client.portal is not None

    settings = client.app.state.settings
    assert settings.classification.auto_accept_enabled is False
    pipeline = MailUnderstandingPipeline(
        orchestrator=_SyntheticOrchestrator(),
        vertical_id="example-triage",
        data_schema_version="synthetic-v1",
        vertical_namespace="example_triage",
        enrichers=[],
        version_provider=ClassificationVersionProvider(
            taxonomy_loader=client.app.state.taxonomy_loader,
            prompt_version="synthetic-prompt-v1",
            model_version="synthetic-model-v1",
            embedding_version=None,
        ),
        auto_accept_enabled=settings.classification.auto_accept_enabled,
    )

    worker_result = client.portal.call(
        classify_job,
        {
            "store": client.app.state.store,
            "mail_understanding_pipeline": pipeline,
            "service": client.app.state.service,
        },
        str(run_id),
    )
    assert worker_result["status"] == RunStatus.WAITING_APPROVAL.value

    saved = client.portal.call(client.app.state.store.get_run, run_id)
    assert saved is not None
    assert saved.status == RunStatus.WAITING_APPROVAL
    assert saved.classification is not None
    assert saved.classification.labels
    assert saved.classification.meta.needs_human_review is True
    assert saved.classification.versions is not None
    assert saved.classification.orchestration_audit is not None
    assert (
        saved.classification.orchestration_audit.details["acceptance"]["reason"]
        == "p0_auto_accept_disabled"
    )

    feedback = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json={
            "final_labels": ["action_required"],
            "error_reasons": ["wrong_label"],
        },
        headers={"X-MailAgent-Reviewer-Id": "reviewer-synthetic-p0"},
    )
    assert feedback.status_code == 201
    assert feedback.json()["predicted_labels"] == ["notification"]
    assert feedback.json()["final_labels"] == ["action_required"]
    assert feedback.json()["versions"] == saved.classification.versions.model_dump()
    listed_feedback = client.get(f"/api/v1/runs/{run_id}/classification-feedback")
    assert listed_feedback.status_code == 200
    assert [item["revision"] for item in listed_feedback.json()] == [1]

    sample_id = "synthetic-p0-001"
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps(
            {
                "predictions": [
                    {
                        "sample_id": sample_id,
                        "labels": [
                            label.l1_code for label in saved.classification.labels
                        ],
                        "needs_human_review": (
                            saved.classification.meta.needs_human_review
                        ),
                        "strategy": saved.classification.orchestration_audit.details[
                            "strategy"
                        ],
                        "versions": saved.classification.versions.model_dump(
                            mode="json"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "gold-manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "corpus_version": "synthetic-p0-v1",
                "taxonomy_version": saved.classification.versions.taxonomy,
                "examples": [
                    {
                        "sample_id": sample_id,
                        "thread_id": "synthetic-thread-p0-001",
                        "labels": feedback.json()["final_labels"],
                        "split": "test",
                        "annotation_refs": [
                            "annotation-synthetic-p0-a",
                            "annotation-synthetic-p0-b",
                        ],
                        "adjudicated": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    valid_labels = client.app.state.taxonomy_loader.get_tree().all_codes()
    manifest = load_gold_manifest(manifest_path, valid_labels)
    predictions = load_prediction_snapshot(prediction_path, valid_labels)
    assert predictions[0].versions == saved.classification.versions

    report = evaluate_predictions(
        manifest,
        predictions,
        ReleaseGate(minimum_eligible=1),
    )
    assert report == evaluate_predictions(
        manifest,
        predictions,
        ReleaseGate(minimum_eligible=1),
    )
    assert report.auto_accepted_examples == 0
    assert report.reviewed_examples == 1
    assert report.coverage == 0.0
    assert report.micro.precision is None
    assert report.micro.false_negative == 1
    assert report.gate.status == "ineligible"
