from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from mailagent.api.main import app
from mailagent.domain.models import (
    ClassificationMeta,
    ClassificationResponse,
    ClassificationVersions,
    RunStatus,
    TaxonomyLabel,
)


TRUSTED_HEADERS = {"X-MailAgent-Reviewer-Id": "reviewer-opaque-1"}


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'feedback-api.db'}"
    monkeypatch.setenv("MAILAGENT_DATABASE_URL", database_url)
    monkeypatch.setenv(
        "MAILAGENT_CLASSIFICATION_FEEDBACK__MODE", "trusted_internal"
    )

    async def unavailable_redis_pool(_redis_url: str):
        raise ConnectionError("Redis unavailable in API integration test")

    monkeypatch.setattr(
        "mailagent.api.main.create_redis_pool",
        unavailable_redis_pool,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def disabled_client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'feedback-disabled.db'}"
    monkeypatch.setenv("MAILAGENT_DATABASE_URL", database_url)
    monkeypatch.delenv("MAILAGENT_CLASSIFICATION_FEEDBACK__MODE", raising=False)

    async def unavailable_redis_pool(_redis_url: str):
        raise ConnectionError("Redis unavailable in API integration test")

    monkeypatch.setattr(
        "mailagent.api.main.create_redis_pool",
        unavailable_redis_pool,
    )
    with TestClient(app) as test_client:
        yield test_client


def _create_run(client: TestClient) -> UUID:
    response = client.post(
        "/api/v1/runs",
        json={
            "email": {
                "message_id": uuid4().hex,
                "sender": "ops@example.com",
                "subject": "STATUS update",
                "body": "STATUS moved to 08:00 UTC.",
            }
        },
    )
    assert response.status_code == 202
    return UUID(response.json()["id"])


def _classification() -> ClassificationResponse:
    return ClassificationResponse(
        labels=[
            TaxonomyLabel(
                l1_code="notification",
                l1_label="Notification",
                confidence=0.91,
                reasoning="STATUS update",
            )
        ],
        meta=ClassificationMeta(overall_confidence=0.91),
        versions=ClassificationVersions(
            taxonomy="sha256:stored-taxonomy",
            rules="sha256:stored-rules",
            prompt="stored-prompt",
            model="stored-model",
            embedding="stored-embedding",
            preprocessing="sha256:stored-preprocessing",
        ),
    )


def _classify_run(client: TestClient, run_id: UUID) -> None:
    assert client.portal is not None
    client.portal.call(
        client.app.state.service.update_classification,
        run_id,
        _classification(),
        RunStatus.COMPLETED,
    )


def _feedback_payload(**overrides) -> dict:
    payload = {
        "final_labels": ["action_required"],
        "error_reasons": ["wrong_label"],
    }
    payload.update(overrides)
    return payload


def test_feedback_disabled_by_default_fails_closed_without_append(
    disabled_client: TestClient,
) -> None:
    run_id = _create_run(disabled_client)
    _classify_run(disabled_client, run_id)

    response = disabled_client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(),
        headers=TRUSTED_HEADERS,
    )
    assert disabled_client.portal is not None
    stored = disabled_client.portal.call(
        disabled_client.app.state.store.list_classification_feedback,
        run_id,
    )

    assert response.status_code == 403
    assert stored == []


def test_enabled_feedback_requires_trusted_reviewer_header(client: TestClient) -> None:
    run_id = _create_run(client)
    _classify_run(client, run_id)

    response = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(),
    )

    assert response.status_code == 401


@pytest.mark.parametrize("reviewer_id", ["   ", "x" * 256])
def test_enabled_feedback_rejects_invalid_trusted_reviewer_header(
    client: TestClient,
    reviewer_id: str,
) -> None:
    run_id = _create_run(client)
    _classify_run(client, run_id)

    response = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(),
        headers={"X-MailAgent-Reviewer-Id": reviewer_id},
    )

    assert response.status_code == 401


def test_enabled_feedback_uses_normalized_trusted_header_identity(
    client: TestClient,
) -> None:
    run_id = _create_run(client)
    _classify_run(client, run_id)

    response = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(),
        headers={"X-MailAgent-Reviewer-Id": "  reviewer-trusted-1  "},
    )

    assert response.status_code == 201
    assert response.json()["reviewer_id"] == "reviewer-trusted-1"
    assert response.json()["eligible_for_sample_proposal"] is False


@pytest.mark.parametrize(
    "caller_controlled",
    [
        {"reviewer_id": "body-reviewer"},
        {"eligible_for_sample_proposal": False},
        {"eligible_for_sample_proposal": True},
    ],
)
def test_feedback_rejects_caller_controlled_identity_and_sample_eligibility(
    client: TestClient,
    caller_controlled: dict[str, object],
) -> None:
    run_id = _create_run(client)
    _classify_run(client, run_id)

    response = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(**caller_controlled),
        headers=TRUSTED_HEADERS,
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"final_labels": []},
        {"final_labels": ["action_required", "action_required"]},
        {"final_labels": [" action_required"]},
        {"final_labels": ["noise", "action_required"]},
        {"error_reasons": []},
        {"error_reasons": ["wrong_label", "wrong_label"]},
        {"error_reasons": ["not_allowed"]},
    ],
)
def test_feedback_request_constraints_return_controlled_422(
    client: TestClient,
    invalid_fields: dict[str, object],
) -> None:
    run_id = _create_run(client)
    _classify_run(client, run_id)

    response = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(**invalid_fields),
        headers=TRUSTED_HEADERS,
    )

    assert response.status_code == 422


def test_feedback_for_unknown_run_returns_404(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/runs/{uuid4()}/classification-feedback",
        json=_feedback_payload(),
        headers=TRUSTED_HEADERS,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "run not found"}


def test_feedback_for_unclassified_run_returns_409(client: TestClient) -> None:
    run_id = _create_run(client)

    response = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(),
        headers=TRUSTED_HEADERS,
    )

    assert response.status_code == 409


def test_feedback_rejects_label_absent_from_active_taxonomy(client: TestClient) -> None:
    run_id = _create_run(client)
    _classify_run(client, run_id)

    response = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(final_labels=["obsolete-category"]),
        headers=TRUSTED_HEADERS,
    )

    assert response.status_code == 422


def test_feedback_rejects_noise_combined_with_business_label(client: TestClient) -> None:
    run_id = _create_run(client)
    _classify_run(client, run_id)

    response = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(final_labels=["noise", "action_required"]),
        headers=TRUSTED_HEADERS,
    )

    assert response.status_code == 422


def test_feedback_appends_and_lists_immutable_revisions(client: TestClient) -> None:
    run_id = _create_run(client)
    stored_classification = _classification()
    assert client.portal is not None
    client.portal.call(
        client.app.state.service.update_classification,
        run_id,
        stored_classification,
        RunStatus.COMPLETED,
    )

    first = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(),
        headers=TRUSTED_HEADERS,
    )
    second = client.post(
        f"/api/v1/runs/{run_id}/classification-feedback",
        json=_feedback_payload(
                final_labels=["notification"],
        ),
        headers={"X-MailAgent-Reviewer-Id": "reviewer-opaque-2"},
    )
    listed = client.get(f"/api/v1/runs/{run_id}/classification-feedback")

    assert first.status_code == 201
    assert first.json()["revision"] == 1
    assert first.json()["predicted_labels"] == ["notification"]
    assert first.json()["versions"] == stored_classification.versions.model_dump()
    assert first.json()["eligible_for_sample_proposal"] is False
    assert second.status_code == 201
    assert second.json()["revision"] == 2
    assert second.json()["eligible_for_sample_proposal"] is False
    assert listed.status_code == 200
    assert [item["revision"] for item in listed.json()] == [1, 2]
    assert [item["final_labels"] for item in listed.json()] == [
        ["action_required"],
        ["notification"],
    ]
