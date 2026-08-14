"""Integration tests for the /api/v1/samples endpoints (Section 17.6-17.8).

Uses a mock VectorStore injected via FastAPI dependency_overrides so no real
database is required for the samples CRUD contract.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from mailagent.api.main import app, get_vector_store
from mailagent.domain.models import SampleRecord


def _make_sample(**overrides: Any) -> SampleRecord:
    defaults: dict[str, Any] = {
        "mail_hash": "hash-001",
        "subject_raw": "STATUS update",
        "subject_clean": "STATUS update",
        "sender": "ops@example.com",
        "sender_domain": "example.com",
        "body": "Entity Berlin Example STATUS Shanghai.",
        "label_l1": "entity",
        "label_l2": "schedule",
        "label_l3": "eta_update",
        "confidence": 0.92,
        "source": "seed",
    }
    defaults.update(overrides)
    return SampleRecord(**defaults)


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(mock_vector_store: AsyncMock) -> TestClient:
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestSamplesList:
    def test_list_returns_samples(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        sample = _make_sample()
        mock_vector_store.get_samples = AsyncMock(return_value=[sample])
        resp = client.get("/api/v1/samples")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["mail_hash"] == "hash-001"
        assert data[0]["label_l3"] == "eta_update"

    def test_list_with_label_filter(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_vector_store.get_samples = AsyncMock(return_value=[])
        resp = client.get("/api/v1/samples", params={"label": "eta_update"})
        assert resp.status_code == 200
        assert resp.json() == []
        mock_vector_store.get_samples.assert_awaited_once_with(
            label="eta_update", source=None, page=1
        )

    def test_list_with_source_and_page(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_vector_store.get_samples = AsyncMock(return_value=[])
        resp = client.get(
            "/api/v1/samples", params={"source": "seed", "page": 3}
        )
        assert resp.status_code == 200
        mock_vector_store.get_samples.assert_awaited_once_with(
            label=None, source="seed", page=3
        )

    def test_list_empty_when_no_vector_store(self) -> None:
        """Without a VectorStore the list endpoint returns an empty list."""
        app.dependency_overrides[get_vector_store] = lambda: None
        with TestClient(app) as c:
            resp = c.get("/api/v1/samples")
            assert resp.status_code == 200
            assert resp.json() == []
        app.dependency_overrides.clear()


class TestSampleGet:
    def test_get_sample_by_id(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        sid = uuid4()
        sample = _make_sample(id=sid)
        mock_vector_store.get_sample = AsyncMock(return_value=sample)
        resp = client.get(f"/api/v1/samples/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(sid)
        assert body["sender"] == "ops@example.com"

    def test_get_sample_not_found(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_vector_store.get_sample = AsyncMock(return_value=None)
        resp = client.get(f"/api/v1/samples/{uuid4()}")
        assert resp.status_code == 404


class TestSampleDelete:
    def test_delete_sample(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        sid = uuid4()
        mock_vector_store.delete_sample = AsyncMock(return_value=None)
        resp = client.delete(f"/api/v1/samples/{sid}")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": str(sid)}
        mock_vector_store.delete_sample.assert_awaited_once_with(sid)


class TestSamplePatch:
    def test_patch_updates_label(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        sid = uuid4()
        updated = _make_sample(id=sid, label_l3="location_request")
        mock_vector_store.update_sample_label = AsyncMock(return_value=None)
        mock_vector_store.get_sample = AsyncMock(return_value=updated)
        resp = client.patch(
            f"/api/v1/samples/{sid}", json={"label": "location_request"}
        )
        assert resp.status_code == 200
        assert resp.json()["label_l3"] == "location_request"
        mock_vector_store.update_sample_label.assert_awaited_once_with(
            sid, "location_request"
        )

    def test_patch_returns_404_when_missing(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_vector_store.update_sample_label = AsyncMock(return_value=None)
        mock_vector_store.get_sample = AsyncMock(return_value=None)
        resp = client.patch(
            f"/api/v1/samples/{uuid4()}", json={"label": "x"}
        )
        assert resp.status_code == 404


class TestSamplesCrudRoundtrip:
    def test_crud_roundtrip(
        self,
        client: TestClient,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Create-via-mock → get → patch → delete roundtrip across endpoints."""
        sid: UUID = uuid4()

        # GET (exists)
        sample = _make_sample(id=sid)
        mock_vector_store.get_sample = AsyncMock(return_value=sample)
        resp = client.get(f"/api/v1/samples/{sid}")
        assert resp.status_code == 200

        # PATCH (update label)
        patched = _make_sample(id=sid, label_l3="new_label")
        mock_vector_store.update_sample_label = AsyncMock(return_value=None)
        mock_vector_store.get_sample = AsyncMock(return_value=patched)
        resp = client.patch(
            f"/api/v1/samples/{sid}", json={"label": "new_label"}
        )
        assert resp.status_code == 200
        assert resp.json()["label_l3"] == "new_label"

        # DELETE
        mock_vector_store.delete_sample = AsyncMock(return_value=None)
        resp = client.delete(f"/api/v1/samples/{sid}")
        assert resp.status_code == 200

        # GET (gone)
        mock_vector_store.get_sample = AsyncMock(return_value=None)
        resp = client.get(f"/api/v1/samples/{sid}")
        assert resp.status_code == 404
