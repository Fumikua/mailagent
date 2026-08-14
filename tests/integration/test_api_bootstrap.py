"""Integration tests for the /api/v1/bootstrap/* and clustering endpoints.

Simplified coverage (Section 17.10/17.11): verifies endpoints exist and return
the expected response shape. The seed/import/confirm endpoints are placeholders
(actual BootstrapPipeline execution is deferred to the CLI), so these tests
focus on the API contract rather than end-to-end pipeline behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mailagent.api.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


class TestBootstrapSeed:
    def test_seed_returns_pending_job(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/bootstrap/seed",
            json={"dir": "/tmp/seed", "force": False, "no_rules": True},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert len(data["job_id"]) > 0

    def test_seed_with_defaults(self, client: TestClient) -> None:
        resp = client.post("/api/v1/bootstrap/seed", json={"dir": "/tmp/seed"})
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"


class TestBootstrapImport:
    def test_import_returns_pending_job(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/bootstrap/import",
            json={"dir": "/tmp/import", "batch_size": 50},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"


class TestBootstrapStatus:
    def test_status_returns_pending(self, client: TestClient) -> None:
        resp = client.get("/api/v1/bootstrap/status/some-job-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "some-job-id"
        assert data["status"] == "pending"
        assert "report_id" in data


class TestBootstrapReport:
    def test_report_404_when_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/v1/bootstrap/report/nonexistent-report-id")
        assert resp.status_code == 404

    def test_report_returns_markdown_and_pending_samples(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        report_id = "testreport01"
        md_path = tmp_path / f"bootstrap_{report_id}.md"
        md_path.write_text("# Bootstrap Report\n\nSample content.", encoding="utf-8")
        json_path = tmp_path / f"bootstrap_{report_id}.json"
        report_data: dict[str, Any] = {
            "job_id": report_id,
            "samples": [
                {"subject": "STATUS update", "sender": "ops@example.com", "tier": "tier1"},
            ],
        }
        json_path.write_text(json.dumps(report_data), encoding="utf-8")

        monkeypatch.setattr(
            client.app.state.settings.bootstrap, "reports_dir", str(tmp_path)
        )

        resp = client.get(f"/api/v1/bootstrap/report/{report_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"] == report_id
        assert "Bootstrap Report" in data["markdown"]
        assert len(data["pending_samples"]) == 1
        assert data["pending_samples"][0]["subject"] == "STATUS update"


class TestBootstrapConfirm:
    def test_confirm_returns_501(self, client: TestClient) -> None:
        """Confirm is deferred to the CLI; API returns 501."""
        resp = client.post(
            "/api/v1/bootstrap/confirm",
            json={"report_id": "abc", "confirmations": []},
        )
        assert resp.status_code == 501


class TestClusteringReport:
    def test_clustering_404_when_no_reports(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            client.app.state.settings.bootstrap, "reports_dir", str(tmp_path)
        )
        resp = client.get("/api/v1/clustering/report")
        assert resp.status_code == 404

    def test_clustering_returns_latest_report(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "intent_discovery_20260701.md").write_text(
            "# Old report", encoding="utf-8"
        )
        (tmp_path / "intent_discovery_20260715.md").write_text(
            "# Latest report", encoding="utf-8"
        )
        monkeypatch.setattr(
            client.app.state.settings.bootstrap, "reports_dir", str(tmp_path)
        )
        resp = client.get("/api/v1/clustering/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "Latest report" in data["markdown"]
        assert "intent_discovery_20260715.md" in data["report_path"]


class TestBootstrapFlow:
    """Seed → status → report → confirm simplified flow (Section 17.11)."""

    def test_full_flow_contract(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 1. Seed → get job_id
        seed_resp = client.post(
            "/api/v1/bootstrap/seed",
            json={"dir": str(tmp_path), "no_rules": True},
        )
        assert seed_resp.status_code == 202
        job_id = seed_resp.json()["job_id"]

        # 2. Status → pending
        status_resp = client.get(f"/api/v1/bootstrap/status/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "pending"

        # 3. Report → create a fake report and read it back
        report_id = "flowreport01"
        (tmp_path / f"bootstrap_{report_id}.md").write_text(
            "# Flow Report", encoding="utf-8"
        )
        monkeypatch.setattr(
            client.app.state.settings.bootstrap, "reports_dir", str(tmp_path)
        )
        report_resp = client.get(f"/api/v1/bootstrap/report/{report_id}")
        assert report_resp.status_code == 200
        assert "Flow Report" in report_resp.json()["markdown"]

        # 4. Confirm → 501 (deferred to CLI)
        confirm_resp = client.post(
            "/api/v1/bootstrap/confirm",
            json={"report_id": report_id, "confirmations": []},
        )
        assert confirm_resp.status_code == 501
