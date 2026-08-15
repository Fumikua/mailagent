"""Integration tests for implemented bootstrap-report and clustering APIs."""

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


class TestUnavailableBootstrapCommands:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/api/v1/bootstrap/seed"),
            ("post", "/api/v1/bootstrap/import"),
            ("get", "/api/v1/bootstrap/status/job-id"),
            ("post", "/api/v1/bootstrap/confirm"),
        ],
    )
    def test_unimplemented_commands_are_not_advertised(
        self,
        client: TestClient,
        method: str,
        path: str,
    ) -> None:
        response = client.request(method, path, json={})
        assert response.status_code == 404


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
                {
                    "subject": "STATUS update",
                    "sender": "ops@example.com",
                    "tier": "tier1",
                },
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
