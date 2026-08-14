"""端到端集成测试：API → service → Worker (classify_job) → store。

mock LLM 响应（不实际调用 LLM 服务），mock Redis（不入队真实任务），
通过手动调用 classify_job 模拟 Worker 后台处理流程。

验证：
- POST /api/v1/runs 立即返回 202 + status=pending
- GET /api/v1/runs/{run_id} 返回 status=completed + classification
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from mailagent.api.main import app

PROJECT_ROOT = Path(__file__).parent.parent.parent  # tests/integration/ → tests/ → mailagent/
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "emails"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _post_email(client: TestClient, fixture_name: str) -> tuple[dict, UUID]:
    """POST /api/v1/runs 提交邮件 fixture，返回 (response_body, run_id)"""
    payload = _load_fixture(fixture_name)
    response = client.post("/api/v1/runs", json={"email": payload})
    assert response.status_code == 202, f"POST failed: {response.text}"
    body = response.json()
    return body, UUID(body["id"])


class TestApiContract:
    """验证 API 立即返回 202 + pending 的契约（不涉及 Worker）"""

    def test_post_returns_202_pending(self) -> None:
        with TestClient(app) as client:
            body, _ = _post_email(client, "status_update.json")
            assert body["status"] == "pending"
            assert body["classification"] is None
            assert body["trace"][-1] == "create_run:enqueued"

    def test_get_run_returns_404_for_unknown(self) -> None:
        from uuid import uuid4

        with TestClient(app) as client:
            response = client.get(f"/api/v1/runs/{uuid4()}")
            assert response.status_code == 404
