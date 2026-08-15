from __future__ import annotations

from fastapi.testclient import TestClient

from mailagent.api.main import app


def test_readiness_fails_closed_when_redis_and_worker_are_unavailable(
    monkeypatch,
) -> None:
    async def unavailable_pool(_url: str):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr("mailagent.api.main.create_redis_pool", unavailable_pool)
    with TestClient(app) as client:
        response = client.get(
            "/readyz",
            headers={"x-request-id": "readiness-test-1"},
        )

    assert response.status_code == 503
    assert response.headers["x-request-id"] == "readiness-test-1"
    detail = response.json()["detail"]
    assert detail["status"] == "not_ready"
    assert detail["checks"]["database"] == "ok"
    assert detail["checks"]["redis"] == "unavailable"
    assert detail["checks"]["worker"] == "unknown"
    assert detail["checks"]["vertical"] == "ok"


def test_invalid_request_id_is_replaced(monkeypatch) -> None:
    async def unavailable_pool(_url: str):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr("mailagent.api.main.create_redis_pool", unavailable_pool)
    with TestClient(app) as client:
        response = client.get("/healthz", headers={"x-request-id": "bad id"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] != "bad id"
    assert len(response.headers["x-request-id"]) == 32
