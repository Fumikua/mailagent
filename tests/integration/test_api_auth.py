from __future__ import annotations

from fastapi.testclient import TestClient

from mailagent.api.main import app
from mailagent.infra.config import ApiAuthSettings, Settings


ROLE_KEYS = {
    "MAILAGENT_SUBMITTER_API_KEY": "submitter-secret",
    "MAILAGENT_REVIEWER_API_KEY": "reviewer-secret",
    "MAILAGENT_OPERATOR_API_KEY": "operator-secret",
    "MAILAGENT_ADMIN_API_KEY": "admin-secret",
}


def test_api_role_boundary_is_enforced(
    monkeypatch,
) -> None:
    for name, value in ROLE_KEYS.items():
        monkeypatch.setenv(name, value)
    settings = Settings.from_yaml().model_copy(
        update={"api_auth": ApiAuthSettings(mode="api_key")}
    )
    monkeypatch.setattr(
        Settings,
        "from_yaml",
        classmethod(lambda cls, *args, **kwargs: settings),
    )

    with TestClient(app) as client:
        assert client.get("/api/v1/skills").status_code == 401
        assert (
            client.get(
                "/api/v1/skills",
                headers={"Authorization": "Bearer submitter-secret"},
            ).status_code
            == 200
        )
        assert (
            client.delete(
                "/api/v1/samples/00000000-0000-0000-0000-000000000001",
                headers={"Authorization": "Bearer submitter-secret"},
            ).status_code
            == 403
        )
        assert (
            client.delete(
                "/api/v1/samples/00000000-0000-0000-0000-000000000001",
                headers={"Authorization": "Bearer admin-secret"},
            ).status_code
            == 200
        )
