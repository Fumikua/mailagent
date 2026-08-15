from __future__ import annotations

import pytest
from fastapi import HTTPException

from mailagent.api.auth import ApiRole, authenticate_bearer, validate_api_auth_secrets
from mailagent.infra.config import ApiAuthSettings, Settings


ROLE_KEYS = {
    "MAILAGENT_SUBMITTER_API_KEY": "submitter-secret",
    "MAILAGENT_REVIEWER_API_KEY": "reviewer-secret",
    "MAILAGENT_OPERATOR_API_KEY": "operator-secret",
    "MAILAGENT_ADMIN_API_KEY": "admin-secret",
}


def test_disabled_auth_is_allowed_as_development_principal() -> None:
    principal = authenticate_bearer(ApiAuthSettings(), None)

    assert principal.subject == "development"
    assert principal.role is ApiRole.ADMIN


def test_api_key_auth_resolves_each_role(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ROLE_KEYS.items():
        monkeypatch.setenv(name, value)
    settings = ApiAuthSettings(mode="api_key")

    validate_api_auth_secrets(settings)
    assert authenticate_bearer(settings, "reviewer-secret").role is ApiRole.REVIEWER


def test_api_key_auth_rejects_missing_or_invalid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ROLE_KEYS.items():
        monkeypatch.setenv(name, value)
    settings = ApiAuthSettings(mode="api_key")

    with pytest.raises(HTTPException) as missing:
        authenticate_bearer(settings, None)
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as invalid:
        authenticate_bearer(settings, "wrong")
    assert invalid.value.status_code == 401


def test_enabled_auth_fails_fast_when_a_role_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in ROLE_KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MAILAGENT_OPERATOR_API_KEY")

    with pytest.raises(RuntimeError, match="MAILAGENT_OPERATOR_API_KEY"):
        validate_api_auth_secrets(ApiAuthSettings(mode="api_key"))


def test_production_rejects_disabled_auth() -> None:
    with pytest.raises(ValueError, match="production requires"):
        Settings(environment="production")
