"""API authentication and role authorization for MailAgent control surfaces."""
from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mailagent.infra.config import ApiAuthSettings


class ApiRole(StrEnum):
    SUBMITTER = "submitter"
    REVIEWER = "reviewer"
    OPERATOR = "operator"
    ADMIN = "admin"


_ROLE_RANK = {
    ApiRole.SUBMITTER: 10,
    ApiRole.REVIEWER: 20,
    ApiRole.OPERATOR: 30,
    ApiRole.ADMIN: 40,
}


@dataclass(frozen=True, slots=True)
class ApiPrincipal:
    subject: str
    role: ApiRole


_bearer = HTTPBearer(auto_error=False)
BearerCredentials = Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]


def _configured_keys(settings: ApiAuthSettings) -> dict[ApiRole, str]:
    configured: dict[ApiRole, str] = {}
    for role_name, env_name in settings.role_key_envs().items():
        value = os.getenv(env_name, "")
        if not value:
            raise RuntimeError(f"API authentication requires environment variable {env_name}")
        configured[ApiRole(role_name)] = value
    if len(set(configured.values())) != len(configured):
        raise RuntimeError("API authentication role keys must be unique")
    return configured


def validate_api_auth_secrets(settings: ApiAuthSettings) -> None:
    """Fail startup when enabled authentication has missing or ambiguous keys."""

    if settings.mode == "api_key":
        _configured_keys(settings)


def authenticate_bearer(
    settings: ApiAuthSettings,
    token: str | None,
) -> ApiPrincipal:
    if settings.mode == "disabled":
        return ApiPrincipal(subject="development", role=ApiRole.ADMIN)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer API key is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    for role, expected in _configured_keys(settings).items():
        if hmac.compare_digest(token, expected):
            return ApiPrincipal(subject=f"api-key:{role.value}", role=role)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(required: ApiRole) -> Callable[..., ApiPrincipal]:
    def dependency(
        request: Request,
        credentials: BearerCredentials,
    ) -> ApiPrincipal:
        token = credentials.credentials if credentials is not None else None
        principal = authenticate_bearer(request.app.state.settings.api_auth, token)
        if _ROLE_RANK[principal.role] < _ROLE_RANK[required]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{required.value} role is required",
            )
        request.state.principal = principal
        return principal

    return dependency


require_submitter = require_role(ApiRole.SUBMITTER)
require_reviewer = require_role(ApiRole.REVIEWER)
require_operator = require_role(ApiRole.OPERATOR)
require_admin = require_role(ApiRole.ADMIN)
