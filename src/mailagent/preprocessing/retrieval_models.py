"""Contracts for deterministic, auditable embedding input preparation."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mailagent.domain.versioning import digest_named_assets


class RetrievalCleaningPolicy(BaseModel):
    version: str
    latest_max_chars: int = Field(default=1800, ge=1)
    context_max_chars: int = Field(default=1200, ge=0)
    attachments_max_chars: int = Field(default=300, ge=0)
    min_meaningful_chars: int = Field(default=40, ge=1)
    signature_delimiters: tuple[str, ...] = ()
    disclaimer_patterns: tuple[str, ...] = ()

    model_config = ConfigDict(frozen=True)


def validated_policy_version(policy: RetrievalCleaningPolicy) -> str:
    """Identify the validated in-memory cleaning policy, never its source file."""

    return digest_named_assets(
        [
            (
                "preprocessing:retrieval_cleaning_policy",
                policy.model_dump_json().encode("utf-8"),
            )
        ]
    )


def load_retrieval_cleaning_policy(path: str | Path) -> RetrievalCleaningPolicy:
    """Load one vertical-owned, deterministic retrieval cleaning policy."""

    policy_path = Path(path)
    try:
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load retrieval cleaning policy: {policy_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("retrieval cleaning policy must be a YAML mapping")
    return RetrievalCleaningPolicy.model_validate(raw)


class RetrievalDocument(BaseModel):
    text: str
    primary_text: str
    context_text: str
    policy_version: str
    flags: list[str] = Field(default_factory=list)
    eligible: bool
    ineligible_reason: str | None = None
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    latest_char_count: int = Field(ge=0)
    context_char_count: int = Field(ge=0)
