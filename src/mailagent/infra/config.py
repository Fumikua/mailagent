from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    url: str = "sqlite+aiosqlite:///./data/mailagent.db"


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"


class ModelSettings(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    enabled: bool = False


class PolicySettings(BaseModel):
    approval_actions: list[str] = Field(default_factory=lambda: ["send_email", "forward_email", "delete_email"])
    blocked_actions: list[str] = Field(default_factory=lambda: ["send_email", "forward_email", "delete_email"])


class ClassificationSettings(BaseModel):
    taxonomy_path: str = "./taxonomy.yaml"
    autonomy_level_default: str = "L0"
    confidence_threshold: float = 0.8
    auto_accept_enabled: bool = False


class ClassificationFeedbackSettings(BaseModel):
    """Administrative correction endpoint trust boundary.

    The endpoint is fail-closed by default. ``trusted_internal`` is intended
    only for deployments where a trusted reverse proxy strips and injects the
    configured reviewer identity header.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["disabled", "trusted_internal"] = "disabled"
    reviewer_identity_header: str = "X-MailAgent-Reviewer-Id"

    @field_validator("reviewer_identity_header")
    @classmethod
    def _validate_header_name(cls, value: str) -> str:
        allowed = set("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if not value or len(value) > 128 or any(char not in allowed for char in value):
            raise ValueError("reviewer_identity_header must be a valid HTTP header name")
        return value


class ApiAuthSettings(BaseModel):
    """Bearer-key role boundary; values are read only from environment variables."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["disabled", "api_key"] = "disabled"
    submitter_key_env: str = "MAILAGENT_SUBMITTER_API_KEY"
    reviewer_key_env: str = "MAILAGENT_REVIEWER_API_KEY"
    operator_key_env: str = "MAILAGENT_OPERATOR_API_KEY"
    admin_key_env: str = "MAILAGENT_ADMIN_API_KEY"

    @field_validator(
        "submitter_key_env",
        "reviewer_key_env",
        "operator_key_env",
        "admin_key_env",
    )
    @classmethod
    def _validate_env_name(cls, value: str) -> str:
        if not value or not value.replace("_", "A").isalnum() or value.upper() != value:
            raise ValueError(
                "API key environment names must use uppercase letters, digits, and underscores"
            )
        return value

    def role_key_envs(self) -> dict[str, str]:
        return {
            "submitter": self.submitter_key_env,
            "reviewer": self.reviewer_key_env,
            "operator": self.operator_key_env,
            "admin": self.admin_key_env,
        }


class VerticalSettings(BaseModel):
    """Selects installed vertical code and its external business profile.

    ``id`` is intentionally required-without-a-business-default: core must not
    assume any specific vertical. Deployments set it explicitly in config.yml
    (or via MAILAGENT_VERTICAL__ID); an empty value fails fast at selection.
    """

    id: str = ""
    verticals_path: str = "./verticals"


# ---------------------------------------------------------------------------
# vector-similarity-path-b settings
# ---------------------------------------------------------------------------


class EmbeddingSettings(BaseModel):
    """Embedding service configuration.

    Supports two providers:
    - ``tei`` (default): HuggingFace Text Embeddings Inference. Uses
      ``POST /embed`` with ``{"inputs": [...]}`` and no auth header.
    - ``openai``: OpenAI-compatible API (e.g. SiliconFlow, Azure OpenAI,
      vLLM OpenAI server). Uses ``POST /v1/embeddings`` with
      ``{"input": [...], "model": "..."}`` and ``Authorization: Bearer <key>``.
      The key is read from the env var named by ``api_key_env``.
    """

    provider: str = "tei"
    model_name: str = "Qwen3-Embedding-8B"
    api_base: str = "http://localhost:8080"
    dimension: int = 4096
    timeout: int = 30
    # OpenAI-compatible provider only: name of the env var holding the API key.
    # Ignored when provider == "tei".
    api_key_env: str = "EMBEDDING_API_KEY"


class VectorStoreSettings(BaseModel):
    """Vector store (pgvector) configuration."""

    top_k: int = 5
    similarity_threshold: float = 0.85
    minimum_support: int = Field(default=1, ge=1)
    minimum_margin: float = Field(default=0.03, ge=0, le=1)
    archive_window_months: int = 12
    stratified_sample_threshold: int = 50000


class FusionSettings(BaseModel):
    """Three-path fusion orchestrator configuration."""

    rule_confidence_threshold: float = 0.9
    vector_confidence_threshold: float = 0.85
    llm_fallback_threshold: float = 0.7
    enable_clustering: bool = True
    cluster_min_size: int = 5
    cluster_interval_days: int = 7
    enabled: bool = False  # feature flag: default off for backward compat


class RulesSettings(BaseModel):
    """Rule classifier configuration.

    ``rules_dir`` is a standalone-CLI fallback only; the runtime resolves rule
    assets from the selected vertical's manifest (``loaded.rules.path``). The
    default is intentionally vertical-agnostic.
    """

    rules_dir: str = "./verticals/rules"
    enable_autolearn: bool = True
    autolearn_min_samples: int = 5
    autolearn_min_ratio: float = 0.8


class ClusteringSettings(BaseModel):
    """HDBSCAN clustering configuration."""

    min_cluster_size: int = 5
    min_samples: int = 3
    metric: str = "cosine"
    max_samples: int = 50000
    window_days: int = 30


class BootstrapSettings(BaseModel):
    """Bootstrap pipeline configuration."""

    weekly_batch_size: int = 4200
    default_batch_size: int = 50
    reports_dir: str = "./reports"


class RetentionSettings(BaseModel):
    """数据保留策略配置（cleanup_job 使用）。

    控制各审计/业务表的过期数据清理周期。设为 0 表示不清理对应表。
    """

    # processing_runs 表保留天数（含完整邮件正文 payload）
    runs_retention_days: int = 90
    # samples_archive 表保留天数（已归档样本，含邮件正文）
    archive_retention_days: int = 365
    # classification_feedback 表保留天数（人工修正审计）
    feedback_retention_days: int = 365
    # mail_gateway_ingest_ledger 表保留天数（去重账本，仅清理终态行）
    ledger_retention_days: int = 90
    # mail_gateway_backfill_audit 表保留天数（手动回填审计）
    backfill_audit_retention_days: int = 365


class MailGatewaySettings(BaseModel):
    enabled: bool = False
    adapter: Literal["imap", "pop3"] = "imap"
    mailbox_id: str = "primary"
    host: str = ""
    port: int = 993
    username: str = ""
    password_env: str = ""
    use_ssl: Literal[True] = True
    mailbox: str = "INBOX"
    poll_interval_seconds: int = Field(default=60, ge=60)
    fetch_batch_size: int = Field(default=50, ge=1, le=100)
    seen_filter: Literal["all", "unseen"] = "all"
    max_message_bytes: int = Field(default=26_214_400, gt=0)
    initial_sync_mode: Literal["from_now", "bounded_backfill", "incremental"] = "from_now"
    backfill_since_days: int = Field(default=1, ge=1, le=7)
    backfill_max_messages: int = Field(default=1000, ge=1, le=1000)
    initial_backfill_confirmed: bool = False
    initial_sync_max_messages: int = Field(default=1000, ge=1)
    sender_domain_allowlist: list[str] = Field(default_factory=list)
    recipient_domain_allowlist: list[str] = Field(default_factory=list)
    subject_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_gateway(self) -> "MailGatewaySettings":
        if self.poll_interval_seconds % 60:
            raise ValueError("poll_interval_seconds must be a multiple of 60")
        if self.enabled and not all((self.host, self.username, self.password_env, self.mailbox_id)):
            raise ValueError("enabled mail gateway requires host, username, password_env, and mailbox_id")
        if self.initial_sync_mode == "bounded_backfill" and not self.initial_backfill_confirmed:
            raise ValueError("bounded_backfill requires initial_backfill_confirmed=true")
        if self.adapter == "pop3":
            if self.port == 993:
                self.port = 995
            if self.initial_sync_mode != "incremental":
                raise ValueError("pop3 adapter requires initial_sync_mode='incremental'")
            if self.mailbox != "INBOX":
                import warnings

                warnings.warn(
                    "pop3 adapter only supports INBOX; mailbox field is ignored",
                    stacklevel=2,
                )
        elif self.adapter == "imap" and self.initial_sync_mode == "incremental":
            raise ValueError("incremental sync mode is pop3-only; use from_now or bounded_backfill for imap")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAILAGENT_", env_nested_delimiter="__", extra="ignore")

    app_name: str = "mailagent"
    environment: str = "development"
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    classification: ClassificationSettings = Field(default_factory=ClassificationSettings)
    classification_feedback: ClassificationFeedbackSettings = Field(
        default_factory=ClassificationFeedbackSettings
    )
    api_auth: ApiAuthSettings = Field(default_factory=ApiAuthSettings)
    vertical: VerticalSettings = Field(default_factory=VerticalSettings)
    vertical_overrides: dict[str, Any] = Field(default_factory=dict)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)
    rules: RulesSettings = Field(default_factory=RulesSettings)
    clustering: ClusteringSettings = Field(default_factory=ClusteringSettings)
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    mail_gateways: list[MailGatewaySettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_mailbox_ids(self) -> "Settings":
        ids = [gw.mailbox_id for gw in self.mail_gateways if gw.enabled]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate mailbox_id among enabled mail_gateways")
        if (
            self.environment.casefold() in {"production", "prod"}
            and self.api_auth.mode == "disabled"
        ):
            raise ValueError("production requires api_auth.mode='api_key'")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yml") -> "Settings":
        # 加载 .env 文件（项目根目录或当前工作目录），不覆盖已存在的 env 变量
        for candidate in (Path.cwd() / ".env", Path.cwd().parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate, override=False)
                break

        config_path = Path(path)
        data: dict[str, Any] = {}
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text()) or {}
        app = data.pop("app", {})
        data["app_name"] = app.get("name", "mailagent")
        data["environment"] = app.get("environment", "development")
        # ``mail_gateways`` is canonical, but retain the original singular
        # object for one compatibility release. Normalize both forms before
        # Pydantic validation so the runtime has one representation.
        legacy_present = "mail_gateway" in data
        canonical_present = "mail_gateways" in data
        if legacy_present and canonical_present:
            raise ValueError("configure either mail_gateway or mail_gateways, not both")

        legacy_gateway = data.pop("mail_gateway", None)
        gateway_overlay: dict[str, Any] = {}
        prefix = "MAILAGENT_MAIL_GATEWAY__"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            field_name = key.removeprefix(prefix).lower()
            gateway_overlay[field_name] = yaml.safe_load(value)

        if legacy_present:
            warnings.warn(
                "mail_gateway is deprecated; use mail_gateways",
                DeprecationWarning,
                stacklevel=2,
            )
            gateway = dict(legacy_gateway or {})
            gateway.update(gateway_overlay)
            data["mail_gateways"] = [gateway]
        elif canonical_present and gateway_overlay:
            gateways = list(data.get("mail_gateways") or [])
            if len(gateways) != 1:
                raise ValueError(
                    "MAILAGENT_MAIL_GATEWAY__* overrides require exactly one mail_gateways entry"
                )
            gateway = dict(gateways[0])
            gateway.update(gateway_overlay)
            data["mail_gateways"] = [gateway]
        elif gateway_overlay:
            data["mail_gateways"] = [gateway_overlay]
        settings = cls(**data)
        # YAML is versioned configuration; deployment environment variables must
        # still take precedence without ever needing to store secrets in YAML.
        database_url = os.getenv("MAILAGENT_DATABASE_URL")
        redis_url = os.getenv("MAILAGENT_REDIS__URL")
        model_base_url = os.getenv("MAILAGENT_MODEL__BASE_URL")
        model_name = os.getenv("MAILAGENT_MODEL__MODEL_NAME")
        model_enabled = os.getenv("MAILAGENT_MODEL__ENABLED")
        if database_url:
            settings.database.url = database_url
        if redis_url:
            settings.redis.url = redis_url
        if model_base_url:
            settings.model.base_url = model_base_url
        if model_name:
            settings.model.model_name = model_name
        if model_enabled is not None:
            settings.model.enabled = model_enabled.lower() in {"1", "true", "yes", "on"}
        return settings
