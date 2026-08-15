from __future__ import annotations

import unicodedata
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RunStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    WAITING_APPROVAL = "waiting_approval"
    REJECTED = "rejected"
    FAILED = "failed"


class MailEvent(BaseModel):
    message_id: str = Field(min_length=1)
    sender: str
    subject: str
    body: str
    recipients: list[str] = Field(default_factory=list)
    mailbox_id: str = "demo"
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attachments: list[str] = Field(default_factory=list)
    attachment_meta: list["AttachmentMeta"] | None = None


class AttachmentMeta(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    size: int = 0


class ProposedAction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: Literal[
        "add_label", "create_draft", "send_email", "forward_email", "delete_email"
    ]
    risk: Literal["low", "high"]
    requires_approval: bool
    status: Literal["proposed", "approved", "rejected", "blocked"] = "proposed"
    preview: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """Deprecated category projection retained for API compatibility.

    Core deliberately does not enumerate business categories. A selected
    vertical owns the taxonomy and may project one of its labels here while
    clients migrate to ``classification.labels``.
    """

    category: str = Field(min_length=1, max_length=128)
    summary: str
    urgency: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    needs_reply: bool


class TaxonomyLabel(BaseModel):
    l1_code: str
    l1_label: str
    l2_code: str | None = None
    l2_label: str | None = None
    l3_code: str | None = None
    l3_label: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str = ""


class CalibrationLog(BaseModel):
    raw: float = Field(ge=0, le=1)
    calibrated: float = Field(ge=0, le=1)
    anchor: str


class ClassificationMeta(BaseModel):
    urgency: Literal["low", "medium", "high", "urgent"] = "medium"
    language: str = "en"
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"
    has_attachments: bool = False
    overall_confidence: float = Field(ge=0, le=1, default=0.0)
    needs_human_review: bool = False
    fallback: bool = False
    model_used: str = ""
    latency_ms: int = 0


class EnrichmentError(BaseModel):
    """A non-fatal vertical enrichment failure retained for audit and review."""

    enricher_id: str
    namespace: str
    message: str


# ---------------------------------------------------------------------------
# vector-similarity-path-b domain models
# ---------------------------------------------------------------------------


class NormalizedSubject(BaseModel):
    """Subject line after generic prefix stripping and whitespace folding."""

    raw: str
    clean: str


class ThreadSegment(BaseModel):
    """One segment of a parsed email thread (latest first)."""

    text: str
    position: int
    is_latest: bool


class RuleMatch(BaseModel):
    """A single rule match from RuleClassifier."""

    rule_type: Literal[
        "sender_domains", "subject_patterns", "body_keywords", "structural"
    ]
    label: str
    confidence: float = Field(ge=0, le=1)
    matched_pattern: str


class RuleResult(BaseModel):
    """Aggregated rule matching result."""

    matches: list[RuleMatch] = Field(default_factory=list)
    selected: RuleMatch | None = None
    conflict_logged: bool = False


class PathBCandidate(BaseModel):
    """One vector similarity candidate from VectorClassifier."""

    label: str
    max_similarity: float = Field(ge=0, le=1)
    count: int = Field(ge=1)
    mean_similarity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class PathBResult(BaseModel):
    """Vector similarity classification result."""

    candidates: list[PathBCandidate] = Field(default_factory=list)
    top1_label: str | None = None
    top1_similarity: float = Field(ge=0, le=1, default=0.0)
    top1_support: int = Field(ge=0, default=0)
    top1_mean_similarity: float = Field(ge=0, le=1, default=0.0)
    next_category_similarity: float | None = Field(default=None, ge=0, le=1)
    similarity_margin: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1, default=0.0)
    reason: Literal[
        "ok",
        "below_threshold",
        "embedding_unavailable",
        "empty_samples",
        "ineligible_document",
        "ambiguous_candidates",
        "stale_category",
    ] = "ok"


class FusionConflict(BaseModel):
    """Conflicting top labels that require a human fusion review."""

    sources: list[Literal["rule", "vector", "llm"]]
    labels: list[str]


class FusionMeta(BaseModel):
    """Audit metadata for three-path fusion decisions."""

    fusion_strategy: Literal[
        "rule_only",
        "rule_vector_confirmed",
        "vector_only",
        "llm_fallback",
        "all_low_review",
    ]
    source: Literal["rule", "vector", "llm"]
    confidence: float = Field(ge=0, le=1)
    rule_result: RuleResult | None = None
    vector_result: PathBResult | None = None
    llm_result: dict[str, Any] | None = None
    vector_confirmed: bool = False
    # P0 (target-label-scoping): records the target label path when scoped vector
    # retrieval was attempted in Step 2, regardless of whether it confirmed.
    # None when no target profile matched (or feature is off). Does NOT add a
    # new fusion_strategy — scoped retrieval is an internal optimization of
    # ``rule_vector_confirmed``.
    target_profile: str | None = None
    conflict: FusionConflict | None = None


class OrchestrationAudit(BaseModel):
    """JSON-safe record of every classifier decision made for a mail."""

    selected_source: str | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ClassificationVersions(BaseModel):
    model_config = ConfigDict(frozen=True)

    taxonomy: str
    rules: str | None = None
    target_profiles: str | None = None
    prompt: str | None = None
    model: str | None = None
    embedding: str | None = None
    preprocessing: str


ClassificationFeedbackErrorReason = Literal[
    "wrong_label",
    "missing_label",
    "extra_label",
    "ambiguous",
    "insufficient_evidence",
    "taxonomy_gap",
    "other",
]


def normalize_reviewer_identity(value: object) -> str:
    """Normalize a trusted reviewer identity without accepting control text."""

    if not isinstance(value, str):
        raise ValueError("trusted reviewer identity must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        raise ValueError("trusted reviewer identity contains disallowed control text")
    normalized = normalized.strip()
    if not normalized:
        raise ValueError("trusted reviewer identity must not be blank")
    if len(normalized) > 255:
        raise ValueError("trusted reviewer identity must be at most 255 characters")
    return normalized


class ClassificationFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    final_labels: list[str] = Field(min_length=1)
    error_reasons: list[ClassificationFeedbackErrorReason] = Field(min_length=1)

    @field_validator("final_labels")
    @classmethod
    def _validate_final_labels(cls, labels: list[str]) -> list[str]:
        if any(not label or label != label.strip() for label in labels):
            raise ValueError("final labels must be non-empty canonical codes")
        if len(labels) != len(set(labels)):
            raise ValueError("final labels must be unique")
        return labels

    @field_validator("error_reasons")
    @classmethod
    def _validate_error_reasons(
        cls,
        reasons: list[ClassificationFeedbackErrorReason],
    ) -> list[ClassificationFeedbackErrorReason]:
        if len(reasons) != len(set(reasons)):
            raise ValueError("error reasons must be unique")
        return reasons


class ClassificationFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    run_id: UUID
    revision: int = Field(ge=1)
    predicted_labels: list[str]
    final_labels: list[str]
    error_reasons: list[ClassificationFeedbackErrorReason]
    reviewer_id: str
    reviewed_at: datetime
    versions: ClassificationVersions | None
    eligible_for_sample_proposal: bool


class SampleQualityAssessment(BaseModel):
    """Quality and taxonomy decision recorded before a sample is persisted."""

    disposition: Literal["accepted", "warned", "rejected", "legacy"]
    reasons: list[str] = Field(default_factory=list)
    fingerprint: str = Field(min_length=64, max_length=64)
    taxonomy_schema_version: str = "flat-v1"
    retrieval_policy_version: str


class SampleRecord(BaseModel):
    """One labeled email sample stored in the vector store for similarity search."""

    id: UUID = Field(default_factory=uuid4)
    mail_hash: str
    subject_raw: str
    subject_clean: str
    sender: str
    sender_domain: str
    body: str
    label_l1: str
    label_l2: str | None = None
    label_l3: str | None = None
    confidence: float = Field(ge=0, le=1)
    source: Literal["seed", "rule_tier1", "rule_tier2", "llm", "human_fix", "auto"]
    reviewed: bool = False
    thread_parsed: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    batch_confirmed_at: datetime | None = None
    taxonomy_schema_version: str = "flat-v1"
    retrieval_document: dict[str, Any] | None = None
    retrieval_fingerprint: str | None = None
    retrieval_policy_version: str | None = None
    quality: SampleQualityAssessment | None = None
    review_override_reason: str | None = None

    @model_validator(mode="after")
    def mark_legacy_three_level_records(self) -> "SampleRecord":
        """Keep callers that still provide L2/L3 readable but out of flat retrieval."""

        if self.taxonomy_schema_version == "flat-v1" and (
            self.label_l2 is not None or self.label_l3 is not None
        ):
            self.taxonomy_schema_version = "legacy-v3"
        return self


class ClassificationResponse(BaseModel):
    labels: list[TaxonomyLabel] = Field(default_factory=list)
    meta: ClassificationMeta = Field(default_factory=ClassificationMeta)
    calibration_log: CalibrationLog | None = None
    autonomy_level: str = "L0"
    vertical_id: str = ""
    data_schema_version: str = "1"
    data: dict[str, Any] = Field(default_factory=dict)
    enrichment_errors: list[EnrichmentError] = Field(default_factory=list)
    orchestration_audit: OrchestrationAudit | None = None
    fusion_meta: FusionMeta | None = None
    versions: ClassificationVersions | None = None


class SkillDefinition(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str
    instructions: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(
        default_factory=lambda: ["add_label", "create_draft"]
    )
    enabled: bool = True


class SkillVersion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    skill_id: UUID
    version: int = Field(ge=1)
    definition: SkillDefinition
    published_at: datetime | None = None


class RunResponse(BaseModel):
    id: UUID
    status: RunStatus
    email: MailEvent
    skill_version_id: UUID | None
    decision: Decision | None
    actions: list[ProposedAction]
    trace: list[str]
    error: str | None = None
    classification: ClassificationResponse | None = None
    created_at: datetime
    updated_at: datetime


class CreateRunRequest(BaseModel):
    email: MailEvent
    skill_id: UUID | None = None


class ApprovalRequest(BaseModel):
    action_ids: list[UUID] | None = None
