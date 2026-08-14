"""Bootstrap pipeline for Path B sample seeding and incremental import.

Stage 1 (seed): LLM full annotation on manually-selected .eml files.
Stage 2 (import_history): tiered labeling (rule → LLM fallback) on weekly batches.
Stage 3 (production feedback): planned/deferred. Runtime classification persists
audit data but does not automatically ingest samples.

Reports are written as dual-format files (markdown + JSON) under ``reports/``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from mailagent.classification.contracts import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationRequest,
    Classifier,
)
from mailagent.classification.rule_classifier import RuleClassifier
from mailagent.domain.mail_parser import parse_email_message
from mailagent.domain.models import MailEvent, PathBCandidate, RuleResult, SampleRecord
from mailagent.infra.config import BootstrapSettings
from mailagent.infra.vector_store import VectorStore
from mailagent.llm.embedding import EmbeddingClient
from mailagent.preprocessing.pipeline import embed_retrieval_document, preprocess_mail
from mailagent.preprocessing.contracts import MailPreprocessingExtension
from mailagent.preprocessing.retrieval_models import RetrievalCleaningPolicy, RetrievalDocument
from mailagent.preprocessing.subject_normalizer import normalize_subject
from mailagent.infra.sample_quality import assess_sample_quality

logger = logging.getLogger(__name__)

# Tier confidence thresholds (aligned with FusionSettings defaults).
_TIER1_THRESHOLD = 0.9
_TIER2_THRESHOLD = 0.7

# Body preview length for reports.
_BODY_PREVIEW_LEN = 200
_LATEST_WEIGHT = 0.6
_CONTEXT_WEIGHT = 0.4


def _compute_mail_hash(subject: str, sender: str, received_at: datetime) -> str:
    """SHA-256 hash of subject + sender + received_at for deduplication."""
    raw = f"{subject}|{sender}|{received_at.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _apply_retrieval_text_override(
    document: RetrievalDocument,
    override_text: object,
) -> RetrievalDocument:
    """Replace only the derived retrieval view after an explicit human review."""

    if not isinstance(override_text, str) or not override_text.strip():
        raise ValueError("retrieval text override must be non-empty text")
    text = override_text.strip()
    flags = [*document.flags]
    if "reviewer_override" not in flags:
        flags.append("reviewer_override")
    return document.model_copy(
        update={
            "text": text,
            "primary_text": text,
            "context_text": "",
            "latest_char_count": len(text),
            "context_char_count": 0,
            "flags": flags,
        }
    )


def _extract_domain(sender: str) -> str:
    """Extract the domain part from a sender string like 'Name <user@domain>'."""
    addr = sender
    if "<" in addr and ">" in addr:
        addr = addr[addr.index("<") + 1 : addr.index(">")]
    addr = addr.strip()
    if "@" not in addr:
        return ""
    return addr.rsplit("@", 1)[1].strip().lower()


def _extract_attachments(message: Message) -> list[str]:
    """Return attachment filenames in MIME message order."""

    attachments: list[str] = []
    for part in message.walk():
        filename = part.get_filename()
        if part.get_content_disposition() == "attachment" and filename:
            attachments.append(str(filename))
    return attachments


def _parse_eml(path: Path) -> MailEvent:
    """Parse historical RFC822 input through the shared gateway parser."""

    raw = path.read_bytes()
    message = BytesParser(policy=policy.default).parsebytes(raw)
    event = parse_email_message(raw, mailbox_id="bootstrap")
    return event.model_copy(update={"attachments": _extract_attachments(message)})


def _split_label_path(label: str) -> tuple[str, str, str]:
    """Split a dot-separated label path into (l1, l2, l3).

    Falls back to using the full label as l1 when no dots are present.
    """
    parts = [p for p in label.split(".") if p]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], parts[1]
    if len(parts) == 1:
        return parts[0], parts[0], parts[0]
    return "unknown", "unknown", "unknown"


def _attempt_to_label_dict(attempt: ClassificationAttempt) -> dict[str, Any]:
    """Convert an LLM ClassificationAttempt into a serializable label dict."""
    if not attempt.labels:
        return {}
    label = attempt.labels[0]
    l1_code = label.l1_code.strip()
    if not l1_code:
        return {}
    return {
        "l1": l1_code,
        "l2": label.l2_code.strip() if label.l2_code else None,
        "l3": label.l3_code.strip() if label.l3_code else None,
        "confidence": label.confidence,
        "reasoning": label.reasoning,
    }


def _labels_match(rule_label: str, llm_match: dict[str, Any]) -> bool:
    """Return whether rule and LLM evidence identify the same label."""
    rule_parts = [part for part in rule_label.split(".") if part]
    llm_parts = [
        str(code)
        for code in (
            llm_match.get("l1"),
            llm_match.get("l2"),
            llm_match.get("l3"),
        )
        if code
    ]
    if not rule_parts or not llm_parts:
        return False
    return rule_parts == llm_parts or rule_parts[-1] == llm_parts[-1]


def _label_dict_to_fields(
    llm_match: dict[str, Any] | None,
    rule_label: str | None,
) -> tuple[str, str, str]:
    """Resolve (label_l1, label_l2, label_l3) from LLM dict or rule label."""
    if llm_match and llm_match.get("l1"):
        return (
            str(llm_match.get("l1", "unknown")),
            str(llm_match.get("l2") or llm_match.get("l1", "unknown")),
            str(llm_match.get("l3") or llm_match.get("l1", "unknown")),
        )
    if rule_label:
        return _split_label_path(rule_label)
    return "unknown", "unknown", "unknown"


class BootstrapPipeline:
    """Orchestrates Path B bootstrap: seed labeling, incremental import, and archival."""

    def __init__(
        self,
        rule_classifier: RuleClassifier,
        llm_classifier: Classifier,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        settings: BootstrapSettings,
        *,
        cleaning_policy: RetrievalCleaningPolicy | None = None,
        preprocessing_extension: MailPreprocessingExtension | None = None,
        taxonomy_loader: Any | None = None,
    ) -> None:
        self.rule_classifier = rule_classifier
        self.llm_classifier = llm_classifier
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.settings = settings
        self.cleaning_policy = cleaning_policy
        self.preprocessing_extension = preprocessing_extension
        self.taxonomy_loader = taxonomy_loader

    # ------------------------------------------------------------------
    # Stage 1: Seed labeling (LLM full annotation, all Tier 3)
    # ------------------------------------------------------------------

    async def seed(
        self,
        dir: Path,
        force: bool = False,
        no_rules: bool = False,
    ) -> str:
        """Stage 1: scan .eml files, run LLM annotation on all, generate report.

        Args:
            dir: Directory containing manually-selected .eml files.
            force: Overwrite existing samples with the same mail_hash.
            no_rules: Skip the empty-rules warning (for CI/scripted use).

        Returns:
            Report ID (used for subsequent review/confirm commands).
        """
        eml_files = sorted(dir.glob("*.eml"))
        if not eml_files:
            logger.warning("No .eml files found in %s", dir)

        if not no_rules:
            self._check_rules_warning()

        samples: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for path in eml_files:
            mail_event = _parse_eml(path)
            mail_hash = _compute_mail_hash(
                mail_event.subject, mail_event.sender, mail_event.received_at
            )

            if not force and await self._mail_hash_exists(mail_hash):
                logger.info("Skipping %s (mail_hash already in DB, use --force)", path.name)
                continue

            preprocessed = await preprocess_mail(
                mail_event,
                self.embedding_client,
                cleaning_policy=self.cleaning_policy,
                extension=self.preprocessing_extension,
                embed=False,
            )
            if not preprocessed.retrieval_document.eligible:
                excluded.append(
                    self._build_excluded_entry(
                        mail_event,
                        mail_hash,
                        preprocessed.retrieval_document,
                        str(path),
                    )
                )
                continue

            attempt = await self.llm_classifier.classify(
                ClassificationRequest(mail=mail_event)
            )
            llm_match = _attempt_to_label_dict(attempt)
            quality = self._assess_proposed_quality(
                preprocessed.retrieval_document, tier="tier3", rule_result=None, llm_match=llm_match
            )
            if quality is not None and quality.disposition != "accepted":
                excluded.append(
                    self._build_excluded_entry(
                        mail_event,
                        mail_hash,
                        preprocessed.retrieval_document,
                        str(path),
                        quality=quality,
                    )
                )
                continue
            emb_thread, emb_seg0 = await embed_retrieval_document(
                preprocessed.retrieval_document, self.embedding_client
            )

            samples.append(
                self._build_sample_entry(
                    tier="tier3",
                    mail_event=mail_event,
                    mail_hash=mail_hash,
                    rule_result=None,
                    llm_match=llm_match,
                    emb_thread=emb_thread,
                    emb_seg0=emb_seg0,
                    retrieval_document=preprocessed.retrieval_document,
                    file_path=str(path),
                    quality=quality,
                )
            )

        report_id = self._generate_report(
            stage="seed",
            samples=samples,
            input_count=len(eml_files),
            excluded=excluded,
        )
        return report_id

    # ------------------------------------------------------------------
    # Stage 2: Incremental import (tiered labeling)
    # ------------------------------------------------------------------

    async def import_history(self, dir: Path, batch_size: int = 50) -> str:
        """Stage 2: import historical emails in batches with tiered labeling.

        Tier 1: rule confidence >= 0.9 (auto-confirm candidates).
        Tier 2: rule confidence 0.7-0.9 (rule + LLM verify).
        Tier 3: no rule or rule < 0.7 (LLM only).

        Args:
            dir: Directory containing .eml files.
            batch_size: Number of emails per processing batch.

        Returns:
            Report ID.
        """
        eml_files = sorted(dir.glob("*.eml"))
        if not eml_files:
            logger.warning("No .eml files found in %s", dir)

        samples: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for batch_start in range(0, len(eml_files), batch_size):
            batch = eml_files[batch_start : batch_start + batch_size]
            for path in batch:
                mail_event = _parse_eml(path)
                mail_hash = _compute_mail_hash(
                    mail_event.subject, mail_event.sender, mail_event.received_at
                )

                preprocessed = await preprocess_mail(
                    mail_event,
                    self.embedding_client,
                    cleaning_policy=self.cleaning_policy,
                    extension=self.preprocessing_extension,
                    embed=False,
                )
                if not preprocessed.retrieval_document.eligible:
                    excluded.append(
                        self._build_excluded_entry(
                            mail_event,
                            mail_hash,
                            preprocessed.retrieval_document,
                            str(path),
                        )
                    )
                    continue

                rule_result = self.rule_classifier.match(mail_event)
                tier, _candidate, evidence = await self._tier_classify(
                    mail_event, rule_result
                )
                if tier == "tier2":
                    llm_match = evidence.get("llm_match") or {}
                    verification_status = evidence["verification_status"]
                    verification_detail = evidence.get("verification_detail")
                else:
                    llm_match = evidence
                    verification_status = None
                    verification_detail = None

                quality = self._assess_proposed_quality(
                    preprocessed.retrieval_document,
                    tier=tier,
                    rule_result=rule_result,
                    llm_match=llm_match,
                )
                if quality is not None and quality.disposition != "accepted":
                    excluded.append(
                        self._build_excluded_entry(
                            mail_event,
                            mail_hash,
                            preprocessed.retrieval_document,
                            str(path),
                            quality=quality,
                        )
                    )
                    continue
                emb_thread, emb_seg0 = await embed_retrieval_document(
                    preprocessed.retrieval_document, self.embedding_client
                )

                samples.append(
                    self._build_sample_entry(
                        tier=tier,
                        mail_event=mail_event,
                        mail_hash=mail_hash,
                        rule_result=rule_result,
                        llm_match=llm_match,
                        emb_thread=emb_thread,
                        emb_seg0=emb_seg0,
                        retrieval_document=preprocessed.retrieval_document,
                        file_path=str(path),
                        verification_status=verification_status,
                        verification_detail=verification_detail,
                        quality=quality,
                    )
                )

        report_id = self._generate_report(
            stage="import",
            samples=samples,
            input_count=len(eml_files),
            excluded=excluded,
        )
        return report_id

    # ------------------------------------------------------------------
    # Tier classification
    # ------------------------------------------------------------------

    async def _tier_classify(
        self,
        mail_event: MailEvent,
        rule_result: RuleResult,
    ) -> tuple[str, PathBCandidate | None, dict[str, Any]]:
        """Classify a mail into Tier 1/2/3 based on rule confidence.

        Returns:
            Tuple of (tier, path_b_candidate, classification evidence).
            path_b_candidate is always None (no vector classifier in bootstrap).
            Tier 2 evidence contains the rule suggestion, structured LLM result,
            verification status, and optional failure detail. Tier 3 evidence is
            the LLM label dict.
        """
        selected = rule_result.selected
        if selected is not None and selected.confidence >= _TIER1_THRESHOLD:
            return "tier1", None, {}

        if selected is not None and selected.confidence >= _TIER2_THRESHOLD:
            evidence: dict[str, Any] = {
                "rule_suggestion": selected.label,
                "llm_match": None,
                "verification_status": "unavailable",
                "verification_detail": None,
            }
            try:
                attempt = await self.llm_classifier.classify(
                    ClassificationRequest(mail=mail_event)
                )
            except Exception as exc:
                logger.warning("Tier 2 LLM verification unavailable: %s", exc)
                evidence["verification_detail"] = str(exc)
                return "tier2", None, evidence

            if not isinstance(attempt, ClassificationAttempt):
                evidence["verification_detail"] = (
                    "Malformed LLM result: expected ClassificationAttempt"
                )
                return "tier2", None, evidence

            if attempt.status != AttemptStatus.SUCCESS:
                evidence["verification_detail"] = (
                    attempt.error or f"LLM attempt status: {attempt.status.value}"
                )
                return "tier2", None, evidence

            llm_match = _attempt_to_label_dict(attempt)
            if not llm_match:
                evidence["verification_detail"] = "LLM returned no usable labels"
                return "tier2", None, evidence

            evidence["llm_match"] = llm_match
            evidence["verification_status"] = (
                "confirmed"
                if _labels_match(selected.label, llm_match)
                else "disagreed"
            )
            return "tier2", None, evidence

        # Tier 3: no rule or low confidence — call LLM.
        attempt = await self.llm_classifier.classify(
            ClassificationRequest(mail=mail_event)
        )
        llm_match = _attempt_to_label_dict(attempt)
        return "tier3", None, llm_match

    # ------------------------------------------------------------------
    # Confirm samples from a report
    # ------------------------------------------------------------------

    async def confirm_tier(
        self,
        report_id: str,
        tier: int,
        all_: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Confirm samples from a bootstrap report by tier.

        Args:
            report_id: Report identifier returned by seed/import.
            tier: Tier number (1, 2, or 3).
            all_: Batch-confirm all samples in this tier.
            dry_run: Preview without writing to DB.

        Returns:
            Number of confirmed samples.

        Raises:
            ValueError: If Tier 2 bypasses review or Tier 3 uses batch confirm.
        """
        if tier == 2 or (all_ and tier == 3):
            raise ValueError(
                "Tier 2/3 样本必须逐条审核, 请使用 mailagent bootstrap review --tier 2,3"
            )

        report = self._load_report(report_id)
        tier_key = f"tier{tier}"
        tier_samples = [s for s in report["samples"] if s["tier"] == tier_key]

        if dry_run:
            for s in tier_samples:
                logger.info(
                    "[dry-run] Would confirm: %s — %s", s["mail_hash"], s["subject"]
                )
            return len(tier_samples)

        source = self._tier_source(tier, report.get("stage", "import"))
        count = 0
        for entry in tier_samples:
            sample, emb_thread, emb_seg0 = await self._prepare_sample_for_persistence(
                entry, source=source
            )
            # Delete any existing sample with the same mail_hash (upsert semantics,
            # supports --force re-runs that overwrite previously confirmed samples).
            await self._delete_by_mail_hash(sample.mail_hash)
            await self._persist_sample(sample, emb_thread, emb_seg0)
            count += 1

        return count

    # ------------------------------------------------------------------
    # Interactive review (placeholder — UI lives in cli.py)
    # ------------------------------------------------------------------

    async def review_interactive(self, report_id: str, tier: str) -> None:
        """Load samples for interactive review. Actual UI is in cli.py."""
        report = self._load_report(report_id)
        tier_key = f"tier{tier}"
        samples = [s for s in report["samples"] if s["tier"] == tier_key]
        logger.info(
            "Loaded %d samples for %s review from report %s",
            len(samples),
            tier_key,
            report_id,
        )

    # ------------------------------------------------------------------
    # Persist a single sample
    # ------------------------------------------------------------------

    async def _persist_sample(
        self,
        sample: SampleRecord,
        embedding_thread: list[float],
        embedding_segment_0: list[float],
    ) -> None:
        """Insert a sample with embeddings into the vector store."""
        await self.vector_store.insert_sample(sample, embedding_thread, embedding_segment_0)

    async def _prepare_sample_for_persistence(
        self,
        entry: dict[str, Any],
        *,
        source: str,
    ) -> tuple[SampleRecord, list[float], list[float]]:
        """Build one reviewed sample and re-embed an explicit text override.

        The report entry and immutable raw email payload are never mutated.
        This shared path keeps CLI review and non-interactive confirmation
        consistent.
        """

        entry_for_sample = dict(entry)
        emb_thread = entry_for_sample.get("embedding_thread", [])
        emb_seg0 = entry_for_sample.get("embedding_segment_0", [])
        override = entry_for_sample.get("retrieval_text_override")
        if override is not None:
            document_raw = entry_for_sample.get("retrieval_document")
            if not isinstance(document_raw, dict):
                raise ValueError("retrieval text override requires a retrieval document")
            document = _apply_retrieval_text_override(
                RetrievalDocument.model_validate(document_raw),
                override,
            )
            embeddings = await self.embedding_client.embed_batch(
                [document.text, document.context_text or document.text]
            )
            emb_seg0 = embeddings[0]
            emb_thread = [
                _LATEST_WEIGHT * latest + _CONTEXT_WEIGHT * context
                for latest, context in zip(embeddings[0], embeddings[1])
            ]
            entry_for_sample["retrieval_document"] = document.model_dump()
        return self._entry_to_sample(entry_for_sample, source=source), emb_thread, emb_seg0

    # ------------------------------------------------------------------
    # Archive old samples
    # ------------------------------------------------------------------

    async def archive_old_samples(self, months: int = 12) -> int:
        """Move samples older than ``months`` to the archive table."""
        return await self.vector_store.archive_old_samples(months)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_rules_warning(self) -> None:
        """Log a warning if no rules are loaded in the rule classifier."""
        has_rules = bool(
            getattr(self.rule_classifier, "_sender_domain_rules", [])
            or getattr(self.rule_classifier, "_subject_pattern_rules", [])
            or getattr(self.rule_classifier, "_body_keyword_rules", [])
            or getattr(self.rule_classifier, "_structural_rules", [])
        )
        if not has_rules:
            logger.warning(
                "未检测到规则文件, 所有邮件将走 Tier 3 LLM 全量标注。"
                "使用 --no-rules 跳过此警告。"
            )

    async def _mail_hash_exists(self, mail_hash: str) -> bool:
        """Check if a sample with the given mail_hash already exists in the DB."""
        from sqlalchemy import func, select

        from mailagent.infra.store import SampleORM

        async with self.vector_store.sessions() as session:
            result = await session.scalar(
                select(func.count())
                .select_from(SampleORM)
                .where(SampleORM.mail_hash == mail_hash)
            )
            return int(result or 0) > 0

    async def _delete_by_mail_hash(self, mail_hash: str) -> None:
        """Delete all samples with the given mail_hash (for --force overwrite)."""
        from sqlalchemy import delete

        from mailagent.infra.store import SampleORM

        async with self.vector_store.sessions() as session:
            await session.execute(
                delete(SampleORM).where(SampleORM.mail_hash == mail_hash)
            )
            await session.commit()

    def _build_sample_entry(
        self,
        tier: str,
        mail_event: MailEvent,
        mail_hash: str,
        rule_result: RuleResult | None,
        llm_match: dict[str, Any],
        emb_thread: list[float],
        emb_seg0: list[float],
        retrieval_document: Any,
        file_path: str,
        verification_status: str | None = None,
        verification_detail: str | None = None,
        quality: Any | None = None,
    ) -> dict[str, Any]:
        """Build a serializable sample entry for the report."""
        rule_match_dict: dict[str, Any] | None = None
        if rule_result is not None and rule_result.selected is not None:
            sel = rule_result.selected
            rule_match_dict = {
                "label": sel.label,
                "confidence": sel.confidence,
                "rule_type": sel.rule_type,
                "matched_pattern": sel.matched_pattern,
            }

        consistency = False
        if rule_match_dict and llm_match:
            consistency = _labels_match(rule_match_dict["label"], llm_match)

        body_preview = mail_event.body[:_BODY_PREVIEW_LEN]
        proposed_flat_category = self._proposed_flat_category(
            tier=tier, rule_result=rule_result, llm_match=llm_match
        )

        return {
            "id": str(uuid.uuid4()),
            "tier": tier,
            "subject": mail_event.subject,
            "sender": mail_event.sender,
            "mail_hash": mail_hash,
            "normalized_subject": normalize_subject(mail_event.subject).clean,
            "proposed_flat_category": proposed_flat_category,
            "taxonomy_schema_version": (
                quality.taxonomy_schema_version if quality is not None else "legacy-v3"
            ),
            "retrieval_fingerprint": quality.fingerprint if quality is not None else None,
            "quality": quality.model_dump() if quality is not None else None,
            "rule_match": rule_match_dict,
            "llm_match": llm_match or None,
            "consistency": consistency,
            "verification_status": verification_status,
            "verification_detail": verification_detail,
            "action": "pending",
            "embedding_thread": emb_thread,
            "embedding_segment_0": emb_seg0,
            "retrieval_document": retrieval_document.model_dump(),
            "body_preview": body_preview,
            "mail_event": {
                "message_id": mail_event.message_id,
                "subject": mail_event.subject,
                "sender": mail_event.sender,
                "body": mail_event.body,
                "recipients": mail_event.recipients,
                "received_at": mail_event.received_at.isoformat(),
            },
            "file_path": file_path,
        }

    def _build_excluded_entry(
        self,
        mail_event: MailEvent,
        mail_hash: str,
        retrieval_document: Any,
        file_path: str,
        quality: Any | None = None,
    ) -> dict[str, Any]:
        """Record a rejected pre-embedding sample without persisting a vector."""

        return {
            "subject": mail_event.subject,
            "sender": mail_event.sender,
            "mail_hash": mail_hash,
            "reason": (
                quality.reasons[0]
                if quality is not None and quality.reasons
                else retrieval_document.ineligible_reason
            ),
            "quality": quality.model_dump() if quality is not None else None,
            "retrieval_document": retrieval_document.model_dump(),
            "body_preview": mail_event.body[:_BODY_PREVIEW_LEN],
            "file_path": file_path,
        }

    def _generate_report(
        self,
        stage: str,
        samples: list[dict[str, Any]],
        input_count: int,
        excluded: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate markdown + JSON report files and return the report ID."""
        report_id = uuid.uuid4().hex[:12]
        reports_dir = Path(self.settings.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        tier_counts: dict[str, dict[str, int]] = {
            "tier1": {"count": 0, "auto_confirmed": 0},
            "tier2": {"count": 0, "auto_confirmed": 0},
            "tier3": {"count": 0, "auto_confirmed": 0},
        }
        for s in samples:
            tier = s["tier"]
            if tier in tier_counts:
                tier_counts[tier]["count"] += 1

        quality_summary = {"accepted": 0, "warned": 0, "rejected": 0}
        for entry in [*samples, *(excluded or [])]:
            quality = entry.get("quality") or {}
            disposition = quality.get("disposition")
            if disposition in quality_summary:
                quality_summary[disposition] += 1

        report_data = {
            "job_id": report_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "input_count": input_count,
            "tiers": tier_counts,
            "quality_summary": quality_summary,
            "samples": samples,
            "excluded": excluded or [],
        }

        json_path = reports_dir / f"bootstrap_{report_id}.json"
        json_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        md_path = reports_dir / f"bootstrap_{report_id}.md"
        md_path.write_text(self._render_markdown(report_data), encoding="utf-8")

        return report_id

    def _proposed_flat_category(
        self,
        *,
        tier: str,
        rule_result: RuleResult | None,
        llm_match: dict[str, Any],
    ) -> str:
        """Resolve the category that the current tier would persist."""

        selected = rule_result.selected if rule_result is not None else None
        if tier in {"tier1", "tier2"} and selected is not None:
            return selected.label
        return str(llm_match.get("l1") or "")

    def _assess_proposed_quality(
        self,
        document: RetrievalDocument,
        *,
        tier: str,
        rule_result: RuleResult | None,
        llm_match: dict[str, Any],
    ) -> Any | None:
        """Apply flat-taxonomy quality gates before scheduling embeddings."""

        if self.taxonomy_loader is None:
            return None
        valid_labels = self.taxonomy_loader.get_tree().all_codes()
        return assess_sample_quality(
            document,
            label_l1=self._proposed_flat_category(
                tier=tier, rule_result=rule_result, llm_match=llm_match
            ),
            valid_labels=valid_labels,
        )

    def _render_markdown(self, data: dict[str, Any]) -> str:
        """Render the markdown report from report data."""
        lines: list[str] = []
        lines.append(f"# Bootstrap Report — {data['job_id']}")
        lines.append("")
        lines.append("## Overview")
        lines.append(f"- Stage: {data['stage']}")
        lines.append(f"- Created: {data['created_at']}")
        lines.append(f"- Input count: {data['input_count']}")
        lines.append("")

        for tier_name, tier_label in [
            ("tier1", "Tier 1 (auto-confirm candidates)"),
            ("tier2", "Tier 2 (rule + LLM verify)"),
            ("tier3", "Tier 3 (LLM only)"),
        ]:
            tier_data = data["tiers"].get(tier_name, {})
            count = tier_data.get("count", 0)
            lines.append(f"## {tier_label}")
            lines.append(f"Count: {count}")
            tier_samples = [s for s in data["samples"] if s["tier"] == tier_name]
            for s in tier_samples:
                summary = f"- [{s['action']}] {s['subject']} — {s['sender']}"
                if tier_name == "tier2":
                    rule_label = (s.get("rule_match") or {}).get("label", "none")
                    llm_match = s.get("llm_match") or {}
                    llm_label = (
                        llm_match.get("l3")
                        or llm_match.get("l2")
                        or llm_match.get("l1")
                        or "unavailable"
                    )
                    summary += (
                        f" | rule suggestion: {rule_label}"
                        f" | LLM suggestion: {llm_label}"
                        f" | verification: {s.get('verification_status', 'unavailable')}"
                        " | mandatory individual review"
                    )
                lines.append(summary)
                retrieval = s.get("retrieval_document") or {}
                if retrieval:
                    lines.append(f"  - Retrieval policy: {retrieval.get('policy_version')}")
                    lines.append(f"  - Retrieval text: {retrieval.get('text', '')}")
            lines.append("")

        lines.append("## Excluded before embedding")
        lines.append(f"Count: {len(data.get('excluded', []))}")
        for excluded in data.get("excluded", []):
            lines.append(
                f"- [excluded] {excluded['subject']} — {excluded['sender']}"
                f" | reason: {excluded.get('reason', 'unknown')}"
            )
        lines.append("")

        lines.append("## Statistics")
        total = sum(t.get("count", 0) for t in data["tiers"].values())
        lines.append(f"- Total samples: {total}")
        for tier_name, tier_data in data["tiers"].items():
            lines.append(f"- {tier_name}: {tier_data.get('count', 0)}")
        quality_summary = data.get("quality_summary", {})
        lines.append(f"- Quality accepted: {quality_summary.get('accepted', 0)}")
        lines.append(f"- Quality warned: {quality_summary.get('warned', 0)}")
        lines.append(f"- Quality rejected: {quality_summary.get('rejected', 0)}")
        lines.append("")

        return "\n".join(lines)

    def _load_report(self, report_id: str) -> dict[str, Any]:
        """Load a report JSON by report_id."""
        report_path = Path(self.settings.reports_dir) / f"bootstrap_{report_id}.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        return json.loads(report_path.read_text(encoding="utf-8"))

    def _tier_source(self, tier: int, stage: str) -> str:
        """Determine the source field for a sample based on tier and stage."""
        if stage == "seed":
            return "seed"
        if tier == 1:
            return "rule_tier1"
        if tier == 2:
            return "rule_tier2"
        return "llm"

    def _entry_to_sample(
        self,
        entry: dict[str, Any],
        source: str,
    ) -> SampleRecord:
        """Convert a report sample entry to a SampleRecord for persistence."""
        mail_event_data = entry["mail_event"]
        llm_match = entry.get("llm_match")
        rule_match = entry.get("rule_match")
        rule_label = rule_match["label"] if rule_match else None

        if source == "rule_tier2" and rule_label:
            label_l1, _label_l2, _label_l3 = _split_label_path(rule_label)
        else:
            label_l1, _label_l2, _label_l3 = _label_dict_to_fields(
                llm_match, rule_label
            )

        confidence = 1.0
        if source == "rule_tier1" and rule_match:
            confidence = rule_match["confidence"]
        elif source == "rule_tier2" and rule_match:
            confidence = rule_match["confidence"]
        elif llm_match:
            confidence = llm_match.get("confidence", 0.8)

        sender = mail_event_data["sender"]
        retrieval_raw = entry.get("retrieval_document")
        if self.taxonomy_loader is not None and isinstance(retrieval_raw, dict):
            document = RetrievalDocument.model_validate(retrieval_raw)
            valid_labels = self.taxonomy_loader.get_tree().all_codes()
            quality = assess_sample_quality(
                document,
                label_l1=label_l1,
                valid_labels=valid_labels,
            )
            if quality.disposition != "accepted":
                raise ValueError(
                    "sample rejected by flat quality gate: " + ", ".join(quality.reasons)
                )
            return SampleRecord(
                mail_hash=entry["mail_hash"],
                subject_raw=mail_event_data["subject"],
                subject_clean=normalize_subject(mail_event_data["subject"]).clean,
                sender=sender,
                sender_domain=_extract_domain(sender),
                body=mail_event_data["body"],
                label_l1=label_l1,
                confidence=confidence,
                source=source,  # type: ignore[arg-type]
                reviewed=True,
                thread_parsed=True,
                taxonomy_schema_version=quality.taxonomy_schema_version,
                retrieval_document=document.model_dump(),
                retrieval_fingerprint=quality.fingerprint,
                retrieval_policy_version=quality.retrieval_policy_version,
                quality=quality,
                review_override_reason=entry.get("override_reason"),
            )

        return SampleRecord(
            mail_hash=entry["mail_hash"],
            subject_raw=mail_event_data["subject"],
            subject_clean=mail_event_data["subject"],
            sender=sender,
            sender_domain=_extract_domain(sender),
            body=mail_event_data["body"],
            label_l1=label_l1,
            label_l2=_label_l2,
            label_l3=_label_l3,
            confidence=confidence,
            source=source,  # type: ignore[arg-type]
            reviewed=True,
            thread_parsed=True,
            # Bootstrap reports retain the legacy record shape until their
            # flat-category review flow can validate taxonomy and quality.
            # Such rows are auditable but excluded from active vector retrieval.
            taxonomy_schema_version="legacy-v3",
            review_override_reason=entry.get("override_reason"),
        )
