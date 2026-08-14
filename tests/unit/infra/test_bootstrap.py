"""Unit tests for BootstrapPipeline.

Covers Stage 1 (seed), Stage 2 (import_history with tiered labeling),
confirm_tier (with tier-1-only batch restriction), dry-run, --force overwrite,
and 12-month archive. EmbeddingClient, LLM classifier, and RuleClassifier are
mocked; VectorStore uses an in-memory SQLite engine.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
)
from mailagent.domain.models import (
    MailEvent,
    RuleMatch,
    RuleResult,
    SampleRecord,
    TaxonomyLabel,
)
from mailagent.infra.bootstrap import (
    BootstrapPipeline,
    _extract_attachments,
    _extract_domain,
)
from mailagent.infra.config import BootstrapSettings, VectorStoreSettings
from mailagent.infra.store import Base, SampleArchiveORM
from mailagent.infra.vector_store import VectorStore
from mailagent.llm.embedding import EmbeddingClient

_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding(seed: float) -> list[float]:
    """Deterministic embedding vector for testing."""
    return [seed + i * 0.01 for i in range(_DIM)]


def _make_llm_attempt(
    l1: str = "schedule",
    l2: str | None = None,
    l3: str | None = None,
    confidence: float = 0.85,
) -> ClassificationAttempt:
    """Build a successful LLM ClassificationAttempt with one label.

    扁平 taxonomy：l2/l3 均为 None，仅 l1_code 有值。
    """
    return ClassificationAttempt(
        source="llm",
        status=AttemptStatus.SUCCESS,
        labels=[
            TaxonomyLabel(
                l1_code=l1,
                l1_label=l1,
                l2_code=None,
                l2_label=None,
                l3_code=None,
                l3_label=None,
                confidence=confidence,
            )
        ],
        confidence=confidence,
    )


def _make_unavailable_llm_attempt(error: str = "model unavailable") -> ClassificationAttempt:
    """Build an unavailable LLM attempt with no usable label."""
    return ClassificationAttempt(
        source="llm",
        status=AttemptStatus.UNAVAILABLE,
        error=error,
    )


def _make_medium_rule_result(label: str = "schedule") -> RuleResult:
    """Build one Tier 2 rule result."""
    match = RuleMatch(
        rule_type="sender_domains",
        label=label,
        confidence=0.80,
        matched_pattern="domain=midconf.com",
    )
    return RuleResult(matches=[match], selected=match)


def _make_fixed_rule_classifier(rule_result: RuleResult) -> MagicMock:
    """Build a rule classifier that always returns one controlled result."""
    classifier = _make_empty_rule_classifier()
    classifier._sender_domain_rules = [MagicMock()]
    classifier.match = MagicMock(return_value=rule_result)
    return classifier


def _write_eml(
    path: Path,
    sender: str,
    subject: str,
    body: str = "Test body content.",
) -> None:
    """Write a minimal .eml file using the stdlib email module."""
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = "recipient@example.com"
    msg["Subject"] = subject
    msg["Date"] = "Mon, 21 Jul 2026 10:00:00 +0000"
    msg["Message-ID"] = f"<{uuid4().hex}@example.com>"
    msg.set_content(body)
    path.write_bytes(bytes(msg))


def _make_empty_rule_classifier() -> MagicMock:
    """A mock RuleClassifier with no rules loaded (triggers empty-rules warning)."""
    clf = MagicMock()
    clf._sender_domain_rules = []
    clf._subject_pattern_rules = []
    clf._body_keyword_rules = []
    clf._structural_rules = []
    clf.match = MagicMock(return_value=RuleResult(matches=[], selected=None))
    return clf


def _make_tiered_rule_classifier() -> MagicMock:
    """A mock RuleClassifier with tier1 (0.95) and tier2 (0.80) sender-domain rules.

    扁平 taxonomy：label 使用单个 code（如 "schedule"）。
    - highconf.com  → label schedule, conf 0.95 (Tier 1)
    - midconf.com   → label operation,  conf 0.80 (Tier 2)
    - any other     → no match (Tier 3, LLM fallback)
    """
    clf = MagicMock()
    # Non-empty _sender_domain_rules so _check_rules_warning does not fire.
    clf._sender_domain_rules = [MagicMock()]
    clf._subject_pattern_rules = []
    clf._body_keyword_rules = []
    clf._structural_rules = []

    def match(mail_event, context=None):  # noqa: ANN001
        domain = _extract_domain(mail_event.sender)
        if domain == "highconf.com":
            rm = RuleMatch(
                rule_type="sender_domains",
                label="schedule",
                confidence=0.95,
                matched_pattern="domain=highconf.com",
            )
            return RuleResult(matches=[rm], selected=rm)
        if domain == "midconf.com":
            rm = RuleMatch(
                rule_type="sender_domains",
                label="operation",
                confidence=0.80,
                matched_pattern="domain=midconf.com",
            )
            return RuleResult(matches=[rm], selected=rm)
        return RuleResult(matches=[], selected=None)

    clf.match = match
    return clf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bootstrap_logger():
    """Re-enable the bootstrap logger in case alembic's fileConfig disabled it.

    Alembic's ``logging.config.fileConfig`` (called by migration tests) sets
    ``disable_existing_loggers=True`` by default, which marks every logger not
    listed in ``alembic.ini``'s ``[loggers]`` section as ``disabled=True``.
    This breaks ``caplog`` assertions in tests that run after any migration
    test. Resetting ``disabled`` here keeps caplog working regardless of test
    ordering.
    """
    logging.getLogger("mailagent.infra.bootstrap").disabled = False
    yield


@pytest.fixture
async def engine():
    """In-memory SQLite engine with all tables created."""
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def vector_store(engine) -> VectorStore:
    return VectorStore(VectorStoreSettings(), engine)


@pytest.fixture
def embedding_client() -> MagicMock:
    """Mock EmbeddingClient whose embed_batch returns two DIM-length vectors."""
    client = MagicMock(spec=EmbeddingClient)
    client.embed_batch = AsyncMock(
        return_value=[_embedding(0.1), _embedding(0.2)]
    )
    return client


@pytest.fixture
def llm_classifier() -> MagicMock:
    """Mock LLM classifier returning a successful attempt with one label."""
    clf = MagicMock()
    clf.source = "llm"
    clf.classify = AsyncMock(return_value=_make_llm_attempt())
    return clf


@pytest.fixture
def bootstrap_settings(tmp_path: Path) -> BootstrapSettings:
    return BootstrapSettings(
        weekly_batch_size=4200,
        default_batch_size=50,
        reports_dir=str(tmp_path / "reports"),
    )


# ---------------------------------------------------------------------------
# Test: _extract_attachments helper
# ---------------------------------------------------------------------------


class TestExtractAttachments:
    """Attachment filename extraction from email.message.Message objects."""

    def test_extracts_filenames_from_multipart_with_attachment_disposition(self) -> None:
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "DG list"
        msg["From"] = "ops@example.com"
        msg.set_content("Please find the DG list attached.")
        msg.add_attachment(
            b"fake bytes",
            maintype="application",
            subtype="octet-stream",
            filename="DG_List_MV_PACIFIC.xlsx",
        )
        msg.add_attachment(
            b"fake bytes",
            maintype="application",
            subtype="pdf",
            filename="Crew_List.pdf",
        )

        attachments = _extract_attachments(msg)

        assert attachments == ["DG_List_MV_PACIFIC.xlsx", "Crew_List.pdf"]

    def test_skips_inline_parts_without_attachment_disposition(self) -> None:
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "Plain"
        msg["From"] = "ops@example.com"
        msg.set_content("Plain body with no attachments.")

        attachments = _extract_attachments(msg)

        assert attachments == []

    def test_skips_inline_image_without_filename(self) -> None:
        """Inline images with no filename are not attachment-bearing for retrieval."""
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "Logo"
        msg["From"] = "ops@example.com"
        msg.set_content("See our logo inline.")
        msg.add_attachment(
            b"fake image bytes",
            maintype="image",
            subtype="png",
            filename=None,
        )

        attachments = _extract_attachments(msg)

        assert attachments == []


# ---------------------------------------------------------------------------
# Test: seed (Stage 1)
# ---------------------------------------------------------------------------


class TestSeed:
    async def test_confirmed_retrieval_text_override_is_reembedded_without_mutating_raw_mail(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        eml_dir = tmp_path / "seed_emails"
        eml_dir.mkdir()
        _write_eml(
            eml_dir / "schedule.eml",
            "ops@example.com",
            "STATUS Update",
            body="Original mail body with the location request.",
        )
        taxonomy_loader = MagicMock()
        taxonomy_loader.get_tree.return_value.all_codes.return_value = {"schedule"}
        pipeline = BootstrapPipeline(
            rule_classifier=_make_empty_rule_classifier(),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
            taxonomy_loader=taxonomy_loader,
        )

        report_id = await pipeline.seed(eml_dir, no_rules=True)
        report_path = Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["samples"][0]["retrieval_text_override"] = (
            "Subject: STATUS Update\nLatest message:\nPlease arrange location at 14:00."
        )
        report["samples"][0]["override_reason"] = "retain operational request"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        await pipeline.confirm_tier(report_id, tier=3)
        sample = (await vector_store.get_samples())[0]

        assert sample.retrieval_document is not None
        assert sample.retrieval_document["text"].endswith("Please arrange location at 14:00.")
        assert sample.review_override_reason == "retain operational request"
        assert sample.body == "Original mail body with the location request.\n"
        assert embedding_client.embed_batch.await_count == 2

    async def test_confirmed_seed_with_taxonomy_becomes_accepted_flat_sample(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        eml_dir = tmp_path / "seed_emails"
        eml_dir.mkdir()
        _write_eml(
            eml_dir / "schedule.eml",
            "ops@example.com",
            "STATUS Update",
            body="STATUS revised to 14:00 tomorrow. Please arrange location.",
        )
        taxonomy_loader = MagicMock()
        taxonomy_loader.get_tree.return_value.all_codes.return_value = {"schedule"}
        pipeline = BootstrapPipeline(
            rule_classifier=_make_empty_rule_classifier(),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
            taxonomy_loader=taxonomy_loader,
        )

        report_id = await pipeline.seed(eml_dir, no_rules=True)
        await pipeline.confirm_tier(report_id, tier=3)
        samples = await vector_store.get_samples()

        assert len(samples) == 1
        assert samples[0].label_l1 == "schedule"
        assert samples[0].label_l2 is None and samples[0].label_l3 is None
        assert samples[0].quality is not None
        assert samples[0].quality.disposition == "accepted"
        assert samples[0].taxonomy_schema_version == "flat-v1"

    async def test_seed_excludes_attachment_only_mail_before_embedding_or_llm(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        eml_dir = tmp_path / "seed_emails"
        eml_dir.mkdir()
        _write_eml(
            eml_dir / "attachment.eml",
            "ops@example.com",
            "Document attached",
            body="Please see attached document.",
        )
        pipeline = BootstrapPipeline(
            rule_classifier=_make_empty_rule_classifier(),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        report_id = await pipeline.seed(eml_dir, no_rules=True)
        report = json.loads(
            (Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id}.json")
            .read_text(encoding="utf-8")
        )

        assert report["samples"] == []
        assert report["excluded"][0]["reason"] == "attachment_dependent"
        embedding_client.embed_batch.assert_not_awaited()
        llm_classifier.classify.assert_not_awaited()

    async def test_seed_excludes_unknown_flat_category_before_embedding(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """A taxonomy-invalid proposal is reported without spending an embedding call."""

        eml_dir = tmp_path / "seed_emails"
        eml_dir.mkdir()
        _write_eml(
            eml_dir / "unknown-category.eml",
            "ops@example.com",
            "STATUS Update",
            body="STATUS revised to 14:00 tomorrow. Please arrange location.",
        )
        llm_classifier.classify = AsyncMock(return_value=_make_llm_attempt("unknown"))
        taxonomy_loader = MagicMock()
        taxonomy_loader.get_tree.return_value.all_codes.return_value = {"schedule"}
        pipeline = BootstrapPipeline(
            rule_classifier=_make_empty_rule_classifier(),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
            taxonomy_loader=taxonomy_loader,
        )

        report_id = await pipeline.seed(eml_dir, no_rules=True)
        report = json.loads(
            (Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id}.json")
            .read_text(encoding="utf-8")
        )

        assert report["samples"] == []
        assert report["excluded"][0]["reason"] == "unknown_taxonomy_category"
        assert report["excluded"][0]["quality"]["disposition"] == "rejected"
        assert report["quality_summary"] == {"accepted": 0, "warned": 0, "rejected": 1}
        embedding_client.embed_batch.assert_not_awaited()

    async def test_seed_generates_report_with_llm_annotation(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Stage 1 seed: no-rules warning fires, LLM annotates, report generated."""
        eml_dir = tmp_path / "seed_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "a.eml", "ops@example.com", "STATUS Update")

        rule_clf = _make_empty_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        with caplog.at_level(logging.WARNING, logger="mailagent.infra.bootstrap"):
            report_id = await pipeline.seed(eml_dir, force=False, no_rules=False)

        # Report ID returned.
        assert isinstance(report_id, str)
        assert len(report_id) == 12

        # Empty-rules warning was logged.
        assert any("未检测到规则文件" in r.getMessage() for r in caplog.records)

        # Both markdown and JSON reports exist.
        reports_dir = Path(bootstrap_settings.reports_dir)
        assert (reports_dir / f"bootstrap_{report_id}.json").exists()
        assert (reports_dir / f"bootstrap_{report_id}.md").exists()

        # JSON report has correct structure.
        report = json.loads(
            (reports_dir / f"bootstrap_{report_id}.json").read_text(encoding="utf-8")
        )
        assert report["stage"] == "seed"
        assert report["input_count"] == 1
        assert len(report["samples"]) == 1
        assert report["samples"][0]["tier"] == "tier3"
        assert report["samples"][0]["llm_match"]["l1"] == "schedule"
        assert report["samples"][0]["llm_match"]["l3"] is None
        assert report["samples"][0]["retrieval_document"]["text"].startswith(
            "Subject: STATUS Update\nLatest message:"
        )

        # LLM classify was called once.
        llm_classifier.classify.assert_awaited_once()

    async def test_seed_no_rules_flag_skips_warning(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """--no-rules flag suppresses the empty-rules warning."""
        eml_dir = tmp_path / "seed_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "a.eml", "ops@example.com", "STATUS Update")

        rule_clf = _make_empty_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        with caplog.at_level(logging.WARNING, logger="mailagent.infra.bootstrap"):
            await pipeline.seed(eml_dir, force=False, no_rules=True)

        # No empty-rules warning.
        assert not any("未检测到规则文件" in r.getMessage() for r in caplog.records)

    async def test_seed_force_overwrites_existing_mail_hash(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """--force re-runs overwrite previously confirmed samples with the same hash."""
        eml_dir = tmp_path / "seed_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "a.eml", "ops@example.com", "STATUS Update")

        rule_clf = _make_empty_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        # First seed → 1 sample in report.
        report_id_1 = await pipeline.seed(eml_dir, force=False, no_rules=True)
        report_1 = json.loads(
            (Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id_1}.json")
            .read_text(encoding="utf-8")
        )
        assert len(report_1["samples"]) == 1

        # Confirm tier 3 → sample persisted to DB.
        count = await pipeline.confirm_tier(report_id_1, tier=3, all_=False)
        assert count == 1
        assert await vector_store.count_samples() == 1

        # Second seed WITHOUT --force → sample skipped (mail_hash exists in DB).
        report_id_2 = await pipeline.seed(eml_dir, force=False, no_rules=True)
        report_2 = json.loads(
            (Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id_2}.json")
            .read_text(encoding="utf-8")
        )
        assert len(report_2["samples"]) == 0  # Skipped.

        # Third seed WITH --force → sample re-annotated (not skipped).
        report_id_3 = await pipeline.seed(eml_dir, force=True, no_rules=True)
        report_3 = json.loads(
            (Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id_3}.json")
            .read_text(encoding="utf-8")
        )
        assert len(report_3["samples"]) == 1  # Not skipped.


# ---------------------------------------------------------------------------
# Test: import_history (Stage 2 — tiered labeling)
# ---------------------------------------------------------------------------


class TestImportHistory:
    async def test_import_history_three_tier_classification(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """Stage 2 import: three .eml files → tier1, tier2, tier3 respectively."""
        eml_dir = tmp_path / "week_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "tier1.eml", "ops@highconf.com", "STATUS Update")
        _write_eml(eml_dir / "tier2.eml", "ops@midconf.com", "Location Plan")
        _write_eml(eml_dir / "tier3.eml", "ops@nolabel.com", "General Inquiry")

        rule_clf = _make_tiered_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        report_id = await pipeline.import_history(eml_dir, batch_size=50)

        report = json.loads(
            (Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id}.json")
            .read_text(encoding="utf-8")
        )
        assert report["stage"] == "import"
        assert report["input_count"] == 3

        tier_counts = {t: 0 for t in ("tier1", "tier2", "tier3")}
        for s in report["samples"]:
            tier_counts[s["tier"]] += 1
        assert tier_counts == {"tier1": 1, "tier2": 1, "tier3": 1}

        # Tier 1 remains rule-only.
        t1 = next(s for s in report["samples"] if s["tier"] == "tier1")
        assert t1["rule_match"]["label"] == "schedule"
        assert t1["rule_match"]["confidence"] == pytest.approx(0.95)
        assert t1["llm_match"] is None

        # Tier 2 retains the rule suggestion and records the LLM disagreement.
        t2 = next(s for s in report["samples"] if s["tier"] == "tier2")
        assert t2["rule_match"]["label"] == "operation"
        assert t2["llm_match"]["l1"] == "schedule"
        assert t2["verification_status"] == "disagreed"

        # Tier 3 has llm_match but no rule_match.
        t3 = next(s for s in report["samples"] if s["tier"] == "tier3")
        assert t3["rule_match"] is None
        assert t3["llm_match"]["l3"] is None

        # LLM verifies Tier 2 and labels Tier 3, but Tier 2 is not persisted.
        assert llm_classifier.classify.await_count == 2
        assert await vector_store.count_samples() == 0


class TestTier2Verification:
    async def test_matching_llm_label_confirms_report_evidence(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """A matching Tier 2 report retains both suggestions without persistence."""
        eml_dir = tmp_path / "matching"
        eml_dir.mkdir()
        _write_eml(eml_dir / "tier2.eml", "ops@midconf.com", "Schedule")
        pipeline = BootstrapPipeline(
            rule_classifier=_make_fixed_rule_classifier(
                _make_medium_rule_result("schedule")
            ),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        report_id = await pipeline.import_history(eml_dir)
        entry = pipeline._load_report(report_id)["samples"][0]

        assert entry["tier"] == "tier2"
        assert entry["rule_match"]["label"] == "schedule"
        assert entry["llm_match"]["l1"] == "schedule"
        assert entry["verification_status"] == "confirmed"
        assert await vector_store.count_samples() == 0

    async def test_disagreeing_llm_label_preserves_report_rule_suggestion(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """A disagreement remains Tier 2 with both suggestions for review."""
        eml_dir = tmp_path / "disagreeing"
        eml_dir.mkdir()
        _write_eml(eml_dir / "tier2.eml", "ops@midconf.com", "Operation")
        pipeline = BootstrapPipeline(
            rule_classifier=_make_fixed_rule_classifier(
                _make_medium_rule_result("operation")
            ),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        report_id = await pipeline.import_history(eml_dir)
        entry = pipeline._load_report(report_id)["samples"][0]

        assert entry["tier"] == "tier2"
        assert entry["rule_match"]["label"] == "operation"
        assert entry["llm_match"]["l1"] == "schedule"
        assert entry["verification_status"] == "disagreed"
        assert await vector_store.count_samples() == 0

    async def test_unavailable_llm_attempt_records_report_failure_detail(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """A non-success LLM attempt remains Tier 2 with unavailable evidence."""
        eml_dir = tmp_path / "unavailable"
        eml_dir.mkdir()
        _write_eml(eml_dir / "tier2.eml", "ops@midconf.com", "Schedule")
        llm_classifier.classify = AsyncMock(
            return_value=_make_unavailable_llm_attempt("model unavailable")
        )
        pipeline = BootstrapPipeline(
            rule_classifier=_make_fixed_rule_classifier(
                _make_medium_rule_result("schedule")
            ),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        report_id = await pipeline.import_history(eml_dir)
        entry = pipeline._load_report(report_id)["samples"][0]

        assert entry["tier"] == "tier2"
        assert entry["rule_match"]["label"] == "schedule"
        assert entry["llm_match"] is None
        assert entry["verification_status"] == "unavailable"
        assert entry["verification_detail"] == "model unavailable"
        assert await vector_store.count_samples() == 0

    async def test_llm_exception_records_unavailable_without_changing_tier(
        self,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """A classifier exception cannot promote, discard, or persist Tier 2."""
        llm_classifier.classify = AsyncMock(side_effect=RuntimeError("transport down"))
        pipeline = BootstrapPipeline(
            rule_classifier=_make_empty_rule_classifier(),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )
        mail = MailEvent(
            message_id="tier2-exception",
            sender="ops@midconf.com",
            subject="Schedule",
            body="Schedule update",
        )

        tier, _, evidence = await pipeline._tier_classify(
            mail, _make_medium_rule_result("schedule")
        )

        assert tier == "tier2"
        assert evidence["llm_match"] is None
        assert evidence["verification_status"] == "unavailable"
        assert evidence["verification_detail"] == "transport down"
        assert await vector_store.count_samples() == 0

    async def test_success_attempt_without_usable_code_is_unavailable(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """A nominal success with a blank label cannot become disagreement."""
        eml_dir = tmp_path / "blank-code"
        eml_dir.mkdir()
        _write_eml(eml_dir / "tier2.eml", "ops@midconf.com", "Schedule")
        llm_classifier.classify = AsyncMock(return_value=_make_llm_attempt(l1=""))
        pipeline = BootstrapPipeline(
            rule_classifier=_make_fixed_rule_classifier(
                _make_medium_rule_result("schedule")
            ),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        report_id = await pipeline.import_history(eml_dir)
        entry = pipeline._load_report(report_id)["samples"][0]

        assert entry["tier"] == "tier2"
        assert entry["rule_match"]["label"] == "schedule"
        assert entry["llm_match"] is None
        assert entry["verification_status"] == "unavailable"
        assert entry["verification_detail"] == "LLM returned no usable labels"
        assert await vector_store.count_samples() == 0

    async def test_invalid_llm_return_object_is_unavailable(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """A malformed classifier return cannot abort the Tier 2 import."""
        eml_dir = tmp_path / "invalid-return"
        eml_dir.mkdir()
        _write_eml(eml_dir / "tier2.eml", "ops@midconf.com", "Schedule")
        llm_classifier.classify = AsyncMock(return_value=None)
        pipeline = BootstrapPipeline(
            rule_classifier=_make_fixed_rule_classifier(
                _make_medium_rule_result("schedule")
            ),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        report_id = await pipeline.import_history(eml_dir)
        entry = pipeline._load_report(report_id)["samples"][0]

        assert entry["tier"] == "tier2"
        assert entry["rule_match"]["label"] == "schedule"
        assert entry["llm_match"] is None
        assert entry["verification_status"] == "unavailable"
        assert entry["verification_detail"] == (
            "Malformed LLM result: expected ClassificationAttempt"
        )
        assert await vector_store.count_samples() == 0


# ---------------------------------------------------------------------------
# Test: confirm_tier
# ---------------------------------------------------------------------------


class TestConfirmTier:
    async def test_confirm_tier1_all_succeeds(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """confirm --tier 1 --all persists all Tier 1 samples with source=rule_tier1."""
        eml_dir = tmp_path / "week_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "t1.eml", "ops@highconf.com", "STATUS Update")

        rule_clf = _make_tiered_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )
        report_id = await pipeline.import_history(eml_dir)

        count = await pipeline.confirm_tier(report_id, tier=1, all_=True)
        assert count == 1
        assert await vector_store.count_samples() == 1

        samples = await vector_store.get_samples()
        assert samples[0].source == "rule_tier1"
        assert samples[0].label_l3 == "schedule"
        assert samples[0].reviewed is True

    async def test_confirm_tier2_all_rejected(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """confirm --tier 2 --all raises ValueError directing user to review."""
        eml_dir = tmp_path / "week_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "t2.eml", "ops@midconf.com", "Location Plan")

        rule_clf = _make_tiered_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )
        report_id = await pipeline.import_history(eml_dir)

        with pytest.raises(ValueError, match="逐条审核"):
            await pipeline.confirm_tier(report_id, tier=2, all_=True)

        # Nothing persisted.
        assert await vector_store.count_samples() == 0

    async def test_confirm_tier2_without_review_rejected(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """The report-wide confirm path cannot masquerade as individual review."""
        eml_dir = tmp_path / "week_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "t2.eml", "ops@midconf.com", "Location Plan")
        pipeline = BootstrapPipeline(
            rule_classifier=_make_tiered_rule_classifier(),
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )
        report_id = await pipeline.import_history(eml_dir)

        with pytest.raises(ValueError, match="逐条审核"):
            await pipeline.confirm_tier(report_id, tier=2, all_=False)

        assert await vector_store.count_samples() == 0

    async def test_confirm_dry_run_no_db_writes(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """confirm --dry-run returns count but does not persist to DB."""
        eml_dir = tmp_path / "week_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "t1.eml", "ops@highconf.com", "STATUS Update")

        rule_clf = _make_tiered_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )
        report_id = await pipeline.import_history(eml_dir)

        count = await pipeline.confirm_tier(
            report_id, tier=1, all_=True, dry_run=True
        )
        assert count == 1
        assert await vector_store.count_samples() == 0


# ---------------------------------------------------------------------------
# Test: archive_old_samples
# ---------------------------------------------------------------------------


class TestArchive:
    async def test_archive_moves_old_samples_and_deletes_from_active(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """archive_old_samples moves old rows to samples_archive and deletes from active."""
        rule_clf = _make_empty_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        # Insert one old sample (400 days ago) and one recent (1 day ago).
        old_sample = SampleRecord(
            mail_hash="hash-old",
            subject_raw="Old subject",
            subject_clean="old subject",
            sender="ops@old.com",
            sender_domain="old.com",
            body="Old body",
            label_l1="schedule",
            label_l2="schedule",
            label_l3="schedule",
            confidence=0.9,
            source="seed",
            created_at=datetime.now(timezone.utc) - timedelta(days=400),
        )
        recent_sample = SampleRecord(
            mail_hash="hash-recent",
            subject_raw="Recent subject",
            subject_clean="recent subject",
            sender="ops@recent.com",
            sender_domain="recent.com",
            body="Recent body",
            label_l1="operation",
            label_l2="operation",
            label_l3="operation",
            confidence=0.9,
            source="seed",
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        emb = _embedding(0.5)
        await vector_store.insert_sample(old_sample, emb, emb)
        await vector_store.insert_sample(recent_sample, emb, emb)
        assert await vector_store.count_samples() == 2

        # Archive samples older than 12 months.
        moved = await pipeline.archive_old_samples(months=12)
        assert moved == 1
        assert await vector_store.count_samples() == 1

        # Verify the old sample is in the archive table.
        async with vector_store.sessions() as session:
            archived = (await session.scalars(select(SampleArchiveORM))).all()
        assert len(archived) == 1
        assert archived[0].mail_hash == "hash-old"
        assert archived[0].label_l3 == "schedule"
