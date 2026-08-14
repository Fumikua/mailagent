"""End-to-end integration tests for BootstrapPipeline.seed (Section 18.5).

Stage 1 full flow:
  .eml files → preprocess_mail (mock embeddings) → LLM annotation (mock)
  → report generation → confirm_tier → samples persisted to in-memory SQLite.

Uses mock LLM + SQLite in-memory database. Embeddings are mocked to a
deterministic vector; no real TEI / LLM calls are made.
"""
from __future__ import annotations

import json
import logging
import typing
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
)
from mailagent.domain.models import TaxonomyLabel
from mailagent.infra.bootstrap import BootstrapPipeline
from mailagent.infra.config import BootstrapSettings, VectorStoreSettings
from mailagent.infra.store import Base
from mailagent.infra.vector_store import VectorStore
from mailagent.llm.embedding import EmbeddingClient

_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding(seed: float) -> list[float]:
    return [seed + i * 0.01 for i in range(_DIM)]


def _make_llm_attempt(
    l1: str = "entity",
    l2: str = "schedule",
    l3: str = "eta_update",
    confidence: float = 0.85,
) -> ClassificationAttempt:
    return ClassificationAttempt(
        source="llm",
        status=AttemptStatus.SUCCESS,
        labels=[
            TaxonomyLabel(
                l1_code=l1,
                l1_label=l1,
                l2_code=l2,
                l2_label=l2,
                l3_code=l3,
                l3_label=l3,
                confidence=confidence,
            )
        ],
        confidence=confidence,
    )


def _write_eml(
    path: Path,
    sender: str,
    subject: str,
    body: str = "Test body content.",
) -> None:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = "recipient@example.com"
    msg["Subject"] = subject
    msg["Date"] = "Mon, 21 Jul 2026 10:00:00 +0000"
    msg["Message-ID"] = f"<{uuid4().hex}@example.com>"
    msg.set_content(body)
    path.write_bytes(bytes(msg))


def _make_empty_rule_classifier() -> MagicMock:
    """A mock RuleClassifier with no rules loaded."""
    from mailagent.domain.models import RuleResult

    clf = MagicMock()
    clf._sender_domain_rules = []
    clf._subject_pattern_rules = []
    clf._body_keyword_rules = []
    clf._structural_rules = []
    clf.match = MagicMock(return_value=RuleResult(matches=[], selected=None))
    return clf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine():
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
    client = MagicMock(spec=EmbeddingClient)
    client.embed_batch = AsyncMock(
        return_value=[_embedding(0.1), _embedding(0.2)]
    )
    return client


@pytest.fixture
def llm_classifier() -> MagicMock:
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


@pytest.fixture(autouse=True)
def _enable_bootstrap_logger() -> typing.Generator[None, None, None]:
    """Re-enable the bootstrap logger if a prior Alembic fileConfig disabled it.

    When TestClient starts up the API lifespan, ``upgrade_database`` triggers
    Alembic's ``fileConfig`` which sets ``disable_existing_loggers=True`` and
    silences the ``mailagent.infra.bootstrap`` logger.  This fixture restores
    the logger state so caplog can capture the empty-rules warning.
    """
    lg = logging.getLogger("mailagent.infra.bootstrap")
    prev_disabled = lg.disabled
    prev_propagate = lg.propagate
    lg.disabled = False
    lg.propagate = True
    yield
    lg.disabled = prev_disabled
    lg.propagate = prev_propagate


# ---------------------------------------------------------------------------
# Tests: Stage 1 seed flow
# ---------------------------------------------------------------------------


class TestBootstrapSeedIntegration:
    """Stage 1: .eml → LLM annotation → report → confirm → DB."""

    async def test_seed_full_flow_persists_samples(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Full Stage 1 flow: 3 .eml files → seed → confirm → 3 samples in DB."""
        # Arrange: 3 synthetic .eml files.
        eml_dir = tmp_path / "seed_emails"
        eml_dir.mkdir()
        _write_eml(eml_dir / "a.eml", "ops@example.com", "STATUS Update")
        _write_eml(eml_dir / "b.eml", "ops@example.com", "Status Report")
        _write_eml(eml_dir / "c.eml", "captain@entity.com", "Location Request")

        rule_clf = _make_empty_rule_classifier()
        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        # Stage 1: seed → all Tier 3 (no rules), LLM annotates every email.
        with caplog.at_level(logging.WARNING, logger="mailagent.infra.bootstrap"):
            report_id = await pipeline.seed(eml_dir, force=False, no_rules=False)

        # Empty-rules warning fires.
        assert any("未检测到规则文件" in r.getMessage() for r in caplog.records)

        # Report ID is a 12-char hex string.
        assert isinstance(report_id, str)
        assert len(report_id) == 12

        # Both markdown + JSON reports exist.
        reports_dir = Path(bootstrap_settings.reports_dir)
        json_path = reports_dir / f"bootstrap_{report_id}.json"
        md_path = reports_dir / f"bootstrap_{report_id}.md"
        assert json_path.exists()
        assert md_path.exists()

        # JSON report has correct structure.
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert report["stage"] == "seed"
        assert report["input_count"] == 3
        assert len(report["samples"]) == 3
        assert all(s["tier"] == "tier3" for s in report["samples"])
        assert all(s["llm_match"] is not None for s in report["samples"])
        assert all(s["llm_match"]["l3"] == "eta_update" for s in report["samples"])

        # LLM classify called once per .eml.
        assert llm_classifier.classify.await_count == 3

        # Pre-confirm: no samples in DB yet.
        assert await vector_store.count_samples() == 0

        # Confirm tier 3 individually (all_=False is required for tier 3).
        count = await pipeline.confirm_tier(report_id, tier=3, all_=False)
        assert count == 3

        # All 3 samples now persisted with source="seed".
        assert await vector_store.count_samples() == 3
        samples = await vector_store.get_samples()
        assert all(s.source == "seed" for s in samples)
        assert all(s.label_l3 == "eta_update" for s in samples)
        assert all(s.reviewed is True for s in samples)

    async def test_seed_no_rules_flag_suppresses_warning(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """--no-rules suppresses the empty-rules warning during seed."""
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

        assert not any("未检测到规则文件" in r.getMessage() for r in caplog.records)

    async def test_seed_force_re_annotates_existing_hash(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """--force re-annotates a .eml whose mail_hash already exists in DB."""
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

        # First seed + confirm → 1 sample in DB.
        report_id_1 = await pipeline.seed(eml_dir, force=False, no_rules=True)
        await pipeline.confirm_tier(report_id_1, tier=3, all_=False)
        assert await vector_store.count_samples() == 1

        # Second seed WITHOUT --force → sample skipped.
        report_id_2 = await pipeline.seed(eml_dir, force=False, no_rules=True)
        report_2 = json.loads(
            (Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id_2}.json")
            .read_text(encoding="utf-8")
        )
        assert len(report_2["samples"]) == 0

        # Third seed WITH --force → sample re-annotated.
        report_id_3 = await pipeline.seed(eml_dir, force=True, no_rules=True)
        report_3 = json.loads(
            (Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id_3}.json")
            .read_text(encoding="utf-8")
        )
        assert len(report_3["samples"]) == 1

    async def test_seed_markdown_report_has_correct_structure(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        embedding_client: MagicMock,
        llm_classifier: MagicMock,
        bootstrap_settings: BootstrapSettings,
    ) -> None:
        """Markdown report has the expected section structure."""
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

        report_id = await pipeline.seed(eml_dir, force=False, no_rules=True)
        md_path = Path(bootstrap_settings.reports_dir) / f"bootstrap_{report_id}.md"
        content = md_path.read_text(encoding="utf-8")

        assert "# Bootstrap Report" in content
        assert "## Overview" in content
        assert "## Tier 3 (LLM only)" in content
        assert "## Statistics" in content
