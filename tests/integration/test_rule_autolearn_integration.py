"""End-to-end integration tests for RuleLearner (Section 18.8).

Cross-domain verification: insert samples with a dominant (sender_domain, label)
pair plus 2+ other domains with the same label, run ``run_weekly_scan``, verify
the proposal report contains the correct checkbox, then apply it through the
real ``rules add`` Click command and verify the rule is appended to
``sender_domains.yaml``.

Uses a real SQLite ``VectorStore`` so the distribution matrix is computed from
real persisted rows.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
import yaml
from click.testing import CliRunner
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from mailagent.classification.rule_classifier import SenderDomainRule
from mailagent.domain.models import SampleQualityAssessment, SampleRecord
from mailagent.infra.cli import main
from mailagent.infra.config import RulesSettings, VectorStoreSettings
from mailagent.infra.rule_learner import RuleLearner
from mailagent.infra.store import Base
from mailagent.infra.vector_store import VectorStore

_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding(seed: float) -> list[float]:
    return [seed + i * 0.01 for i in range(_DIM)]


def _make_sample(
    sender_domain: str,
    label_l1: str,
    subject: str = "Test subject",
    days_ago: int = 1,
) -> SampleRecord:
    mail_hash = f"hash-{uuid4()}"
    fingerprint = hashlib.sha256(mail_hash.encode("utf-8")).hexdigest()
    return SampleRecord(
        mail_hash=mail_hash,
        subject_raw=subject,
        subject_clean=subject.lower(),
        sender=f"ops@{sender_domain}",
        sender_domain=sender_domain,
        body="Test body",
        label_l1=label_l1,
        label_l2=None,
        label_l3=None,
        confidence=0.9,
        source="seed",
        created_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        taxonomy_schema_version="flat-v1",
        retrieval_document={"text": subject},
        retrieval_fingerprint=fingerprint,
        retrieval_policy_version="example-triage-v1",
        quality=SampleQualityAssessment(
            disposition="accepted",
            fingerprint=fingerprint,
            retrieval_policy_version="example-triage-v1",
        ),
    )


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
def rules_settings(tmp_path: Path) -> RulesSettings:
    """RulesSettings pointing at the selected test vertical's rules asset."""
    vertical_dir = tmp_path / "verticals" / "example_triage"
    rules_dir = vertical_dir / "rules"
    rules_dir.mkdir(parents=True)
    (vertical_dir / "manifest.yaml").write_text(
        "id: example-triage\n"
        "namespace: example_triage\n"
        'data_schema_version: "1"\n'
        "taxonomy: taxonomy.yaml\n"
        "data_schema: data-schema.json\n"
        "runtime_factory: mailagent.verticals.runtime:build_empty_runtime\n"
        "rules:\n"
        "  path: rules\n"
        '  version: "1"\n',
        encoding="utf-8",
    )
    (vertical_dir / "taxonomy.yaml").write_text(
        "nodes:\n"
        "  - code: eta_update\n"
        "    label: STATUS update\n",
        encoding="utf-8",
    )
    (vertical_dir / "data-schema.json").write_text("{}", encoding="utf-8")
    return RulesSettings(
        rules_dir=str(rules_dir),
        enable_autolearn=True,
        autolearn_min_samples=5,
        autolearn_min_ratio=0.8,
    )


@pytest.fixture
def learner(
    vector_store: VectorStore, rules_settings: RulesSettings, tmp_path: Path
) -> RuleLearner:
    """RuleLearner wired to the real SQLite vector_store and tmp dirs."""
    learner = RuleLearner(vector_store, rules_settings)
    learner._reports_dir = tmp_path / "reports"  # type: ignore[assignment]
    learner._rules_dir = Path(rules_settings.rules_dir)  # type: ignore[assignment]
    return learner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRuleAutolearnIntegration:
    """Cross-domain verification + report generation + rules add command."""

    async def test_cross_domain_verified_proposal_generated(
        self,
        learner: RuleLearner,
        vector_store: VectorStore,
    ) -> None:
        """8 samples at newshipping.com (7 eta_update) + cross-domain → proposal."""
        # newshipping.com: 7 eta_update + 1 other (87.5% ratio).
        for _ in range(7):
            s = _make_sample("newshipping.com", "eta_update")
            await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))
        s = _make_sample("newshipping.com", "location_plan")
        await vector_store.insert_sample(s, _embedding(0.2), _embedding(0.2))

        # Cross-domain: 2 other domains also have eta_update.
        for _ in range(2):
            s = _make_sample("domain2.com", "eta_update")
            await vector_store.insert_sample(s, _embedding(0.3), _embedding(0.3))
        for _ in range(2):
            s = _make_sample("domain3.com", "eta_update")
            await vector_store.insert_sample(s, _embedding(0.4), _embedding(0.4))

        # Run scan.
        report_path_str = await learner.run_weekly_scan()
        report_path = Path(report_path_str)
        assert report_path.exists()

        content = report_path.read_text(encoding="utf-8")
        assert "# Rule Proposals Report" in content
        assert "newshipping.com" in content
        assert "eta_update" in content
        assert "- [ ] 确认添加" in content
        assert "```yaml" in content
        # Cross-domain verified → True.
        assert "True" in content

    async def test_cross_domain_failure_no_proposal(
        self,
        learner: RuleLearner,
        vector_store: VectorStore,
    ) -> None:
        """Cross-domain fails + ratio < 90% → no proposal."""
        for _ in range(7):
            s = _make_sample("rare.com", "exotic_label")
            await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))
        s = _make_sample("rare.com", "other_label")
        await vector_store.insert_sample(s, _embedding(0.2), _embedding(0.2))

        report_path_str = await learner.run_weekly_scan()
        content = Path(report_path_str).read_text(encoding="utf-8")
        assert "无规则提议" in content

    async def test_below_min_samples_no_proposal(
        self,
        learner: RuleLearner,
        vector_store: VectorStore,
    ) -> None:
        """< 5 samples → no proposal even with 100% ratio and cross-domain."""
        for _ in range(4):
            s = _make_sample("small.com", "small_label")
            await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))
        for _ in range(2):
            s = _make_sample("d2.com", "small_label")
            await vector_store.insert_sample(s, _embedding(0.2), _embedding(0.2))

        report_path_str = await learner.run_weekly_scan()
        content = Path(report_path_str).read_text(encoding="utf-8")
        assert "无规则提议" in content

    async def test_single_domain_discount_proposal(
        self,
        learner: RuleLearner,
        vector_store: VectorStore,
    ) -> None:
        """Cross-domain fails + ratio ≥ 90% → proposal with ×0.9 discount."""
        # 10 samples at single.com, all unique_label (100%), no cross-domain.
        for _ in range(10):
            s = _make_sample("single.com", "unique_label")
            await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        report_path_str = await learner.run_weekly_scan()
        content = Path(report_path_str).read_text(encoding="utf-8")
        assert "single.com" in content
        assert "unique_label" in content
        assert "Single-domain" in content

    async def test_rules_add_command_appends_to_yaml(
        self,
        learner: RuleLearner,
        vector_store: VectorStore,
        rules_settings: RulesSettings,
    ) -> None:
        """The real rules-add command validates and deduplicates a learner report."""
        # Seed the proposal by running a scan with cross-domain verified data.
        for _ in range(8):
            s = _make_sample("confirmed.com", "eta_update")
            await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))
        for _ in range(2):
            s = _make_sample("d2.com", "eta_update")
            await vector_store.insert_sample(s, _embedding(0.2), _embedding(0.2))
        for _ in range(2):
            s = _make_sample("d3.com", "eta_update")
            await vector_store.insert_sample(s, _embedding(0.3), _embedding(0.3))

        report_path_str = await learner.run_weekly_scan()
        report_path = Path(report_path_str)

        # Initially no rules file exists.
        rules_file = learner._rules_dir / "sender_domains.yaml"
        assert not rules_file.exists()

        # Confirm the proposal by replacing its "- [ ]" with "- [x]".
        content = report_path.read_text(encoding="utf-8")
        confirmed_content = content.replace(
            "- [ ] 确认添加 (Confirm and append)",
            "- [x] 确认添加 (Confirm and append)",
        )
        report_path.write_text(confirmed_content, encoding="utf-8")

        cli_settings = SimpleNamespace(
            vertical=SimpleNamespace(
                id="example-triage",
                verticals_path=str(Path(rules_settings.rules_dir).parent.parent),
            ),
            rules=rules_settings,
        )
        with patch("mailagent.infra.cli._build_settings", return_value=cli_settings):
            first_result = CliRunner().invoke(
                main, ["rules", "add", "--from-report", str(report_path)]
            )
            second_result = CliRunner().invoke(
                main, ["rules", "add", "--from-report", str(report_path)]
            )

        assert first_result.exit_code == 0, first_result.output
        assert "Added 1 rules" in first_result.output
        assert second_result.exit_code == 0, second_result.output
        assert "Added 0 rules" in second_result.output

        stored_rules = yaml.safe_load(rules_file.read_text(encoding="utf-8"))
        assert len(stored_rules) == 1
        validated_rule = SenderDomainRule.model_validate(stored_rules[0])
        assert validated_rule.domain == "confirmed.com"
        assert validated_rule.label == "eta_update"
