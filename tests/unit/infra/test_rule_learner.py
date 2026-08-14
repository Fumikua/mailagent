"""Unit tests for RuleLearner.

All tests mock vector_store — no real database calls. The report parsing
and YAML appending tests use temporary directories.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml

from mailagent.domain.models import SampleRecord
from mailagent.infra.config import RulesSettings
from mailagent.infra.rule_learner import (
    RuleLearner,
    _parse_checked_sender_domain_rules,
    append_confirmed_sender_domain_rules,
)
from mailagent.classification.taxonomy import TaxonomyLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    sender_domain: str = "example.com",
    label_l1: str = "schedule",
    subject: str = "Test subject",
) -> SampleRecord:
    return SampleRecord(
        mail_hash=f"hash-{uuid4()}",
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
    )


def _make_learner(
    tmp_path: Path,
    vector_store: MagicMock | None = None,
    settings: RulesSettings | None = None,
) -> RuleLearner:
    """Build a RuleLearner with tmp reports and rules directories."""
    vs = vector_store or MagicMock()
    learner = RuleLearner(vs, settings or RulesSettings())
    learner._reports_dir = tmp_path / "reports"  # type: ignore[assignment]
    learner._rules_dir = tmp_path / "rules"  # type: ignore[assignment]
    return learner


def _active_taxonomy_snapshot(tmp_path: Path):
    """Load the literal flat labels used by rule-application tests."""
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        "nodes:\n"
        "  - code: schedule\n    label: Schedule\n"
        "  - code: old\n    label: Old\n"
        "  - code: eta_update\n    label: STATUS update\n"
        "  - code: other_label\n    label: Other\n"
        "  - code: label\n    label: Label\n"
        "  - code: label1\n    label: Label 1\n"
        "  - code: label2\n    label: Label 2\n",
        encoding="utf-8",
    )
    return TaxonomyLoader(taxonomy_path).get_snapshot()


# ---------------------------------------------------------------------------
# Test: _compute_distribution
# ---------------------------------------------------------------------------


class TestComputeDistribution:
    def test_flat_samples_are_grouped_by_l1_category(self, tmp_path: Path) -> None:
        """Flat samples must produce proposals for their active l1 category."""
        learner = _make_learner(tmp_path)
        samples = [
            _make_sample("a.com", "schedule"),
            _make_sample("a.com", "schedule"),
        ]

        assert learner._compute_distribution(samples) == {("a.com", "schedule"): 2}

    def test_basic_distribution(self, tmp_path: Path) -> None:
        """Distribution should count (sender_domain, label_l1) pairs."""
        learner = _make_learner(tmp_path)
        samples = [
            _make_sample("a.com", "label1"),
            _make_sample("a.com", "label1"),
            _make_sample("a.com", "label2"),
            _make_sample("b.com", "label1"),
        ]
        dist = learner._compute_distribution(samples)
        assert dist[("a.com", "label1")] == 2
        assert dist[("a.com", "label2")] == 1
        assert dist[("b.com", "label1")] == 1

    def test_empty_samples(self, tmp_path: Path) -> None:
        """No samples → empty distribution."""
        learner = _make_learner(tmp_path)
        dist = learner._compute_distribution([])
        assert dist == {}


# ---------------------------------------------------------------------------
# Test: _check_cross_domain
# ---------------------------------------------------------------------------


class TestCheckCrossDomain:
    def test_cross_domain_true_with_two_other_domains(
        self, tmp_path: Path
    ) -> None:
        """≥ 2 other domains with the same label → True."""
        learner = _make_learner(tmp_path)
        distribution = {
            ("target.com", "label1"): 5,
            ("other1.com", "label1"): 3,
            ("other2.com", "label1"): 2,
            ("target.com", "label2"): 1,
        }
        assert learner._check_cross_domain("label1", distribution, "target.com")

    def test_cross_domain_false_with_one_other_domain(
        self, tmp_path: Path
    ) -> None:
        """Only 1 other domain → False."""
        learner = _make_learner(tmp_path)
        distribution = {
            ("target.com", "label1"): 5,
            ("other1.com", "label1"): 3,
        }
        assert not learner._check_cross_domain("label1", distribution, "target.com")

    def test_cross_domain_false_no_other_domains(
        self, tmp_path: Path
    ) -> None:
        """No other domains → False."""
        learner = _make_learner(tmp_path)
        distribution = {
            ("target.com", "label1"): 5,
            ("target.com", "label2"): 1,
        }
        assert not learner._check_cross_domain("label1", distribution, "target.com")


# ---------------------------------------------------------------------------
# Test: _generate_proposal
# ---------------------------------------------------------------------------


class TestGenerateProposal:
    def test_cross_domain_proposal(self, tmp_path: Path) -> None:
        """Cross-domain verified → confidence = ratio, no single_domain flag."""
        learner = _make_learner(tmp_path)
        proposal = learner._generate_proposal(
            "example.com", "eta_update", 0.875, cross_domain=True
        )
        assert proposal["domain"] == "example.com"
        assert proposal["label"] == "eta_update"
        assert proposal["ratio"] == 0.875
        assert proposal["confidence"] == 0.875
        assert proposal["cross_domain"] is True
        assert proposal["single_domain"] is False
        assert "domain: example.com" in proposal["yaml_fragment"]
        assert "label: eta_update" in proposal["yaml_fragment"]

    def test_single_domain_discount_proposal(self, tmp_path: Path) -> None:
        """Cross-domain fails → confidence = ratio * 0.9, single_domain = True."""
        learner = _make_learner(tmp_path)
        proposal = learner._generate_proposal(
            "single.com", "unique_label", 1.0, cross_domain=False
        )
        assert proposal["confidence"] == pytest.approx(0.9, abs=1e-6)
        assert proposal["cross_domain"] is False
        assert proposal["single_domain"] is True


# ---------------------------------------------------------------------------
# Test: run_weekly_scan
# ---------------------------------------------------------------------------


class TestRunWeeklyScan:
    async def test_trigger_conditions_met_with_cross_domain(
        self, tmp_path: Path
    ) -> None:
        """8 emails + 87.5% + cross-domain → proposal generated."""
        # newshipping.com: 8 emails, 7 with "eta_update" (87.5%)
        # 2+ other domains also have "eta_update" → cross-domain True
        samples = []
        for _ in range(7):
            samples.append(_make_sample("newshipping.com", "eta_update"))
        samples.append(_make_sample("newshipping.com", "other_label"))
        # Cross-domain: 2 other domains with "eta_update"
        for _ in range(2):
            samples.append(_make_sample("domain2.com", "eta_update"))
        for _ in range(2):
            samples.append(_make_sample("domain3.com", "eta_update"))

        vs = MagicMock()
        vs.get_samples = AsyncMock(return_value=samples)
        learner = _make_learner(tmp_path, vs)

        result = await learner.run_weekly_scan()
        report_path = Path(result)
        assert report_path.exists()

        content = report_path.read_text(encoding="utf-8")
        assert "newshipping.com" in content
        assert "eta_update" in content
        assert "- [ ] 确认添加" in content
        # Cross-domain verified should be True
        assert "True" in content

    async def test_cross_domain_failure_no_proposal(self, tmp_path: Path) -> None:
        """Cross-domain fails and ratio < 90% → no proposal."""
        # rare.com: 8 emails, 7 with "exotic" (87.5%), no other domains
        samples = []
        for _ in range(7):
            samples.append(_make_sample("rare.com", "exotic_label"))
        samples.append(_make_sample("rare.com", "other_label"))

        vs = MagicMock()
        vs.get_samples = AsyncMock(return_value=samples)
        learner = _make_learner(tmp_path, vs)

        result = await learner.run_weekly_scan()
        report_path = Path(result)
        content = report_path.read_text(encoding="utf-8")
        assert "无规则提议" in content

    async def test_below_min_samples_no_proposal(self, tmp_path: Path) -> None:
        """< 5 samples → no proposal even with 100% ratio and cross-domain."""
        samples = [
            _make_sample("small.com", "small_label"),
            _make_sample("small.com", "small_label"),
            _make_sample("small.com", "small_label"),
            _make_sample("small.com", "small_label"),
        ]
        # Cross-domain: 2 other domains
        samples.append(_make_sample("d2.com", "small_label"))
        samples.append(_make_sample("d3.com", "small_label"))

        vs = MagicMock()
        vs.get_samples = AsyncMock(return_value=samples)
        learner = _make_learner(tmp_path, vs)

        result = await learner.run_weekly_scan()
        report_path = Path(result)
        content = report_path.read_text(encoding="utf-8")
        assert "无规则提议" in content

    async def test_single_domain_discount_proposal(self, tmp_path: Path) -> None:
        """Cross-domain fails but ratio >= 90% → proposal with discount."""
        # single.com: 10 emails, all 10 with "unique_label" (100%)
        # No other domains have "unique_label"
        samples = []
        for _ in range(10):
            samples.append(_make_sample("single.com", "unique_label"))

        vs = MagicMock()
        vs.get_samples = AsyncMock(return_value=samples)
        learner = _make_learner(tmp_path, vs)

        result = await learner.run_weekly_scan()
        report_path = Path(result)
        content = report_path.read_text(encoding="utf-8")
        assert "single.com" in content
        assert "unique_label" in content
        # Single-domain discount marker
        assert "Single-domain" in content
        assert "single_domain" not in content or "True" in content

    async def test_report_format_with_checkbox(self, tmp_path: Path) -> None:
        """Report should contain markdown checkbox for confirmation."""
        samples = []
        for _ in range(8):
            samples.append(_make_sample("checkbox.com", "eta_update"))
        samples.append(_make_sample("checkbox.com", "other"))
        # Cross-domain
        samples.append(_make_sample("d2.com", "eta_update"))
        samples.append(_make_sample("d3.com", "eta_update"))

        vs = MagicMock()
        vs.get_samples = AsyncMock(return_value=samples)
        learner = _make_learner(tmp_path, vs)

        result = await learner.run_weekly_scan()
        content = Path(result).read_text(encoding="utf-8")
        assert "# Rule Proposals Report" in content
        assert "- [ ] 确认添加" in content
        assert "```yaml" in content
        assert "```" in content


# ---------------------------------------------------------------------------
# Test: parse_report_and_append
# ---------------------------------------------------------------------------


class TestParseReportAndAppend:
    def test_append_confirmed_sender_domain_rules_writes_validated_rule(
        self, tmp_path: Path
    ) -> None:
        """A checked Markdown proposal becomes one sender-domain rule."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            """- [x] 确认添加 (Confirm and append)

```yaml
- domain: confirmed.example
  label: schedule
  confidence: 0.92
```
""",
            encoding="utf-8",
        )
        rules_dir = tmp_path / "rules"

        assert (
            append_confirmed_sender_domain_rules(
                report_path, rules_dir, _active_taxonomy_snapshot(tmp_path)
            )
            == 1
        )
        assert yaml.safe_load(
            (rules_dir / "sender_domains.yaml").read_text(encoding="utf-8")
        ) == [{"domain": "confirmed.example", "label": "schedule", "confidence": 0.92}]

    def test_append_confirmed_sender_domain_rules_is_idempotent(
        self, tmp_path: Path
    ) -> None:
        """Reapplying a report does not duplicate its normalized domain/label."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            """- [x] 确认添加

```yaml
- domain: confirmed.example
  label: schedule
  confidence: 0.92
```
""",
            encoding="utf-8",
        )
        rules_dir = tmp_path / "rules"
        rules_file = rules_dir / "sender_domains.yaml"

        taxonomy_snapshot = _active_taxonomy_snapshot(tmp_path)
        assert append_confirmed_sender_domain_rules(
            report_path, rules_dir, taxonomy_snapshot
        ) == 1
        assert append_confirmed_sender_domain_rules(
            report_path, rules_dir, taxonomy_snapshot
        ) == 0
        assert len(yaml.safe_load(rules_file.read_text(encoding="utf-8"))) == 1

    @pytest.mark.parametrize(
        "proposal",
        [
            "- domain: invalid.example\n  label: schedule\n  confidence: 0.92\n  unexpected: value",
            "- domain: '   '\n  label: schedule\n  confidence: 0.92",
            "- domain: invalid.example\n  label: schedule\n  confidence: 1.1",
            "- domain: [unterminated",
        ],
    )
    def test_append_confirmed_sender_domain_rules_rejects_invalid_proposals_atomically(
        self, tmp_path: Path, proposal: str
    ) -> None:
        """Invalid checked data cannot change an already-valid rules file."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            f"""- [x] 确认添加

```yaml
{proposal}
```
""",
            encoding="utf-8",
        )
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        rules_file = rules_dir / "sender_domains.yaml"
        original = "- domain: existing.example\n  label: schedule\n  confidence: 0.95\n"
        rules_file.write_text(original, encoding="utf-8")

        with pytest.raises(ValueError):
            append_confirmed_sender_domain_rules(
                report_path, rules_dir, _active_taxonomy_snapshot(tmp_path)
            )

        assert rules_file.read_text(encoding="utf-8") == original

    def test_rejects_inactive_candidate_label_before_replacement(
        self, tmp_path: Path
    ) -> None:
        """An operator-approved inactive label cannot alter the live rule file."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            """- [x] 确认添加

```yaml
- domain: invalid.example
  label: definitely_not_active
  confidence: 0.92
```
""",
            encoding="utf-8",
        )
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        rules_file = rules_dir / "sender_domains.yaml"
        original = b"- domain: existing.example\n  label: schedule\n  confidence: 0.95\n"
        rules_file.write_bytes(original)

        with pytest.raises(ValueError, match="absent from active taxonomy"):
            append_confirmed_sender_domain_rules(
                report_path, rules_dir, _active_taxonomy_snapshot(tmp_path)
            )

        assert rules_file.read_bytes() == original

    def test_validates_complete_post_update_set_before_replacement(
        self, tmp_path: Path
    ) -> None:
        """A pre-existing inactive label blocks replacement by a valid addition."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            """- [x] 确认添加

```yaml
- domain: valid.example
  label: schedule
  confidence: 0.92
```
""",
            encoding="utf-8",
        )
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        rules_file = rules_dir / "sender_domains.yaml"
        original = b"- domain: stale.example\n  label: inactive_existing\n  confidence: 0.95\n"
        rules_file.write_bytes(original)

        with pytest.raises(ValueError, match="absent from active taxonomy"):
            append_confirmed_sender_domain_rules(
                report_path, rules_dir, _active_taxonomy_snapshot(tmp_path)
            )

        assert rules_file.read_bytes() == original

    @pytest.mark.parametrize(
        "checked_section",
        [
            "- [x] 确认添加\n\nproposal text without a fence\n",
            "- [x] 确认添加\n\n```yaml\n[]\n```\n",
            (
                "- [x] 确认添加\n\n"
                "```yaml\n- domain: one.example\n  label: schedule\n```\n"
                "```yaml\n- domain: two.example\n  label: schedule\n```\n"
            ),
        ],
        ids=["missing-fence", "empty-list", "multiple-fences"],
    )
    def test_checked_proposal_requires_exactly_one_non_empty_yaml_fence(
        self, tmp_path: Path, checked_section: str
    ) -> None:
        """A malformed checked section must fail closed at the parser boundary."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(checked_section, encoding="utf-8")

        with pytest.raises(ValueError):
            _parse_checked_sender_domain_rules(report_path)

    @pytest.mark.parametrize(
        "checked_section",
        [
            "- [x] 确认添加\n\n```yaml\n- domain: incomplete.example\n",
            "- [x] 确认添加\n\n```yaml\n\n```\n",
        ],
        ids=["unterminated-fence", "empty-fence"],
    )
    def test_checked_proposal_rejects_incomplete_or_empty_yaml_fence(
        self, tmp_path: Path, checked_section: str
    ) -> None:
        """Existing strict YAML failures remain controlled and atomic."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(checked_section, encoding="utf-8")

        with pytest.raises(ValueError):
            _parse_checked_sender_domain_rules(report_path)

    def test_appends_confirmed_rules_to_yaml(self, tmp_path: Path) -> None:
        """Confirmed proposals (checkboxes [x]) should be appended to YAML."""
        learner = _make_learner(tmp_path)
        rules_file = learner._rules_dir / "sender_domains.yaml"
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        rules_file.write_text("- domain: existing.com\n  label: old\n", encoding="utf-8")

        # Create a report with 2 proposals: 1 confirmed, 1 not
        report_content = """# Rule Proposals Report

## Proposals

### Proposal 1: confirmed.com → eta_update

- [x] 确认添加 (Confirm and append)

```yaml
- domain: confirmed.com
  label: eta_update
  confidence: 0.88
```

### Proposal 2: unconfirmed.com → other_label

- [ ] 确认添加 (Confirm and append)

```yaml
- domain: unconfirmed.com
  label: other_label
  confidence: 0.90
```
"""
        report_path = tmp_path / "test_report.md"
        report_path.write_text(report_content, encoding="utf-8")

        count = learner.parse_report_and_append(
            report_path, _active_taxonomy_snapshot(tmp_path)
        )
        assert count == 1

        yaml_content = rules_file.read_text(encoding="utf-8")
        assert "confirmed.com" in yaml_content
        assert "unconfirmed.com" not in yaml_content
        assert "existing.com" in yaml_content  # original content preserved

    def test_no_confirmed_returns_zero(self, tmp_path: Path) -> None:
        """No confirmed checkboxes → 0 rules appended."""
        learner = _make_learner(tmp_path)
        rules_file = learner._rules_dir / "sender_domains.yaml"
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        rules_file.write_text("", encoding="utf-8")

        report_content = """# Rule Proposals Report

## Proposals

### Proposal 1: example.com → label

- [ ] 确认添加

```yaml
- domain: example.com
  label: label
  confidence: 0.85
```
"""
        report_path = tmp_path / "test_report.md"
        report_path.write_text(report_content, encoding="utf-8")

        count = learner.parse_report_and_append(
            report_path, _active_taxonomy_snapshot(tmp_path)
        )
        assert count == 0

    def test_multiple_confirmed_appended(self, tmp_path: Path) -> None:
        """Multiple confirmed proposals → all appended."""
        learner = _make_learner(tmp_path)
        rules_file = learner._rules_dir / "sender_domains.yaml"
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        rules_file.write_text("", encoding="utf-8")

        report_content = """# Rule Proposals Report

### Proposal 1: a.com → label1

- [x] 确认添加

```yaml
- domain: a.com
  label: label1
  confidence: 0.85
```

### Proposal 2: b.com → label2

- [x] 确认添加

```yaml
- domain: b.com
  label: label2
  confidence: 0.90
```
"""
        report_path = tmp_path / "test_report.md"
        report_path.write_text(report_content, encoding="utf-8")

        count = learner.parse_report_and_append(
            report_path, _active_taxonomy_snapshot(tmp_path)
        )
        assert count == 2

        yaml_content = rules_file.read_text(encoding="utf-8")
        assert "a.com" in yaml_content
        assert "b.com" in yaml_content


# ---------------------------------------------------------------------------
# Test: _generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_format(self, tmp_path: Path) -> None:
        """Report should have correct markdown structure."""
        learner = _make_learner(tmp_path)
        proposals = [
            {
                "domain": "test.com",
                "label": "eta_update",
                "ratio": 0.875,
                "confidence": 0.875,
                "cross_domain": True,
                "single_domain": False,
                "yaml_fragment": "- domain: test.com\n  label: eta_update\n  confidence: 0.88\n",
            }
        ]
        report_path = tmp_path / "report.md"
        learner._generate_report(proposals, 100, report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "# Rule Proposals Report" in content
        assert "**Total samples scanned**: 100" in content
        assert "test.com" in content
        assert "eta_update" in content
        assert "- [ ] 确认添加" in content
        assert "```yaml" in content
        assert "- domain: test.com" in content

    def test_empty_proposals_report(self, tmp_path: Path) -> None:
        """No proposals → report says 'no proposals'."""
        learner = _make_learner(tmp_path)
        report_path = tmp_path / "empty.md"
        learner._generate_report([], 50, report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "无规则提议" in content
