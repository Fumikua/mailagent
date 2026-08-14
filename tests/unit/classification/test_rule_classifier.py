"""Unit tests for RuleClassifier: four rule types, conflict resolution, hot reload,
Protocol compliance, and error tolerance.

Covers:
    - Each rule type matching independently (sender_domains / subject_patterns /
      body_keywords / structural).
    - Conflict resolution: highest confidence, tie-break by rule-type priority,
      same-label max confidence.
    - Hot reload: mtime change triggers reload.
    - Invalid YAML preserves old rules + warning.
    - Missing rule files tolerated (empty rule set + warning).
    - RuleMatch field completeness.
    - Classifier Protocol contract: source attribute + async classify method.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from mailagent.classification import (
    AttemptStatus,
    ClassificationRequest,
)
from mailagent.classification.rule_classifier import (
    BodyKeywordRule,
    RuleClassifier,
    SenderDomainRule,
    StructuralRule,
    SubjectPatternRule,
)
from mailagent.domain.models import MailEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_rules(
    rules_dir: Path,
    sender_domains: list[dict[str, Any]] | None = None,
    subject_patterns: list[dict[str, Any]] | None = None,
    body_keywords: list[dict[str, Any]] | None = None,
    structural: list[dict[str, Any]] | None = None,
) -> None:
    """Write rule YAML files into *rules_dir*."""
    rules_dir.mkdir(parents=True, exist_ok=True)
    for filename, rules in {
        "sender_domains.yaml": sender_domains,
        "subject_patterns.yaml": subject_patterns,
        "body_keywords.yaml": body_keywords,
        "structural.yaml": structural,
    }.items():
        (rules_dir / filename).write_text(
            yaml.safe_dump(rules if rules is not None else []), encoding="utf-8"
        )


def _mail(
    sender: str = "ops@example.com",
    subject: str = "Test subject",
    body: str = "Plain body",
    recipients: list[str] | None = None,
) -> MailEvent:
    return MailEvent(
        message_id="m1",
        sender=sender,
        subject=subject,
        body=body,
        recipients=recipients or [],
    )


@pytest.fixture
def taxonomy_loader() -> MagicMock:
    loader = MagicMock()
    loader.get_tree.return_value.all_codes.return_value = {"schedule"}
    return loader


def _valid_rules(rules_dir: Path, label: str = "schedule") -> Path:
    _write_rules(
        rules_dir,
        sender_domains=[{"domain": "example.com", "label": label}],
        subject_patterns=[{"pattern": "STATUS", "label": label}],
        body_keywords=[{"keywords": ["DG"], "label": label}],
        structural=[{"condition": "has_attachments", "label": label}],
    )
    return rules_dir


def _write_subject_rules(rules_dir: Path, pattern: str, label: str) -> None:
    _write_rules(rules_dir, subject_patterns=[{"pattern": pattern, "label": label}])


# ---------------------------------------------------------------------------
# Sender domain matching
# ---------------------------------------------------------------------------


class TestSenderDomainMatching:
    def test_domain_match(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[
                {"domain": "example.com", "label": "locationing", "confidence": 0.95}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="ops@example.com"))

        assert result.selected is not None
        assert result.selected.label == "locationing"
        assert result.selected.rule_type == "sender_domains"
        assert result.selected.confidence == 0.95

    def test_domain_match_with_display_name(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "example.com", "label": "locationing"}],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="Example Ops <ops@example.com>"))

        assert result.selected is not None
        assert result.selected.label == "locationing"

    def test_domain_no_match(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "partner.example", "label": "locationing"}],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="ops@example.com"))

        assert result.selected is None
        assert result.matches == []

    def test_local_part_pattern_match(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[
                {
                    "domain": "example.com",
                    "label": "locationing",
                    "local_part_pattern": "ops.*",
                }
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="ops@example.com"))

        assert result.selected is not None
        assert result.selected.label == "locationing"

    def test_local_part_pattern_no_match(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[
                {
                    "domain": "service.example",
                    "label": "locationing",
                    "local_part_pattern": "ops.*",
                }
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="info@service.example"))

        assert result.selected is None

    def test_domain_case_insensitive(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "Example.Com", "label": "locationing"}],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="ops@EXAMPLE.COM"))

        assert result.selected is not None


# ---------------------------------------------------------------------------
# Subject pattern matching
# ---------------------------------------------------------------------------


class TestSubjectPatternMatching:
    def test_subject_regex_match(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            subject_patterns=[
                {"pattern": "STATUS.*", "label": "arrival_notice", "confidence": 0.85}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(subject="STATUS change for Berlin Example"))

        assert result.selected is not None
        assert result.selected.label == "arrival_notice"
        assert result.selected.rule_type == "subject_patterns"
        assert result.selected.confidence == 0.85

    def test_subject_match_after_prefix_strip(self, tmp_path: Path) -> None:
        """Subject patterns run against the normalized (prefix-stripped) subject."""
        _write_rules(
            tmp_path,
            subject_patterns=[{"pattern": "Status Report", "label": "status_report"}],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(subject="Re[2]: Re: Status Report"))

        assert result.selected is not None
        assert result.selected.label == "status_report"

    def test_subject_no_match(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            subject_patterns=[{"pattern": "STATUS.*", "label": "arrival_notice"}],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(subject="Staff update confirmation"))

        assert result.selected is None


# ---------------------------------------------------------------------------
# Body keyword matching
# ---------------------------------------------------------------------------


class TestBodyKeywordMatching:
    def test_body_keyword_or_match(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            body_keywords=[
                {
                    "keywords": ["PRIORITY GOODS", "PG"],
                    "label": "priority_review",
                    "confidence": 0.80,
                    "match_all": False,
                }
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(body="This message contains a PG list."))

        assert result.selected is not None
        assert result.selected.label == "priority_review"
        assert result.selected.rule_type == "body_keywords"
        assert result.selected.confidence == 0.80

    def test_body_keyword_and_match_all_true(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            body_keywords=[
                {
                    "keywords": ["staff update", "crew list"],
                    "label": "staff_update",
                    "match_all": True,
                }
            ],
        )
        clf = RuleClassifier(tmp_path)
        # Both keywords present → match.
        r1 = clf.match(_mail(body="Please find staff update and crew list attached."))
        assert r1.selected is not None
        assert r1.selected.label == "staff_update"

        # Only one keyword → no match.
        r2 = clf.match(_mail(body="Please process the staff update request."))
        assert r2.selected is None

    def test_body_keyword_case_insensitive(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            body_keywords=[{"keywords": ["priority goods"], "label": "priority_review"}],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(body="PRIORITY GOODS manifest attached"))

        assert result.selected is not None
        assert result.selected.label == "priority_review"

    def test_body_keyword_no_match(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            body_keywords=[{"keywords": ["nonexistent"], "label": "x"}],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(body="Plain body text"))

        assert result.selected is None


# ---------------------------------------------------------------------------
# Structural matching
# ---------------------------------------------------------------------------


class TestStructuralMatching:
    def test_structural_has_attachments(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            structural=[
                {"condition": "has_attachments", "label": "document_processing"}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(
            _mail(body="See attached."), context={"has_attachments": True}
        )

        assert result.selected is not None
        assert result.selected.label == "document_processing"
        assert result.selected.rule_type == "structural"

    def test_structural_has_attachments_false(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            structural=[
                {"condition": "has_attachments", "label": "document_processing"}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(body="No attachments"), context={"has_attachments": False})

        assert result.selected is None

    def test_structural_body_length_comparison(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            structural=[
                {"condition": "body_length > 100", "label": "long_email", "confidence": 0.70}
            ],
        )
        clf = RuleClassifier(tmp_path)
        long_body = "x" * 200
        r1 = clf.match(_mail(body=long_body))
        assert r1.selected is not None
        assert r1.selected.label == "long_email"

        r2 = clf.match(_mail(body="short"))
        assert r2.selected is None

    def test_structural_and_combination(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            structural=[
                {
                    "condition": "has_attachments and body_length > 50",
                    "label": "doc_with_body",
                }
            ],
        )
        clf = RuleClassifier(tmp_path)
        # Both conditions met.
        r1 = clf.match(
            _mail(body="y" * 100), context={"has_attachments": True}
        )
        assert r1.selected is not None
        assert r1.selected.label == "doc_with_body"

        # Only body_length met (no attachments).
        r2 = clf.match(
            _mail(body="y" * 100), context={"has_attachments": False}
        )
        assert r2.selected is None


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


class TestConflictResolution:
    def test_highest_confidence_wins(self, tmp_path: Path) -> None:
        """Different confidence: highest wins regardless of rule type."""
        _write_rules(
            tmp_path,
            sender_domains=[
                {"domain": "example.com", "label": "locationing", "confidence": 0.95}
            ],
            body_keywords=[
                {"keywords": ["hello"], "label": "greeting", "confidence": 0.80}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="ops@example.com", body="hello world"))

        assert result.selected is not None
        assert result.selected.label == "locationing"
        assert result.selected.confidence == 0.95
        assert len(result.matches) == 2
        assert result.conflict_logged is True

    def test_same_confidence_priority_order(self, tmp_path: Path) -> None:
        """Same confidence: structural > sender_domains > subject_patterns > body_keywords."""
        _write_rules(
            tmp_path,
            sender_domains=[
                {"domain": "example.com", "label": "domain_label", "confidence": 0.90}
            ],
            subject_patterns=[
                {"pattern": "Test", "label": "subject_label", "confidence": 0.90}
            ],
            body_keywords=[
                {"keywords": ["body"], "label": "body_label", "confidence": 0.90}
            ],
            structural=[
                {"condition": "body_length > 0", "label": "struct_label", "confidence": 0.90}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(
            _mail(sender="ops@example.com", subject="Test subject", body="body text")
        )

        assert result.selected is not None
        assert result.selected.rule_type == "structural"
        assert len(result.matches) == 4

    def test_same_confidence_sender_over_subject(self, tmp_path: Path) -> None:
        """At equal confidence, sender_domains beats subject_patterns."""
        _write_rules(
            tmp_path,
            sender_domains=[
                {"domain": "example.com", "label": "domain_lbl", "confidence": 0.90}
            ],
            subject_patterns=[
                {"pattern": "Test", "label": "subject_lbl", "confidence": 0.90}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="ops@example.com", subject="Test subject"))

        assert result.selected is not None
        assert result.selected.rule_type == "sender_domains"

    def test_same_label_max_confidence(self, tmp_path: Path) -> None:
        """Same label from multiple rule types: max confidence wins."""
        _write_rules(
            tmp_path,
            sender_domains=[
                {"domain": "example.com", "label": "locationing", "confidence": 0.85}
            ],
            subject_patterns=[
                {"pattern": "STATUS", "label": "locationing", "confidence": 0.92}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(
            _mail(sender="ops@example.com", subject="STATUS change", body="x")
        )

        assert result.selected is not None
        assert result.selected.label == "locationing"
        assert result.selected.confidence == 0.92

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "example.com", "label": "locationing"}],
        )
        clf = RuleClassifier(tmp_path)
        assert clf.resolve_conflict([]) is None


# ---------------------------------------------------------------------------
# Hot reload
# ---------------------------------------------------------------------------


class TestHotReload:
    async def test_classification_consumes_run_bound_rule_snapshot(
        self, tmp_path: Path
    ) -> None:
        _write_subject_rules(tmp_path, pattern="STATUS", label="schedule")
        classifier = RuleClassifier(tmp_path)
        bound = classifier.get_snapshot()

        _write_subject_rules(tmp_path, pattern="CREW", label="schedule")
        classifier._last_check = 0.0
        request = ClassificationRequest(
            mail=_mail(subject="STATUS update"),
            asset_snapshots={"rules": bound},
        )

        attempt = await classifier.classify(request)

        assert attempt.status == AttemptStatus.SUCCESS
        assert attempt.evidence["rules_version"] == bound.version

    def test_mtime_change_triggers_reload(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "example.com", "label": "locationing"}],
        )
        clf = RuleClassifier(tmp_path)

        r1 = clf.match(_mail(sender="ops@example.com"))
        assert r1.selected is not None
        assert r1.selected.label == "locationing"

        # Modify the rule file.
        time.sleep(0.05)
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "example.com", "label": "arrival"}],
        )

        # Force past the throttle interval.
        clf._last_check = 0.0

        r2 = clf.match(_mail(sender="ops@example.com"))
        assert r2.selected is not None
        assert r2.selected.label == "arrival"

    def test_successful_reload_replaces_rules_and_version_atomically(
        self, tmp_path: Path
    ) -> None:
        _write_subject_rules(tmp_path, pattern="STATUS", label="schedule")
        classifier = RuleClassifier(tmp_path)
        before = classifier.get_snapshot()

        _write_subject_rules(tmp_path, pattern="CREW", label="schedule")
        classifier._last_check = 0.0
        after = classifier.get_snapshot()

        assert after.version != before.version
        assert classifier.match(_mail(subject="STATUS update"), snapshot=after).selected is None
        selected = classifier.match(
            _mail(subject="CREW update"),
            snapshot=after,
        ).selected
        assert selected is not None
        assert selected.label == "schedule"

    def test_throttle_prevents_rapid_reload(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "example.com", "label": "locationing"}],
        )
        clf = RuleClassifier(tmp_path)
        clf._last_check = time.monotonic()  # set to now

        # Modify the rule file.
        time.sleep(0.05)
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "example.com", "label": "arrival"}],
        )

        # Throttle should prevent reload.
        r = clf.match(_mail(sender="ops@example.com"))
        assert r.selected is not None
        assert r.selected.label == "locationing"  # old rules still active

    def test_throttle_preserves_rules_and_version_together(
        self, tmp_path: Path
    ) -> None:
        _write_subject_rules(tmp_path, pattern="STATUS", label="schedule")
        classifier = RuleClassifier(tmp_path)
        before = classifier.get_snapshot()
        classifier._last_check = time.monotonic()

        _write_subject_rules(tmp_path, pattern="CREW", label="schedule")
        after = classifier.get_snapshot()

        assert after is before
        selected = classifier.match(
            _mail(subject="STATUS update"),
            snapshot=after,
        ).selected
        assert selected is not None


# ---------------------------------------------------------------------------
# Invalid YAML / missing files
# ---------------------------------------------------------------------------


class TestErrorTolerance:
    def test_invalid_yaml_keeps_old_rules(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[{"domain": "example.com", "label": "locationing"}],
        )
        clf = RuleClassifier(tmp_path)

        # Overwrite with invalid YAML.
        time.sleep(0.05)
        (tmp_path / "sender_domains.yaml").write_text(
            "{invalid yaml", encoding="utf-8"
        )

        # Force past throttle.
        clf._last_check = 0.0

        # Old rules should still be active.
        result = clf.match(_mail(sender="ops@example.com"))
        assert result.selected is not None
        assert result.selected.label == "locationing"

    def test_invalid_reload_preserves_rules_and_version_together(
        self, tmp_path: Path
    ) -> None:
        _write_subject_rules(tmp_path, pattern="STATUS", label="schedule")
        classifier = RuleClassifier(tmp_path)
        before = classifier.get_snapshot()
        (tmp_path / "subject_patterns.yaml").write_text(
            "{invalid yaml",
            encoding="utf-8",
        )
        classifier._last_check = 0.0

        after = classifier.get_snapshot()

        assert after is before
        selected = classifier.match(
            _mail(subject="STATUS update"),
            snapshot=after,
        ).selected
        assert selected is not None

    def test_missing_rule_file_rejects_startup(self, tmp_path: Path) -> None:
        """A deployment cannot start with an unintended empty ruleset."""
        with pytest.raises(ValueError, match="rule file is missing"):
            RuleClassifier(tmp_path)

    def test_partial_rule_assets_reject_startup(self, tmp_path: Path) -> None:
        """All four rule files are required for a valid deployment snapshot."""
        (tmp_path / "sender_domains.yaml").write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError, match="rule file is missing"):
            RuleClassifier(tmp_path)


class TestAtomicRuleLoading:
    def test_misspelled_rule_field_rejects_startup(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            subject_patterns=[{"pattern": "STATUS", "lable": "schedule"}],
        )

        with pytest.raises(ValueError, match="failed to validate rules"):
            RuleClassifier(tmp_path)

    def test_misspelled_rule_field_does_not_replace_active_snapshot(
        self,
        tmp_path: Path,
    ) -> None:
        _write_subject_rules(tmp_path, pattern="STATUS", label="schedule")
        classifier = RuleClassifier(tmp_path)
        before = classifier.get_snapshot()

        _write_rules(
            tmp_path,
            subject_patterns=[
                {"pattern": "CREW", "label": "schedule", "confidnce": 0.99}
            ],
        )
        classifier._last_check = 0.0
        after = classifier.get_snapshot()

        assert after is before
        assert classifier.match(_mail(subject="STATUS update")).selected is not None
        assert classifier.match(_mail(subject="CREW update")).selected is None

    def test_empty_match_all_keyword_rule_rejects_startup(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            body_keywords=[
                {"keywords": [], "label": "schedule", "match_all": True}
            ],
        )

        with pytest.raises(ValueError, match="failed to validate rules"):
            RuleClassifier(tmp_path)

    def test_empty_match_all_keyword_rule_never_matches_defensively(
        self,
        tmp_path: Path,
    ) -> None:
        _write_rules(tmp_path)
        classifier = RuleClassifier(tmp_path)
        invalid_legacy_rule = BodyKeywordRule.model_construct(
            keywords=(),
            label="schedule",
            confidence=0.8,
            match_all=True,
        )

        assert classifier._match_body_keywords(
            _mail(body="any body"),
            [invalid_legacy_rule],
        ) == []

    def test_invalid_taxonomy_label_keeps_previous_rules(
        self, tmp_path: Path, taxonomy_loader: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bad changed file cannot partially replace the active rule snapshot."""
        rules_dir = _valid_rules(tmp_path)
        classifier = RuleClassifier(rules_dir, taxonomy_loader)
        _write_subject_rules(rules_dir, pattern="STATUS", label="old_schedule")
        monkeypatch.setattr(time, "monotonic", lambda: 10.0)

        result = classifier.match(_mail(subject="STATUS update"))

        assert result.selected is not None
        assert result.selected.label == "schedule"

    def test_invalid_regex_keeps_complete_previous_snapshot(
        self, tmp_path: Path, taxonomy_loader: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regex compilation failure leaves every prior rule type active."""
        rules_dir = _valid_rules(tmp_path)
        classifier = RuleClassifier(rules_dir, taxonomy_loader)
        _write_subject_rules(rules_dir, pattern="[", label="schedule")
        monkeypatch.setattr(time, "monotonic", lambda: 10.0)

        result = classifier.match(_mail(subject="STATUS update"))

        assert result.selected is not None
        assert result.selected.label == "schedule"
        assert classifier.match(_mail(sender="ops@example.com")).selected is not None
        assert classifier.match(_mail(body="DG manifest")).selected is not None
        assert classifier.match(
            _mail(), context={"has_attachments": True}
        ).selected is not None

    def test_invalid_utf8_keeps_complete_previous_snapshot(
        self, tmp_path: Path, taxonomy_loader: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed replacement file cannot escape the atomic reload boundary."""
        rules_dir = _valid_rules(tmp_path)
        classifier = RuleClassifier(rules_dir, taxonomy_loader)
        rules_file = rules_dir / "subject_patterns.yaml"
        prior_mtime = rules_file.stat().st_mtime
        rules_file.write_bytes(b"\xff")
        os.utime(rules_file, (prior_mtime + 1, prior_mtime + 1))
        monkeypatch.setattr(time, "monotonic", lambda: 10.0)

        result = classifier.match(_mail(subject="STATUS update"))

        assert result.selected is not None
        assert result.selected.label == "schedule"

    def test_out_of_range_confidence_does_not_replace_snapshot(
        self, tmp_path: Path, taxonomy_loader: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confidence outside [0, 1] is rejected while old rules keep matching."""
        rules_dir = _valid_rules(tmp_path)
        classifier = RuleClassifier(rules_dir, taxonomy_loader)
        _write_rules(
            rules_dir,
            sender_domains=[
                {"domain": "example.com", "label": "schedule", "confidence": 1.1}
            ],
        )
        monkeypatch.setattr(time, "monotonic", lambda: 10.0)

        result = classifier.match(_mail(sender="ops@example.com"))

        assert result.selected is not None
        assert result.selected.confidence == 0.95
        assert len(classifier._sender_domain_rules) == 1
        assert len(classifier._subject_pattern_rules) == 1
        assert len(classifier._body_keyword_rules) == 1
        assert len(classifier._structural_rules) == 1

    def test_rule_confidence_above_one_is_rejected(self) -> None:
        """Schema bounds prevent impossible confidence values before matching."""
        with pytest.raises(ValidationError):
            SenderDomainRule(domain="example.com", label="schedule", confidence=1.1)


# ---------------------------------------------------------------------------
# RuleMatch field integrity
# ---------------------------------------------------------------------------


class TestRuleMatchFields:
    def test_sender_domain_match_fields(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[
                {
                    "domain": "example.com",
                    "label": "locationing",
                    "confidence": 0.95,
                    "local_part_pattern": "ops.*",
                }
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(sender="ops@example.com"))

        assert result.selected is not None
        m = result.selected
        assert m.rule_type == "sender_domains"
        assert m.label == "locationing"
        assert m.confidence == 0.95
        assert "example.com" in m.matched_pattern
        assert "ops.*" in m.matched_pattern

    def test_subject_pattern_match_fields(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            subject_patterns=[{"pattern": "STATUS.*", "label": "arrival_notice"}],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(subject="STATUS change"))

        assert result.selected is not None
        m = result.selected
        assert m.rule_type == "subject_patterns"
        assert m.label == "arrival_notice"
        assert m.matched_pattern == "STATUS.*"

    def test_body_keyword_match_fields(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            body_keywords=[
                {"keywords": ["DG", "danger"], "label": "dg_compliance"}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(body="This is DG cargo"))

        assert result.selected is not None
        m = result.selected
        assert m.rule_type == "body_keywords"
        assert m.label == "dg_compliance"
        assert "DG" in m.matched_pattern

    def test_structural_match_fields(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            structural=[
                {"condition": "body_length > 10", "label": "long_email"}
            ],
        )
        clf = RuleClassifier(tmp_path)
        result = clf.match(_mail(body="x" * 50))

        assert result.selected is not None
        m = result.selected
        assert m.rule_type == "structural"
        assert m.label == "long_email"
        assert m.matched_pattern == "body_length > 10"


# ---------------------------------------------------------------------------
# Classifier Protocol contract
# ---------------------------------------------------------------------------


class TestClassifierProtocol:
    def test_source_attribute(self, tmp_path: Path) -> None:
        _write_rules(tmp_path)
        clf = RuleClassifier(tmp_path)
        assert clf.source == "rules"

    def test_has_classify_method(self, tmp_path: Path) -> None:
        _write_rules(tmp_path)
        clf = RuleClassifier(tmp_path)
        assert callable(getattr(clf, "classify", None))

    async def test_classify_no_match(self, tmp_path: Path) -> None:
        _write_rules(tmp_path)
        clf = RuleClassifier(tmp_path)
        request = ClassificationRequest(mail=_mail())
        attempt = await clf.classify(request)

        assert attempt.source == "rules"
        assert attempt.status == AttemptStatus.NO_MATCH
        assert attempt.confidence == 0.0
        assert attempt.labels == []

    async def test_classify_success(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            sender_domains=[
                {"domain": "example.com", "label": "locationing", "confidence": 0.95}
            ],
        )
        clf = RuleClassifier(tmp_path)
        request = ClassificationRequest(mail=_mail(sender="ops@example.com"))
        attempt = await clf.classify(request)

        assert attempt.source == "rules"
        assert attempt.status == AttemptStatus.SUCCESS
        assert len(attempt.labels) == 1
        assert attempt.labels[0].l1_code == "locationing"
        assert attempt.labels[0].l1_label == "locationing"
        assert attempt.labels[0].confidence == 0.95
        assert attempt.confidence == 0.95
        assert attempt.evidence["rule_type"] == "sender_domains"
        assert "matched_pattern" in attempt.evidence

    async def test_classify_passes_context_for_structural(self, tmp_path: Path) -> None:
        _write_rules(
            tmp_path,
            structural=[
                {"condition": "has_attachments", "label": "document_processing"}
            ],
        )
        clf = RuleClassifier(tmp_path)
        request = ClassificationRequest(
            mail=_mail(body="See attached."),
            context={"has_attachments": True},
        )
        attempt = await clf.classify(request)

        assert attempt.status == AttemptStatus.SUCCESS
        assert attempt.labels[0].l1_code == "document_processing"
        assert attempt.evidence["rule_type"] == "structural"


# ---------------------------------------------------------------------------
# Rule schema model defaults
# ---------------------------------------------------------------------------


class TestRuleSchemaDefaults:
    @pytest.mark.parametrize(
        ("model", "payload"),
        [
            (
                SenderDomainRule,
                {"domain": "example.com", "label": "schedule", "unknown": 1},
            ),
            (
                SubjectPatternRule,
                {"pattern": "STATUS", "label": "schedule", "unknown": 1},
            ),
            (
                BodyKeywordRule,
                {"keywords": ["STATUS"], "label": "schedule", "unknown": 1},
            ),
            (
                StructuralRule,
                {
                    "condition": "has_attachments",
                    "label": "schedule",
                    "unknown": 1,
                },
            ),
        ],
    )
    def test_unknown_fields_are_rejected(
        self,
        model: type[
            SenderDomainRule
            | SubjectPatternRule
            | BodyKeywordRule
            | StructuralRule
        ],
        payload: dict[str, object],
    ) -> None:
        with pytest.raises(ValidationError):
            model.model_validate(payload)

    @pytest.mark.parametrize(
        ("model", "payload"),
        [
            (SenderDomainRule, {"domain": " ", "label": "schedule"}),
            (SenderDomainRule, {"domain": "example.com", "label": " "}),
            (SubjectPatternRule, {"pattern": " ", "label": "schedule"}),
            (BodyKeywordRule, {"keywords": [], "label": "schedule"}),
            (BodyKeywordRule, {"keywords": [" "], "label": "schedule"}),
            (StructuralRule, {"condition": " ", "label": "schedule"}),
        ],
    )
    def test_blank_or_empty_match_fields_are_rejected(
        self,
        model: type[
            SenderDomainRule
            | SubjectPatternRule
            | BodyKeywordRule
            | StructuralRule
        ],
        payload: dict[str, object],
    ) -> None:
        with pytest.raises(ValidationError):
            model.model_validate(payload)

    def test_rule_match_fields_are_normalized(self) -> None:
        domain = SenderDomainRule(domain="  Example.COM  ", label=" schedule ")
        keywords = BodyKeywordRule(
            keywords=["  STATUS  ", " location "],
            label=" schedule ",
        )

        assert domain.domain == "example.com"
        assert domain.label == "schedule"
        assert keywords.keywords == ("STATUS", "location")
        assert keywords.label == "schedule"

    def test_sender_domain_rule_defaults(self) -> None:
        rule = SenderDomainRule(domain="example.com", label="locationing")
        assert rule.confidence == 0.95
        assert rule.local_part_pattern is None

    def test_subject_pattern_rule_defaults(self) -> None:
        rule = SubjectPatternRule(pattern="STATUS.*", label="arrival")
        assert rule.confidence == 0.85

    def test_body_keyword_rule_defaults(self) -> None:
        rule = BodyKeywordRule(keywords=["DG"], label="dg")
        assert rule.confidence == 0.80
        assert rule.match_all is False

    def test_structural_rule_defaults(self) -> None:
        rule = StructuralRule(condition="has_attachments", label="doc")
        assert rule.confidence == 0.90
