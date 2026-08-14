"""End-to-end integration tests for the rule classifier (Section 18.3).

Loads the real ``verticals/example_triage/rules/`` YAML files into a
``RuleClassifier`` instance and matches synthetic emails against all four rule
types (sender_domains / subject_patterns / body_keywords / structural), plus
conflict resolution across multiple matches.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mailagent.classification.rule_classifier import RuleClassifier
from mailagent.domain.models import MailEvent

PROJECT_ROOT = Path(__file__).parent.parent.parent
RULES_DIR = PROJECT_ROOT / "verticals" / "example_triage" / "rules"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rule_classifier() -> RuleClassifier:
    """Load the real example_triage rules directory."""
    return RuleClassifier(RULES_DIR)


@pytest.fixture
def rules_dir_exists() -> Path:
    """Ensure the rules directory and at least one rule file exist."""
    assert RULES_DIR.exists(), f"Rules dir missing: {RULES_DIR}"
    yaml_files = list(RULES_DIR.glob("*.yaml"))
    assert yaml_files, f"No YAML rule files found in {RULES_DIR}"
    return RULES_DIR


# ---------------------------------------------------------------------------
# Tests: single-rule-type matches against real rules
# ---------------------------------------------------------------------------


class TestSenderDomainRulesIntegration:
    """sender_domains.yaml is loaded and matches real domain rules."""

    def test_github_domain_matches(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Email from @noreply.github.com matches the notification sender-domain rule."""
        mail = MailEvent(
            message_id="m1",
            sender="notify@noreply.github.com",
            subject="Anything",
            body="Body text that is long enough to avoid the structural rule.",
        )
        result = rule_classifier.match(mail)
        labels = {m.label for m in result.matches}
        assert "notification" in labels
        assert result.selected is not None
        assert result.selected.label == "notification"
        assert result.selected.rule_type == "sender_domains"
        assert result.selected.confidence == pytest.approx(0.85)

    def test_unknown_domain_does_not_match(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Email from an unlisted domain does not match any sender-domain rule."""
        mail = MailEvent(
            message_id="m2",
            sender="someone@unknown.com",
            subject="Anything",
            body="Body text that is long enough to avoid the structural rule.",
        )
        result = rule_classifier.match(mail)
        domain_matches = [m for m in result.matches if m.rule_type == "sender_domains"]
        assert domain_matches == []


class TestSubjectPatternRulesIntegration:
    """subject_patterns.yaml is loaded and matches clean subject regexes."""

    def test_urgent_subject_matches(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Subject containing 'URGENT' matches the action_required rule."""
        mail = MailEvent(
            message_id="s1",
            sender="ops@unknown.com",
            subject="URGENT: please review",
            body="Body text that is long enough to avoid the structural rule.",
        )
        result = rule_classifier.match(mail)
        labels = {m.label for m in result.matches}
        assert "action_required" in labels

    def test_out_of_office_subject_matches(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Subject containing 'Out of Office' matches the noise rule."""
        mail = MailEvent(
            message_id="s2",
            sender="captain@entity.com",
            subject="Out of Office",
            body="Body text that is long enough to avoid the structural rule.",
        )
        result = rule_classifier.match(mail)
        labels = {m.label for m in result.matches}
        assert "noise" in labels

    def test_re_prefixed_subject_still_matches(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Reply-prefixed subject is normalized before matching."""
        mail = MailEvent(
            message_id="s3",
            sender="ops@unknown.com",
            subject="Re[2]: URGENT update",
            body="Body text that is long enough to avoid the structural rule.",
        )
        result = rule_classifier.match(mail)
        labels = {m.label for m in result.matches}
        assert "action_required" in labels


class TestBodyKeywordRulesIntegration:
    """body_keywords.yaml is loaded and matches keyword OR rules."""

    def test_action_required_or_match(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Body containing 'please approve' matches the action_required OR rule."""
        mail = MailEvent(
            message_id="b1",
            sender="ops@unknown.com",
            subject="Subject",
            body="Please approve the attached document for the upcoming deadline.",
        )
        result = rule_classifier.match(mail)
        labels = {m.label for m in result.matches}
        assert "action_required" in labels

    def test_noise_automated_message_match(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Body containing 'do not reply' matches the noise rule."""
        mail = MailEvent(
            message_id="b2",
            sender="ops@unknown.com",
            subject="Subject",
            body="This is an automated message. Do not reply to this email.",
        )
        result = rule_classifier.match(mail)
        labels = {m.label for m in result.matches}
        assert "noise" in labels


class TestStructuralRulesIntegration:
    """structural.yaml is loaded and conditions are evaluated."""

    def test_short_body_matches_noise(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Structural rule with body_length < 50 matches when body is short."""
        mail = MailEvent(
            message_id="st1",
            sender="ops@unknown.com",
            subject="Subject",
            body="Hi.",
        )
        result = rule_classifier.match(mail)
        labels = {m.label for m in result.matches}
        assert "noise" in labels

    def test_long_body_no_structural_match(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Long body (>= 50 chars) matches no structural rule."""
        mail = MailEvent(
            message_id="st2",
            sender="ops@unknown.com",
            subject="Subject",
            body="A" * 600,
        )
        result = rule_classifier.match(mail)
        structural_matches = [m for m in result.matches if m.rule_type == "structural"]
        assert structural_matches == []


# ---------------------------------------------------------------------------
# Tests: conflict resolution across multiple rule types
# ---------------------------------------------------------------------------


class TestConflictResolutionIntegration:
    """Multiple matches are resolved via confidence + rule-type priority."""

    def test_high_confidence_rule_wins(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Subject pattern (0.89) wins over sender-domain rule (0.85)."""
        # @noreply.github.com → notification (0.85, sender_domains)
        # Subject "Delivery Status Notification" → noise (0.89, subject_patterns)
        mail = MailEvent(
            message_id="c1",
            sender="notify@noreply.github.com",
            subject="Delivery Status Notification",
            body="Body text that is long enough to avoid the structural rule.",
        )
        result = rule_classifier.match(mail)
        assert result.selected is not None
        assert result.selected.label == "noise"
        assert result.selected.confidence == pytest.approx(0.89)
        assert result.selected.rule_type == "subject_patterns"
        assert len(result.matches) >= 2
        assert result.conflict_logged is True

    def test_no_match_returns_empty_result(
        self, rule_classifier: RuleClassifier, rules_dir_exists: Path
    ) -> None:
        """Email matching no rules returns an empty result with selected=None."""
        mail = MailEvent(
            message_id="c2",
            sender="unknown@nowhere.com",
            subject="Random unrelated subject",
            body="Random unrelated body text that is long enough to avoid structural rules.",
        )
        result = rule_classifier.match(mail)
        assert result.matches == []
        assert result.selected is None
        assert result.conflict_logged is False
