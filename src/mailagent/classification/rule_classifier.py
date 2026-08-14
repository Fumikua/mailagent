"""Rule-based classifier: YAML-driven sender-domain / subject-pattern / body-keyword / structural rules.

The :class:`RuleClassifier` loads four YAML rule files from a configurable
directory, matches incoming :class:`MailEvent` instances against all rule
types, resolves conflicts via confidence + rule-type priority, and implements
the :class:`Classifier` Protocol (``source="rules"``).

Rule files are hot-reloadable: file mtimes are polled every 5 seconds and
changed files are reloaded automatically. Invalid YAML preserves the
previously loaded rules and logs a warning.

Rule type priority (for conflict resolution at equal confidence):
    ``structural > sender_domains > subject_patterns > body_keywords``
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, Sequence, TypeVar

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, ValidationError

from mailagent.domain.models import MailEvent, RuleMatch, RuleResult, TaxonomyLabel
from mailagent.domain.versioning import (
    ValidatedAssetSnapshot,
    digest_named_assets,
)
from mailagent.preprocessing.subject_normalizer import normalize_subject

from .contracts import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationRequest,
)

logger = logging.getLogger(__name__)

_RuleT = TypeVar("_RuleT", bound=BaseModel)

if TYPE_CHECKING:
    from .taxonomy import TaxonomyLoader

# ---------------------------------------------------------------------------
# Rule YAML schema models
# ---------------------------------------------------------------------------


class _FrozenRuleModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _normalize_rule_text(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("rule text must not be blank")
    return normalized


def _normalize_domain(value: str) -> str:
    return _normalize_rule_text(value).lower()


_RuleText = Annotated[str, AfterValidator(_normalize_rule_text)]
_RuleDomain = Annotated[str, AfterValidator(_normalize_domain)]


class SenderDomainRule(_FrozenRuleModel):
    """Match by sender email domain, optionally restricted by local-part regex."""

    domain: _RuleDomain
    label: _RuleText
    confidence: float = Field(default=0.95, ge=0, le=1)
    local_part_pattern: _RuleText | None = None


class SubjectPatternRule(_FrozenRuleModel):
    """Regex matched against the normalized clean subject (case-insensitive)."""

    pattern: _RuleText
    label: _RuleText
    confidence: float = Field(default=0.85, ge=0, le=1)


class BodyKeywordRule(_FrozenRuleModel):
    """Keyword matching against the email body (OR or AND semantics)."""

    keywords: tuple[_RuleText, ...] = Field(min_length=1)
    label: _RuleText
    confidence: float = Field(default=0.80, ge=0, le=1)
    match_all: StrictBool = False


class StructuralRule(_FrozenRuleModel):
    """Simple condition expression evaluated against mail-derived variables.

    Supported variables: ``has_attachments`` (bool), ``body_length`` (int),
    ``has_recipients`` (bool).
    Supported operators: ``>`` ``<`` ``>=`` ``<=`` ``==`` ``!=`` and
    boolean ``and`` / ``or`` combinations. No ``eval()`` is used.
    """

    condition: _RuleText
    label: _RuleText
    confidence: float = Field(default=0.90, ge=0, le=1)


# ---------------------------------------------------------------------------
# RuleClassifier
# ---------------------------------------------------------------------------

# File names for each rule type, in priority order for reload checks.
_RULE_FILES: dict[str, str] = {
    "structural": "structural.yaml",
    "sender_domains": "sender_domains.yaml",
    "subject_patterns": "subject_patterns.yaml",
    "body_keywords": "body_keywords.yaml",
}

# Priority for conflict resolution at equal confidence (lower = higher priority).
_PRIORITY: dict[str, int] = {
    "structural": 0,
    "sender_domains": 1,
    "subject_patterns": 2,
    "body_keywords": 3,
}

# Hot-reload poll interval (seconds).
_RELOAD_INTERVAL: float = 5.0

RuleDefinition = (
    SenderDomainRule | SubjectPatternRule | BodyKeywordRule | StructuralRule
)


def validate_rule_labels(
    rules: Sequence[RuleDefinition], valid_labels: set[str]
) -> None:
    """Reject any rule label absent from the active taxonomy snapshot."""
    invalid_labels = sorted({rule.label for rule in rules} - valid_labels)
    if invalid_labels:
        raise ValueError(
            f"rule labels are absent from active taxonomy: {', '.join(invalid_labels)}"
        )


class _RuleSnapshot(NamedTuple):
    """A fully parsed and validated set of rule assets."""

    sender_domains: tuple[SenderDomainRule, ...]
    subject_patterns: tuple[SubjectPatternRule, ...]
    body_keywords: tuple[BodyKeywordRule, ...]
    structural: tuple[StructuralRule, ...]
    compiled_subject_patterns: tuple[tuple[re.Pattern[str], SubjectPatternRule], ...]
    mtimes: tuple[tuple[str, float], ...]
    taxonomy_version: str | None


def _canonical_rules(rules: Sequence[BaseModel]) -> bytes:
    return json.dumps(
        [rule.model_dump(mode="json") for rule in rules],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _version_rule_snapshot(snapshot: _RuleSnapshot) -> str:
    return digest_named_assets(
        [
            ("rules:sender_domains", _canonical_rules(snapshot.sender_domains)),
            ("rules:subject_patterns", _canonical_rules(snapshot.subject_patterns)),
            ("rules:body_keywords", _canonical_rules(snapshot.body_keywords)),
            ("rules:structural", _canonical_rules(snapshot.structural)),
        ]
    )


class RuleClassifier:
    """YAML-driven rule classifier implementing the ``Classifier`` Protocol."""

    source = "rules"

    def __init__(
        self, rules_dir: Path, taxonomy_loader: TaxonomyLoader | None = None
    ) -> None:
        self._rules_dir = Path(rules_dir)
        self._taxonomy_loader = taxonomy_loader
        self._mtimes: dict[str, float] = {}
        self._last_check: float = 0.0
        self._sender_domain_rules: list[SenderDomainRule] = []
        self._subject_pattern_rules: list[SubjectPatternRule] = []
        self._body_keyword_rules: list[BodyKeywordRule] = []
        self._structural_rules: list[StructuralRule] = []
        self._compiled_subject_patterns: list[tuple[re.Pattern[str], SubjectPatternRule]] = []
        self._has_loaded_snapshot = False
        self._active_snapshot: ValidatedAssetSnapshot[_RuleSnapshot] | None = None
        self._load_all()

    # ------------------------------------------------------------------
    # Rule file loading
    # ------------------------------------------------------------------

    def _load_all(
        self,
        taxonomy_snapshot: ValidatedAssetSnapshot[Any] | None = None,
    ) -> None:
        """Load a complete snapshot without exposing a partially valid one."""
        try:
            snapshot = self._load_snapshot(taxonomy_snapshot)
        except ValueError as exc:
            if not self._has_loaded_snapshot:
                raise
            logger.warning("Failed to reload rule snapshot; keeping previous rules: %s", exc)
            return

        self._active_snapshot = ValidatedAssetSnapshot(
            value=snapshot,
            version=_version_rule_snapshot(snapshot),
        )
        self._sender_domain_rules = list(snapshot.sender_domains)
        self._subject_pattern_rules = list(snapshot.subject_patterns)
        self._body_keyword_rules = list(snapshot.body_keywords)
        self._structural_rules = list(snapshot.structural)
        self._compiled_subject_patterns = list(snapshot.compiled_subject_patterns)
        self._mtimes = dict(snapshot.mtimes)
        self._has_loaded_snapshot = True

    def _load_snapshot(
        self,
        taxonomy_snapshot: ValidatedAssetSnapshot[Any] | None = None,
    ) -> _RuleSnapshot:
        """Parse, validate, and compile all rule assets before committing them."""
        sender_domains, sender_mtime = self._load_file(
            SenderDomainRule, _RULE_FILES["sender_domains"]
        )
        subject_patterns, subject_mtime = self._load_file(
            SubjectPatternRule, _RULE_FILES["subject_patterns"]
        )
        body_keywords, body_mtime = self._load_file(
            BodyKeywordRule, _RULE_FILES["body_keywords"]
        )
        structural, structural_mtime = self._load_file(
            StructuralRule, _RULE_FILES["structural"]
        )

        all_rules: list[
            SenderDomainRule
            | SubjectPatternRule
            | BodyKeywordRule
            | StructuralRule
        ] = [*sender_domains, *subject_patterns, *body_keywords, *structural]
        taxonomy_version: str | None = None
        if self._taxonomy_loader is not None:
            if taxonomy_snapshot is None:
                candidate = getattr(self._taxonomy_loader, "get_snapshot", lambda: None)()
                if isinstance(candidate, ValidatedAssetSnapshot):
                    taxonomy_snapshot = candidate
            if taxonomy_snapshot is not None:
                taxonomy_tree = taxonomy_snapshot.value
                taxonomy_version = taxonomy_snapshot.version
            else:
                taxonomy_tree = self._taxonomy_loader.get_tree()
            validate_rule_labels(all_rules, taxonomy_tree.all_codes())

        try:
            compiled_subject_patterns = [
                (re.compile(rule.pattern, re.IGNORECASE), rule) for rule in subject_patterns
            ]
            for rule in sender_domains:
                if rule.local_part_pattern is not None:
                    re.compile(rule.local_part_pattern)
        except re.error as exc:
            raise ValueError(f"invalid rule regex: {exc}") from exc

        return _RuleSnapshot(
            sender_domains=tuple(sender_domains),
            subject_patterns=tuple(subject_patterns),
            body_keywords=tuple(body_keywords),
            structural=tuple(structural),
            compiled_subject_patterns=tuple(compiled_subject_patterns),
            mtimes=tuple(
                {
                    _RULE_FILES["sender_domains"]: sender_mtime,
                    _RULE_FILES["subject_patterns"]: subject_mtime,
                    _RULE_FILES["body_keywords"]: body_mtime,
                    _RULE_FILES["structural"]: structural_mtime,
                }.items()
            ),
            taxonomy_version=taxonomy_version,
        )

    def _load_file(
        self,
        model_cls: type[_RuleT],
        filename: str,
    ) -> tuple[list[_RuleT], float]:
        """Load one YAML rule file.

        Invalid or missing assets raise so the caller can preserve the preceding
        complete snapshot or fail startup.
        """
        path = self._rules_dir / filename
        if not path.exists():
            raise ValueError(f"rule file is missing: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError, UnicodeError) as exc:
            raise ValueError(f"failed to parse {path}: {exc}") from exc

        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise ValueError(f"rule file {path} must contain a list")

        try:
            rules = [model_cls(**item) for item in raw if isinstance(item, dict)]
        except ValidationError as exc:
            raise ValueError(f"failed to validate rules in {path}: {exc}") from exc

        if len(rules) != len(raw):
            raise ValueError(f"rule file {path} contains a non-mapping entry")
        try:
            return rules, path.stat().st_mtime
        except OSError as exc:
            raise ValueError(f"failed to stat {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def _check_reload(
        self,
        taxonomy_snapshot: ValidatedAssetSnapshot[Any] | None = None,
    ) -> None:
        """Poll rule file mtimes; reload all if any changed.

        Throttled to once per ``_RELOAD_INTERVAL`` seconds.
        """
        active_taxonomy_version = (
            self._active_snapshot.value.taxonomy_version
            if self._active_snapshot is not None
            else None
        )
        taxonomy_changed = (
            taxonomy_snapshot is not None
            and taxonomy_snapshot.version != active_taxonomy_version
        )
        now = time.monotonic()
        if not taxonomy_changed and now - self._last_check < _RELOAD_INTERVAL:
            return
        self._last_check = now

        changed = False
        for filename in _RULE_FILES.values():
            path = self._rules_dir / filename
            current_mtime = path.stat().st_mtime if path.exists() else 0.0
            if current_mtime != self._mtimes.get(filename, 0.0):
                changed = True
                break

        if changed or taxonomy_changed:
            logger.info("Rule file change detected; reloading all rules")
            self._load_all(taxonomy_snapshot)

    def get_snapshot(
        self,
        taxonomy_snapshot: ValidatedAssetSnapshot[Any] | None = None,
    ) -> ValidatedAssetSnapshot[_RuleSnapshot]:
        """Return the exact complete validated rule state active for a run."""

        if taxonomy_snapshot is None and self._taxonomy_loader is not None:
            candidate = getattr(self._taxonomy_loader, "get_snapshot", lambda: None)()
            if isinstance(candidate, ValidatedAssetSnapshot):
                taxonomy_snapshot = candidate
        self._check_reload(taxonomy_snapshot)
        if self._active_snapshot is None:
            raise RuntimeError("rule snapshot was not initialized")
        return self._active_snapshot

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(
        self,
        mail_event: MailEvent,
        context: dict[str, Any] | None = None,
        *,
        snapshot: ValidatedAssetSnapshot[_RuleSnapshot] | None = None,
    ) -> RuleResult:
        """Match a mail event against all loaded rule types.

        Args:
            mail_event: The incoming mail to classify.
            context: Optional context dict (e.g. from ``ClassificationRequest``)
                providing ``has_attachments`` for structural rules.

        Returns:
            A :class:`RuleResult` with all matches and the conflict-resolved
            selection.
        """
        if snapshot is None:
            self._check_reload()
            sender_domains: Sequence[SenderDomainRule] = self._sender_domain_rules
            compiled_subject_patterns: Sequence[
                tuple[re.Pattern[str], SubjectPatternRule]
            ] = self._compiled_subject_patterns
            body_keywords: Sequence[BodyKeywordRule] = self._body_keyword_rules
            structural: Sequence[StructuralRule] = self._structural_rules
        else:
            sender_domains = snapshot.value.sender_domains
            compiled_subject_patterns = snapshot.value.compiled_subject_patterns
            body_keywords = snapshot.value.body_keywords
            structural = snapshot.value.structural

        matches: list[RuleMatch] = []
        matches.extend(self._match_sender_domains(mail_event, sender_domains))
        matches.extend(
            self._match_subject_patterns(mail_event, compiled_subject_patterns)
        )
        matches.extend(self._match_body_keywords(mail_event, body_keywords))
        matches.extend(self._match_structural(mail_event, context, structural))

        selected = self.resolve_conflict(matches)
        return RuleResult(
            matches=matches,
            selected=selected,
            conflict_logged=len(matches) > 1,
        )

    def _match_sender_domains(
        self,
        mail_event: MailEvent,
        rules: Sequence[SenderDomainRule],
    ) -> list[RuleMatch]:
        """Match sender email domain (and optional local-part pattern)."""
        results: list[RuleMatch] = []
        domain = self._extract_domain(mail_event.sender)
        local_part = self._extract_local_part(mail_event.sender)
        if not domain:
            return results
        for rule in rules:
            if domain != rule.domain.lower():
                continue
            if rule.local_part_pattern:
                if not re.match(rule.local_part_pattern, local_part):
                    continue
            pattern_desc = f"domain={rule.domain}"
            if rule.local_part_pattern:
                pattern_desc += f",local_part={rule.local_part_pattern}"
            results.append(
                RuleMatch(
                    rule_type="sender_domains",
                    label=rule.label,
                    confidence=rule.confidence,
                    matched_pattern=pattern_desc,
                )
            )
        return results

    def _match_subject_patterns(
        self,
        mail_event: MailEvent,
        patterns: Sequence[tuple[re.Pattern[str], SubjectPatternRule]],
    ) -> list[RuleMatch]:
        """Match normalized clean subject against compiled regex patterns."""
        results: list[RuleMatch] = []
        normalized = normalize_subject(mail_event.subject)
        clean = normalized.clean
        if not clean:
            return results
        for compiled, rule in patterns:
            if compiled.search(clean):
                results.append(
                    RuleMatch(
                        rule_type="subject_patterns",
                        label=rule.label,
                        confidence=rule.confidence,
                        matched_pattern=rule.pattern,
                    )
                )
        return results

    def _match_body_keywords(
        self,
        mail_event: MailEvent,
        rules: Sequence[BodyKeywordRule],
    ) -> list[RuleMatch]:
        """Match body against keyword sets (OR when match_all=False, AND when True)."""
        results: list[RuleMatch] = []
        body = mail_event.body
        if not body:
            return results
        body_upper = body.upper()
        for rule in rules:
            if not rule.keywords:
                # Defensive guard for legacy/model_construct snapshots. Empty
                # match-all sets are otherwise vacuously true in Python.
                continue
            keywords_upper = [kw.upper() for kw in rule.keywords]
            if rule.match_all:
                hit = all(kw in body_upper for kw in keywords_upper)
            else:
                hit = any(kw in body_upper for kw in keywords_upper)
            if hit:
                results.append(
                    RuleMatch(
                        rule_type="body_keywords",
                        label=rule.label,
                        confidence=rule.confidence,
                        matched_pattern=",".join(rule.keywords),
                    )
                )
        return results

    def _match_structural(
        self,
        mail_event: MailEvent,
        context: dict[str, Any] | None,
        rules: Sequence[StructuralRule],
    ) -> list[RuleMatch]:
        """Match structural rules against mail-derived variables."""
        results: list[RuleMatch] = []
        variables: dict[str, Any] = {
            "has_attachments": bool((context or {}).get("has_attachments", False)),
            "body_length": len(mail_event.body),
            "has_recipients": len(mail_event.recipients) > 0,
        }
        for rule in rules:
            if self._evaluate_condition(rule.condition, variables):
                results.append(
                    RuleMatch(
                        rule_type="structural",
                        label=rule.label,
                        confidence=rule.confidence,
                        matched_pattern=rule.condition,
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def resolve_conflict(self, matches: list[RuleMatch]) -> RuleMatch | None:
        """Resolve multiple matches to a single selected match.

        Priority rules:
            1. Highest confidence wins.
            2. Ties broken by rule type priority:
               ``structural > sender_domains > subject_patterns > body_keywords``.
            3. Same label + same priority → max confidence (satisfied by sort).

        Returns ``None`` when *matches* is empty.
        """
        if not matches:
            return None
        # Sort by confidence DESC, then rule-type priority ASC (lower = higher).
        sorted_matches = sorted(
            matches,
            key=lambda m: (-m.confidence, _PRIORITY.get(m.rule_type, 99)),
        )
        return sorted_matches[0]

    # ------------------------------------------------------------------
    # Classifier Protocol implementation
    # ------------------------------------------------------------------

    async def classify(self, request: ClassificationRequest) -> ClassificationAttempt:
        """Classify via rules, returning one :class:`ClassificationAttempt`."""
        if "rules" in request.asset_snapshots:
            bound = request.asset_snapshots["rules"]
            if not isinstance(bound, ValidatedAssetSnapshot):
                return ClassificationAttempt(
                    source=self.source,
                    status=AttemptStatus.UNAVAILABLE,
                    error="no rule snapshot is compatible with the bound taxonomy",
                )
            snapshot = bound
        else:
            snapshot = self.get_snapshot()
        bound_taxonomy = request.asset_snapshots.get("taxonomy")
        if (
            isinstance(bound_taxonomy, ValidatedAssetSnapshot)
            and snapshot.value.taxonomy_version is not None
            and snapshot.value.taxonomy_version != bound_taxonomy.version
        ):
            return ClassificationAttempt(
                source=self.source,
                status=AttemptStatus.UNAVAILABLE,
                error="rule snapshot taxonomy version does not match the run",
            )

        result = self.match(
            request.mail,
            context=request.context,
            snapshot=snapshot,
        )
        if result.selected is None:
            return ClassificationAttempt(
                source=self.source,
                status=AttemptStatus.NO_MATCH,
                confidence=0.0,
                evidence={"rules_version": snapshot.version},
            )
        selected = result.selected
        label = TaxonomyLabel(
            l1_code=selected.label,
            l1_label=selected.label,
            confidence=selected.confidence,
        )
        return ClassificationAttempt(
            source=self.source,
            status=AttemptStatus.SUCCESS,
            labels=[label],
            confidence=selected.confidence,
            evidence={
                "rule_type": selected.rule_type,
                "matched_pattern": selected.matched_pattern,
                "rules_version": snapshot.version,
            },
        )

    # ------------------------------------------------------------------
    # Sender address parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_email_address(sender: str) -> str:
        """Extract the bare email address from a sender string.

        Handles ``Name <user@domain>`` and bare ``user@domain`` formats.
        """
        if "<" in sender and ">" in sender:
            start = sender.index("<") + 1
            end = sender.index(">", start)
            return sender[start:end].strip()
        return sender.strip()

    @classmethod
    def _extract_domain(cls, sender: str) -> str:
        """Extract the domain part (after ``@``) from the sender string."""
        addr = cls._extract_email_address(sender)
        if "@" not in addr:
            return ""
        return addr.rsplit("@", 1)[1].strip().lower()

    @classmethod
    def _extract_local_part(cls, sender: str) -> str:
        """Extract the local part (before ``@``) from the sender string."""
        addr = cls._extract_email_address(sender)
        if "@" not in addr:
            return ""
        return addr.rsplit("@", 1)[0].strip()

    # ------------------------------------------------------------------
    # Safe structural condition evaluator
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_condition(condition: str, variables: dict[str, Any]) -> bool:
        """Safely evaluate a simple structural condition.

        Supports:
            - Bare boolean variables: ``has_attachments``
            - Numeric comparisons: ``body_length > 100``, ``body_length >= 50``
            - AND / OR combinations: ``has_attachments and body_length > 100``

        No ``eval()`` is used; the expression is split and compared manually.
        Unsupported syntax evaluates to ``False``.
        """
        # Split on ' or ' (lower precedence) first, then ' and '.
        or_parts = re.split(r"\s+or\s+", condition, flags=re.IGNORECASE)
        for or_part in or_parts:
            and_parts = re.split(r"\s+and\s+", or_part, flags=re.IGNORECASE)
            if all(RuleClassifier._eval_atom(part.strip(), variables) for part in and_parts):
                return True
        return False

    @staticmethod
    def _eval_atom(atom: str, variables: dict[str, Any]) -> bool:
        """Evaluate a single comparison atom or bare boolean variable."""
        atom = atom.strip()
        if not atom:
            return False
        # Check comparison operators (two-char operators first to avoid prefix matches).
        for op in (">=", "<=", "==", "!=", ">", "<"):
            idx = atom.find(op)
            if idx >= 0:
                left_str = atom[:idx].strip()
                right_str = atom[idx + len(op):].strip()
                left_val: Any = variables.get(left_str, 0)
                try:
                    right_val: Any = float(right_str)
                except ValueError:
                    right_val = variables.get(right_str, 0)
                try:
                    left_num = float(left_val)
                    right_num = float(right_val)
                except (TypeError, ValueError):
                    return False
                if op == ">=":
                    return left_num >= right_num
                if op == "<=":
                    return left_num <= right_num
                if op == "==":
                    return left_num == right_num
                if op == "!=":
                    return left_num != right_num
                if op == ">":
                    return left_num > right_num
                if op == "<":
                    return left_num < right_num
        # Bare boolean variable.
        return bool(variables.get(atom, False))
