"""Rule learner for weekly sender-domain rule discovery.

Runs as an arq cron job, scanning the last 30 days of samples to compute a
``sender_domain × label`` distribution matrix. When a domain has enough samples
(≥ 5) and a single label dominates (≥ 80%), a rule proposal is generated —
with cross-domain verification and single-domain discount applied as needed.

Proposals are written to a markdown report with checkboxes for human review.
``parse_report_and_append`` reads confirmed (checked) proposals and appends them
to ``sender_domains.yaml``.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from ..core.rule_classifier import SenderDomainRule, validate_rule_labels
from ..domain.models import SampleRecord
from ..domain.versioning import ValidatedAssetSnapshot
from .config import RulesSettings
from .vector_store import VectorStore

if TYPE_CHECKING:
    from ..llm.taxonomy import TaxonomyTree

logger = logging.getLogger(__name__)


def append_confirmed_sender_domain_rules(
    report_path: Path,
    rules_dir: Path,
    taxonomy_snapshot: ValidatedAssetSnapshot[TaxonomyTree],
) -> int:
    """Append checked Markdown proposals after strict validation and dedupe."""
    candidates = _parse_checked_sender_domain_rules(report_path)
    rules_file = rules_dir / "sender_domains.yaml"
    existing_rules = _load_sender_domain_rules(rules_file) if rules_file.exists() else []

    seen = {_rule_identity(rule) for rule in existing_rules}
    additions: list[SenderDomainRule] = []
    for candidate in candidates:
        identity = _rule_identity(candidate)
        if identity not in seen:
            additions.append(candidate)
            seen.add(identity)

    all_rules = [*existing_rules, *additions]
    validate_rule_labels(all_rules, taxonomy_snapshot.value.all_codes())
    if not additions:
        return 0

    content = yaml.safe_dump(
        [rule.model_dump(exclude_none=True) for rule in all_rules],
        allow_unicode=True,
        sort_keys=False,
    )
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=rules_file.parent,
            prefix=f".{rules_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(rules_file)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return len(additions)


def _parse_checked_sender_domain_rules(report_path: Path) -> list[SenderDomainRule]:
    """Read exactly one non-empty YAML list from every checked proposal."""
    lines = report_path.read_text(encoding="utf-8").splitlines()
    candidates: list[SenderDomainRule] = []
    index = 0

    while index < len(lines):
        checkbox = lines[index].strip().lower()
        if not checkbox.startswith("- ["):
            index += 1
            continue

        section_start = index + 1
        section_end = section_start
        while section_end < len(lines) and not lines[section_end].strip().startswith("- ["):
            section_end += 1

        if checkbox.startswith("- [x]"):
            section = lines[section_start:section_end]
            yaml_fences = [
                offset for offset, line in enumerate(section) if line.strip() == "```yaml"
            ]
            if len(yaml_fences) != 1:
                raise ValueError(
                    "checked proposal must contain exactly one YAML fence"
                )

            fence_start = yaml_fences[0] + 1
            fence_end = fence_start
            while fence_end < len(section) and section[fence_end].strip() != "```":
                fence_end += 1
            if fence_end == len(section):
                raise ValueError("unterminated YAML fence in checked proposal")

            parsed = _decode_sender_domain_rule_list(
                "\n".join(section[fence_start:fence_end]), "checked proposal"
            )
            if not parsed:
                raise ValueError(
                    "checked proposal YAML fence must contain a non-empty rule list"
                )
            candidates.extend(parsed)

        index = section_end

    return candidates


def _load_sender_domain_rules(rules_file: Path) -> list[SenderDomainRule]:
    """Load a complete sender-domain rules file through the shared schema."""
    return _decode_sender_domain_rule_list(
        rules_file.read_text(encoding="utf-8"),
        f"existing rules file {rules_file}",
        allow_empty_document=True,
    )


def _decode_sender_domain_rule_list(
    yaml_text: str,
    source: str,
    *,
    allow_empty_document: bool = False,
) -> list[SenderDomainRule]:
    try:
        raw_rules = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}") from exc

    if raw_rules is None and allow_empty_document:
        return []
    if not isinstance(raw_rules, list):
        raise ValueError(f"{source} must contain a list of sender-domain rules")

    rules: list[SenderDomainRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError(f"{source} must contain sender-domain rule mappings")
        try:
            rules.append(SenderDomainRule.model_validate(raw_rule))
        except ValidationError as exc:
            raise ValueError(f"invalid sender-domain rule in {source}") from exc
    return rules


def _rule_identity(rule: SenderDomainRule) -> tuple[str, str]:
    """Return the normalized identity used for sender-domain rule deduplication."""
    return rule.domain, rule.label


class RuleLearner:
    """Weekly scanner that proposes sender-domain rules from sample data."""

    def __init__(
        self,
        vector_store: VectorStore,
        settings: RulesSettings,
    ) -> None:
        self.vector_store = vector_store
        self.settings = settings
        self._rules_dir = Path(settings.rules_dir)
        self._reports_dir = Path("./reports")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_weekly_scan(self) -> str:
        """Scan recent samples and generate a rule proposal report.

        Returns the report file path.
        """
        samples = await self._fetch_all_samples()
        distribution = self._compute_distribution(samples)

        # Compute per-domain totals for ratio calculation.
        domain_totals: dict[str, int] = {}
        for (domain, _), count in distribution.items():
            domain_totals[domain] = domain_totals.get(domain, 0) + count

        proposals: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()

        for (domain, label), count in sorted(distribution.items()):
            if (domain, label) in seen_pairs:
                continue
            seen_pairs.add((domain, label))

            total = domain_totals.get(domain, 0)
            if total == 0:
                continue

            ratio = count / total
            min_samples = self.settings.autolearn_min_samples
            min_ratio = self.settings.autolearn_min_ratio

            # Basic trigger: enough samples and dominant ratio.
            if count < min_samples or ratio < min_ratio:
                continue

            cross_domain = self._check_cross_domain(label, distribution, domain)

            if not cross_domain and ratio < 0.9:
                # Cross-domain failed and ratio not high enough for discount.
                continue

            proposal = self._generate_proposal(domain, label, ratio, cross_domain)
            proposals.append(proposal)

        # Generate report.
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = (
            self._reports_dir
            / f"rule_proposals_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
        )
        self._generate_report(proposals, len(samples), report_path)
        logger.info(
            "rule scan complete: %d proposals, report at %s",
            len(proposals),
            report_path,
        )
        return str(report_path)

    def parse_report_and_append(
        self,
        report_path: Path,
        taxonomy_snapshot: ValidatedAssetSnapshot[TaxonomyTree],
    ) -> int:
        """Append strictly validated, checked proposals to this learner's rules."""
        return append_confirmed_sender_domain_rules(
            report_path, self._rules_dir, taxonomy_snapshot
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_distribution(
        self, samples: list[SampleRecord]
    ) -> dict[tuple[str, str], int]:
        """Build a ``(sender_domain, label_l1) → count`` distribution matrix."""
        distribution: dict[tuple[str, str], int] = {}
        for s in samples:
            key = (s.sender_domain, s.label_l1)
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def _check_cross_domain(
        self,
        label: str,
        distribution: dict[tuple[str, str], int],
        target_domain: str,
    ) -> bool:
        """Check if ≥ 2 other sender_domains also have this label."""
        other_domains: set[str] = set()
        for (domain, lbl), count in distribution.items():
            if lbl == label and domain != target_domain and count > 0:
                other_domains.add(domain)
        return len(other_domains) >= 2

    def _generate_proposal(
        self,
        domain: str,
        label: str,
        ratio: float,
        cross_domain: bool,
    ) -> dict:
        """Generate a rule proposal dict with YAML fragment and metadata."""
        confidence = ratio
        single_domain = False

        if not cross_domain:
            confidence = ratio * 0.9
            single_domain = True

        yaml_fragment = (
            f"- domain: {domain}\n"
            f"  label: {label}\n"
            f"  confidence: {confidence:.2f}\n"
        )

        return {
            "domain": domain,
            "label": label,
            "ratio": ratio,
            "confidence": confidence,
            "cross_domain": cross_domain,
            "single_domain": single_domain,
            "yaml_fragment": yaml_fragment,
        }

    def _generate_report(
        self,
        proposals: list[dict],
        total_samples: int,
        output_path: Path,
    ) -> None:
        """Write a markdown report of rule proposals with confirmation checkboxes."""
        now = datetime.now(timezone.utc)
        lines: list[str] = [
            f"# Rule Proposals Report — {now.strftime('%Y-%m-%d')}",
            "",
            f"**Date**: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Total samples scanned**: {total_samples}",
            f"**Trigger conditions**: "
            f"min_samples={self.settings.autolearn_min_samples}, "
            f"min_ratio={self.settings.autolearn_min_ratio}",
            "",
        ]

        if proposals:
            lines.append("## Proposals")
            lines.append("")
            for idx, p in enumerate(proposals, 1):
                lines.append(
                    f"### Proposal {idx}: {p['domain']} → {p['label']}"
                )
                lines.append("")
                lines.append(f"- **Domain**: {p['domain']}")
                lines.append(f"- **Label**: {p['label']}")
                lines.append(f"- **Ratio**: {p['ratio']:.1%}")
                lines.append(f"- **Confidence**: {p['confidence']:.2f}")
                lines.append(
                    f"- **Cross-domain verified**: {p['cross_domain']}"
                )
                if p["single_domain"]:
                    lines.append(
                        f"- **Single-domain**: {p['single_domain']} "
                        "(confidence discounted × 0.9)"
                    )
                lines.append("")
                lines.append("- [ ] 确认添加 (Confirm and append)")
                lines.append("")
                lines.append("```yaml")
                lines.append(p["yaml_fragment"].rstrip())
                lines.append("```")
                lines.append("")
        else:
            lines.append("## Proposals")
            lines.append("")
            lines.append("无规则提议 (No rule proposals).")
            lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    async def _fetch_all_samples(self) -> list[SampleRecord]:
        """Paginate through ``get_samples`` to fetch all samples."""
        samples: list[SampleRecord] = []
        page = 1
        page_size = 1000
        while True:
            batch = await self.vector_store.get_samples(
                page=page, page_size=page_size
            )
            if not batch:
                break
            samples.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return samples
