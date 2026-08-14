"""Stable JSON and Markdown rendering for classification evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import EvaluationReport, MetricResult

JSON_REPORT_NAME = "classification-evaluation.json"
MARKDOWN_REPORT_NAME = "classification-evaluation.md"


def _format_ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _format_interval(metric: MetricResult) -> str:
    if metric.wilson_interval is None:
        return "N/A"
    lower, upper = metric.wilson_interval
    return f"[{lower:.6f}, {upper:.6f}]"


def render_evaluation_markdown(report: EvaluationReport) -> str:
    """Render all report diagnostics in deterministic, metadata-only Markdown."""

    lines = [
        "# Classification Evaluation",
        "",
        f"- Corpus version: `{report.corpus_version}`",
        f"- Taxonomy version: `{report.taxonomy_version}`",
        f"- Evaluated split: `{report.evaluated_split}`",
        f"- Excluded non-test examples: {report.excluded_non_test_examples}",
        f"- Total examples: {report.total_examples}",
        f"- Auto-accepted examples: {report.auto_accepted_examples}",
        f"- Reviewed examples: {report.reviewed_examples}",
        f"- Missing predictions: {report.missing_predictions}",
        f"- Coverage: {_format_ratio(report.coverage)}",
        f"- Review rate: {_format_ratio(report.review_rate)}",
        f"- Suggestion recall: {_format_ratio(report.suggestion_recall)}",
        "",
        "## Prediction Versions",
        "",
        "| Component | Version | Count |",
        "| --- | --- | ---: |",
    ]
    for component, versions in sorted(report.prediction_version_distribution.items()):
        if not versions:
            lines.append(f"| {component} | N/A | 0 |")
            continue
        for version, count in sorted(versions.items()):
            lines.append(f"| {component} | `{version}` | {count} |")

    lines.extend(
        [
            "",
            "## Automatic Decision Metrics",
            "",
            "| Label | TP | FP | FN | Precision | Recall | Wilson 95% CI | False-positive sample IDs |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            _metric_row("micro", report.micro),
        ]
    )
    for label, metric in sorted(report.per_label.items()):
        lines.append(_metric_row(label, metric))

    lines.extend(["", "## Automatic Mismatches", ""])
    if report.mismatches:
        lines.extend(
            [
                "| Sample ID | Gold labels | Predicted labels | Strategy |",
                "| --- | --- | --- | --- |",
            ]
        )
        for mismatch in report.mismatches:
            lines.append(
                f"| `{mismatch.sample_id}` | {', '.join(mismatch.gold_labels)} | "
                f"{', '.join(mismatch.predicted_labels)} | {mismatch.strategy} |"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Taxonomy Version Mismatches", ""])
    if report.taxonomy_version_mismatches:
        lines.extend(
            [
                "| Sample ID | Expected | Prediction |",
                "| --- | --- | --- |",
            ]
        )
        for taxonomy_mismatch in report.taxonomy_version_mismatches:
            lines.append(
                f"| `{taxonomy_mismatch.sample_id}` | "
                f"`{taxonomy_mismatch.expected_taxonomy_version}` | "
                f"`{taxonomy_mismatch.prediction_taxonomy_version}` |"
            )
    else:
        lines.append("None.")

    reasons = ", ".join(report.gate.reason_codes) or "none"
    lines.extend(
        [
            "",
            "## Release Gate",
            "",
            f"- Status: **{report.gate.status.upper()}**",
            f"- Eligible: {str(report.gate.eligible).lower()}",
            f"- Eligible label decisions: {report.gate.eligible_decisions}",
            "- Precision gate statistic: Wilson 95% lower bound",
            "- Per-label eligible decisions: "
            + (
                ", ".join(
                    f"{label}={count}"
                    for label, count in sorted(
                        report.gate.label_eligible_decisions.items()
                    )
                )
                or "none"
            ),
            f"- Reason codes: {reasons}",
            "",
            "| Reason | Label | Actual support / Wilson lower bound | Threshold |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    if report.gate.failures:
        for failure in report.gate.failures:
            lines.append(
                f"| {failure.reason_code} | {failure.label or '-'} | "
                f"{_format_gate_value(failure.actual)} | "
                f"{_format_gate_value(failure.threshold)} |"
            )
    else:
        lines.append("| none | - | - | - |")
    return "\n".join(lines) + "\n"


def _metric_row(label: str, metric: MetricResult) -> str:
    false_positive_ids = ", ".join(metric.false_positive_sample_ids) or "-"
    return (
        f"| {label} | {metric.true_positive} | {metric.false_positive} | "
        f"{metric.false_negative} | {_format_ratio(metric.precision)} | "
        f"{_format_ratio(metric.recall)} | {_format_interval(metric)} | "
        f"{false_positive_ids} |"
    )


def _format_gate_value(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_evaluation_reports(
    report: EvaluationReport, output_dir: Path
) -> tuple[Path, Path]:
    """Write deterministic report artifacts under fixed filenames."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_REPORT_NAME
    markdown_path = output_dir / MARKDOWN_REPORT_NAME
    payload = report.model_dump(mode="json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_evaluation_markdown(report),
        encoding="utf-8",
    )
    return json_path, markdown_path
