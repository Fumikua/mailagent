"""Precision-first metrics and fail-closed release gates for saved predictions."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mailagent.domain.models import ClassificationVersions

from .corpus import GoldCorpusManifest, PredictionRecord


class ReleaseGate(BaseModel):
    """Thresholds for automatic classification release decisions.

    A ``None`` precision threshold disables that part of the gate. Volume and
    taxonomy-version checks remain fail-closed regardless of threshold choices.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_precision: float | None = Field(default=0.98, ge=0, le=1)
    label_precision: float | None = Field(default=0.95, ge=0, le=1)
    noise_precision: float | None = Field(default=0.99, ge=0, le=1)
    minimum_eligible: int = Field(default=100, ge=1)


class MetricResult(BaseModel):
    """Set-based label-decision metrics for automatic predictions."""

    model_config = ConfigDict(frozen=True)

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float | None
    recall: float | None
    wilson_interval: tuple[float, float] | None
    false_positive_sample_ids: list[str]


class EvaluationMismatch(BaseModel):
    """Metadata-only details for one incorrect automatic prediction."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    gold_labels: list[str]
    predicted_labels: list[str]
    strategy: str


class TaxonomyVersionMismatch(BaseModel):
    """Metadata-only details for one stale or future prediction snapshot row."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    expected_taxonomy_version: str
    prediction_taxonomy_version: str


GateReasonCode = Literal[
    "taxonomy_version_mismatch",
    "insufficient_volume",
    "insufficient_label_support",
    "overall_precision_below_threshold",
    "label_precision_below_threshold",
    "noise_precision_below_threshold",
]


class GateFailure(BaseModel):
    """Structured details for a release-gate failure."""

    model_config = ConfigDict(frozen=True)

    reason_code: GateReasonCode
    label: str | None = None
    actual: float | int | None = None
    threshold: float | int | None = None


class GateResult(BaseModel):
    """Fail-closed outcome of applying a release gate to diagnostics."""

    model_config = ConfigDict(frozen=True)

    status: Literal["passed", "failed", "ineligible"]
    passed: bool
    eligible: bool
    eligible_decisions: int
    label_eligible_decisions: dict[str, int]
    reason_codes: list[GateReasonCode]
    failures: list[GateFailure]


class EvaluationReport(BaseModel):
    """Complete deterministic diagnostics for one prediction snapshot."""

    model_config = ConfigDict(frozen=True)

    corpus_version: str
    taxonomy_version: str
    evaluated_split: Literal["test"] = "test"
    excluded_non_test_examples: int
    prediction_version_distribution: dict[str, dict[str, int]]
    total_examples: int
    auto_accepted_examples: int
    reviewed_examples: int
    missing_predictions: int
    review_rate: float
    coverage: float
    suggestion_recall: float | None
    micro: MetricResult
    per_label: dict[str, MetricResult]
    mismatches: list[EvaluationMismatch]
    taxonomy_version_mismatches: list[TaxonomyVersionMismatch]
    gate: GateResult


class _Counts:
    """Mutable accumulator kept private so report models remain immutable."""

    def __init__(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.false_positive_sample_ids: set[str] = set()


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        (p * (1 - p) + z * z / (4 * total)) / total
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _metric_result(counts: _Counts) -> MetricResult:
    precision_denominator = counts.true_positive + counts.false_positive
    recall_denominator = counts.true_positive + counts.false_negative
    precision = (
        counts.true_positive / precision_denominator
        if precision_denominator
        else None
    )
    recall = counts.true_positive / recall_denominator if recall_denominator else None
    return MetricResult(
        true_positive=counts.true_positive,
        false_positive=counts.false_positive,
        false_negative=counts.false_negative,
        precision=precision,
        recall=recall,
        wilson_interval=wilson_interval(counts.true_positive, precision_denominator),
        false_positive_sample_ids=sorted(counts.false_positive_sample_ids),
    )


def _version_distribution(
    predictions: list[PredictionRecord],
) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {}
    for field_name in ClassificationVersions.model_fields:
        counts = Counter(
            str(value) if (value := getattr(prediction.versions, field_name)) is not None else "<none>"
            for prediction in predictions
        )
        distribution[field_name] = dict(sorted(counts.items()))
    return distribution


def _append_reason(
    reason_codes: list[GateReasonCode], reason_code: GateReasonCode
) -> None:
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)


def _apply_gate(
    micro: MetricResult,
    per_label: dict[str, MetricResult],
    taxonomy_mismatches: list[TaxonomyVersionMismatch],
    gate: ReleaseGate,
) -> GateResult:
    eligible_decisions = micro.true_positive + micro.false_positive
    micro_supported = eligible_decisions >= gate.minimum_eligible
    label_eligible_decisions = {
        label: metric.true_positive + metric.false_positive
        for label, metric in sorted(per_label.items())
    }
    labels_supported = True
    reason_codes: list[GateReasonCode] = []
    failures: list[GateFailure] = []

    if taxonomy_mismatches:
        _append_reason(reason_codes, "taxonomy_version_mismatch")

    if not micro_supported:
        _append_reason(reason_codes, "insufficient_volume")
        failures.append(
            GateFailure(
                reason_code="insufficient_volume",
                actual=eligible_decisions,
                threshold=gate.minimum_eligible,
            )
        )
    elif gate.overall_precision is not None:
        micro_lower_bound = (
            micro.wilson_interval[0]
            if micro.wilson_interval is not None
            else None
        )
        if (
            micro_lower_bound is None
            or micro_lower_bound < gate.overall_precision
        ):
            _append_reason(reason_codes, "overall_precision_below_threshold")
            failures.append(
                GateFailure(
                    reason_code="overall_precision_below_threshold",
                    actual=micro_lower_bound,
                    threshold=gate.overall_precision,
                )
            )

    ordered_label_metrics = sorted(
        per_label.items(), key=lambda item: (item[0] == "noise", item[0])
    )
    for label, metric in ordered_label_metrics:
        if label == "noise":
            threshold = gate.noise_precision
            reason_code: GateReasonCode = "noise_precision_below_threshold"
        else:
            threshold = gate.label_precision
            reason_code = "label_precision_below_threshold"
        if threshold is None:
            continue
        label_decisions = label_eligible_decisions[label]
        if label_decisions < gate.minimum_eligible:
            labels_supported = False
            _append_reason(reason_codes, "insufficient_label_support")
            failures.append(
                GateFailure(
                    reason_code="insufficient_label_support",
                    label=label,
                    actual=label_decisions,
                    threshold=gate.minimum_eligible,
                )
            )
            # Support and quality are separate states. Do not claim a Wilson
            # failure until this label has the governed decision volume.
            continue
        lower_bound = (
            metric.wilson_interval[0]
            if metric.wilson_interval is not None
            else None
        )
        if lower_bound is None or lower_bound < threshold:
            _append_reason(reason_codes, reason_code)
            failures.append(
                GateFailure(
                    reason_code=reason_code,
                    label=label,
                    actual=lower_bound,
                    threshold=threshold,
                )
            )

    eligible = micro_supported and labels_supported
    passed = eligible and not reason_codes
    status: Literal["passed", "failed", "ineligible"]
    if not eligible:
        status = "ineligible"
    elif passed:
        status = "passed"
    else:
        status = "failed"
    return GateResult(
        status=status,
        passed=passed,
        eligible=eligible,
        eligible_decisions=eligible_decisions,
        label_eligible_decisions=label_eligible_decisions,
        reason_codes=reason_codes,
        failures=failures,
    )


def evaluate_predictions(
    manifest: GoldCorpusManifest,
    predictions: list[PredictionRecord],
    gate: ReleaseGate,
) -> EvaluationReport:
    """Evaluate a saved snapshot without calling Rules, Vector, or LLM providers."""

    all_gold_by_id: dict[str, set[str]] = {}
    for example in manifest.examples:
        if example.sample_id in all_gold_by_id:
            raise ValueError(f"duplicate gold sample_id: {example.sample_id}")
        all_gold_by_id[example.sample_id] = set(example.labels)

    prediction_by_id: dict[str, PredictionRecord] = {}
    for prediction in predictions:
        if prediction.sample_id in prediction_by_id:
            raise ValueError(f"duplicate prediction sample_id: {prediction.sample_id}")
        if prediction.sample_id not in all_gold_by_id:
            raise ValueError(f"unknown prediction sample_id: {prediction.sample_id}")
        prediction_by_id[prediction.sample_id] = prediction

    test_examples = [
        example for example in manifest.examples if example.split == "test"
    ]
    gold_by_id = {
        example.sample_id: set(example.labels) for example in test_examples
    }
    test_sample_ids = set(gold_by_id)
    test_predictions = [
        prediction
        for prediction in predictions
        if prediction.sample_id in test_sample_ids
    ]

    label_names = sorted(
        {
            label
            for labels in gold_by_id.values()
            for label in labels
        }
        | {
            label
            for prediction in test_predictions
            for label in prediction.labels
        }
    )
    micro_counts = _Counts()
    label_counts = {label: _Counts() for label in label_names}
    mismatches: list[EvaluationMismatch] = []

    auto_predictions = sorted(
        (
            prediction
            for prediction in test_predictions
            if not prediction.needs_human_review
        ),
        key=lambda prediction: prediction.sample_id,
    )
    for sample_id, gold in sorted(gold_by_id.items()):
        candidate_prediction = prediction_by_id.get(sample_id)
        automatic_prediction = (
            candidate_prediction
            if candidate_prediction is not None
            and not candidate_prediction.needs_human_review
            else None
        )
        predicted = (
            set(automatic_prediction.labels)
            if automatic_prediction is not None
            else set()
        )
        true_positive = gold & predicted
        false_positive = predicted - gold
        false_negative = gold - predicted

        micro_counts.true_positive += len(true_positive)
        micro_counts.false_positive += len(false_positive)
        micro_counts.false_negative += len(false_negative)
        if false_positive:
            micro_counts.false_positive_sample_ids.add(sample_id)

        for label in true_positive:
            label_counts[label].true_positive += 1
        for label in false_positive:
            label_counts[label].false_positive += 1
            label_counts[label].false_positive_sample_ids.add(sample_id)
        for label in false_negative:
            label_counts[label].false_negative += 1

        if automatic_prediction is not None and gold != predicted:
            mismatches.append(
                EvaluationMismatch(
                    sample_id=sample_id,
                    gold_labels=sorted(gold),
                    predicted_labels=sorted(predicted),
                    strategy=automatic_prediction.strategy,
                )
            )

    taxonomy_mismatches = sorted(
        (
            TaxonomyVersionMismatch(
                sample_id=prediction.sample_id,
                expected_taxonomy_version=manifest.taxonomy_version,
                prediction_taxonomy_version=prediction.versions.taxonomy,
            )
            for prediction in test_predictions
            if prediction.versions.taxonomy != manifest.taxonomy_version
        ),
        key=lambda mismatch: mismatch.sample_id,
    )

    suggested_true_positive = sum(
        len(gold_by_id[sample_id] & set(prediction.labels))
        for sample_id, prediction in (
            (prediction.sample_id, prediction) for prediction in test_predictions
        )
    )
    gold_label_decisions = sum(len(labels) for labels in gold_by_id.values())
    suggestion_recall = (
        suggested_true_positive / gold_label_decisions
        if gold_label_decisions
        else None
    )

    micro = _metric_result(micro_counts)
    per_label = {
        label: _metric_result(label_counts[label])
        for label in label_names
    }
    gate_result = _apply_gate(micro, per_label, taxonomy_mismatches, gate)
    total_examples = len(test_examples)
    reviewed_examples = sum(
        prediction.needs_human_review for prediction in test_predictions
    )
    auto_accepted_examples = len(auto_predictions)

    return EvaluationReport(
        corpus_version=manifest.corpus_version,
        taxonomy_version=manifest.taxonomy_version,
        evaluated_split="test",
        excluded_non_test_examples=len(manifest.examples) - total_examples,
        prediction_version_distribution=_version_distribution(test_predictions),
        total_examples=total_examples,
        auto_accepted_examples=auto_accepted_examples,
        reviewed_examples=reviewed_examples,
        missing_predictions=total_examples - len(test_predictions),
        review_rate=reviewed_examples / total_examples if total_examples else 0.0,
        coverage=auto_accepted_examples / total_examples if total_examples else 0.0,
        suggestion_recall=suggestion_recall,
        micro=micro,
        per_label=per_label,
        mismatches=mismatches,
        taxonomy_version_mismatches=taxonomy_mismatches,
        gate=gate_result,
    )
