"""Deterministic classification evaluation metric tests."""

from __future__ import annotations

import pytest

from mailagent.domain.models import ClassificationVersions
from mailagent.evaluation import (
    GoldCorpusManifest,
    GoldExample,
    PredictionRecord,
    ReleaseGate,
    evaluate_predictions,
    wilson_interval,
)


def _versions(taxonomy: str = "taxonomy-v1", model: str = "model-v1") -> ClassificationVersions:
    return ClassificationVersions(
        taxonomy=taxonomy,
        rules="rules-v1",
        prompt="prompt-v1",
        model=model,
        embedding="embedding-v1",
        preprocessing="preprocessing-v1",
    )


def _example(
    sample_id: str,
    labels: list[str],
    *,
    split: str = "test",
) -> GoldExample:
    return GoldExample(
        sample_id=sample_id,
        thread_id=f"thread-{sample_id}",
        labels=labels,
        split=split,
        annotation_refs=[f"annotation-{sample_id}-a", f"annotation-{sample_id}-b"],
        adjudicated=True,
    )


def _manifest(*examples: GoldExample) -> GoldCorpusManifest:
    selected = examples or (
        _example("sample-1", ["schedule"]),
        _example("sample-2", ["schedule"]),
        _example("sample-3", ["operation"]),
        _example("sample-4", ["document"]),
    )
    return GoldCorpusManifest(
        corpus_version="corpus-v1",
        taxonomy_version="taxonomy-v1",
        examples=list(selected),
    )


def _prediction(
    sample_id: str,
    labels: list[str],
    *,
    review: bool = False,
    taxonomy: str = "taxonomy-v1",
    strategy: str = "rule_only",
    model: str = "model-v1",
) -> PredictionRecord:
    return PredictionRecord(
        sample_id=sample_id,
        labels=labels,
        needs_human_review=review,
        strategy=strategy,
        versions=_versions(taxonomy, model),
    )


def _predictions() -> list[PredictionRecord]:
    return [
        _prediction("sample-1", ["schedule"]),
        _prediction("sample-2", ["noise"]),
        _prediction("sample-3", ["operation"], review=True, strategy="all_low_review"),
    ]


def _small_gate(**overrides: object) -> ReleaseGate:
    values: dict[str, object] = {
        "overall_precision": 0.0,
        "label_precision": 0.0,
        "noise_precision": 0.0,
        "minimum_eligible": 1,
    }
    values.update(overrides)
    return ReleaseGate(**values)


def test_evaluation_counts_only_non_review_predictions_as_auto_accepts() -> None:
    report = evaluate_predictions(_manifest(), _predictions(), _small_gate())

    assert report.total_examples == 4
    assert report.auto_accepted_examples == 2
    assert report.reviewed_examples == 1
    assert report.missing_predictions == 1
    assert report.micro.true_positive == 1
    assert report.micro.false_positive == 1
    assert report.micro.false_negative == 3
    assert report.micro.precision == 0.5
    assert report.micro.recall == 0.25
    assert report.coverage == 0.5
    assert report.review_rate == 0.25
    assert report.suggestion_recall == 0.5


def test_multi_label_metrics_use_set_based_true_false_positive_and_negative_counts() -> None:
    manifest = _manifest(
        _example("sample-a", ["alpha", "beta"]),
        _example("sample-b", ["beta"]),
        _example("sample-c", ["alpha", "beta"]),
        _example("sample-d", ["beta", "gamma"]),
    )
    predictions = [
        _prediction("sample-a", ["alpha", "alpha", "gamma"]),
        _prediction("sample-b", ["beta"]),
        _prediction("sample-c", ["alpha", "beta"], review=True),
    ]

    report = evaluate_predictions(
        manifest,
        predictions,
        _small_gate(label_precision=None, noise_precision=None),
    )

    assert report.micro.true_positive == 2
    assert report.micro.false_positive == 1
    assert report.micro.false_negative == 5
    assert report.micro.precision == pytest.approx(2 / 3)
    assert report.micro.recall == pytest.approx(2 / 7)
    assert report.per_label["beta"].true_positive == 1
    assert report.per_label["beta"].false_negative == 3
    assert report.per_label["beta"].recall == 0.25
    assert report.per_label["alpha"].false_negative == 1
    assert report.per_label["gamma"].false_negative == 1
    assert report.per_label["gamma"].false_positive_sample_ids == ["sample-a"]


def test_zero_auto_accepts_have_undefined_precision_and_wilson_interval() -> None:
    manifest = _manifest(_example("sample-a", ["alpha"]))
    predictions = [_prediction("sample-a", ["alpha"], review=True)]

    report = evaluate_predictions(
        manifest,
        predictions,
        _small_gate(label_precision=None, noise_precision=None),
    )

    assert report.micro.precision is None
    assert report.micro.wilson_interval is None
    assert report.coverage == 0.0
    assert report.suggestion_recall == 1.0
    assert report.gate.status == "ineligible"
    assert report.gate.reason_codes == ["insufficient_volume"]


def test_wilson_interval_uses_the_standard_95_percent_formula() -> None:
    interval = wilson_interval(1, 2)

    assert interval is not None
    assert interval[0] == pytest.approx(0.09453120573423074)
    assert interval[1] == pytest.approx(0.9054687942657693)
    assert wilson_interval(0, 0) is None


def test_per_label_precision_and_false_positive_ids_are_deterministic() -> None:
    report = evaluate_predictions(_manifest(), _predictions(), _small_gate())

    assert list(report.per_label) == ["document", "noise", "operation", "schedule"]
    assert report.per_label["schedule"].precision == 1.0
    assert report.per_label["schedule"].false_negative == 1
    assert report.per_label["noise"].precision == 0.0
    assert report.per_label["operation"].false_negative == 1
    assert report.per_label["document"].false_negative == 1
    assert report.micro.false_positive_sample_ids == ["sample-2"]
    assert [mismatch.sample_id for mismatch in report.mismatches] == ["sample-2"]
    assert report.mismatches[0].gold_labels == ["schedule"]
    assert report.mismatches[0].predicted_labels == ["noise"]


def test_noise_precision_gate_uses_the_99_percent_threshold() -> None:
    correct = [_example(f"noise-{index:03d}", ["noise"]) for index in range(99)]
    first_error = _example("not-noise-100", ["schedule"])
    manifest = _manifest(*correct, first_error)
    predictions = [
        *[_prediction(example.sample_id, ["noise"]) for example in correct],
        _prediction(first_error.sample_id, ["noise"]),
    ]
    gate = ReleaseGate(
        overall_precision=None,
        label_precision=None,
        noise_precision=0.99,
        minimum_eligible=1,
    )

    wilson_failing = evaluate_predictions(manifest, predictions, gate)
    assert wilson_failing.per_label["noise"].precision == 0.99
    assert wilson_failing.per_label["noise"].wilson_interval is not None
    assert wilson_failing.per_label["noise"].wilson_interval[0] < 0.99
    assert wilson_failing.gate.passed is False
    assert wilson_failing.gate.reason_codes == [
        "noise_precision_below_threshold"
    ]
    assert wilson_failing.gate.failures[0].actual == pytest.approx(
        wilson_failing.per_label["noise"].wilson_interval[0]
    )

    second_error = _example("not-noise-101", ["operation"])
    failing = evaluate_predictions(
        _manifest(*correct, first_error, second_error),
        [*predictions, _prediction(second_error.sample_id, ["noise"])],
        gate,
    )
    assert failing.gate.passed is False
    assert failing.gate.reason_codes == ["noise_precision_below_threshold"]
    assert failing.gate.failures[0].label == "noise"


def test_release_gate_excludes_development_and_calibration_examples() -> None:
    development = [
        _example(f"dev-{index:03d}", ["schedule"], split="development")
        for index in range(100)
    ]
    calibration = _example("calibration-001", ["schedule"], split="calibration")
    test_example = _example("test-001", ["schedule"])
    manifest = _manifest(*development, calibration, test_example)
    predictions = [
        *[
            _prediction(example.sample_id, ["schedule"])
            for example in development
        ],
        _prediction(calibration.sample_id, ["noise"]),
        _prediction(test_example.sample_id, ["schedule"]),
    ]

    report = evaluate_predictions(
        manifest,
        predictions,
        _small_gate(minimum_eligible=100),
    )

    assert report.evaluated_split == "test"
    assert report.total_examples == 1
    assert report.excluded_non_test_examples == 101
    assert report.micro.true_positive == 1
    assert report.micro.false_positive == 0
    assert report.prediction_version_distribution["model"] == {"model-v1": 1}
    assert report.gate.status == "ineligible"
    assert "insufficient_volume" in report.gate.reason_codes


def test_every_enabled_label_requires_its_own_minimum_support() -> None:
    schedule_examples = [
        _example(f"schedule-{index:03d}", ["schedule"])
        for index in range(100)
    ]
    noise_example = _example("noise-001", ["noise"])
    manifest = _manifest(*schedule_examples, noise_example)
    predictions = [
        *[
            _prediction(example.sample_id, ["schedule"])
            for example in schedule_examples
        ],
        _prediction(noise_example.sample_id, ["noise"]),
    ]

    report = evaluate_predictions(
        manifest,
        predictions,
        ReleaseGate(
            overall_precision=0.0,
            label_precision=0.0,
            noise_precision=0.0,
            minimum_eligible=100,
        ),
    )

    assert report.gate.eligible_decisions == 101
    assert report.gate.label_eligible_decisions == {
        "noise": 1,
        "schedule": 100,
    }
    assert report.gate.status == "ineligible"
    assert report.gate.reason_codes == ["insufficient_label_support"]
    assert report.gate.failures[0].model_dump() == {
        "reason_code": "insufficient_label_support",
        "label": "noise",
        "actual": 1,
        "threshold": 100,
    }


def test_wilson_lower_bound_controls_micro_and_label_precision_gates() -> None:
    examples = [
        _example(f"schedule-{index:03d}", ["schedule"])
        for index in range(100)
    ]
    predictions = [
        _prediction(example.sample_id, ["schedule"]) for example in examples
    ]

    report = evaluate_predictions(
        _manifest(*examples),
        predictions,
        ReleaseGate(
            overall_precision=0.98,
            label_precision=0.95,
            noise_precision=None,
            minimum_eligible=100,
        ),
    )

    assert report.micro.precision == 1.0
    assert report.micro.wilson_interval is not None
    assert report.micro.wilson_interval[0] < 0.98
    assert report.per_label["schedule"].wilson_interval is not None
    assert report.per_label["schedule"].wilson_interval[0] >= 0.95
    assert report.gate.reason_codes == ["overall_precision_below_threshold"]
    assert report.gate.failures[0].actual == pytest.approx(
        report.micro.wilson_interval[0]
    )


def test_gate_is_ineligible_below_minimum_auto_accepted_label_decisions() -> None:
    report = evaluate_predictions(
        _manifest(_example("sample-a", ["alpha"])),
        [_prediction("sample-a", ["alpha"])],
        _small_gate(
            minimum_eligible=2,
            label_precision=None,
            noise_precision=None,
        ),
    )

    assert report.gate.eligible_decisions == 1
    assert report.gate.eligible is False
    assert report.gate.passed is False
    assert report.gate.status == "ineligible"
    assert report.gate.reason_codes == ["insufficient_volume"]


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            [_prediction("sample-1", ["schedule"]), _prediction("sample-1", ["schedule"])],
            "duplicate prediction sample_id: sample-1",
        ),
        ([_prediction("unknown", ["schedule"])], "unknown prediction sample_id: unknown"),
    ],
)
def test_duplicate_or_unknown_prediction_ids_fail_validation(
    predictions: list[PredictionRecord], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(_manifest(), predictions, _small_gate())


def test_prediction_version_distribution_counts_all_saved_predictions() -> None:
    predictions = [
        _prediction("sample-1", ["schedule"], model="model-a"),
        _prediction("sample-2", ["schedule"], model="model-b"),
        _prediction("sample-3", ["operation"], review=True, model="model-a"),
    ]

    report = evaluate_predictions(
        _manifest(),
        predictions,
        _small_gate(label_precision=None, noise_precision=None),
    )

    assert report.prediction_version_distribution["taxonomy"] == {"taxonomy-v1": 3}
    assert report.prediction_version_distribution["model"] == {"model-a": 2, "model-b": 1}
    assert report.prediction_version_distribution["preprocessing"] == {
        "preprocessing-v1": 3
    }


def test_taxonomy_version_mismatch_keeps_metrics_but_fails_closed() -> None:
    predictions = [
        _prediction("sample-1", ["schedule"], taxonomy="taxonomy-v2"),
        _prediction("sample-2", ["noise"]),
        _prediction("sample-3", ["operation"], review=True),
    ]

    report = evaluate_predictions(
        _manifest(),
        predictions,
        _small_gate(label_precision=None, noise_precision=None),
    )

    assert report.micro.precision == 0.5
    assert report.gate.passed is False
    assert report.gate.reason_codes == ["taxonomy_version_mismatch"]
    assert [mismatch.model_dump() for mismatch in report.taxonomy_version_mismatches] == [
        {
            "sample_id": "sample-1",
            "expected_taxonomy_version": "taxonomy-v1",
            "prediction_taxonomy_version": "taxonomy-v2",
        }
    ]


def test_gate_reports_stable_reason_codes_and_label_failure_details() -> None:
    manifest = _manifest(
        _example("sample-a", ["schedule"]),
        _example("sample-b", ["operation"]),
        _example("sample-c", ["noise"]),
    )
    predictions = [
        _prediction("sample-a", ["operation"]),
        _prediction("sample-b", ["noise"]),
        _prediction("sample-c", ["schedule"]),
    ]
    gate = ReleaseGate(
        overall_precision=0.98,
        label_precision=0.95,
        noise_precision=0.99,
        minimum_eligible=1,
    )

    report = evaluate_predictions(manifest, predictions, gate)

    assert report.gate.reason_codes == [
        "overall_precision_below_threshold",
        "label_precision_below_threshold",
        "noise_precision_below_threshold",
    ]
    assert [(failure.reason_code, failure.label) for failure in report.gate.failures] == [
        ("overall_precision_below_threshold", None),
        ("label_precision_below_threshold", "operation"),
        ("label_precision_below_threshold", "schedule"),
        ("noise_precision_below_threshold", "noise"),
    ]
