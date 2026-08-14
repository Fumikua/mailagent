"""Privacy-safe deterministic classification evaluation contracts."""

from .corpus import (
    GoldCorpusManifest,
    GoldExample,
    PredictionRecord,
    load_gold_manifest,
    load_prediction_snapshot,
)
from .metrics import (
    EvaluationReport,
    GateFailure,
    GateResult,
    MetricResult,
    ReleaseGate,
    evaluate_predictions,
    wilson_interval,
)
from .report import render_evaluation_markdown, write_evaluation_reports

__all__ = [
    "EvaluationReport",
    "GateFailure",
    "GateResult",
    "GoldCorpusManifest",
    "GoldExample",
    "MetricResult",
    "PredictionRecord",
    "ReleaseGate",
    "evaluate_predictions",
    "load_gold_manifest",
    "load_prediction_snapshot",
    "render_evaluation_markdown",
    "wilson_interval",
    "write_evaluation_reports",
]
