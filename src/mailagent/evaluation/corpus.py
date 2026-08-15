"""Validated, privacy-safe contracts for classification gold corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from mailagent.domain.models import ClassificationVersions


class GoldExample(BaseModel):
    """One adjudicated, metadata-only gold corpus example."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^[a-zA-Z0-9._-]+$")
    thread_id: str
    labels: list[str] = Field(min_length=1)
    split: Literal["development", "calibration", "test"]
    annotation_refs: list[str] = Field(min_length=2)
    adjudicated: StrictBool

    @field_validator("annotation_refs")
    @classmethod
    def reject_surrounding_annotation_reference_whitespace(
        cls, references: list[str]
    ) -> list[str]:
        if any(reference != reference.strip() for reference in references):
            raise ValueError(
                "annotation references must not contain surrounding whitespace"
            )
        if len(references) < 2:
            raise ValueError("annotation_refs must contain at least two references")
        if len(references) != len(set(references)):
            raise ValueError("annotation_refs must be unique")
        if any(not reference for reference in references):
            raise ValueError("examples require two independent annotation references")
        return references

    @field_validator("adjudicated")
    @classmethod
    def require_adjudication(cls, adjudicated: bool) -> bool:
        if not adjudicated:
            raise ValueError("examples must be adjudicated")
        return adjudicated


class GoldCorpusManifest(BaseModel):
    """A versioned collection of metadata-only gold corpus examples."""

    model_config = ConfigDict(extra="forbid")

    corpus_version: str
    taxonomy_version: str
    examples: list[GoldExample] = Field(min_length=1)


class PredictionRecord(BaseModel):
    """One saved classification result used for deterministic offline replay."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^[a-zA-Z0-9._-]+$")
    labels: list[str]
    needs_human_review: bool
    strategy: str
    versions: ClassificationVersions


def load_gold_manifest(
    path: Path,
    valid_labels: set[str],
    exclusive_labels: set[str] | None = None,
) -> GoldCorpusManifest:
    """Load and validate an authorized gold corpus manifest.

    ``valid_labels`` is supplied by the active taxonomy so the manifest never
    embeds taxonomy source content or a duplicate taxonomy definition.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = GoldCorpusManifest.model_validate(raw)

    exclusive = exclusive_labels or set()
    sample_ids: set[str] = set()
    thread_splits: dict[str, str] = {}
    for example in manifest.examples:
        if example.sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {example.sample_id}")
        sample_ids.add(example.sample_id)

        unknown_labels = set(example.labels) - valid_labels
        if unknown_labels:
            raise ValueError(f"unknown label: {sorted(unknown_labels)[0]}")

        selected_exclusive = sorted(set(example.labels) & exclusive)
        if selected_exclusive and len(set(example.labels)) != 1:
            raise ValueError(f"label {selected_exclusive[0]} must be exclusive")

        if len(set(example.annotation_refs)) < 2 or any(
            not reference.strip() for reference in example.annotation_refs
        ):
            raise ValueError("examples require two independent annotation references")

        if not example.adjudicated:
            raise ValueError("examples must be adjudicated")

        prior_split = thread_splits.setdefault(example.thread_id, example.split)
        if prior_split != example.split:
            raise ValueError("thread crosses dataset splits")

    return manifest


def load_prediction_snapshot(
    path: Path,
    valid_labels: set[str],
    exclusive_labels: set[str] | None = None,
) -> list[PredictionRecord]:
    """Load a saved prediction snapshot without invoking classification providers.

    The canonical snapshot is a JSON list. A ``{"predictions": [...]}`` envelope
    is also accepted so snapshots can carry an explicit collection boundary.
    Sample-ID membership is validated later against the selected gold manifest.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict) and set(raw) == {"predictions"}:
        rows = raw["predictions"]
    else:
        raise ValueError("prediction snapshot must be a list or predictions envelope")
    if not isinstance(rows, list):
        raise ValueError("prediction snapshot predictions must be a list")

    exclusive = exclusive_labels or set()
    predictions = [PredictionRecord.model_validate(row) for row in rows]
    for prediction in predictions:
        unknown_labels = set(prediction.labels) - valid_labels
        if unknown_labels:
            raise ValueError(f"unknown prediction label: {sorted(unknown_labels)[0]}")
        selected_exclusive = sorted(set(prediction.labels) & exclusive)
        if selected_exclusive and len(set(prediction.labels)) != 1:
            raise ValueError(
                f"prediction label {selected_exclusive[0]} must be exclusive"
            )
    return predictions
