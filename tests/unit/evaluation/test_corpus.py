from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from mailagent.evaluation.corpus import GoldExample, load_gold_manifest


VALID_LABELS = {
    "entity_report",
    "schedule",
    "operation",
    "document",
    "notification",
    "noise",
}


def _gold_schema() -> dict:
    """Minimal gold-manifest JSON schema mirroring the Pydantic constraints."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "adjudicated": {
                            "type": "boolean",
                            "enum": [True],
                        },
                        "annotation_refs": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "pattern": r"^\S(.*\S)?$",
                            },
                        },
                    },
                },
            },
        },
    }


def _example(
    sample_id: str,
    *,
    thread_id: str = "thread-1",
    labels: list[str] | None = None,
    split: str = "development",
    annotation_refs: list[str] | None = None,
    adjudicated: object = True,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "thread_id": thread_id,
        "labels": labels or ["schedule"],
        "split": split,
        "annotation_refs": annotation_refs or ["annotation-a", "annotation-b"],
        "adjudicated": adjudicated,
    }


def _write_manifest(tmp_path: Path, *, examples: list[dict[str, object]]) -> Path:
    path = tmp_path / "gold-manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "corpus_version": "gold-v1",
                "taxonomy_version": "taxonomy-sha256:opaque-version",
                "examples": examples,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_rejects_thread_leakage_across_splits(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        examples=[
            _example("a", thread_id="thread-1", split="development"),
            _example("b", thread_id="thread-1", split="test"),
        ],
    )

    with pytest.raises(ValueError, match="thread crosses dataset splits"):
        load_gold_manifest(path, VALID_LABELS)


def test_manifest_rejects_noise_with_business_label(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        examples=[_example("a", labels=["noise", "schedule"])],
    )

    with pytest.raises(ValueError, match="label noise must be exclusive"):
        load_gold_manifest(path, VALID_LABELS, {"noise"})


def test_manifest_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path, examples=[_example("a"), _example("a", thread_id="thread-2")]
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_gold_manifest(path, VALID_LABELS)


def test_manifest_rejects_unknown_labels(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, examples=[_example("a", labels=["unknown_label"])])

    with pytest.raises(ValueError, match="unknown label"):
        load_gold_manifest(path, VALID_LABELS)


def test_manifest_rejects_missing_two_independent_annotation_references(
    tmp_path: Path,
) -> None:
    path = _write_manifest(
        tmp_path,
        examples=[_example("a", annotation_refs=["annotation-a", "annotation-a"])],
    )

    with pytest.raises(ValidationError, match="annotation_refs must be unique"):
        GoldExample.model_validate(
            _example("a", annotation_refs=["annotation-a", "annotation-a"])
        )
    with pytest.raises(ValueError, match="annotation_refs must be unique"):
        load_gold_manifest(path, VALID_LABELS)


def test_gold_example_rejects_partial_duplicate_annotation_refs() -> None:
    payload = _example(
        "a",
        annotation_refs=["ref-a", "ref-b", "ref-b"],
    )

    with pytest.raises(ValidationError, match="annotation_refs must be unique"):
        GoldExample.model_validate(payload)


def test_loader_rejects_partial_duplicate_annotation_refs(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        examples=[_example("a", annotation_refs=["ref-a", "ref-b", "ref-b"])],
    )

    with pytest.raises(ValueError, match="annotation_refs must be unique"):
        load_gold_manifest(path, VALID_LABELS)


def test_annotation_references_reject_surrounding_whitespace_consistently(
    tmp_path: Path,
) -> None:
    example = _example(
        "a",
        annotation_refs=["annotation-a", " annotation-a "],
    )
    path = _write_manifest(tmp_path, examples=[example])

    with pytest.raises(ValidationError, match="surrounding whitespace"):
        GoldExample.model_validate(example)
    with pytest.raises(ValueError, match="surrounding whitespace"):
        load_gold_manifest(path, VALID_LABELS)

    schema = _gold_schema()
    errors = list(
        Draft202012Validator(schema).iter_errors(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    )
    assert errors
    assert list(errors[0].absolute_path)[-1] == 1


def test_manifest_rejects_unadjudicated_examples(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, examples=[_example("a", adjudicated=False)])

    with pytest.raises(ValueError, match="must be adjudicated"):
        load_gold_manifest(path, VALID_LABELS)


@pytest.mark.parametrize("adjudicated", ["true", "false", 1, 0])
def test_adjudicated_requires_a_strict_json_boolean_true(
    tmp_path: Path,
    adjudicated: object,
) -> None:
    example = _example("a", adjudicated=adjudicated)
    path = _write_manifest(tmp_path, examples=[example])

    with pytest.raises(ValidationError):
        GoldExample.model_validate(example)
    with pytest.raises(ValueError):
        load_gold_manifest(path, VALID_LABELS)

    schema = _gold_schema()
    assert list(
        Draft202012Validator(schema).iter_errors(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    )


def test_manifest_loads_valid_multi_label_business_example(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        examples=[_example("synthetic-001", labels=["schedule", "operation"])],
    )

    manifest = load_gold_manifest(path, VALID_LABELS)

    assert manifest.corpus_version == "gold-v1"
    assert manifest.examples[0].labels == ["schedule", "operation"]
    assert manifest.examples[0].adjudicated is True
