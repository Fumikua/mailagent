from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from mailagent.classification import AttemptStatus, ClassificationRequest
from mailagent.classification.llm_classifier import LLMClassifier
from mailagent.domain.models import MailEvent
from mailagent.llm.client import LLMClient, LLMTimeoutError
from mailagent.classification.taxonomy import TaxonomyLoader


@pytest.fixture
def taxonomy_loader(tmp_path: Path) -> TaxonomyLoader:
    taxonomy_file = tmp_path / "taxonomy.yaml"
    taxonomy_file.write_text(
        """
nodes:
  - code: schedule
    label: 船期
    description: STATUS 变更/船期调整
    selection_guidance:
      - Prefer this category when its configured intent is primary
  - code: noise
    label: 噪声
    description: 不处理
    exclusive: true
""",
        encoding="utf-8",
    )
    return TaxonomyLoader(taxonomy_file)


def _request() -> ClassificationRequest:
    return ClassificationRequest(
        mail=MailEvent(
            message_id="llm-classifier-1",
            sender="ops@example.com",
            subject="Berlin Example STATUS CHANGE",
            body="STATUS updated to 2026-08-01.",
        )
    )


def _tool_response(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(arguments),
                            }
                        }
                    ]
                }
            }
        ]
    }


def _complete_arguments(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "labels": [
            {
                "l1_code": "schedule",
                "confidence": 0.91,
                "reasoning": "STATUS update",
            }
        ],
        "meta": {
            "urgency": "medium",
            "language": "en",
            "sentiment": "neutral",
            "has_attachments": False,
            "needs_human_review": False,
        },
    }
    arguments.update(overrides)
    return arguments


async def test_llm_confidence_remains_raw_for_acceptance(
    taxonomy_loader: TaxonomyLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance confidence is the LLM's clamped raw score, not its audit calibration."""
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="test-model")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="test-model")

    async def fake_chat_completion(**kwargs: Any) -> dict[str, Any]:
        return _tool_response(
            _complete_arguments(
                labels=[
                    {
                        "l1_code": "schedule",
                        "confidence": 0.60,
                        "reasoning": "STATUS mentioned",
                    }
                ]
            )
        )

    monkeypatch.setattr(client, "chat_completion", fake_chat_completion)

    attempt = await classifier.classify(_request())

    assert attempt.confidence == 0.60
    assert attempt.labels[0].confidence == 0.60
    assert attempt.calibration_log is not None
    assert attempt.calibration_log.raw == 0.60
    assert attempt.calibration_log.calibrated == 0.70


def test_calibration_log_uses_highest_valid_label(taxonomy_loader: TaxonomyLoader) -> None:
    """Calibration audit follows the top retained label, not an invalid response entry."""
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="test-model")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="test-model")

    attempt = classifier._post_process(
        _complete_arguments(
            labels=[
                {"l1_code": "unknown", "confidence": 0.99, "reasoning": "invalid"},
                {"l1_code": "schedule", "confidence": 0.82, "reasoning": "valid"},
            ]
        ),
        latency_ms=1,
    )

    assert attempt.calibration_log is not None
    assert attempt.calibration_log.raw == 0.82


async def test_llm_classifier_returns_one_source_tagged_attempt(
    taxonomy_loader: TaxonomyLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="test-model")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="test-model")

    async def fake_chat_completion(**kwargs: Any) -> dict[str, Any]:
        return _tool_response(
            {
                "labels": [
                    {
                        "l1_code": "schedule",
                        "l2_code": None,
                        "l3_code": None,
                        "confidence": 0.96,
                        "reasoning": "STATUS change email",
                    }
                ],
                "meta": {
                    "urgency": "medium",
                    "language": "en",
                    "sentiment": "neutral",
                    "has_attachments": False,
                    "needs_human_review": False,
                },
            }
        )

    monkeypatch.setattr(client, "chat_completion", fake_chat_completion)
    attempt = await classifier.classify(_request())

    assert attempt.source == "llm"
    assert attempt.status == AttemptStatus.SUCCESS
    assert attempt.labels[0].l1_code == "schedule"
    assert attempt.labels[0].l2_code is None
    assert attempt.labels[0].l3_code is None
    assert attempt.confidence == 0.96
    assert "entitys" not in attempt.evidence


async def test_llm_classifier_uses_one_bound_taxonomy_for_prompt_and_validation(
    taxonomy_loader: TaxonomyLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="test-model")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="test-model")
    bound = taxonomy_loader.get_snapshot()
    request = _request().model_copy(
        update={"asset_snapshots": {"taxonomy": bound}}
    )

    async def change_taxonomy_after_prompt(**kwargs: Any) -> dict[str, Any]:
        assert "[schedule]" in kwargs["messages"][-1]["content"]
        taxonomy_loader.path.write_text(
            "nodes:\n  - code: notification\n    label: 通知\n",
            encoding="utf-8",
        )
        taxonomy_loader._last_check = 0
        taxonomy_loader.get_tree()
        return _tool_response(
            _complete_arguments(
                labels=[
                    {
                        "l1_code": "schedule",
                        "confidence": 0.91,
                        "reasoning": "STATUS change",
                    }
                ]
            )
        )

    monkeypatch.setattr(client, "chat_completion", change_taxonomy_after_prompt)

    attempt = await classifier.classify(request)

    assert attempt.labels[0].l1_code == "schedule"
    assert attempt.evidence["taxonomy_version"] == bound.version


async def test_llm_classifier_reports_provider_failure_without_final_review_decision(
    taxonomy_loader: TaxonomyLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="test-model")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="test-model")

    async def fail_chat_completion(**kwargs: Any) -> dict[str, Any]:
        raise LLMTimeoutError("simulated timeout")

    monkeypatch.setattr(client, "chat_completion", fail_chat_completion)
    attempt = await classifier.classify(_request())

    assert attempt.source == "llm"
    assert attempt.status == AttemptStatus.UNAVAILABLE
    assert attempt.error == "simulated timeout"
    assert not hasattr(attempt, "needs_human_review")


@pytest.mark.parametrize(
    "arguments",
    [
        _complete_arguments(
            labels=[
                {
                    "l1_code": "schedule",
                    "confidence": None,
                    "reasoning": "STATUS update",
                }
            ]
        ),
        _complete_arguments(
            labels=[
                {
                    "l1_code": "schedule",
                    "confidence": "0.99",
                    "reasoning": "STATUS update",
                }
            ]
        ),
        _complete_arguments(
            labels=[
                {
                    "l1_code": "schedule",
                    "confidence": math.nan,
                    "reasoning": "STATUS update",
                }
            ]
        ),
        _complete_arguments(
            labels=[
                {
                    "l1_code": "schedule",
                    "confidence": math.inf,
                    "reasoning": "STATUS update",
                }
            ]
        ),
        _complete_arguments(
            labels=[
                {
                    "l1_code": "schedule",
                    "confidence": -math.inf,
                    "reasoning": "STATUS update",
                }
            ]
        ),
        _complete_arguments(labels=None),
        _complete_arguments(labels=["schedule"]),
        _complete_arguments(
            labels=[{"l1_code": "schedule", "reasoning": "missing confidence"}]
        ),
        _complete_arguments(meta=None),
        _complete_arguments(meta={"needs_human_review": "false"}),
    ],
    ids=[
        "null-confidence",
        "string-confidence",
        "nan-confidence",
        "positive-infinity",
        "negative-infinity",
        "null-labels",
        "malformed-label",
        "missing-confidence",
        "null-meta",
        "malformed-meta",
    ],
)
async def test_malformed_tool_output_is_unavailable_and_review_safe(
    taxonomy_loader: TaxonomyLoader,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, Any],
) -> None:
    client = LLMClient(
        base_url="https://api.example.com", api_key="test", model="test-model"
    )
    classifier = LLMClassifier(client, taxonomy_loader, model_name="test-model")

    async def malformed_chat_completion(**kwargs: Any) -> dict[str, Any]:
        return _tool_response(arguments)

    monkeypatch.setattr(client, "chat_completion", malformed_chat_completion)

    attempt = await classifier.classify(_request())

    assert attempt.status == AttemptStatus.UNAVAILABLE
    assert attempt.labels == []
    assert attempt.confidence == 0.0
    assert attempt.meta.needs_human_review is True
    assert attempt.error is not None
    assert attempt.error.startswith("malformed_classify_email_output")


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [None]},
        {"choices": [{"message": {"tool_calls": [None]}}]},
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "[]"}}]}}]},
    ],
)
async def test_malformed_response_envelope_never_escapes_classification(
    taxonomy_loader: TaxonomyLoader,
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
) -> None:
    client = LLMClient(
        base_url="https://api.example.com", api_key="test", model="test-model"
    )
    classifier = LLMClassifier(client, taxonomy_loader, model_name="test-model")

    async def malformed_chat_completion(**kwargs: Any) -> dict[str, Any]:
        return response

    monkeypatch.setattr(client, "chat_completion", malformed_chat_completion)

    attempt = await classifier.classify(_request())

    assert attempt.status == AttemptStatus.UNAVAILABLE
    assert attempt.meta.needs_human_review is True
    assert attempt.error is not None
    assert attempt.error.startswith("malformed_classify_email_output")


def test_truncate_body_returns_full_body_under_cap(taxonomy_loader: TaxonomyLoader) -> None:
    """Bodies at or below ``body_max_chars`` pass through untouched."""
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="m")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="m", body_max_chars=1000)

    body = "x" * 1000
    assert classifier._truncate_body(body) == body
    assert classifier._truncate_body("short") == "short"


def test_truncate_body_caps_overlong_body_with_head_tail(taxonomy_loader: TaxonomyLoader) -> None:
    """Overlong bodies are head+tail truncated with a ``[truncated]`` marker."""
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="m")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="m", body_max_chars=1000)

    body = "HEAD" + "x" * 2000 + "TAIL"
    truncated = classifier._truncate_body(body)

    # Result must be strictly shorter than the cap, contain the marker, and
    # preserve both the original head and the original tail.
    assert len(truncated) < len(body)
    assert "[truncated]" in truncated
    assert truncated.startswith("HEAD")
    assert truncated.endswith("TAIL")
    # head_len + marker + tail_len should not exceed the configured cap.
    head_len = 1000 - classifier._body_tail_chars
    assert truncated == f"{'HEAD' + 'x' * (head_len - 4)}\n[truncated]\n{'x' * (classifier._body_tail_chars - 4) + 'TAIL'}"


def test_truncate_body_respects_vertical_specific_threshold(taxonomy_loader: TaxonomyLoader) -> None:
    """Different verticals can set different caps; 8000 vs 16000 must yield different truncation points."""
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="m")
    tight_cap_clf = LLMClassifier(client, taxonomy_loader, model_name="m", body_max_chars=8000)
    default_clf = LLMClassifier(client, taxonomy_loader, model_name="m", body_max_chars=16_000)

    body = "A" * 12_000  # exceeds 8000 but not 16000
    assert "[truncated]" in tight_cap_clf._truncate_body(body)
    assert default_clf._truncate_body(body) == body  # no truncation under default cap


def test_build_messages_uses_truncated_body(taxonomy_loader: TaxonomyLoader) -> None:
    """The user prompt fed to the LLM must carry the truncated body, not the raw one."""
    from mailagent.classification import ClassificationRequest

    client = LLMClient(base_url="https://api.example.com", api_key="test", model="m")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="m", body_max_chars=500)

    long_body = "B" * 5000
    request = ClassificationRequest(
        mail=MailEvent(
            message_id="trunc-1",
            sender="ops@example.com",
            subject="long body",
            body=long_body,
        )
    )
    messages = classifier._build_messages(request)
    user_content = messages[-1]["content"]
    assert "[truncated]" in user_content
    assert "B" * 5000 not in user_content  # full body must not leak through


def test_build_messages_keeps_core_vertical_agnostic_and_uses_configured_guidance(
    taxonomy_loader: TaxonomyLoader,
) -> None:
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="m")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="m")

    system_message, user_message = classifier._build_messages(_request())

    assert "authoritative business configuration" in system_message["content"]
    assert "selection guidance" in system_message["content"]
    for vertical_term in (
        "PRE-EVENT",
        "entity_report",
        "Location cancellation",
        "加班船",
        "Staff update",
        "Delivery Status Notification",
    ):
        assert vertical_term not in system_message["content"]
    assert "Prefer this category when its configured intent is primary" in user_message["content"]


def test_exclusive_label_behavior_comes_from_taxonomy_config(
    taxonomy_loader: TaxonomyLoader,
) -> None:
    client = LLMClient(base_url="https://api.example.com", api_key="test", model="m")
    classifier = LLMClassifier(client, taxonomy_loader, model_name="m")

    attempt = classifier._post_process(
        _complete_arguments(
            labels=[
                {"l1_code": "schedule", "confidence": 0.85, "reasoning": "configured"},
                {"l1_code": "noise", "confidence": 0.90, "reasoning": "exclusive"},
            ]
        ),
        latency_ms=1,
    )

    assert [label.l1_code for label in attempt.labels] == ["noise"]
