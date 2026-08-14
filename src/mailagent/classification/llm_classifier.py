"""LLM-backed generic classifier with no vertical-specific enrichment."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mailagent.domain.calibration import calibration_log
from mailagent.domain.models import ClassificationMeta, TaxonomyLabel
from mailagent.domain.versioning import ValidatedAssetSnapshot
from mailagent.llm.client import LLMClient, LLMError, LLMResponseError
from .contracts import AttemptStatus, ClassificationAttempt, ClassificationRequest
from .taxonomy import TaxonomyLoader, TaxonomyTree

logger = logging.getLogger(__name__)

_BODY_MAX_CHARS = 16_000
_BODY_TAIL_CHARS = 8_000
_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_email",
        "description": "Classify an email against the supplied flat taxonomy.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "labels": {
                    "type": "array",
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "l1_code": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["l1_code", "confidence", "reasoning"],
                    },
                },
                "meta": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "urgency": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                        "language": {"type": "string"},
                        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                        "has_attachments": {"type": "boolean"},
                        "needs_human_review": {"type": "boolean"},
                    },
                    "required": ["urgency", "language", "sentiment", "needs_human_review"],
                },
            },
            "required": ["labels", "meta"],
        },
    },
}


class _LLMLabelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    l1_code: str = Field(min_length=1)
    l2_code: str | None = None
    l3_code: str | None = None
    confidence: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)
    reasoning: str


class _LLMMetaOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    urgency: Literal["low", "medium", "high", "urgent"]
    language: str = Field(min_length=1)
    sentiment: Literal["positive", "neutral", "negative"]
    has_attachments: bool = False
    needs_human_review: bool


class _LLMToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    labels: list[_LLMLabelOutput] = Field(max_length=5)
    meta: _LLMMetaOutput


def _validate_tool_output(arguments: object) -> _LLMToolOutput:
    try:
        return _LLMToolOutput.model_validate(arguments)
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first_error["loc"])
        raise LLMResponseError(
            "malformed_classify_email_output: "
            f"{location or '<root>'}: {first_error['msg']}"
        ) from exc


def _normalize_codes(l1: str, l2: str | None, l3: str | None) -> tuple[str, str | None, str | None]:
    """Normalize LLM-returned taxonomy codes into (l1, l2, l3) short codes.

    LLMs sometimes return full dotted paths (e.g. ``l2_code="l1.l2"``)
    instead of short codes (``l2_code="l2"``). We split each input on
    ``.``, flatten, then **deduplicate while preserving order** so that
    ``l1="a", l2="a.b", l3="a.b.c"``
    collapses to ``["a", "b", "c"]`` instead of
    ``["a", "a", "b"]``.
    """
    raw_parts: list[str] = []
    for value in (l1, l2, l3):
        if not value:
            continue
        raw_parts.extend(
            item
            for item in (part.strip() for part in value.split("."))
            if item and item.lower() not in {"-", "*", "none", "n/a", "null"}
        )
    # Deduplicate while preserving order (handles LLM returning full paths).
    seen: set[str] = set()
    parts: list[str] = []
    for item in raw_parts:
        if item not in seen:
            seen.add(item)
            parts.append(item)
    return (
        parts[0] if parts else "",
        parts[1] if len(parts) > 1 else None,
        parts[2] if len(parts) > 2 else None,
    )


class LLMClassifier:
    """Returns one LLM classification attempt; it never makes a routing decision."""

    source = "llm"
    prompt_version = "llm-classifier-v1"

    def __init__(
        self,
        llm_client: LLMClient,
        taxonomy_loader: TaxonomyLoader,
        model_name: str = "",
        body_max_chars: int = _BODY_MAX_CHARS,
    ) -> None:
        self.llm = llm_client
        self.taxonomy = taxonomy_loader
        self.model_name = model_name
        self.body_max_chars = body_max_chars
        # Tail kept on truncation. Bounded so head+tail never exceeds the cap.
        self._body_tail_chars = min(_BODY_TAIL_CHARS, body_max_chars // 2)

    async def classify(self, request: ClassificationRequest) -> ClassificationAttempt:
        start = time.perf_counter()
        try:
            bound_taxonomy = request.asset_snapshots.get("taxonomy")
            taxonomy_snapshot = (
                bound_taxonomy
                if isinstance(bound_taxonomy, ValidatedAssetSnapshot)
                else self.taxonomy.get_snapshot()
            )
            response = await self.llm.chat_completion(
                messages=self._build_messages(request, taxonomy_snapshot.value),
                tools=[_CLASSIFY_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_email"}},
            )
            return self._parse_response(
                response,
                int((time.perf_counter() - start) * 1000),
                taxonomy_tree=taxonomy_snapshot.value,
                taxonomy_version=taxonomy_snapshot.version,
            )
        except (LLMError, LLMResponseError, ValueError) as exc:
            logger.warning("LLM classifier unavailable: %s", exc)
            return ClassificationAttempt(
                source=self.source,
                status=AttemptStatus.UNAVAILABLE,
                meta=ClassificationMeta(
                    needs_human_review=True,
                    model_used=self.model_name,
                    latency_ms=int((time.perf_counter() - start) * 1000),
                ),
                error=str(exc),
            )

    def _build_messages(
        self,
        request: ClassificationRequest,
        taxonomy_tree: TaxonomyTree | None = None,
    ) -> list[dict[str, str]]:
        mail = request.mail
        body = self._truncate_body(mail.body)
        tree = taxonomy_tree or self.taxonomy.get_snapshot().value
        return [
            {
                "role": "system",
                "content": (
                    "You are a mail classification assistant. The taxonomy is flat: every label is a single l1_code. "
                    "Treat the supplied taxonomy as the authoritative business configuration. "
                    "Use only its codes, descriptions, keywords, and selection guidance; do not rely on unstated "
                    "domain assumptions. Return one to five labels from that taxonomy. When several labels match, "
                    "give the highest confidence to the primary intent defined by the configuration and lower "
                    "confidence to secondary aspects. If the configuration does not resolve the mail unambiguously, "
                    "return no labels and set needs_human_review to true."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Taxonomy\n{self.taxonomy.serialize_for_prompt(tree)}\n\n"
                    f"## Email\n- message_id: {mail.message_id}\n- sender: {mail.sender}\n"
                    f"- subject: {mail.subject}\n- body:\n{body}"
                ),
            },
        ]

    def _truncate_body(self, body: str) -> str:
        """Truncate mail body to ``body_max_chars`` keeping head + tail context.

        Reply chains can grow long; feeding the full body to the LLM inflates
        tokens and slows responses. We keep the opening (often the latest
        segment after thread parsing) and the tail (older context), dropping
        the middle.
        """
        if len(body) <= self.body_max_chars:
            return body
        head_len = self.body_max_chars - self._body_tail_chars
        return f"{body[:head_len]}\n[truncated]\n{body[-self._body_tail_chars:]}"

    def _parse_response(
        self,
        response: dict[str, Any],
        latency_ms: int,
        *,
        taxonomy_tree: TaxonomyTree | None = None,
        taxonomy_version: str | None = None,
    ) -> ClassificationAttempt:
        try:
            if not isinstance(response, dict):
                raise TypeError("response must be an object")
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                raise TypeError("choices must be a non-empty list")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError("choice must be an object")
            message = choice.get("message")
            if not isinstance(message, dict):
                raise TypeError("message must be an object")
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                raise TypeError("tool_calls must be a non-empty list")
            tool_call = tool_calls[0]
            if not isinstance(tool_call, dict):
                raise TypeError("tool call must be an object")
            function = tool_call.get("function")
            if not isinstance(function, dict):
                raise TypeError("tool function must be an object")
            raw_arguments = function.get("arguments")
            if not isinstance(raw_arguments, str):
                raise TypeError("tool arguments must be a JSON string")
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LLMResponseError(
                f"malformed_classify_email_output: invalid response envelope: {exc}"
            ) from exc
        return self._post_process(
            _validate_tool_output(arguments),
            latency_ms,
            taxonomy_tree=taxonomy_tree,
            taxonomy_version=taxonomy_version,
        )

    def _post_process(
        self,
        arguments: dict[str, Any] | _LLMToolOutput,
        latency_ms: int,
        *,
        taxonomy_tree: TaxonomyTree | None = None,
        taxonomy_version: str | None = None,
    ) -> ClassificationAttempt:
        output = (
            arguments
            if isinstance(arguments, _LLMToolOutput)
            else _validate_tool_output(arguments)
        )
        active_version: str | None
        if taxonomy_tree is None:
            snapshot = self.taxonomy.get_snapshot()
            tree = snapshot.value
            active_version = taxonomy_version or snapshot.version
        else:
            tree = taxonomy_tree
            active_version = taxonomy_version
        valid_codes = tree.all_codes()
        parsed_labels: list[tuple[TaxonomyLabel, float]] = []
        seen_l1: set[str] = set()
        has_exclusive_label = False

        for raw_label in output.labels:
            # Flat taxonomy: only L1 is meaningful; retain nullable L2/L3
            # fields in the response for contract compatibility.
            l1, _, _ = _normalize_codes(
                raw_label.l1_code,
                raw_label.l2_code,
                raw_label.l3_code,
            )
            if not l1 or l1 not in valid_codes or l1 in seen_l1:
                continue
            seen_l1.add(l1)
            l1_node = tree.find_l1(l1)
            has_exclusive_label = has_exclusive_label or bool(
                l1_node and l1_node.exclusive
            )
            raw_confidence = raw_label.confidence
            parsed_labels.append(
                (
                    TaxonomyLabel(
                        l1_code=l1,
                        l1_label=l1_node.label if l1_node else l1,
                        l2_code=None,
                        l2_label=None,
                        l3_code=None,
                        l3_label=None,
                        confidence=raw_confidence,
                        reasoning=raw_label.reasoning,
                    ),
                    raw_confidence,
                )
            )

        if has_exclusive_label:
            parsed_labels = [
                (label, raw_confidence)
                for label, raw_confidence in parsed_labels
                if (node := tree.find_l1(label.l1_code)) is not None
                and node.exclusive
            ]

        labels = [label for label, _ in parsed_labels]
        top_raw = max((raw_confidence for _, raw_confidence in parsed_labels), default=None)
        overall_confidence = max((label.confidence for label in labels), default=0.0)
        raw_meta = output.meta
        meta = ClassificationMeta(
            urgency=raw_meta.urgency,
            language=raw_meta.language,
            sentiment=raw_meta.sentiment,
            has_attachments=raw_meta.has_attachments,
            overall_confidence=overall_confidence,
            needs_human_review=raw_meta.needs_human_review or overall_confidence < 0.5,
            model_used=self.model_name,
            latency_ms=latency_ms,
        )
        return ClassificationAttempt(
            source=self.source,
            status=AttemptStatus.SUCCESS,
            labels=labels,
            confidence=overall_confidence,
            meta=meta,
            calibration_log=calibration_log(top_raw) if top_raw is not None else None,
            evidence=(
                {"taxonomy_version": active_version}
                if active_version is not None
                else {}
            ),
        )
