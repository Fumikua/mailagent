"""Vertical-neutral mail-understanding pipeline."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from enum import Enum
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from mailagent.domain.models import (
    ClassificationResponse,
    EnrichmentError,
    FusionMeta,
    MailEvent,
    OrchestrationAudit,
)

from mailagent.classification.contracts import ClassificationOrchestrator, ClassificationRequest, Enricher
from .versioning import ClassificationVersionProvider

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert diagnostic values to JSON-safe data without blocking classification."""

    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class MailUnderstandingPipeline:
    """Combines a generic classification result with selected vertical enrichers."""

    def __init__(
        self,
        orchestrator: ClassificationOrchestrator,
        vertical_id: str,
        data_schema_version: str,
        vertical_namespace: str,
        enrichers: Iterable[Enricher],
        data_schema: dict[str, object] | None = None,
        version_provider: ClassificationVersionProvider | None = None,
        auto_accept_enabled: bool = False,
    ) -> None:
        self._orchestrator = orchestrator
        self._vertical_id = vertical_id
        self._data_schema_version = data_schema_version
        self._vertical_namespace = vertical_namespace
        self._enrichers = list(enrichers)
        self._version_provider = version_provider
        self._auto_accept_enabled = auto_accept_enabled
        # Multiple enrichers may write the same vertical namespace (e.g. one
        # enricher writes data.<namespace>.entities, another writes
        # data.<namespace>.signals). We merge their data dicts per namespace.
        # But every enricher must still write the selected vertical's namespace
        # — no cross-namespace leakage.
        for enricher in self._enrichers:
            if enricher.namespace != self._vertical_namespace:
                raise ValueError(
                    f"enricher {enricher.id} namespace {enricher.namespace} must write "
                    f"selected namespace {self._vertical_namespace}"
                )
        if data_schema is not None:
            Draft202012Validator.check_schema(data_schema)
        self._data_validator = (
            Draft202012Validator(data_schema) if data_schema is not None else None
        )

    async def process(self, mail: MailEvent) -> ClassificationResponse:
        binding = self._version_provider.bind() if self._version_provider else None
        versions = binding.versions if binding else None
        request = ClassificationRequest(
            mail=mail,
            context={
                "has_attachments": bool(mail.attachments or mail.attachment_meta),
            },
            asset_snapshots=(
                dict(binding.asset_snapshots)
                if binding is not None
                else {}
            ),
        )
        core = await self._orchestrator.classify(request)
        data: dict[str, object] = {}
        enrichment_errors: list[EnrichmentError] = []

        for enricher in self._enrichers:
            try:
                patch = await enricher.enrich(request, core.model_copy(deep=True))
                if patch.namespace != enricher.namespace:
                    raise ValueError(f"enricher {enricher.id} returned a different namespace")
                # Build the merged candidate without committing, so a schema
                # validation failure leaves the namespace bucket untouched.
                bucket = data.get(patch.namespace)
                if bucket is None:
                    candidate = dict(patch.data)
                elif isinstance(bucket, dict) and isinstance(patch.data, dict):
                    candidate = {**bucket, **patch.data}
                else:
                    raise ValueError(
                        f"enricher {enricher.id} cannot merge non-dict data into namespace {patch.namespace}"
                    )
                if self._data_validator is not None:
                    self._data_validator.validate(candidate)
                data[patch.namespace] = candidate
            except Exception as exc:
                logger.exception("enricher %s failed", enricher.id)
                core.meta = core.meta.model_copy(update={"needs_human_review": True})
                enrichment_errors.append(
                    EnrichmentError(
                        enricher_id=enricher.id,
                        namespace=enricher.namespace,
                        message=str(exc),
                    )
                )

        # P0 acceptance is a final response invariant. Enrichers receive a
        # defensive copy, and the gate is still re-applied after every
        # enrichment-side mutation immediately before response projection.
        if not self._auto_accept_enabled:
            core.meta = core.meta.model_copy(update={"needs_human_review": True})
            core.audit = {
                **core.audit,
                "acceptance": {
                    "status": "review",
                    "reason": "p0_auto_accept_disabled",
                },
            }

        try:
            fusion_meta = FusionMeta.model_validate(core.audit) if core.audit else None
        except ValidationError:
            fusion_meta = None
        orchestration_audit = None
        if core.attempts or core.audit:
            orchestration_audit = OrchestrationAudit(
                selected_source=core.selected_source,
                attempts=[
                    _json_safe(attempt.model_dump(mode="json"))
                    for attempt in core.attempts
                ],
                details=_json_safe(core.audit),
            )

        return ClassificationResponse(
            labels=core.labels,
            meta=core.meta,
            calibration_log=core.calibration_log,
            vertical_id=self._vertical_id,
            data_schema_version=self._data_schema_version,
            data=data,
            enrichment_errors=enrichment_errors,
            orchestration_audit=orchestration_audit,
            fusion_meta=fusion_meta,
            versions=versions,
        )
