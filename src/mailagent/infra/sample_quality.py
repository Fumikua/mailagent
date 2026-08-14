"""Pure quality assessment for samples proposed for vector retrieval."""
from __future__ import annotations

import hashlib

from mailagent.domain.models import SampleQualityAssessment
from mailagent.preprocessing.retrieval_models import RetrievalDocument


def _fingerprint(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def assess_sample_quality(
    document: RetrievalDocument,
    *,
    label_l1: str,
    valid_labels: set[str],
) -> SampleQualityAssessment:
    """Assess a generated retrieval document against the active flat taxonomy."""

    reasons: list[str] = []
    if not document.eligible:
        reasons.append(document.ineligible_reason or "ineligible_document")
    if not label_l1 or label_l1 not in valid_labels:
        reasons.append("unknown_taxonomy_category")
    return SampleQualityAssessment(
        disposition="rejected" if reasons else "accepted",
        reasons=reasons,
        fingerprint=_fingerprint(document.text),
        retrieval_policy_version=document.policy_version,
    )
