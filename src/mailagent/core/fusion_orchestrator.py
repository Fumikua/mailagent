"""Three-path fusion orchestrator: rules → vector confirmation → LLM fallback.

The :class:`FusionOrchestrator` implements the :class:`ClassificationOrchestrator`
Protocol. It coordinates three classifier paths via five fusion strategies:

    1. ``rule_only`` — rule confidence ≥ ``rule_confidence_threshold`` (0.9),
       use rule result directly.
    2. ``rule_vector_confirmed`` — rule confidence in
       ``[llm_fallback_threshold, rule_confidence_threshold)`` (0.7-0.9), vector
       confirms the same label; use rule result with ``vector_confirmed=True``.
    3. ``vector_only`` — no eligible rule, vector confidence ≥
       ``vector_confidence_threshold`` (0.85).
    4. ``llm_fallback`` — all else fails, LLM produces a successful result.
    5. ``all_low_review`` — no path produced sufficient confidence; requires
       human review.

Missing classifiers are silently skipped — e.g., if only ``llm`` is configured,
the orchestrator degrades gracefully to LLM-only classification.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from mailagent.domain.models import (
    FusionConflict,
    FusionMeta,
    PathBResult,
    RuleMatch,
    RuleResult,
    TaxonomyLabel,
)
from mailagent.domain.versioning import ValidatedAssetSnapshot

from mailagent.classification.contracts import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationCoreResult,
    ClassificationRequest,
    Classifier,
)
from mailagent.domain.models import ClassificationMeta
from .target_profile import TargetProfileLoader, TargetProfileSet

if TYPE_CHECKING:
    from mailagent.infra.config import FusionSettings

logger = logging.getLogger(__name__)

# Type aliases for FusionMeta literal fields.
FusionStrategy = Literal[
    "rule_only",
    "rule_vector_confirmed",
    "vector_only",
    "llm_fallback",
    "all_low_review",
]
FusionSource = Literal["rule", "vector", "llm"]


def _to_fusion_source(source: str) -> FusionSource:
    """Map a classifier ``source`` to the FusionMeta ``source`` literal."""
    if source == "rules":
        return "rule"
    if source == "vector":
        return "vector"
    return "llm"


def _extract_rule_result(attempt: ClassificationAttempt | None) -> RuleResult | None:
    """Reconstruct a partial :class:`RuleResult` from a rule attempt's evidence.

    The :class:`RuleClassifier` stores ``rule_type`` and ``matched_pattern`` in
    the attempt's ``evidence`` dict. This helper builds a minimal
    :class:`RuleResult` with a single selected match for audit purposes.
    """
    if attempt is None or attempt.status != AttemptStatus.SUCCESS or not attempt.labels:
        return None
    rule_type = attempt.evidence.get("rule_type", "subject_patterns")
    matched_pattern = attempt.evidence.get("matched_pattern", "")
    return RuleResult(
        selected=RuleMatch(
            rule_type=rule_type,  # type: ignore[arg-type]
            label=attempt.labels[0].l1_code,
            confidence=attempt.confidence,
            matched_pattern=matched_pattern,
        ),
    )


def _extract_vector_result(attempt: ClassificationAttempt | None) -> PathBResult | None:
    """Reconstruct a :class:`PathBResult` from a vector attempt's evidence."""
    if attempt is None:
        return None
    raw = attempt.evidence.get("path_b_result")
    if raw is None:
        return None
    return PathBResult.model_validate(raw)


def _extract_llm_result(attempt: ClassificationAttempt | None) -> dict[str, Any] | None:
    """Build an ``llm_result`` dict from an LLM attempt for audit."""
    if attempt is None or attempt.status != AttemptStatus.SUCCESS:
        return None
    return {
        "labels": [label.model_dump() for label in attempt.labels],
        "confidence": attempt.confidence,
    }


def _pick_best_attempt(attempts: list[ClassificationAttempt]) -> ClassificationAttempt | None:
    """Pick the attempt with the highest confidence among SUCCESS attempts."""
    best: ClassificationAttempt | None = None
    for attempt in attempts:
        if attempt.status != AttemptStatus.SUCCESS:
            continue
        if best is None or attempt.confidence > best.confidence:
            best = attempt
    return best


def _unavailable_attempt(source: str) -> ClassificationAttempt:
    return ClassificationAttempt(
        source=source,
        status=AttemptStatus.UNAVAILABLE,
        error="classifier is not configured",
    )


def _label_path(label: TaxonomyLabel) -> str:
    """Build a dotted taxonomy path (``l1.l2.l3``) from a TaxonomyLabel.

    Only non-None levels are joined, so a label with only ``l1_code`` returns
    that code alone. Used to match against ``TargetProfile.label``.
    """

    parts = [label.l1_code, label.l2_code, label.l3_code]
    return ".".join(part for part in parts if part)


class FusionOrchestrator:
    """Three-path fusion orchestrator implementing ``ClassificationOrchestrator``.

    Coordinates rule / vector / LLM classifiers via five fusion strategies.
    Missing classifiers are silently skipped — the orchestrator degrades
    gracefully to whatever paths are available.
    """

    def __init__(
        self,
        classifiers: Iterable[Classifier],
        settings: FusionSettings,
        order: tuple[str, ...] = ("rules", "vector", "llm"),
        target_profile_loader: TargetProfileLoader | None = None,
    ) -> None:
        self._classifiers = {c.source: c for c in classifiers}
        self._settings = settings
        self._order = order
        self._target_profile_loader = target_profile_loader

    async def classify(self, request: ClassificationRequest) -> ClassificationCoreResult:
        """Run the three-path fusion cascade and return the selected result."""
        attempts: list[ClassificationAttempt] = []
        rule_attempt: ClassificationAttempt | None = None
        vector_attempt: ClassificationAttempt | None = None
        llm_attempt: ClassificationAttempt | None = None
        vector_called = False
        # ``target_profile`` is set when Step 2 matches a target label and the
        # scoped vector retrieval was attempted (regardless of confirmation).
        # It is threaded into FusionMeta for audit on all downstream strategies.
        target_profile: str | None = None
        conflict: FusionConflict | None = None

        # ------------------------------------------------------------------
        # Step 1: Try rules classifier → rule_only if high confidence
        # ------------------------------------------------------------------
        rules_clf = self._classifiers.get("rules")
        if rules_clf is not None:
            rule_attempt = await rules_clf.classify(request)
            attempts.append(rule_attempt)

            if (
                rule_attempt.status == AttemptStatus.SUCCESS
                and rule_attempt.confidence >= self._settings.rule_confidence_threshold
            ):
                return self._build_result(
                    strategy="rule_only",
                    attempt=rule_attempt,
                    attempts=attempts,
                    rule_attempt=rule_attempt,
                    vector_attempt=None,
                    llm_attempt=None,
                    vector_confirmed=False,
                    target_profile=None,  # rule_only never triggers target branch
                )
        else:
            attempts.append(_unavailable_attempt("rules"))

        # ------------------------------------------------------------------
        # Step 2: Medium-confidence rule (0.7-0.9) → vector confirmation
        # ------------------------------------------------------------------
        if (
            rule_attempt is not None
            and rule_attempt.status == AttemptStatus.SUCCESS
            and rule_attempt.confidence >= self._settings.llm_fallback_threshold
        ):
            # P0 target scoped retrieval: if the rule candidate's label matches
            # a target profile, set request.context["vector_scope"] so the
            # upcoming VectorClassifier call runs a scoped knn_search instead of
            # a global scan. Record the matched label path for audit.
            if (
                self._target_profile_loader is not None
                and rule_attempt.labels
            ):
                label_path = _label_path(rule_attempt.labels[0])
                if "target_profiles" in request.asset_snapshots:
                    bound_profiles = request.asset_snapshots["target_profiles"]
                    matched = (
                        self._target_profile_loader.find_match(
                            label_path,
                            snapshot=bound_profiles,
                        )
                        if isinstance(bound_profiles, ValidatedAssetSnapshot)
                        and isinstance(bound_profiles.value, TargetProfileSet)
                        else None
                    )
                else:
                    matched = self._target_profile_loader.find_match(label_path)
                if matched is not None:
                    request.context["vector_scope"] = list(matched.vector_scope)
                    target_profile = matched.label

            vector_clf = self._classifiers.get("vector")
            if vector_clf is not None:
                vector_attempt = await vector_clf.classify(request)
                attempts.append(vector_attempt)
                vector_called = True

                if (
                    vector_attempt.status == AttemptStatus.SUCCESS
                    and vector_attempt.confidence >= self._settings.vector_confidence_threshold
                    and vector_attempt.labels
                    and rule_attempt.labels
                    and vector_attempt.labels[0].l1_code == rule_attempt.labels[0].l1_code
                ):
                    # rule_vector_confirmed: vector agrees with rule
                    return self._build_result(
                        strategy="rule_vector_confirmed",
                        attempt=rule_attempt,
                        attempts=attempts,
                        rule_attempt=rule_attempt,
                        vector_attempt=vector_attempt,
                        llm_attempt=None,
                        vector_confirmed=True,
                        target_profile=target_profile,
                    )
                # Disagreement is handled below after all vector paths complete.
            else:
                vector_attempt = _unavailable_attempt("vector")
                attempts.append(vector_attempt)
                vector_called = True

        # ------------------------------------------------------------------
        # Step 3: vector_only (no eligible rule)
        # ------------------------------------------------------------------
        if not vector_called:
            vector_clf = self._classifiers.get("vector")
            if vector_clf is not None:
                vector_attempt = await vector_clf.classify(request)
                attempts.append(vector_attempt)
                vector_called = True
            else:
                vector_attempt = _unavailable_attempt("vector")
                attempts.append(vector_attempt)
                vector_called = True

        if (
            rule_attempt is not None
            and rule_attempt.status == AttemptStatus.SUCCESS
            and rule_attempt.labels
            and vector_attempt is not None
            and vector_attempt.status == AttemptStatus.SUCCESS
            and vector_attempt.labels
            and rule_attempt.labels[0].l1_code != vector_attempt.labels[0].l1_code
        ):
            conflict = FusionConflict(
                sources=["rule", "vector"],
                labels=[
                    rule_attempt.labels[0].l1_code,
                    vector_attempt.labels[0].l1_code,
                ],
            )

        if (
            vector_attempt is not None
            and vector_attempt.status == AttemptStatus.SUCCESS
            and vector_attempt.confidence >= self._settings.vector_confidence_threshold
            and conflict is None
        ):
            return self._build_result(
                strategy="vector_only",
                attempt=vector_attempt,
                attempts=attempts,
                rule_attempt=rule_attempt,
                vector_attempt=vector_attempt,
                llm_attempt=None,
                vector_confirmed=False,
                target_profile=target_profile,
            )

        # ------------------------------------------------------------------
        # Step 4: LLM fallback
        # ------------------------------------------------------------------
        llm_clf = self._classifiers.get("llm")
        if llm_clf is not None:
            llm_attempt = await llm_clf.classify(request)
            attempts.append(llm_attempt)

            if llm_attempt.status == AttemptStatus.SUCCESS:
                needs_review = (
                    conflict is not None
                    or
                    llm_attempt.meta.needs_human_review
                    or llm_attempt.confidence < self._settings.llm_fallback_threshold
                )
                return self._build_result(
                    strategy="all_low_review" if needs_review else "llm_fallback",
                    attempt=llm_attempt,
                    attempts=attempts,
                    rule_attempt=rule_attempt,
                    vector_attempt=vector_attempt,
                    llm_attempt=llm_attempt,
                    vector_confirmed=False,
                    needs_human_review=needs_review,
                    target_profile=target_profile,
                    conflict=conflict,
                )
        else:
            attempts.append(_unavailable_attempt("llm"))

        # ------------------------------------------------------------------
        # Step 5: All low confidence → needs human review
        # ------------------------------------------------------------------
        best = _pick_best_attempt(attempts)
        if best is not None:
            return self._build_result(
                strategy="all_low_review",
                attempt=best,
                attempts=attempts,
                rule_attempt=rule_attempt,
                vector_attempt=vector_attempt,
                llm_attempt=llm_attempt,
                vector_confirmed=False,
                needs_human_review=True,
                target_profile=target_profile,
                conflict=conflict,
            )

        # No classifier produced any result — return empty with review flag
        fusion_meta = FusionMeta(
            fusion_strategy="all_low_review",
            source="llm",
            confidence=0.0,
            rule_result=_extract_rule_result(rule_attempt),
            vector_result=_extract_vector_result(vector_attempt),
            llm_result=_extract_llm_result(llm_attempt),
            vector_confirmed=False,
            target_profile=target_profile,
            conflict=conflict,
        )
        return ClassificationCoreResult(
            attempts=attempts,
            meta=ClassificationMeta(needs_human_review=True, fallback=True),
            audit=fusion_meta.model_dump(),
        )

    def _build_result(
        self,
        strategy: FusionStrategy,
        attempt: ClassificationAttempt,
        attempts: list[ClassificationAttempt],
        rule_attempt: ClassificationAttempt | None,
        vector_attempt: ClassificationAttempt | None,
        llm_attempt: ClassificationAttempt | None,
        vector_confirmed: bool,
        needs_human_review: bool = False,
        target_profile: str | None = None,
        conflict: FusionConflict | None = None,
    ) -> ClassificationCoreResult:
        """Build the final :class:`ClassificationCoreResult` with FusionMeta audit."""
        fusion_source = _to_fusion_source(attempt.source)
        fusion_meta = FusionMeta(
            fusion_strategy=strategy,
            source=fusion_source,
            confidence=attempt.confidence,
            rule_result=_extract_rule_result(rule_attempt),
            vector_result=_extract_vector_result(vector_attempt),
            llm_result=_extract_llm_result(llm_attempt),
            vector_confirmed=vector_confirmed,
            target_profile=target_profile,
            conflict=conflict,
        )
        meta = attempt.meta.model_copy(
            update={
                "overall_confidence": attempt.confidence,
                "needs_human_review": needs_human_review or attempt.meta.needs_human_review,
            }
        )
        return ClassificationCoreResult(
            labels=attempt.labels,
            meta=meta,
            calibration_log=attempt.calibration_log,
            selected_source=attempt.source,
            attempts=attempts,
            audit=fusion_meta.model_dump(),
        )
