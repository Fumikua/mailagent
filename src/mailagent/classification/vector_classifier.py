"""Vector similarity classifier: embedding-based Path B classification.

The :class:`VectorClassifier` embeds the incoming mail via the preprocessing
pipeline, queries the :class:`VectorStore` for k-nearest labeled samples, and
returns one :class:`ClassificationAttempt` with the top candidate's label.

Degradation paths:
    - Embedding service unavailable → ``status=UNAVAILABLE``,
      ``reason="embedding_unavailable"``
    - Empty sample library (zero samples) → ``status=NO_MATCH``,
      ``reason="empty_samples"``
    - Samples exist but none above threshold → ``status=NO_MATCH``,
      ``reason="below_threshold"``
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

from mailagent.domain.models import ClassificationMeta, PathBResult, TaxonomyLabel
from mailagent.domain.versioning import ValidatedAssetSnapshot
from mailagent.preprocessing.pipeline import preprocess_mail

from .contracts import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationRequest,
)

if TYPE_CHECKING:
    from mailagent.infra.config import VectorStoreSettings
    from mailagent.infra.vector_store import VectorStore
    from mailagent.llm.embedding import EmbeddingClient
    from mailagent.preprocessing.contracts import MailPreprocessingExtension
    from mailagent.preprocessing.retrieval_models import RetrievalCleaningPolicy
    from .taxonomy import TaxonomyLoader

logger = logging.getLogger(__name__)

# TTL (seconds) for the cached "is the samples table empty?" probe.
# Empty-table detection only matters for the choice between
# ``reason="empty_samples"`` and ``reason="below_threshold"``; the answer
# does not change between requests within this window, so a short cache is
# safe and avoids an N+1 SELECT per classified mail.
_EMPTY_SAMPLES_CACHE_TTL: float = 60.0


class VectorClassifier:
    """Embedding-based classifier implementing the ``Classifier`` Protocol.

    Implements the vector similarity path (Path B): embeds the mail via the
    preprocessing pipeline, performs kNN search over labeled samples, and
    returns the top candidate's label as a :class:`ClassificationAttempt`.
    """

    source = "vector"

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        settings: VectorStoreSettings,
        *,
        cleaning_policy: RetrievalCleaningPolicy | None = None,
        preprocessing_extension: MailPreprocessingExtension | None = None,
        taxonomy_loader: TaxonomyLoader | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._settings = settings
        self._cleaning_policy = cleaning_policy
        self._preprocessing_extension = preprocessing_extension
        self._taxonomy_loader = taxonomy_loader
        # Cache: ``(monotonic_ts, sample_count)``. ``None`` means not yet probed.
        self._empty_samples_cache: tuple[float, int] | None = None

    async def _samples_table_empty(self) -> bool:
        """Return True iff the samples table currently has zero rows.

        Caches the result for :data:`_EMPTY_SAMPLES_CACHE_TTL` seconds to
        avoid an N+1 ``SELECT COUNT(*)`` per classified email. The cache is
        invalidated automatically after the TTL expires; the next call
        re-probes the database. New ``insert_sample`` / ``delete_sample`` /
        ``archive_old_samples`` calls do not invalidate the cache — this is
        acceptable because (a) the empty/non-empty distinction is the only
        thing that depends on the count, and (b) bootstrap batches add
        hundreds of samples per minute, so the brief window where the cache
        is stale is dominated by the soon-to-be-overridden probe anyway.
        """
        now = time.monotonic()
        if self._empty_samples_cache is not None:
            ts, count = self._empty_samples_cache
            if now - ts < _EMPTY_SAMPLES_CACHE_TTL:
                return count == 0
        count = await self._vector_store.count_samples()
        self._empty_samples_cache = (now, count)
        return count == 0

    async def classify(self, request: ClassificationRequest) -> ClassificationAttempt:
        """Classify via vector similarity search over labeled samples.

        Steps:
            1. Preprocess the mail (subject normalization + thread parsing +
               batch embedding) via :func:`preprocess_mail`.
            2. Query the vector store for k-nearest labeled samples.
            3. Build a :class:`PathBResult` from the candidates.
            4. Return a :class:`ClassificationAttempt` with the top label.
        """
        bound_taxonomy = request.asset_snapshots.get("taxonomy")
        taxonomy_tree = None
        taxonomy_version: str | None = None
        if isinstance(bound_taxonomy, ValidatedAssetSnapshot):
            taxonomy_tree = bound_taxonomy.value
            taxonomy_version = bound_taxonomy.version
        elif self._taxonomy_loader is not None:
            get_snapshot = getattr(self._taxonomy_loader, "get_snapshot", None)
            candidate = get_snapshot() if callable(get_snapshot) else None
            if isinstance(candidate, ValidatedAssetSnapshot):
                taxonomy_tree = candidate.value
                taxonomy_version = candidate.version
            else:
                taxonomy_tree = self._taxonomy_loader.get_tree()

        bound_policy = request.asset_snapshots.get("retrieval_cleaning")
        cleaning_policy = (
            bound_policy.value
            if isinstance(bound_policy, ValidatedAssetSnapshot)
            else self._cleaning_policy
        )
        bound_preprocessing = request.asset_snapshots.get("preprocessing")
        preprocessing_snapshot = (
            bound_preprocessing
            if isinstance(bound_preprocessing, ValidatedAssetSnapshot)
            else None
        )

        # Step 1: preprocess mail → embedding_thread
        try:
            preprocessed = await preprocess_mail(
                request.mail,
                self._embedding_client,
                cleaning_policy=cleaning_policy,
                extension=self._preprocessing_extension,
                extension_snapshot=preprocessing_snapshot,
            )
            if not preprocessed.retrieval_document.eligible:
                path_b = PathBResult(
                    candidates=[],
                    top1_label=None,
                    top1_similarity=0.0,
                    confidence=0.0,
                    reason="ineligible_document",
                )
                return ClassificationAttempt(
                    source=self.source,
                    status=AttemptStatus.NO_MATCH,
                    confidence=0.0,
                    meta=ClassificationMeta(overall_confidence=0.0),
                    evidence={
                        "path_b_result": path_b.model_dump(),
                        "retrieval_document": preprocessed.retrieval_document.model_dump(),
                    },
                )
            embedding_thread = preprocessed.embedding_thread
        except Exception as exc:
            logger.warning("VectorClassifier embedding unavailable: %s", exc)
            path_b = PathBResult(
                candidates=[],
                top1_label=None,
                top1_similarity=0.0,
                confidence=0.0,
                reason="embedding_unavailable",
            )
            return ClassificationAttempt(
                source=self.source,
                status=AttemptStatus.UNAVAILABLE,
                confidence=0.0,
                evidence={"path_b_result": path_b.model_dump()},
                error=str(exc),
            )

        # Step 2: knn search over labeled samples.
        # When ``request.context["vector_scope"]`` is set (by FusionOrchestrator
        # when a target profile matches the rule candidate), restrict the search
        # to those L3 codes. If the scoped search returns no candidates, retry
        # once with the global scope (label_scope=None) and record the fallback.
        vector_scope = request.context.get("vector_scope")
        scoped_fallback = False
        candidates = await self._vector_store.knn_search(
            embedding_thread,
            top_k=self._settings.top_k,
            threshold=self._settings.similarity_threshold,
            label_scope=vector_scope,
        )
        if not candidates and vector_scope:
            # Scoped search returned nothing — retry with global scope.
            scoped_fallback = True
            candidates = await self._vector_store.knn_search(
                embedding_thread,
                top_k=self._settings.top_k,
                threshold=self._settings.similarity_threshold,
                label_scope=None,
            )

        # Step 3: handle empty results with appropriate reason
        if not candidates:
            is_empty = await self._samples_table_empty()
            reason: Literal["empty_samples", "below_threshold"] = (
                "empty_samples" if is_empty else "below_threshold"
            )
            path_b = PathBResult(
                candidates=[],
                top1_label=None,
                top1_similarity=0.0,
                confidence=0.0,
                reason=reason,
            )
            return ClassificationAttempt(
                source=self.source,
                status=AttemptStatus.NO_MATCH,
                confidence=0.0,
                meta=ClassificationMeta(overall_confidence=0.0),
                evidence={
                    "path_b_result": path_b.model_dump(),
                    "scoped_fallback": scoped_fallback,
                    "vector_scope": vector_scope,
                },
            )

        # Step 4: build result from candidates.
        # Multi-label policy (threshold-based): each candidate whose
        # ``count >= minimum_support`` independently qualifies as a label.
        # ``minimum_margin`` is intentionally NOT applied — a small margin
        # between top1 and top2 is the signal of a genuine multi-label mail,
        # not ambiguity to be discarded.
        top1 = candidates[0]
        next_similarity = candidates[1].max_similarity if len(candidates) > 1 else None
        margin = (
            max(top1.max_similarity - next_similarity, 0.0)
            if next_similarity is not None
            else None
        )
        path_b = PathBResult(
            candidates=candidates,
            top1_label=top1.label,
            top1_similarity=top1.max_similarity,
            top1_support=top1.count,
            top1_mean_similarity=top1.mean_similarity,
            next_category_similarity=next_similarity,
            similarity_margin=margin,
            confidence=top1.max_similarity,
            reason="ok",
        )

        # Filter candidates that independently pass minimum_support, then
        # resolve their taxonomy label (skip stale categories).  Candidates
        # are already ordered by max_similarity descending, so the resulting
        # labels list is too.
        supported_labels: list[TaxonomyLabel] = []
        for candidate in candidates:
            if candidate.count < self._settings.minimum_support:
                continue
            l1_label = candidate.label
            if taxonomy_tree is not None:
                node = taxonomy_tree.find_l1(candidate.label)
                if node is None:
                    # Stale category for this candidate — skip it but keep
                    # evaluating the rest. If nothing survives, fall through
                    # to the empty-result path below.
                    continue
                l1_label = node.label
            supported_labels.append(
                TaxonomyLabel(
                    l1_code=candidate.label,
                    l1_label=l1_label,
                    confidence=candidate.max_similarity,
                )
            )

        if not supported_labels:
            # Either top1 lacked support, or every supported candidate was a
            # stale category. Either way, no usable label was produced.
            top1_unsupported = top1.count < self._settings.minimum_support
            path_b.reason = "ambiguous_candidates" if top1_unsupported else "stale_category"
            return ClassificationAttempt(
                source=self.source,
                status=AttemptStatus.NO_MATCH,
                confidence=0.0,
                meta=ClassificationMeta(overall_confidence=0.0),
                evidence={
                    "path_b_result": path_b.model_dump(),
                    "minimum_support": self._settings.minimum_support,
                    "scoped_fallback": scoped_fallback,
                    "vector_scope": vector_scope,
                },
            )

        return ClassificationAttempt(
            source=self.source,
            status=AttemptStatus.SUCCESS,
            labels=supported_labels,
            confidence=top1.max_similarity,
            meta=ClassificationMeta(overall_confidence=top1.max_similarity),
            evidence={
                "path_b_result": path_b.model_dump(),
                "scoped_fallback": scoped_fallback,
                "vector_scope": vector_scope,
                "retrieval_document": preprocessed.retrieval_document.model_dump(),
                **(
                    {"taxonomy_version": taxonomy_version}
                    if taxonomy_version is not None
                    else {}
                ),
                **(
                    {"preprocessing_version": preprocessing_snapshot.version}
                    if preprocessing_snapshot is not None
                    else {}
                ),
                **(
                    {"retrieval_cleaning_version": bound_policy.version}
                    if isinstance(bound_policy, ValidatedAssetSnapshot)
                    else {}
                ),
            },
        )
