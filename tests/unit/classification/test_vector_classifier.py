"""Unit tests for VectorClassifier: Path B embedding-based classification.

Covers:
    - Normal classification returning the top-K candidates' top-1 label.
    - Threshold filtering is delegated to the vector store (knn_search).
    - Single-label majority vs mixed-label candidate ordering.
    - ``embed_batch`` is called exactly once via ``preprocess_mail``.
    - Three degradation paths: embedding unavailable / empty samples /
      below threshold.
    - :class:`PathBResult` field completeness in the attempt evidence.
    - Classifier Protocol contract: ``source`` attribute + ``classify`` signature.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mailagent.classification import (
    AttemptStatus,
    ClassificationAttempt,
    ClassificationRequest,
)
from mailagent.classification.vector_classifier import VectorClassifier
from mailagent.domain.models import MailEvent, PathBCandidate, PathBResult
from mailagent.infra.config import VectorStoreSettings
from mailagent.classification.taxonomy import TaxonomyNode, TaxonomyTree


# ---------------------------------------------------------------------------
# Fake EmbeddingClient — records embed_batch invocations.
# ---------------------------------------------------------------------------


class _FakeEmbeddingClient:
    """Fake EmbeddingClient for preprocess_mail; returns canned vectors."""

    def __init__(
        self,
        return_vectors: list[list[float]] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._return_vectors = return_vectors or [[1.0, 2.0], [1.0, 2.0]]
        self._raise_exc = raise_exc
        self.calls: list[list[str]] = []

    async def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        if self._raise_exc is not None:
            raise self._raise_exc
        self.calls.append(list(texts))
        return self._return_vectors


def _mail(subject: str = "STATUS update", body: str = "Entity STATUS changed to 18:00") -> MailEvent:
    return MailEvent(
        message_id="vec-1",
        sender="ops@example.com",
        subject=subject,
        body=body,
    )


def _request() -> ClassificationRequest:
    return ClassificationRequest(mail=_mail())


def _candidate(label: str, sim: float, count: int = 1) -> PathBCandidate:
    return PathBCandidate(
        label=label,
        max_similarity=sim,
        count=count,
        mean_similarity=sim,
        confidence=sim,
    )


def _build_classifier(
    emb_client: _FakeEmbeddingClient,
    candidates: list[PathBCandidate] | None = None,
    sample_count: int = 10,
    settings: VectorStoreSettings | None = None,
    taxonomy_loader: MagicMock | None = None,
) -> VectorClassifier:
    vector_store = MagicMock()
    vector_store.knn_search = AsyncMock(return_value=candidates or [])
    vector_store.count_samples = AsyncMock(return_value=sample_count)
    return VectorClassifier(
        vector_store,
        emb_client,
        settings or VectorStoreSettings(),
        taxonomy_loader=taxonomy_loader,
    )


# ---------------------------------------------------------------------------
# Normal classification
# ---------------------------------------------------------------------------


class TestNormalClassification:
    """Happy path: knn_search returns candidates → SUCCESS attempt."""

    async def test_returns_success_with_top1_label(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("schedule", 0.95),
                _candidate("operation", 0.85),
            ],
        )

        attempt = await clf.classify(_request())

        assert attempt.source == "vector"
        assert attempt.status == AttemptStatus.SUCCESS
        assert attempt.confidence == pytest.approx(0.95)
        # Multi-label policy (threshold-based): both candidates pass
        # minimum_support=1, so both are emitted as labels ordered by
        # similarity descending.
        assert len(attempt.labels) == 2
        assert attempt.labels[0].l1_code == "schedule"
        assert attempt.labels[0].l1_label == "schedule"
        assert attempt.labels[0].confidence == pytest.approx(0.95)
        assert attempt.labels[1].l1_code == "operation"
        assert attempt.labels[1].confidence == pytest.approx(0.85)

    async def test_evidence_contains_full_path_b_result(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        candidates = [
            _candidate("schedule", 0.95, count=3),
            _candidate("operation", 0.85, count=1),
        ]
        clf = _build_classifier(emb_client, candidates=candidates)

        attempt = await clf.classify(_request())

        path_b_raw = attempt.evidence.get("path_b_result")
        assert path_b_raw is not None
        path_b = PathBResult.model_validate(path_b_raw)
        assert path_b.reason == "ok"
        assert path_b.top1_label == "schedule"
        assert path_b.top1_similarity == pytest.approx(0.95)
        assert path_b.confidence == pytest.approx(0.95)
        assert len(path_b.candidates) == 2
        assert path_b.candidates[0].label == "schedule"

    async def test_knn_search_uses_settings_top_k_and_threshold(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        vector_store = MagicMock()
        vector_store.knn_search = AsyncMock(return_value=[_candidate("schedule", 0.95)])
        vector_store.count_samples = AsyncMock(return_value=10)
        settings = VectorStoreSettings(top_k=7, similarity_threshold=0.88)
        clf = VectorClassifier(vector_store, emb_client, settings)

        await clf.classify(_request())

        vector_store.knn_search.assert_awaited_once()
        call_args = vector_store.knn_search.call_args
        # positional: embedding_thread, then kwargs top_k + threshold
        assert call_args.kwargs["top_k"] == 7
        assert call_args.kwargs["threshold"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Threshold filtering (delegated to vector store)
# ---------------------------------------------------------------------------


class TestThresholdFiltering:
    """Threshold filtering is the vector store's responsibility; an empty
    candidate list from knn_search indicates nothing cleared the threshold."""

    async def test_empty_candidates_triggers_below_threshold_or_empty(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        # samples exist, but knn_search returns nothing → below_threshold
        clf = _build_classifier(
            emb_client, candidates=[], sample_count=5
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.NO_MATCH
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "below_threshold"


class TestEvidenceGates:
    async def test_near_tie_between_flat_categories_emits_multi_label(self) -> None:
        """Multi-label policy: a near-tie between two supported labels is a
        genuine multi-label signal, not ambiguity to be discarded.

        With ``minimum_margin`` intentionally not applied (see VectorClassifier
        Step 4), two candidates with sufficient support and close similarity
        are both emitted as labels.
        """
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        settings = VectorStoreSettings(minimum_margin=0.03)
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("schedule", 0.91, count=3),
                _candidate("operation", 0.90, count=2),
            ],
            settings=settings,
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.SUCCESS
        assert len(attempt.labels) == 2
        assert attempt.labels[0].l1_code == "schedule"
        assert attempt.labels[0].confidence == pytest.approx(0.91)
        assert attempt.labels[1].l1_code == "operation"
        assert attempt.labels[1].confidence == pytest.approx(0.90)
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "ok"
        # The margin is still recorded for audit even though it no longer
        # gates the success decision.
        assert path_b.similarity_margin == pytest.approx(0.01)

    async def test_stale_category_is_not_returned_when_taxonomy_is_available(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        taxonomy_loader = MagicMock()
        taxonomy_loader.get_tree.return_value = TaxonomyTree(
            nodes=[TaxonomyNode(code="schedule", label="Schedule")]
        )
        clf = _build_classifier(
            emb_client,
            candidates=[_candidate("retired_category", 0.95, count=2)],
            taxonomy_loader=taxonomy_loader,
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.NO_MATCH
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "stale_category"

    async def test_insufficient_support_returns_ambiguous_no_match(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        settings = VectorStoreSettings(minimum_support=2)
        clf = _build_classifier(
            emb_client,
            candidates=[_candidate("schedule", 0.95, count=1)],
            settings=settings,
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.NO_MATCH
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "ambiguous_candidates"


class TestIneligibleRetrievalDocument:
    async def test_attachment_only_mail_skips_knn_and_uses_normal_fallback(self) -> None:
        emb_client = _FakeEmbeddingClient()
        clf = _build_classifier(emb_client, candidates=[_candidate("document", 0.99)])
        request = ClassificationRequest(mail=_mail(subject="Document", body="Please see attached document."))

        attempt = await clf.classify(request)

        assert attempt.status == AttemptStatus.NO_MATCH
        assert emb_client.calls == []
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "ineligible_document"


# ---------------------------------------------------------------------------
# Single-label majority vs mixed-label candidate ordering
# ---------------------------------------------------------------------------


class TestCandidateAggregation:
    """The classifier relies on knn_search to aggregate by label and order
    by similarity; here we verify the top-1 selection against varied inputs."""

    async def test_single_label_majority_top1(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        # All candidates share the same label (eta_update) — top1 is the
        # highest similarity one.
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("schedule", 0.97, count=4),
                _candidate("schedule", 0.91, count=2),
            ],
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.SUCCESS
        assert attempt.labels[0].l1_code == "schedule"
        assert attempt.confidence == pytest.approx(0.97)

    async def test_mixed_labels_top1_is_highest_similarity(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("operation", 0.92, count=2),
                _candidate("schedule", 0.88, count=3),
                _candidate("entity_report", 0.80, count=1),
            ],
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.SUCCESS
        assert attempt.labels[0].l1_code == "operation"
        assert attempt.confidence == pytest.approx(0.92)
        # Multi-label policy: all three candidates pass minimum_support=1
        # and are emitted ordered by similarity.
        assert len(attempt.labels) == 3
        assert [lbl.l1_code for lbl in attempt.labels] == [
            "operation",
            "schedule",
            "entity_report",
        ]
        assert [lbl.confidence for lbl in attempt.labels] == [
            pytest.approx(0.92),
            pytest.approx(0.88),
            pytest.approx(0.80),
        ]

    async def test_all_candidates_preserved_in_path_b_result(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        candidates = [
            _candidate("operation", 0.92, count=2),
            _candidate("schedule", 0.88, count=3),
        ]
        clf = _build_classifier(emb_client, candidates=candidates)

        attempt = await clf.classify(_request())

        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert len(path_b.candidates) == 2
        # candidates list is preserved verbatim from knn_search output
        assert path_b.candidates[0].label == "operation"
        assert path_b.candidates[1].label == "schedule"


# ---------------------------------------------------------------------------
# Multi-label policy (threshold-based, scheme B)
# ---------------------------------------------------------------------------


class TestMultiLabelPolicy:
    """Threshold-based multi-label output: each candidate whose support clears
    ``minimum_support`` independently becomes a label."""

    async def test_partial_support_emits_only_qualified_labels(self) -> None:
        """Candidates below minimum_support are dropped; the rest are emitted."""
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        settings = VectorStoreSettings(minimum_support=2)
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("schedule", 0.92, count=3),  # passes
                _candidate("operation", 0.88, count=1),  # filtered out
                _candidate("entity_report", 0.85, count=2),  # passes
            ],
            settings=settings,
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.SUCCESS
        assert [lbl.l1_code for lbl in attempt.labels] == [
            "schedule",
            "entity_report",
        ]
        assert [lbl.confidence for lbl in attempt.labels] == [
            pytest.approx(0.92),
            pytest.approx(0.85),
        ]

    async def test_all_candidates_below_support_returns_no_match(self) -> None:
        """When no candidate clears minimum_support, the attempt degrades."""
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        settings = VectorStoreSettings(minimum_support=5)
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("schedule", 0.95, count=2),
                _candidate("operation", 0.90, count=1),
            ],
            settings=settings,
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.NO_MATCH
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "ambiguous_candidates"

    async def test_stale_category_skipped_but_other_labels_emitted(self) -> None:
        """A stale-category candidate is skipped without dropping the rest."""
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        taxonomy_loader = MagicMock()
        taxonomy_loader.get_tree.return_value = TaxonomyTree(
            nodes=[TaxonomyNode(code="schedule", label="Schedule")]
        )
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("retired_category", 0.95, count=2),
                _candidate("schedule", 0.90, count=2),
            ],
            taxonomy_loader=taxonomy_loader,
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.SUCCESS
        assert [lbl.l1_code for lbl in attempt.labels] == ["schedule"]

    async def test_all_stale_categories_returns_no_match_with_stale_reason(self) -> None:
        """When every supported candidate is stale, the attempt degrades with
        the ``stale_category`` reason rather than ``ambiguous_candidates``."""
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        taxonomy_loader = MagicMock()
        taxonomy_loader.get_tree.return_value = TaxonomyTree(
            nodes=[TaxonomyNode(code="schedule", label="Schedule")]
        )
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("retired_one", 0.95, count=2),
                _candidate("retired_two", 0.90, count=2),
            ],
            taxonomy_loader=taxonomy_loader,
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.NO_MATCH
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "stale_category"

    async def test_confidence_is_top1_max_similarity_regardless_of_label_count(self) -> None:
        """The attempt-level confidence stays anchored to top1 for backward
        compatibility with FusionOrchestrator's threshold gates, even when
        multiple labels are emitted."""
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        clf = _build_classifier(
            emb_client,
            candidates=[
                _candidate("schedule", 0.92, count=2),
                _candidate("operation", 0.88, count=2),
            ],
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.SUCCESS
        assert len(attempt.labels) == 2
        assert attempt.confidence == pytest.approx(0.92)
        assert attempt.meta.overall_confidence == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# embed_batch call count
# ---------------------------------------------------------------------------


class TestEmbedBatchCallCount:
    """embed_batch must be invoked exactly once via preprocess_mail."""

    async def test_embed_batch_called_once(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        clf = _build_classifier(emb_client, candidates=[_candidate("schedule", 0.9)])

        await clf.classify(_request())

        assert len(emb_client.calls) == 1
        # preprocess_mail sends [seg0_text, ctx_text] in a single batch
        assert len(emb_client.calls[0]) == 2


# ---------------------------------------------------------------------------
# Degradation paths
# ---------------------------------------------------------------------------


class TestDegradationEmbeddingUnavailable:
    """When preprocess_mail raises (embedding service down), the classifier
    returns UNAVAILABLE with reason='embedding_unavailable'."""

    async def test_embedding_exception_returns_unavailable(self) -> None:
        emb_client = _FakeEmbeddingClient(raise_exc=RuntimeError("TEI offline"))
        clf = _build_classifier(emb_client, candidates=[])

        attempt = await clf.classify(_request())

        assert attempt.source == "vector"
        assert attempt.status == AttemptStatus.UNAVAILABLE
        assert attempt.confidence == 0.0
        assert attempt.error is not None
        assert "TEI offline" in attempt.error
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "embedding_unavailable"
        assert path_b.top1_label is None
        assert path_b.candidates == []

    async def test_embedding_unavailable_does_not_call_knn_search(self) -> None:
        emb_client = _FakeEmbeddingClient(raise_exc=RuntimeError("timeout"))
        vector_store = MagicMock()
        vector_store.knn_search = AsyncMock(return_value=[])
        vector_store.count_samples = AsyncMock(return_value=0)
        clf = VectorClassifier(vector_store, emb_client, VectorStoreSettings())

        await clf.classify(_request())

        vector_store.knn_search.assert_not_awaited()


class TestDegradationEmptySamples:
    """When the sample library is empty (count=0) and knn returns nothing,
    reason='empty_samples'."""

    async def test_zero_samples_returns_no_match_empty_samples(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        clf = _build_classifier(
            emb_client, candidates=[], sample_count=0
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.NO_MATCH
        assert attempt.confidence == 0.0
        assert attempt.labels == []
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "empty_samples"
        assert path_b.top1_label is None


class TestDegradationBelowThreshold:
    """When samples exist but none clear the similarity threshold,
    reason='below_threshold'."""

    async def test_samples_exist_but_no_match_returns_below_threshold(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        clf = _build_classifier(
            emb_client, candidates=[], sample_count=42
        )

        attempt = await clf.classify(_request())

        assert attempt.status == AttemptStatus.NO_MATCH
        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "below_threshold"
        assert path_b.top1_label is None


# ---------------------------------------------------------------------------
# PathBResult field completeness
# ---------------------------------------------------------------------------


class TestPathBResultFields:
    """Verify all PathBResult fields are populated on the success path."""

    async def test_success_path_b_result_fields(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        candidates = [
            _candidate("schedule", 0.95, count=3),
            _candidate("operation", 0.80, count=1),
        ]
        clf = _build_classifier(emb_client, candidates=candidates)

        attempt = await clf.classify(_request())

        path_b = PathBResult.model_validate(attempt.evidence["path_b_result"])
        assert path_b.reason == "ok"
        assert path_b.top1_label == "schedule"
        assert path_b.top1_similarity == pytest.approx(0.95)
        assert path_b.confidence == pytest.approx(0.95)
        assert len(path_b.candidates) == 2
        # candidate fields are preserved
        top1_candidate = path_b.candidates[0]
        assert top1_candidate.label == "schedule"
        assert top1_candidate.max_similarity == pytest.approx(0.95)
        assert top1_candidate.count == 3
        assert top1_candidate.mean_similarity == pytest.approx(0.95)
        assert top1_candidate.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Classifier Protocol contract
# ---------------------------------------------------------------------------


class TestClassifierProtocolContract:
    """VectorClassifier structurally satisfies the Classifier Protocol."""

    def test_source_attribute_is_vector(self) -> None:
        emb_client = _FakeEmbeddingClient()
        clf = _build_classifier(emb_client)
        assert clf.source == "vector"

    async def test_classify_returns_classification_attempt(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        clf = _build_classifier(
            emb_client, candidates=[_candidate("schedule", 0.9)]
        )

        attempt = await clf.classify(_request())

        assert isinstance(attempt, ClassificationAttempt)

    def test_protocol_structural_contract(self) -> None:
        """Structural subtyping: VectorClassifier exposes ``source`` + ``classify``.

        Note: the Classifier Protocol is not ``@runtime_checkable``, so we verify
        the structural contract via attribute presence rather than isinstance.
        """
        emb_client = _FakeEmbeddingClient()
        clf = _build_classifier(emb_client)
        assert hasattr(clf, "source")
        assert hasattr(clf, "classify")
        assert callable(clf.classify)


# ---------------------------------------------------------------------------
# Settings propagation
# ---------------------------------------------------------------------------


class TestSettingsPropagation:
    """VectorStoreSettings flow into knn_search kwargs."""

    async def test_default_settings_applied(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        vector_store = MagicMock()
        vector_store.knn_search = AsyncMock(return_value=[_candidate("schedule", 0.9)])
        vector_store.count_samples = AsyncMock(return_value=10)
        clf = VectorClassifier(vector_store, emb_client, VectorStoreSettings())

        await clf.classify(_request())

        call_kwargs = vector_store.knn_search.call_args.kwargs
        assert call_kwargs["top_k"] == 5  # VectorStoreSettings default
        assert call_kwargs["threshold"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Scoped vector search (P0 — target label scoping)
# ---------------------------------------------------------------------------


def _request_with_scope(scope: list[str]) -> ClassificationRequest:
    """Build a request whose context carries ``vector_scope``."""

    return ClassificationRequest(mail=_mail(), context={"vector_scope": scope})


class TestScopedVectorSearch:
    """VectorClassifier reads ``request.context["vector_scope"]`` and passes it
    to ``knn_search(label_scope=...)``; empty scoped results trigger a global
    fallback with ``scoped_fallback=true`` recorded in evidence."""

    async def test_context_with_scope_passes_label_scope_to_knn(self) -> None:
        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        vector_store = MagicMock()
        vector_store.knn_search = AsyncMock(
            return_value=[_candidate("entity_report", 0.95)]
        )
        vector_store.count_samples = AsyncMock(return_value=10)
        clf = VectorClassifier(vector_store, emb_client, VectorStoreSettings())

        await clf.classify(_request_with_scope(["entity_report", "document"]))

        # First call must carry label_scope
        call_kwargs = vector_store.knn_search.call_args.kwargs
        assert call_kwargs["label_scope"] == ["entity_report", "document"]

    async def test_context_without_scope_passes_none(self) -> None:
        """Backward compatible: no vector_scope in context → label_scope=None."""

        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        vector_store = MagicMock()
        vector_store.knn_search = AsyncMock(
            return_value=[_candidate("schedule", 0.9)]
        )
        vector_store.count_samples = AsyncMock(return_value=10)
        clf = VectorClassifier(vector_store, emb_client, VectorStoreSettings())

        await clf.classify(_request())  # no context.vector_scope

        call_kwargs = vector_store.knn_search.call_args.kwargs
        assert call_kwargs["label_scope"] is None

    async def test_scoped_empty_triggers_global_fallback(self) -> None:
        """Scoped search returns [] → retry with label_scope=None → SUCCESS."""

        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        vector_store = MagicMock()
        # First call (scoped) returns []; second call (global) returns a hit
        vector_store.knn_search = AsyncMock(
            side_effect=[[], [_candidate("schedule", 0.9)]]
        )
        vector_store.count_samples = AsyncMock(return_value=10)
        clf = VectorClassifier(vector_store, emb_client, VectorStoreSettings())

        attempt = await clf.classify(_request_with_scope(["entity_report"]))

        assert attempt.status == AttemptStatus.SUCCESS
        assert vector_store.knn_search.await_count == 2
        # Second call must use label_scope=None (global fallback)
        second_call_kwargs = vector_store.knn_search.call_args_list[1].kwargs
        assert second_call_kwargs["label_scope"] is None
        # Evidence records the fallback
        assert attempt.evidence["scoped_fallback"] is True
        assert attempt.evidence["vector_scope"] == ["entity_report"]

    async def test_scoped_with_results_does_not_trigger_fallback(self) -> None:
        """Scoped search returns candidates → no retry → scoped_fallback=False."""

        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        vector_store = MagicMock()
        vector_store.knn_search = AsyncMock(
            return_value=[_candidate("entity_report", 0.92)]
        )
        vector_store.count_samples = AsyncMock(return_value=10)
        clf = VectorClassifier(vector_store, emb_client, VectorStoreSettings())

        attempt = await clf.classify(_request_with_scope(["entity_report"]))

        assert attempt.status == AttemptStatus.SUCCESS
        assert vector_store.knn_search.await_count == 1
        assert attempt.evidence["scoped_fallback"] is False
        assert attempt.evidence["vector_scope"] == ["entity_report"]

    async def test_scoped_empty_and_global_empty_records_fallback(self) -> None:
        """Both scoped and global return [] → NO_MATCH with scoped_fallback=True."""

        emb_client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        vector_store = MagicMock()
        vector_store.knn_search = AsyncMock(return_value=[])
        vector_store.count_samples = AsyncMock(return_value=5)
        clf = VectorClassifier(vector_store, emb_client, VectorStoreSettings())

        attempt = await clf.classify(_request_with_scope(["entity_report"]))

        assert attempt.status == AttemptStatus.NO_MATCH
        assert vector_store.knn_search.await_count == 2
        assert attempt.evidence["scoped_fallback"] is True
        assert attempt.evidence["vector_scope"] == ["entity_report"]
