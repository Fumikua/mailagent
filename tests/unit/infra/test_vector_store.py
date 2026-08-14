"""VectorStore unit tests (SQLite brute-force path).

Validates Path B sample storage, KNN search, archival, centroid computation,
and label backfill using an in-memory SQLite database.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from mailagent.domain import SampleQualityAssessment, SampleRecord
from mailagent.infra.config import VectorStoreSettings
from mailagent.infra.store import Base, SampleArchiveORM
from mailagent.infra.vector_store import VectorStore

DIM = 8


@pytest.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def settings() -> VectorStoreSettings:
    return VectorStoreSettings(top_k=5, similarity_threshold=0.85)


@pytest.fixture
def vector_store(engine, settings: VectorStoreSettings) -> VectorStore:
    return VectorStore(settings, engine)


def _embedding(seed: float) -> list[float]:
    """Deterministic embedding vector for testing (DIM dimensions)."""
    return [seed + i * 0.01 for i in range(DIM)]


def _make_sample(
    label_l1: str = "eta_update",
    source: str = "seed",
    days_ago: int = 0,
    mail_hash: str | None = None,
) -> SampleRecord:
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    resolved_mail_hash = mail_hash or f"hash-{uuid4()}"
    fingerprint = hashlib.sha256(resolved_mail_hash.encode("utf-8")).hexdigest()
    return SampleRecord(
        mail_hash=resolved_mail_hash,
        subject_raw="Test subject",
        subject_clean="test subject",
        sender="ops@example.com",
        sender_domain="example.com",
        body="Entity STATUS update body",
        label_l1=label_l1,
        label_l2=None,
        label_l3=None,
        confidence=0.9,
        source=source,  # type: ignore[arg-type]
        reviewed=True,
        thread_parsed=True,
        created_at=created,
        taxonomy_schema_version="flat-v1",
        retrieval_document={"text": "Subject: STATUS\nLatest message:\nETA revised."},
        retrieval_fingerprint=fingerprint,
        retrieval_policy_version="example-triage-v1",
        quality=SampleQualityAssessment(
            disposition="accepted",
            fingerprint=fingerprint,
            retrieval_policy_version="example-triage-v1",
        ),
    )


class TestInsertAndKnn:
    async def test_insert_rejects_duplicate_mail_hash_before_database_write(
        self, vector_store: VectorStore
    ) -> None:
        from mailagent.infra.vector_store import SampleAdmissionError

        first = _make_sample("schedule", mail_hash="same-mail")
        duplicate = _make_sample("operation", mail_hash="same-mail")
        await vector_store.insert_sample(first, _embedding(0.1), _embedding(0.1))

        with pytest.raises(SampleAdmissionError, match="duplicate_mail_hash"):
            await vector_store.insert_sample(duplicate, _embedding(0.2), _embedding(0.2))

    async def test_insert_rejects_duplicate_retrieval_fingerprint(
        self, vector_store: VectorStore
    ) -> None:
        from mailagent.infra.vector_store import SampleAdmissionError

        first = _make_sample("schedule")
        duplicate = _make_sample("operation").model_copy(
            update={
                "retrieval_fingerprint": first.retrieval_fingerprint,
                "quality": first.quality,
            }
        )
        await vector_store.insert_sample(first, _embedding(0.1), _embedding(0.1))

        with pytest.raises(SampleAdmissionError, match="duplicate_retrieval_fingerprint"):
            await vector_store.insert_sample(duplicate, _embedding(0.2), _embedding(0.2))

    async def test_insert_rejects_sample_without_accepted_quality(
        self, vector_store: VectorStore
    ) -> None:
        from mailagent.infra.vector_store import SampleAdmissionError

        rejected = _make_sample("schedule").model_copy(update={"quality": None})

        with pytest.raises(SampleAdmissionError, match="quality_not_accepted"):
            await vector_store.insert_sample(rejected, _embedding(0.1), _embedding(0.1))

    async def test_insert_rejects_warned_disposition(self, vector_store: VectorStore) -> None:
        """warned samples are not yet admissible to active retrieval."""
        from mailagent.infra.vector_store import SampleAdmissionError

        warned = _make_sample("schedule").model_copy(
            update={
                "quality": SampleQualityAssessment(
                    disposition="warned",
                    reasons=["soft_warning"],
                    fingerprint="b" * 64,
                    retrieval_policy_version="example-triage-v1",
                ),
            }
        )

        with pytest.raises(SampleAdmissionError, match="quality_not_accepted"):
            await vector_store.insert_sample(warned, _embedding(0.1), _embedding(0.1))

    async def test_insert_rejects_rejected_disposition(self, vector_store: VectorStore) -> None:
        """rejected samples must never enter the active sample library."""
        from mailagent.infra.vector_store import SampleAdmissionError

        rejected = _make_sample("schedule").model_copy(
            update={
                "quality": SampleQualityAssessment(
                    disposition="rejected",
                    reasons=["unknown_taxonomy_category"],
                    fingerprint="c" * 64,
                    retrieval_policy_version="example-triage-v1",
                ),
            }
        )

        with pytest.raises(SampleAdmissionError, match="quality_not_accepted"):
            await vector_store.insert_sample(rejected, _embedding(0.1), _embedding(0.1))

    async def test_flat_sample_round_trip_preserves_quality_provenance(
        self, vector_store: VectorStore
    ) -> None:
        sample = _make_sample().model_copy(
            update={
                "label_l1": "schedule",
                "label_l2": None,
                "label_l3": None,
                "taxonomy_schema_version": "flat-v1",
                "retrieval_document": {"text": "Subject: STATUS\nLatest message:\nETA revised."},
                "retrieval_fingerprint": "a" * 64,
                "retrieval_policy_version": "example-triage-v1",
                "quality": SampleQualityAssessment(
                    disposition="accepted",
                    fingerprint="a" * 64,
                    retrieval_policy_version="example-triage-v1",
                ),
                "review_override_reason": "operator confirmed clean retrieval text",
            }
        )

        await vector_store.insert_sample(sample, _embedding(0.1), _embedding(0.1))
        restored = await vector_store.get_sample(sample.id)

        assert restored is not None
        assert restored.label_l1 == "schedule"
        assert restored.label_l2 is None and restored.label_l3 is None
        assert restored.quality is not None
        assert restored.quality.disposition == "accepted"
        assert restored.retrieval_fingerprint == "a" * 64
        assert restored.review_override_reason == "operator confirmed clean retrieval text"

    async def test_insert_and_knn_returns_matching_label(
        self, vector_store: VectorStore
    ) -> None:
        """Insert a sample and query with the same embedding → cosine similarity ≈ 1.0."""
        sample = _make_sample("eta_update")
        emb = _embedding(0.1)
        await vector_store.insert_sample(sample, emb, emb)

        candidates = await vector_store.knn_search(emb, top_k=5, threshold=0.5)
        assert len(candidates) == 1
        assert candidates[0].label == "eta_update"
        assert candidates[0].count == 1
        assert candidates[0].max_similarity == pytest.approx(1.0, abs=1e-6)
        assert candidates[0].confidence == pytest.approx(1.0, abs=1e-6)

    async def test_knn_excludes_legacy_three_level_rows(
        self, vector_store: VectorStore
    ) -> None:
        flat = _make_sample("schedule")
        legacy = _make_sample("entity").model_copy(
            update={
                "label_l2": "schedule",
                "label_l3": "legacy_eta_update",
                "taxonomy_schema_version": "legacy-v3",
                "quality": None,
                "reviewed": False,
            }
        )
        emb = _embedding(0.1)
        await vector_store.insert_sample(flat, emb, emb)
        await vector_store.insert_sample(legacy, emb, emb)

        candidates = await vector_store.knn_search(emb, top_k=5, threshold=0.5)

        assert [candidate.label for candidate in candidates] == ["schedule"]

    async def test_knn_threshold_filters_low_similarity(
        self, vector_store: VectorStore
    ) -> None:
        """Orthogonal embeddings should be filtered out by the similarity threshold."""
        sample = _make_sample("eta_update")
        # Embedding along axis 0
        emb_stored = [0.0] * DIM
        emb_stored[0] = 1.0
        await vector_store.insert_sample(sample, emb_stored, emb_stored)

        # Orthogonal query along axis 1 → cosine similarity = 0
        query = [0.0] * DIM
        query[1] = 1.0
        candidates = await vector_store.knn_search(query, top_k=5, threshold=0.5)
        assert candidates == []

    async def test_knn_groups_by_label_and_aggregates(
        self, vector_store: VectorStore
    ) -> None:
        """Multiple samples with the same label should be grouped into one candidate."""
        # Three near-identical embeddings for label "eta_update"
        for i in range(3):
            s = _make_sample("eta_update")
            emb = _embedding(0.1 + i * 0.0001)
            await vector_store.insert_sample(s, emb, emb)
        # One different label with a dissimilar embedding
        s2 = _make_sample("location_plan")
        emb2 = _embedding(0.9)
        await vector_store.insert_sample(s2, emb2, emb2)

        candidates = await vector_store.knn_search(_embedding(0.1), top_k=5, threshold=0.5)
        labels = {c.label for c in candidates}
        assert "eta_update" in labels
        assert "location_plan" in labels

        eta = next(c for c in candidates if c.label == "eta_update")
        assert eta.count == 3
        assert eta.max_similarity == pytest.approx(1.0, abs=1e-3)

    async def test_knn_top_k_limits_results(
        self, vector_store: VectorStore
    ) -> None:
        """top_k should cap the number of returned candidates."""
        for i in range(10):
            s = _make_sample(f"label_{i}")
            emb = _embedding(float(i))
            await vector_store.insert_sample(s, emb, emb)

        candidates = await vector_store.knn_search(_embedding(0.0), top_k=3, threshold=0.0)
        assert len(candidates) <= 3

    async def test_category_repair_restores_knn_visibility(
        self, vector_store: VectorStore
    ) -> None:
        """A stale label updated via update_sample_label re-enters kNN results.

        Simulates the remediation flow described in spec.md:
        stale stored category is rewritten to a valid flat code, after which
        the sample participates in retrieval again.
        """
        sample = _make_sample("stale_code")
        emb = _embedding(0.1)
        await vector_store.insert_sample(sample, emb, emb)

        # Stale code is still searchable as long as it's flat-v1 + accepted.
        candidates = await vector_store.knn_search(emb, top_k=5, threshold=0.5)
        assert any(c.label == "stale_code" for c in candidates)

        # Operator repairs the label to a valid code.
        await vector_store.update_sample_label(sample.id, "schedule", source="human_fix")

        # Now the same query returns the repaired label.
        candidates = await vector_store.knn_search(emb, top_k=5, threshold=0.5)
        assert any(c.label == "schedule" for c in candidates)
        assert all(c.label != "stale_code" for c in candidates)

    async def test_knn_excludes_unreviewed_samples(
        self, vector_store: VectorStore
    ) -> None:
        """Unreviewed samples must not participate in kNN search."""
        reviewed = _make_sample("schedule")
        unreviewed = _make_sample("schedule").model_copy(update={"reviewed": False})
        emb = _embedding(0.1)
        await vector_store.insert_sample(reviewed, emb, emb)
        await vector_store.insert_sample(unreviewed, emb, emb)

        candidates = await vector_store.knn_search(emb, top_k=5, threshold=0.5)

        # Only the reviewed sample's label should appear, with count == 1.
        assert len(candidates) == 1
        assert candidates[0].label == "schedule"
        assert candidates[0].count == 1


class TestGetSamplesAndCount:
    async def test_get_samples_pagination(
        self, vector_store: VectorStore
    ) -> None:
        """Paginated listing returns the correct slice."""
        for i in range(10):
            s = _make_sample(f"label_{i}")
            await vector_store.insert_sample(s, _embedding(float(i)), _embedding(float(i)))

        page1 = await vector_store.get_samples(page=1, page_size=4)
        page2 = await vector_store.get_samples(page=2, page_size=4)
        page3 = await vector_store.get_samples(page=3, page_size=4)
        assert len(page1) == 4
        assert len(page2) == 4
        assert len(page3) == 2

    async def test_get_samples_filter_by_label(
        self, vector_store: VectorStore
    ) -> None:
        """label filter restricts results to matching flat label_l1."""
        for label in ("eta_update", "location_plan", "eta_update"):
            s = _make_sample(label)
            await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        results = await vector_store.get_samples(label="eta_update")
        assert len(results) == 2
        assert all(r.label_l1 == "eta_update" for r in results)

    async def test_get_samples_filter_by_source(
        self, vector_store: VectorStore
    ) -> None:
        """source filter restricts results to matching source."""
        s1 = _make_sample("eta_update", source="seed")
        s2 = _make_sample("location_plan", source="llm")
        await vector_store.insert_sample(s1, _embedding(0.1), _embedding(0.1))
        await vector_store.insert_sample(s2, _embedding(0.2), _embedding(0.2))

        results = await vector_store.get_samples(source="llm")
        assert len(results) == 1
        assert results[0].source == "llm"

    async def test_count_samples_total_and_windowed(
        self, vector_store: VectorStore
    ) -> None:
        """count_samples returns total or windowed count."""
        s_new = _make_sample("eta_update", days_ago=1)
        s_old = _make_sample("eta_update", days_ago=100)
        await vector_store.insert_sample(s_new, _embedding(0.1), _embedding(0.1))
        await vector_store.insert_sample(s_old, _embedding(0.2), _embedding(0.2))

        total = await vector_store.count_samples()
        assert total == 2

        recent = await vector_store.count_samples(days=30)
        assert recent == 1


class TestDeleteAndUpdate:
    async def test_delete_sample_removes_row(
        self, vector_store: VectorStore
    ) -> None:
        """delete_sample removes the row; count drops to 0."""
        s = _make_sample("eta_update")
        await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        await vector_store.delete_sample(s.id)
        assert await vector_store.count_samples() == 0

    async def test_delete_unknown_id_is_noop(
        self, vector_store: VectorStore
    ) -> None:
        """Deleting a non-existent id should not raise."""
        await vector_store.delete_sample(uuid4())

    async def test_update_sample_label(self, vector_store: VectorStore) -> None:
        """update_sample_label rewrites label_l1, source, confidence, and marks reviewed."""
        s = _make_sample("eta_update")
        await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        await vector_store.update_sample_label(s.id, "location_plan", source="human_fix", confidence=1.0)
        results = await vector_store.get_samples(label="location_plan")
        assert len(results) == 1
        assert results[0].source == "human_fix"
        assert results[0].confidence == 1.0
        assert results[0].reviewed is True

    async def test_update_sample_label_unknown_id_is_noop(
        self, vector_store: VectorStore
    ) -> None:
        """Updating a non-existent id should not raise."""
        await vector_store.update_sample_label(uuid4(), "location_plan")


class TestArchive:
    async def test_archive_moves_old_samples_and_deletes_from_active(
        self, vector_store: VectorStore, engine
    ) -> None:
        """archive_old_samples moves rows older than the cutoff to samples_archive."""
        # Insert one old sample (> 12 months ago) and one recent sample.
        old = _make_sample("eta_update", days_ago=400)
        recent = _make_sample("location_plan", days_ago=1)
        await vector_store.insert_sample(old, _embedding(0.1), _embedding(0.1))
        await vector_store.insert_sample(recent, _embedding(0.2), _embedding(0.2))

        moved = await vector_store.archive_old_samples(months=12)
        assert moved == 1
        assert await vector_store.count_samples() == 1

        # Verify the old sample is in the archive table.
        from sqlalchemy import select

        async with vector_store.sessions() as session:
            archived = (await session.scalars(select(SampleArchiveORM))).all()
        assert len(archived) == 1
        assert archived[0].id == str(old.id)
        assert archived[0].label_l1 == "eta_update"

    async def test_archive_with_no_old_samples_returns_zero(
        self, vector_store: VectorStore
    ) -> None:
        """If no samples are past the retention window, archive returns 0."""
        s = _make_sample("eta_update", days_ago=1)
        await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))
        moved = await vector_store.archive_old_samples(months=12)
        assert moved == 0


class TestCentroids:
    async def test_get_centroids_averages_embeddings_per_label(
        self, vector_store: VectorStore
    ) -> None:
        """Centroid should be the element-wise mean of all embeddings for a label."""
        # Three samples with the same label but different embeddings.
        emb1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        emb3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for emb in (emb1, emb2, emb3):
            s = _make_sample("eta_update")
            await vector_store.insert_sample(s, emb, emb)

        centroids = await vector_store.get_centroids()
        assert "eta_update" in centroids
        centroid = centroids["eta_update"]
        assert len(centroid) == DIM
        # Mean of emb1, emb2, emb3 → each axis gets 1/3
        assert centroid[0] == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert centroid[1] == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert centroid[2] == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert centroid[3] == pytest.approx(0.0, abs=1e-6)

    async def test_get_centroids_empty_when_no_samples(
        self, vector_store: VectorStore
    ) -> None:
        """No samples → empty centroids dict."""
        centroids = await vector_store.get_centroids()
        assert centroids == {}


class TestBackfill:
    async def test_backfill_rewrites_label(self, vector_store: VectorStore) -> None:
        """backfill_samples_label updates label_l1 for the given sample ids."""
        s1 = _make_sample("eta_update")
        s2 = _make_sample("location_plan")
        await vector_store.insert_sample(s1, _embedding(0.1), _embedding(0.1))
        await vector_store.insert_sample(s2, _embedding(0.2), _embedding(0.2))

        await vector_store.backfill_samples_label("new_code", [s1.id, s2.id])
        results = await vector_store.get_samples(label="new_code")
        assert len(results) == 2

    async def test_backfill_is_idempotent(
        self, vector_store: VectorStore
    ) -> None:
        """Calling backfill twice with the same code produces the same state."""
        s = _make_sample("eta_update")
        await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        await vector_store.backfill_samples_label("renamed", [s.id])
        await vector_store.backfill_samples_label("renamed", [s.id])

        results = await vector_store.get_samples(label="renamed")
        assert len(results) == 1

    async def test_backfill_with_empty_list_is_noop(
        self, vector_store: VectorStore
    ) -> None:
        """Empty sample_ids list should not raise."""
        await vector_store.backfill_samples_label("renamed", [])


class TestGetEmbeddingsAndStratified:
    async def test_get_embeddings_returns_tuples(
        self, vector_store: VectorStore
    ) -> None:
        """get_embeddings returns (UUID, embedding, label) for all samples with embeddings."""
        s = _make_sample("eta_update")
        await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        embeddings = await vector_store.get_embeddings()
        assert len(embeddings) == 1
        uid, emb, label = embeddings[0]
        assert uid == s.id
        assert label == "eta_update"
        assert len(emb) == DIM

    async def test_stratified_sample_caps_per_label(
        self, vector_store: VectorStore
    ) -> None:
        """stratified_sample returns at most max_per_label per label."""
        # 5 samples for label A, 2 for label B
        for _ in range(5):
            await vector_store.insert_sample(
                _make_sample("label_a", days_ago=1), _embedding(0.1), _embedding(0.1)
            )
        for _ in range(2):
            await vector_store.insert_sample(
                _make_sample("label_b", days_ago=1), _embedding(0.2), _embedding(0.2)
            )

        result = await vector_store.stratified_sample(days=30, max_per_label=3)
        label_a_count = sum(1 for _, _, label in result if label == "label_a")
        label_b_count = sum(1 for _, _, label in result if label == "label_b")
        assert label_a_count == 3
        assert label_b_count == 2


class TestQualityStats:
    async def test_quality_stats_aggregates_disposition_and_taxonomy(
        self, vector_store: VectorStore
    ) -> None:
        """get_quality_stats returns per-disposition / schema / policy / label counts."""
        s1 = _make_sample("schedule")
        s2 = _make_sample("operation")
        await vector_store.insert_sample(s1, _embedding(0.1), _embedding(0.1))
        await vector_store.insert_sample(s2, _embedding(0.2), _embedding(0.2))

        stats = await vector_store.get_quality_stats()

        assert stats["by_disposition"].get("accepted") == 2
        assert stats["by_taxonomy_schema"].get("flat-v1") == 2
        assert stats["by_retrieval_policy"].get("example-triage-v1") == 2
        assert stats["by_label_l1"].get("schedule") == 1
        assert stats["by_label_l1"].get("operation") == 1
        assert stats["duplicate_fingerprint_rows"] == 0

    async def test_quality_stats_reports_zero_when_empty(
        self, vector_store: VectorStore
    ) -> None:
        """Empty sample library yields zeroed quality stats."""
        stats = await vector_store.get_quality_stats()

        assert stats["by_disposition"] == {}
        assert stats["by_taxonomy_schema"] == {}
        assert stats["by_retrieval_policy"] == {}
        assert stats["by_label_l1"] == {}
        assert stats["duplicate_fingerprint_rows"] == 0


class TestReembedCandidates:
    async def test_get_reembed_candidates_returns_outdated_samples(
        self, vector_store: VectorStore
    ) -> None:
        """Samples whose policy_version differs from target are returned."""
        s1 = _make_sample("schedule")  # policy_version = example-triage-v1
        await vector_store.insert_sample(s1, _embedding(0.1), _embedding(0.1))

        candidates = await vector_store.get_reembed_candidates(
            target_policy_version="example-triage-v2", batch_size=10
        )
        assert len(candidates) == 1
        assert candidates[0] == s1.id

    async def test_get_reembed_candidates_excludes_matching_policy(
        self, vector_store: VectorStore
    ) -> None:
        """Samples already on the target policy version are skipped."""
        s = _make_sample("schedule")
        await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        candidates = await vector_store.get_reembed_candidates(
            target_policy_version="example-triage-v1", batch_size=10
        )
        assert candidates == []

    async def test_mark_reembed_complete_updates_policy_version(
        self, vector_store: VectorStore
    ) -> None:
        """After mark_reembed_complete, the sample is no longer a candidate."""
        s = _make_sample("schedule")
        await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        await vector_store.mark_reembed_complete(
            s.id,
            embedding_thread=_embedding(0.5),
            embedding_segment_0=_embedding(0.5),
            retrieval_policy_version="example-triage-v2",
        )

        candidates = await vector_store.get_reembed_candidates(
            target_policy_version="example-triage-v2", batch_size=10
        )
        assert candidates == []

        updated = await vector_store.get_sample(s.id)
        assert updated is not None
        assert updated.retrieval_policy_version == "example-triage-v2"
