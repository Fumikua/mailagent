"""Unit tests for ClusteringEngine.

All tests mock vector_store, llm_client, and taxonomy_loader — no real
database or LLM calls. HDBSCAN is mocked via module-level patches when
testing the full run_weekly_clustering flow.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from mailagent.domain.models import SampleRecord
from mailagent.infra.clustering import ClusteringEngine
from mailagent.infra.config import ClusteringSettings

DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding(seed: float) -> list[float]:
    """Deterministic embedding vector for testing (DIM dimensions)."""
    return [seed + i * 0.01 for i in range(DIM)]


def _make_sample(
    sample_id: UUID | None = None,
    label_l3: str = "schedule",
    subject: str = "Test subject",
    body: str = "Test body content",
    sender: str = "ops@example.com",
    sender_domain: str = "example.com",
) -> SampleRecord:
    return SampleRecord(
        id=sample_id or uuid4(),
        mail_hash=f"hash-{uuid4()}",
        subject_raw=subject,
        subject_clean=subject.lower(),
        sender=sender,
        sender_domain=sender_domain,
        body=body,
        label_l1=label_l3,
        label_l2=label_l3,
        label_l3=label_l3,
        confidence=0.9,
        source="seed",
    )


def _make_engine(
    tmp_path: Path,
    vector_store: MagicMock | None = None,
    llm_client: MagicMock | None = None,
    taxonomy_loader: MagicMock | None = None,
    settings: ClusteringSettings | None = None,
) -> ClusteringEngine:
    """Build a ClusteringEngine with mocked dependencies and tmp reports dir."""
    vs = vector_store or MagicMock()
    llm = llm_client or MagicMock()
    tax = taxonomy_loader or MagicMock()
    engine = ClusteringEngine(vs, tax, llm, settings or ClusteringSettings())
    engine._reports_dir = tmp_path  # type: ignore[assignment]
    return engine


def _mock_llm(description: str = "Entity STATUS update intent") -> MagicMock:
    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        return_value={
            "choices": [{"message": {"content": description}}]
        }
    )
    return llm


def _mock_taxonomy(codes: set[str] | None = None) -> MagicMock:
    tax = MagicMock()
    tree = MagicMock()
    tree.all_codes = MagicMock(return_value=codes or {"valid_label"})
    tax.get_tree = MagicMock(return_value=tree)
    return tax


# ---------------------------------------------------------------------------
# Test: _classify_cluster
# ---------------------------------------------------------------------------


class TestClassifyCluster:
    def test_known_cluster_high_similarity(self, tmp_path: Path) -> None:
        """Cluster centroid very similar to taxonomy centroid → known."""
        engine = _make_engine(tmp_path)
        # Cluster embeddings that average to [1, 0, 0, ...]
        cluster_emb = [[1.0] + [0.0] * (DIM - 1)] * 3
        centroids = {"known_label": [1.0] + [0.0] * (DIM - 1)}

        cluster_type, max_sim = engine._classify_cluster(cluster_emb, centroids)
        assert cluster_type == "known"
        assert max_sim >= 0.85

    def test_drift_cluster_moderate_similarity(self, tmp_path: Path) -> None:
        """Cluster centroid moderately similar → drift."""
        engine = _make_engine(tmp_path)
        # Centroid [0.7, 0.7, 0, ...] vs taxonomy [1, 0, 0, ...]
        # cosine sim = 0.7 / sqrt(0.98) ≈ 0.707
        cluster_emb = [[0.7, 0.7] + [0.0] * (DIM - 2)] * 3
        centroids = {"known_label": [1.0] + [0.0] * (DIM - 1)}

        cluster_type, max_sim = engine._classify_cluster(cluster_emb, centroids)
        assert cluster_type == "drift"
        assert 0.6 <= max_sim < 0.85

    def test_new_intent_cluster_low_similarity(self, tmp_path: Path) -> None:
        """Cluster centroid orthogonal to taxonomy centroid → new_intent."""
        engine = _make_engine(tmp_path)
        cluster_emb = [[0.0, 0.0, 1.0] + [0.0] * (DIM - 3)] * 3
        centroids = {"known_label": [1.0] + [0.0] * (DIM - 1)}

        cluster_type, max_sim = engine._classify_cluster(cluster_emb, centroids)
        assert cluster_type == "new_intent"
        assert max_sim < 0.6

    def test_empty_centroids_returns_new_intent(self, tmp_path: Path) -> None:
        """No taxonomy centroids → everything is new_intent."""
        engine = _make_engine(tmp_path)
        cluster_emb = [[1.0] + [0.0] * (DIM - 1)] * 3
        cluster_type, max_sim = engine._classify_cluster(cluster_emb, {})
        assert cluster_type == "new_intent"
        assert max_sim == 0.0


# ---------------------------------------------------------------------------
# Test: _extract_representatives
# ---------------------------------------------------------------------------


class TestExtractRepresentatives:
    def test_returns_k_closest_to_centroid(self, tmp_path: Path) -> None:
        """Should return the k samples closest to the centroid."""
        engine = _make_engine(tmp_path)
        centroid = [1.0] + [0.0] * (DIM - 1)

        # 5 samples identical to centroid (sim=1.0)
        close_samples = [_make_sample(subject=f"Close {i}") for i in range(5)]
        close_emb = [[1.0] + [0.0] * (DIM - 1)] * 5

        # 5 samples orthogonal (sim=0.0)
        far_samples = [_make_sample(subject=f"Far {i}") for i in range(5)]
        far_emb = [[0.0, 0.0, 1.0] + [0.0] * (DIM - 3)] * 5

        all_samples = close_samples + far_samples
        all_emb = close_emb + far_emb

        representatives = engine._extract_representatives(
            all_samples, all_emb, centroid, k=5
        )
        assert len(representatives) == 5
        close_subjects = {s.subject_raw for s in representatives}
        assert all("Close" in s for s in close_subjects)

    def test_empty_cluster_returns_empty(self, tmp_path: Path) -> None:
        """Empty cluster samples → empty list."""
        engine = _make_engine(tmp_path)
        result = engine._extract_representatives([], [], [1.0] * DIM, k=5)
        assert result == []

    def test_fewer_than_k_returns_all(self, tmp_path: Path) -> None:
        """Fewer than k samples → return all."""
        engine = _make_engine(tmp_path)
        samples = [_make_sample() for _ in range(3)]
        emb = [[1.0] + [0.0] * (DIM - 1)] * 3
        centroid = [1.0] + [0.0] * (DIM - 1)

        result = engine._extract_representatives(samples, emb, centroid, k=5)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Test: _llm_describe_intent
# ---------------------------------------------------------------------------


class TestLLMDescribeIntent:
    async def test_describes_intent_successfully(self, tmp_path: Path) -> None:
        """LLM returns a description string."""
        llm = _mock_llm("Entity schedule STATUS notification")
        engine = _make_engine(tmp_path, llm_client=llm)
        representatives = [_make_sample(subject="STATUS Update") for _ in range(3)]

        intent = await engine._llm_describe_intent(representatives)
        assert intent == "Entity schedule STATUS notification"

    async def test_returns_unknown_on_failure(self, tmp_path: Path) -> None:
        """LLM exception → returns 'Unknown intent'."""
        llm = MagicMock()
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
        engine = _make_engine(tmp_path, llm_client=llm)
        representatives = [_make_sample()]

        intent = await engine._llm_describe_intent(representatives)
        assert intent == "Unknown intent"

    async def test_empty_representatives_returns_unknown(self, tmp_path: Path) -> None:
        """No representatives → 'Unknown intent' without calling LLM."""
        llm = _mock_llm()
        engine = _make_engine(tmp_path, llm_client=llm)
        intent = await engine._llm_describe_intent([])
        assert intent == "Unknown intent"
        llm.chat_completion.assert_not_called()


# ---------------------------------------------------------------------------
# Test: _generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_report_with_new_intents_and_drifts(self, tmp_path: Path) -> None:
        """Report should contain new intents, drifts, and checkboxes."""
        engine = _make_engine(tmp_path)
        engine._total_samples = 100

        new_intents = [
            {
                "cluster_id": 2,
                "sample_count": 8,
                "intent": "New warranty claim emails",
                "representatives": [
                    _make_sample(subject="Warranty Claim #123"),
                    _make_sample(subject="Re: Warranty Issue"),
                ],
            }
        ]
        drifts = [
            {
                "cluster_id": 1,
                "sample_count": 5,
                "intent": "STATUS drift variant",
                "max_similarity": 0.7234,
                "representatives": [_make_sample(subject="STATUS Update Variant")],
            }
        ]

        report_path = tmp_path / "test_report.md"
        engine._generate_report(new_intents, drifts, report_path)

        content = report_path.read_text(encoding="utf-8")
        # Header and metadata
        assert "# Intent Discovery Report" in content
        assert "**Total samples analyzed**: 100" in content
        assert "**Clustering parameters**" in content
        # New intents section
        assert "## New Intents" in content
        assert "Cluster #2" in content
        assert "New warranty claim emails" in content
        assert "Warranty Claim #123" in content
        # Drifts section
        assert "## Drifts" in content
        assert "Cluster #1" in content
        assert "STATUS drift variant" in content
        assert "0.7234" in content
        # Checkboxes
        assert "- [ ]" in content

    def test_report_empty_when_no_findings(self, tmp_path: Path) -> None:
        """Report with no new intents or drifts should have 'no findings' notes."""
        engine = _make_engine(tmp_path)
        engine._total_samples = 50
        report_path = tmp_path / "empty_report.md"
        engine._generate_report([], [], report_path)

        content = report_path.read_text(encoding="utf-8")
        assert "无新意图候选" in content
        assert "无漂移检测" in content


# ---------------------------------------------------------------------------
# Test: run_weekly_clustering
# ---------------------------------------------------------------------------


class TestRunWeeklyClustering:
    async def test_hdbscan_unavailable_degradation(self, tmp_path: Path) -> None:
        """When hdbscan is not installed, should return skip message."""
        engine = _make_engine(tmp_path)
        with patch("mailagent.infra.clustering.HAS_HDBSCAN", False):
            result = await engine.run_weekly_clustering()
        assert result == "clustering skipped: hdbscan not installed"

    async def test_stratified_sampling_triggered(self, tmp_path: Path) -> None:
        """When embeddings > max_samples, stratified_sample should be called."""
        vs = MagicMock()
        # Create more embeddings than max_samples
        large_data = [
            (uuid4(), _embedding(0.1), f"label_{i % 5}")
            for i in range(60)
        ]
        vs.get_embeddings = AsyncMock(return_value=large_data)
        sampled = large_data[:50]
        vs.stratified_sample = AsyncMock(return_value=sampled)
        vs.get_centroids = AsyncMock(return_value={})
        vs.get_samples = AsyncMock(return_value=[])

        llm = _mock_llm()
        tax = _mock_taxonomy()
        settings = ClusteringSettings(max_samples=50, min_cluster_size=5)
        engine = _make_engine(tmp_path, vs, llm, tax, settings)

        mock_np = MagicMock()
        mock_np.array = MagicMock(side_effect=lambda x: x)
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict = MagicMock(return_value=[-1] * 50)
        mock_hdbscan = MagicMock()
        mock_hdbscan.HDBSCAN = MagicMock(return_value=mock_clusterer)

        with (
            patch("mailagent.infra.clustering.HAS_HDBSCAN", True),
            patch("mailagent.infra.clustering.hdbscan", mock_hdbscan),
            patch("mailagent.infra.clustering.np", mock_np),
        ):
            await engine.run_weekly_clustering()

        vs.stratified_sample.assert_called_once()

    async def test_known_drift_new_intent_in_report(self, tmp_path: Path) -> None:
        """Full run: known cluster skipped, drift + new_intent in report."""
        # 15 samples in 3 clusters + 5 noise = 20 total
        # Cluster 0 (5 samples): known (centroid = taxonomy centroid)
        # Cluster 1 (5 samples): drift (centroid ~0.707 sim)
        # Cluster 2 (5 samples): new_intent (centroid orthogonal)
        # 5 noise points

        ids = [uuid4() for _ in range(20)]
        known_emb = [1.0] + [0.0] * (DIM - 1)
        drift_emb = [0.7, 0.7] + [0.0] * (DIM - 2)
        new_emb = [0.0, 0.0, 1.0] + [0.0] * (DIM - 3)

        embeddings_data: list[tuple[UUID, list[float], str]] = []
        for i in range(5):
            embeddings_data.append((ids[i], list(known_emb), "known_label"))
        for i in range(5, 10):
            embeddings_data.append((ids[i], list(drift_emb), "drift_label"))
        for i in range(10, 15):
            embeddings_data.append((ids[i], list(new_emb), "new_label"))
        for i in range(15, 20):
            embeddings_data.append((ids[i], _embedding(float(i)), "noise"))

        # HDBSCAN labels: cluster 0, 1, 2, then noise
        hdbscan_labels = [0] * 5 + [1] * 5 + [2] * 5 + [-1] * 5

        # Samples for representative extraction
        all_samples = [
            _make_sample(
                sample_id=ids[i],
                subject=f"Email {i}",
                body=f"Body {i}",
                label_l3=embeddings_data[i][2],
            )
            for i in range(20)
        ]

        vs = MagicMock()
        vs.get_embeddings = AsyncMock(return_value=embeddings_data)
        vs.get_centroids = AsyncMock(
            return_value={"known_label": list(known_emb)}
        )
        vs.get_samples = AsyncMock(return_value=all_samples)

        llm = _mock_llm("Test intent description")
        tax = _mock_taxonomy()
        engine = _make_engine(tmp_path, vs, llm, tax)

        mock_np = MagicMock()
        mock_np.array = MagicMock(side_effect=lambda x: x)
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict = MagicMock(return_value=hdbscan_labels)
        mock_hdbscan = MagicMock()
        mock_hdbscan.HDBSCAN = MagicMock(return_value=mock_clusterer)

        with (
            patch("mailagent.infra.clustering.HAS_HDBSCAN", True),
            patch("mailagent.infra.clustering.hdbscan", mock_hdbscan),
            patch("mailagent.infra.clustering.np", mock_np),
        ):
            result = await engine.run_weekly_clustering()

        # Should return a report path, not skip message
        assert result != "clustering skipped: hdbscan not installed"
        report_path = Path(result)
        assert report_path.exists()

        content = report_path.read_text(encoding="utf-8")
        # Known cluster should NOT appear in report (skipped)
        assert "Cluster #0" not in content
        # Drift cluster should appear
        assert "Cluster #1" in content
        assert "## Drifts" in content
        # New intent cluster should appear
        assert "Cluster #2" in content
        assert "## New Intents" in content
        # LLM description should be in report
        assert "Test intent description" in content
        # Checkboxes present
        assert "- [ ]" in content

    async def test_insufficient_samples_generates_empty_report(
        self, tmp_path: Path
    ) -> None:
        """When sample count < min_cluster_size, generate empty report."""
        vs = MagicMock()
        vs.get_embeddings = AsyncMock(return_value=[])
        vs.get_centroids = AsyncMock(return_value={})
        vs.get_samples = AsyncMock(return_value=[])

        llm = _mock_llm()
        tax = _mock_taxonomy()
        engine = _make_engine(tmp_path, vs, llm, tax)

        mock_np = MagicMock()
        mock_hdbscan = MagicMock()

        with (
            patch("mailagent.infra.clustering.HAS_HDBSCAN", True),
            patch("mailagent.infra.clustering.hdbscan", mock_hdbscan),
            patch("mailagent.infra.clustering.np", mock_np),
        ):
            result = await engine.run_weekly_clustering()

        report_path = Path(result)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "无新意图候选" in content


# ---------------------------------------------------------------------------
# Test: backfill_after_taxonomy_change
# ---------------------------------------------------------------------------


class TestBackfill:
    async def test_backfill_finds_stale_labels(self, tmp_path: Path) -> None:
        """Samples with labels not in taxonomy should be backfilled."""
        stale_sample = _make_sample(label_l3="old_removed_label")
        valid_sample = _make_sample(label_l3="valid_label")

        vs = MagicMock()
        vs.get_samples = AsyncMock(return_value=[stale_sample, valid_sample])
        vs.backfill_samples_label = AsyncMock()

        tax = _mock_taxonomy(codes={"valid_label", "new_code"})
        engine = _make_engine(tmp_path, vs, taxonomy_loader=tax)

        count = await engine.backfill_after_taxonomy_change("new_code")
        assert count == 1
        vs.backfill_samples_label.assert_called_once_with(
            "new_code", [stale_sample.id]
        )

    async def test_backfill_idempotent(self, tmp_path: Path) -> None:
        """Second call with no stale samples should return 0."""
        stale_sample = _make_sample(label_l3="old_label")

        vs = MagicMock()
        # First call: 1 stale sample; second call: 0 stale samples
        vs.get_samples = AsyncMock(
            side_effect=[[stale_sample], []]
        )
        vs.backfill_samples_label = AsyncMock()

        tax = _mock_taxonomy(codes={"valid_label", "new_code"})
        engine = _make_engine(tmp_path, vs, taxonomy_loader=tax)

        count1 = await engine.backfill_after_taxonomy_change("new_code")
        assert count1 == 1

        count2 = await engine.backfill_after_taxonomy_change("new_code")
        assert count2 == 0

        # backfill_samples_label should only be called once
        assert vs.backfill_samples_label.call_count == 1

    async def test_backfill_no_stale_returns_zero(self, tmp_path: Path) -> None:
        """All labels valid → no backfill needed."""
        valid_sample = _make_sample(label_l3="valid_label")
        vs = MagicMock()
        vs.get_samples = AsyncMock(return_value=[valid_sample])
        vs.backfill_samples_label = AsyncMock()

        tax = _mock_taxonomy(codes={"valid_label"})
        engine = _make_engine(tmp_path, vs, taxonomy_loader=tax)

        count = await engine.backfill_after_taxonomy_change("new_code")
        assert count == 0
        vs.backfill_samples_label.assert_not_called()


# ---------------------------------------------------------------------------
# Test: HDBSCAN input data assembly
# ---------------------------------------------------------------------------


class TestHdbscanInputAssembly:
    async def test_hdbscan_receives_correct_array(self, tmp_path: Path) -> None:
        """HDBSCAN should receive an array built from embedding vectors."""
        ids = [uuid4() for _ in range(7)]
        embeddings_data = [
            (ids[i], _embedding(0.1 + i * 0.01), "label_a") for i in range(7)
        ]
        all_samples = [
            _make_sample(sample_id=ids[i]) for i in range(7)
        ]

        vs = MagicMock()
        vs.get_embeddings = AsyncMock(return_value=embeddings_data)
        vs.get_centroids = AsyncMock(return_value={})
        vs.get_samples = AsyncMock(return_value=all_samples)

        llm = _mock_llm()
        tax = _mock_taxonomy()
        engine = _make_engine(tmp_path, vs, llm, tax)

        captured_array: list | None = None

        def capture_array(arr):
            nonlocal captured_array
            captured_array = arr
            return [0] * 5 + [-1] * 2  # 5 in cluster, 2 noise

        mock_np = MagicMock()
        mock_np.array = MagicMock(side_effect=capture_array)
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict = MagicMock(return_value=[0] * 5 + [-1] * 2)
        mock_hdbscan = MagicMock()
        mock_hdbscan.HDBSCAN = MagicMock(return_value=mock_clusterer)

        with (
            patch("mailagent.infra.clustering.HAS_HDBSCAN", True),
            patch("mailagent.infra.clustering.hdbscan", mock_hdbscan),
            patch("mailagent.infra.clustering.np", mock_np),
        ):
            await engine.run_weekly_clustering()

        # np.array should have been called with the embedding vectors
        assert captured_array is not None
        assert len(captured_array) == 7
        # Each element should be the embedding vector
        assert captured_array[0] == embeddings_data[0][1]

    async def test_hdbscan_called_with_correct_params(self, tmp_path: Path) -> None:
        """HDBSCAN should be called with settings parameters."""
        embeddings_data = [
            (uuid4(), _embedding(0.1), "label_a") for _ in range(7)
        ]
        all_samples = [_make_sample() for _ in range(7)]

        vs = MagicMock()
        vs.get_embeddings = AsyncMock(return_value=embeddings_data)
        vs.get_centroids = AsyncMock(return_value={})
        vs.get_samples = AsyncMock(return_value=all_samples)

        llm = _mock_llm()
        tax = _mock_taxonomy()
        settings = ClusteringSettings(min_cluster_size=5, min_samples=3, metric="cosine")
        engine = _make_engine(tmp_path, vs, llm, tax, settings)

        mock_np = MagicMock()
        mock_np.array = MagicMock(side_effect=lambda x: x)
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict = MagicMock(return_value=[-1] * 7)
        mock_hdbscan = MagicMock()
        mock_hdbscan.HDBSCAN = MagicMock(return_value=mock_clusterer)

        with (
            patch("mailagent.infra.clustering.HAS_HDBSCAN", True),
            patch("mailagent.infra.clustering.hdbscan", mock_hdbscan),
            patch("mailagent.infra.clustering.np", mock_np),
        ):
            await engine.run_weekly_clustering()

        mock_hdbscan.HDBSCAN.assert_called_once_with(
            min_cluster_size=5, min_samples=3, metric="cosine"
        )
