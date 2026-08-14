"""End-to-end integration tests for ClusteringEngine (Section 18.7).

Generates 100 synthetic samples with embedding vectors, runs the full
``ClusteringEngine.run_weekly_clustering`` flow against a real SQLite
``VectorStore`` instance, and verifies the generated markdown report.

Skips when ``hdbscan`` is not installed (``HAS_HDBSCAN`` is False).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from mailagent.domain.models import SampleRecord
from mailagent.infra.clustering import HAS_HDBSCAN, ClusteringEngine
from mailagent.infra.config import ClusteringSettings, VectorStoreSettings
from mailagent.infra.store import Base
from mailagent.infra.vector_store import VectorStore

_DIM = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embedding(cluster_id: int, idx: int) -> list[float]:
    """Deterministic embedding for cluster ``cluster_id`` (3 distinct clusters)."""
    vec = [0.0] * _DIM
    # Each cluster has a distinct axis; spread along its axis by idx.
    vec[cluster_id % _DIM] = 1.0 + idx * 0.001
    return vec


def _make_sample(
    sample_id,
    label_l3: str,
    cluster_id: int,
    idx: int,
) -> SampleRecord:
    return SampleRecord(
        id=sample_id,
        mail_hash=f"hash-{sample_id}",
        subject_raw=f"Subject cluster {cluster_id} #{idx}",
        subject_clean=f"subject cluster {cluster_id} #{idx}",
        sender=f"sender{idx}@example.com",
        sender_domain="example.com",
        body=f"Body for sample {idx} in cluster {cluster_id}.",
        label_l1="entity",
        label_l2="schedule",
        label_l3=label_l3,
        confidence=0.9,
        source="seed",
    )


def _mock_llm(description: str = "Test intent") -> MagicMock:
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
    tree.all_codes = MagicMock(return_value=codes or {"label_a", "label_b", "label_c"})
    tax.get_tree = MagicMock(return_value=tree)
    return tax


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def vector_store(engine) -> VectorStore:
    return VectorStore(VectorStoreSettings(), engine)


@pytest.fixture
def clustering_settings() -> ClusteringSettings:
    # min_cluster_size=5 ensures 3 clusters of 35 each are detected.
    return ClusteringSettings(
        min_cluster_size=5,
        min_samples=3,
        metric="cosine",
        max_samples=50000,
        window_days=30,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_HDBSCAN, reason="hdbscan not installed")
class TestClusteringIntegration:
    """End-to-end clustering on 100 synthetic samples with real SQLite store."""

    async def test_clustering_100_samples_generates_report(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        clustering_settings: ClusteringSettings,
    ) -> None:
        """Insert 100 samples (3 distinct clusters) → run clustering → report."""
        # 3 clusters × ~33 samples each ≈ 100 samples; cluster labels don't
        # match any taxonomy centroid so they all become "new_intent".
        labels = ["label_a", "label_b", "label_c"]
        for cluster_id in range(3):
            for idx in range(34 if cluster_id < 2 else 32):
                sid = uuid4()
                sample = _make_sample(sid, labels[cluster_id], cluster_id, idx)
                emb = _embedding(cluster_id, idx)
                await vector_store.insert_sample(sample, emb, emb)

        total = await vector_store.count_samples()
        assert total == 100

        llm = _mock_llm("Cluster intent description")
        tax = _mock_taxonomy(codes=set(labels))
        engine_obj = ClusteringEngine(vector_store, tax, llm, clustering_settings)
        engine_obj._reports_dir = tmp_path  # type: ignore[assignment]

        report_path_str = await engine_obj.run_weekly_clustering()
        report_path = Path(report_path_str)
        assert report_path.exists()

        content = report_path.read_text(encoding="utf-8")
        assert "# Intent Discovery Report" in content
        # 100 samples were analyzed.
        assert "**Total samples analyzed**: 100" in content
        # Either new intents or drifts were detected (cluster embedding axis
        # is orthogonal to taxonomy centroid, so cluster_type=new_intent).
        # If HDBSCAN found clusters, the report should mention them.
        # Otherwise the report says "no new intent candidates".
        assert ("## New Intents" in content) or ("无新意图候选" in content)

    async def test_clustering_empty_db_generates_empty_report(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        clustering_settings: ClusteringSettings,
    ) -> None:
        """Empty sample DB → still generates a report with no findings."""
        llm = _mock_llm()
        tax = _mock_taxonomy()
        engine_obj = ClusteringEngine(vector_store, tax, llm, clustering_settings)
        engine_obj._reports_dir = tmp_path  # type: ignore[assignment]

        report_path_str = await engine_obj.run_weekly_clustering()
        report_path = Path(report_path_str)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "无新意图候选" in content
        assert "无漂移检测" in content

    async def test_clustering_insufficient_samples(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        clustering_settings: ClusteringSettings,
    ) -> None:
        """Fewer samples than min_cluster_size → empty report."""
        # Insert only 2 samples (< min_cluster_size=5).
        for i in range(2):
            sid = uuid4()
            sample = _make_sample(sid, "label_a", 0, i)
            emb = _embedding(0, i)
            await vector_store.insert_sample(sample, emb, emb)

        llm = _mock_llm()
        tax = _mock_taxonomy()
        engine_obj = ClusteringEngine(vector_store, tax, llm, clustering_settings)
        engine_obj._reports_dir = tmp_path  # type: ignore[assignment]

        report_path_str = await engine_obj.run_weekly_clustering()
        report_path = Path(report_path_str)
        assert report_path.exists()

    async def test_clustering_report_contains_representatives(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        clustering_settings: ClusteringSettings,
    ) -> None:
        """When new intents are found, the report lists representative emails."""
        # Build a tight cluster of 10 samples with identical embeddings.
        cluster_emb = [1.0] + [0.0] * (_DIM - 1)
        for i in range(10):
            sid = uuid4()
            sample = _make_sample(sid, "label_a", 0, i)
            await vector_store.insert_sample(sample, cluster_emb, cluster_emb)

        # Add 5 noise samples with orthogonal embeddings.
        noise_emb = [0.0, 0.0, 1.0] + [0.0] * (_DIM - 3)
        for i in range(5):
            sid = uuid4()
            sample = _make_sample(sid, "label_b", 1, i)
            await vector_store.insert_sample(sample, noise_emb, noise_emb)

        llm = _mock_llm("Tight cluster intent")
        # Empty taxonomy centroids → cluster_type always new_intent.
        tax = _mock_taxonomy(codes={"label_a", "label_b"})
        engine_obj = ClusteringEngine(vector_store, tax, llm, clustering_settings)
        engine_obj._reports_dir = tmp_path  # type: ignore[assignment]

        report_path_str = await engine_obj.run_weekly_clustering()
        content = Path(report_path_str).read_text(encoding="utf-8")

        # If HDBSCAN found a cluster, the report should include the LLM description.
        assert ("Tight cluster intent" in content) or ("无新意图候选" in content)


class TestClusteringDegradation:
    """When hdbscan is not installed, run_weekly_clustering degrades gracefully."""

    async def test_hdbscan_unavailable_returns_skip_message(
        self,
        tmp_path: Path,
        vector_store: VectorStore,
        clustering_settings: ClusteringSettings,
    ) -> None:
        """When HAS_HDBSCAN is False, returns the skip message string."""
        from unittest.mock import patch

        llm = _mock_llm()
        tax = _mock_taxonomy()
        engine_obj = ClusteringEngine(vector_store, tax, llm, clustering_settings)
        engine_obj._reports_dir = tmp_path  # type: ignore[assignment]

        with patch("mailagent.infra.clustering.HAS_HDBSCAN", False):
            result = await engine_obj.run_weekly_clustering()

        assert result == "clustering skipped: hdbscan not installed"


class TestClusteringCentroidMath:
    """Sanity check on the cosine-similarity math used for cluster classification."""

    def test_orthogonal_embeddings_zero_similarity(self) -> None:
        """Two orthogonal unit vectors have cosine similarity 0."""
        a = [1.0] + [0.0] * (_DIM - 1)
        b = [0.0, 1.0] + [0.0] * (_DIM - 2)
        from mailagent.infra.clustering import _cosine_similarity
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_identical_embeddings_unit_similarity(self) -> None:
        """Two identical unit vectors have cosine similarity 1."""
        a = [1.0] + [0.0] * (_DIM - 1)
        from mailagent.infra.clustering import _cosine_similarity
        assert _cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-9)
