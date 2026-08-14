"""End-to-end integration tests for sample archival (Section 18.9).

Inserts a sample created 13 months ago, runs ``archive_old_samples(months=12)``,
and verifies the sample is moved to ``samples_archive`` and deleted from the
active ``samples`` table.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from mailagent.domain.models import SampleRecord
from mailagent.infra.config import VectorStoreSettings
from mailagent.infra.store import Base, SampleArchiveORM
from mailagent.infra.vector_store import VectorStore

_DIM = 8


def _embedding(seed: float) -> list[float]:
    return [seed + i * 0.01 for i in range(_DIM)]


def _make_sample(days_ago: int, mail_hash: str, label_l3: str = "eta_update") -> SampleRecord:
    return SampleRecord(
        mail_hash=mail_hash,
        subject_raw=f"Subject {mail_hash}",
        subject_clean=f"subject {mail_hash}",
        sender="ops@example.com",
        sender_domain="example.com",
        body=f"Body for {mail_hash}",
        label_l1="entity",
        label_l2="schedule",
        label_l3=label_l3,
        confidence=0.9,
        source="seed",
        created_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestArchiveIntegration:
    """archive_old_samples moves old rows to samples_archive and deletes from active."""

    async def test_archive_moves_13_month_old_sample(
        self, vector_store: VectorStore
    ) -> None:
        """A sample created 13 months ago is archived when window=12 months."""
        # Insert a 13-month-old sample (≈ 396 days ago).
        old_sample = _make_sample(days_ago=396, mail_hash="hash-old")
        emb = _embedding(0.1)
        await vector_store.insert_sample(old_sample, emb, emb)

        # Sanity: 1 sample in active table.
        assert await vector_store.count_samples() == 1
        # No rows in archive table.
        async with vector_store.sessions() as session:
            archived_pre = (await session.scalars(select(SampleArchiveORM))).all()
        assert len(archived_pre) == 0

        # Archive with window=12 months.
        moved = await vector_store.archive_old_samples(months=12)
        assert moved == 1

        # Active table is now empty.
        assert await vector_store.count_samples() == 0

        # Archive table contains the moved row with the original fields.
        async with vector_store.sessions() as session:
            archived = (await session.scalars(select(SampleArchiveORM))).all()
        assert len(archived) == 1
        archived_row = archived[0]
        assert archived_row.mail_hash == "hash-old"
        assert archived_row.label_l3 == "eta_update"
        assert archived_row.sender_domain == "example.com"
        # Embeddings preserved.
        assert archived_row.embedding_thread == emb
        assert archived_row.embedding_segment_0 == emb

    async def test_archive_keeps_recent_sample(
        self, vector_store: VectorStore
    ) -> None:
        """A 1-day-old sample is NOT archived when window=12 months."""
        recent_sample = _make_sample(days_ago=1, mail_hash="hash-recent")
        await vector_store.insert_sample(recent_sample, _embedding(0.2), _embedding(0.2))

        moved = await vector_store.archive_old_samples(months=12)
        assert moved == 0
        assert await vector_store.count_samples() == 1

    async def test_archive_mixed_old_and_recent(
        self, vector_store: VectorStore
    ) -> None:
        """Only old samples are archived; recent ones stay in the active table."""
        old1 = _make_sample(days_ago=400, mail_hash="hash-old-1", label_l3="eta_update")
        old2 = _make_sample(days_ago=500, mail_hash="hash-old-2", label_l3="location_plan")
        recent = _make_sample(days_ago=5, mail_hash="hash-recent", label_l3="status_report")

        for s in (old1, old2, recent):
            await vector_store.insert_sample(s, _embedding(0.1), _embedding(0.1))

        assert await vector_store.count_samples() == 3

        moved = await vector_store.archive_old_samples(months=12)
        assert moved == 2
        assert await vector_store.count_samples() == 1

        # Active table contains only the recent sample.
        remaining = await vector_store.get_samples()
        assert len(remaining) == 1
        assert remaining[0].mail_hash == "hash-recent"
        assert remaining[0].label_l3 == "status_report"

        # Archive table contains both old samples.
        async with vector_store.sessions() as session:
            archived = (await session.scalars(select(SampleArchiveORM))).all()
        archived_hashes = {a.mail_hash for a in archived}
        assert archived_hashes == {"hash-old-1", "hash-old-2"}

    async def test_archive_boundary_12_months_exact(
        self, vector_store: VectorStore
    ) -> None:
        """Sample created exactly 365 days ago (12 months ≈ 360 days) is archived."""
        # 365 > 360 (12 * 30) → archived.
        boundary_sample = _make_sample(days_ago=365, mail_hash="hash-boundary")
        await vector_store.insert_sample(boundary_sample, _embedding(0.1), _embedding(0.1))

        moved = await vector_store.archive_old_samples(months=12)
        assert moved == 1
        assert await vector_store.count_samples() == 0

    async def test_archive_idempotent_second_call_no_op(
        self, vector_store: VectorStore
    ) -> None:
        """A second archive call moves 0 samples (no old samples left)."""
        old = _make_sample(days_ago=400, mail_hash="hash-old")
        await vector_store.insert_sample(old, _embedding(0.1), _embedding(0.1))

        first = await vector_store.archive_old_samples(months=12)
        assert first == 1

        second = await vector_store.archive_old_samples(months=12)
        assert second == 0
