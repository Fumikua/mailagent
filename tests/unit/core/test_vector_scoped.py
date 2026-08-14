"""Unit tests for ``VectorStore.knn_search`` ``label_scope`` filtering (P0).

Validates that ``label_scope`` restricts the SQLite brute-force search to
accepted flat samples whose ``label_l1`` is in the scope list, that
``label_scope=None`` preserves backward-compatible global scan, and that the
PostgreSQL path's ``ANY(:scope)`` SQL clause is constructed correctly (via
dialect inspection).
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
from mailagent.infra.store import Base
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
    return VectorStoreSettings(top_k=5, similarity_threshold=0.5)


@pytest.fixture
def vector_store(engine, settings: VectorStoreSettings) -> VectorStore:
    return VectorStore(settings, engine)


def _embedding(seed: float) -> list[float]:
    return [seed + i * 0.01 for i in range(DIM)]


def _make_sample(label: str) -> SampleRecord:
    fingerprint = hashlib.sha256(label.encode()).hexdigest()
    return SampleRecord(
        mail_hash=f"hash-{uuid4()}",
        subject_raw="Test subject",
        subject_clean="test subject",
        sender="ops@example.com",
        sender_domain="example.com",
        body="body",
        label_l1=label,
        confidence=0.9,
        source="seed",
        reviewed=True,
        thread_parsed=True,
        retrieval_fingerprint=fingerprint,
        retrieval_policy_version="test-v1",
        quality=SampleQualityAssessment(
            disposition="accepted",
            fingerprint=fingerprint,
            retrieval_policy_version="test-v1",
        ),
        created_at=datetime.now(timezone.utc) - timedelta(days=0),
    )


async def _seed_multi_label(store: VectorStore) -> None:
    """Seed three labels from the flat example-triage taxonomy."""

    for label in ("entity_report", "schedule", "operation"):
        s = _make_sample(label)
        emb = _embedding(0.1 if label == "entity_report" else 0.9)
        await store.insert_sample(s, emb, emb)


class TestScopedSearch:
    async def test_label_scope_filters_to_single_label(
        self, vector_store: VectorStore
    ) -> None:
        await _seed_multi_label(vector_store)
        # Query near entity_report's embedding; scope to entity_report only
        candidates = await vector_store.knn_search(
            _embedding(0.1), top_k=5, threshold=0.5, label_scope=["entity_report"]
        )
        labels = {c.label for c in candidates}
        assert labels == {"entity_report"}

    async def test_label_scope_none_returns_all_labels(
        self, vector_store: VectorStore
    ) -> None:
        """label_scope=None must be backward compatible (global scan)."""

        await _seed_multi_label(vector_store)
        # Use a low threshold so all labels pass
        candidates = await vector_store.knn_search(
            _embedding(0.1), top_k=5, threshold=0.0, label_scope=None
        )
        labels = {c.label for c in candidates}
        assert labels == {"entity_report", "schedule", "operation"}

    async def test_label_scope_nonexistent_returns_empty(
        self, vector_store: VectorStore
    ) -> None:
        await _seed_multi_label(vector_store)
        candidates = await vector_store.knn_search(
            _embedding(0.1), top_k=5, threshold=0.5, label_scope=["nonexistent_label"]
        )
        assert candidates == []

    async def test_label_scope_multiple_labels(
        self, vector_store: VectorStore
    ) -> None:
        await _seed_multi_label(vector_store)
        # Scope to two labels that share the same embedding seed (0.9)
        candidates = await vector_store.knn_search(
            _embedding(0.9),
            top_k=5,
            threshold=0.5,
            label_scope=["schedule", "operation"],
        )
        labels = {c.label for c in candidates}
        assert labels == {"schedule", "operation"}


class TestPgScopeClause:
    """Validate the PostgreSQL ``ANY(:scope)`` SQL construction without a real PG.

    We mock the dialect to look like PostgreSQL + pgvector and capture the
    generated SQL / params. This verifies the scope clause is emitted when
    ``label_scope`` is set and omitted when None.
    """

    def _make_pg_store(self) -> VectorStore:
        # Build a store but we won't actually execute — we patch internals.
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        store = VectorStore(VectorStoreSettings(), eng)
        return store

    async def test_pg_scope_clause_present_when_scope_set(self) -> None:
        store = self._make_pg_store()
        captured: dict = {}

        class ResultStub:
            def __init__(self, rows):
                self._rows = rows

            def all(self):
                return self._rows

        class SessionStub:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def execute(self, sql, params):
                captured["sql"] = str(sql)
                captured["params"] = params
                return ResultStub([])

        # Force the PG path
        store._dialect_name = "postgresql"
        # Patch HAS_PGVECTOR in the module where knn_search checks it
        import mailagent.infra.vector_store as vs_mod

        original = vs_mod.HAS_PGVECTOR
        vs_mod.HAS_PGVECTOR = True
        original_sessions = store.sessions
        store.sessions = lambda: SessionStub()  # type: ignore[assignment]
        try:
            await store._knn_pg(_embedding(0.1), 5, 0.5, label_scope=["entity_report"])
        finally:
            vs_mod.HAS_PGVECTOR = original
            store.sessions = original_sessions  # type: ignore[assignment]
        assert "ANY(:scope)" in captured["sql"]
        assert captured["params"]["scope"] == ["entity_report"]

    async def test_pg_scope_clause_absent_when_scope_none(self) -> None:
        store = self._make_pg_store()
        captured: dict = {}

        class ResultStub:
            def all(self):
                return []

        class SessionStub:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def execute(self, sql, params):
                captured["sql"] = str(sql)
                captured["params"] = params
                return ResultStub()

        store._dialect_name = "postgresql"
        import mailagent.infra.vector_store as vs_mod

        original = vs_mod.HAS_PGVECTOR
        vs_mod.HAS_PGVECTOR = True
        original_sessions = store.sessions
        store.sessions = lambda: SessionStub()  # type: ignore[assignment]
        try:
            await store._knn_pg(_embedding(0.1), 5, 0.5, label_scope=None)
        finally:
            vs_mod.HAS_PGVECTOR = original
            store.sessions = original_sessions  # type: ignore[assignment]
        assert "ANY(:scope)" not in captured["sql"]
        assert "scope" not in captured["params"]
