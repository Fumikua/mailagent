"""Vector store for Path B similarity search.

Persists labeled email samples with their embeddings and provides kNN search,
stratified sampling for clustering, centroid computation, and archival.

Dialect behavior:
- PostgreSQL + pgvector: native ``<=>`` cosine distance with HNSW index.
- SQLite (dev / tests): brute-force cosine scan over JSON-stored embeddings.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from ..domain.models import PathBCandidate, SampleQualityAssessment, SampleRecord
from .config import VectorStoreSettings
from .store import HAS_PGVECTOR, SampleArchiveORM, SampleORM


class SampleAdmissionError(ValueError):
    """Raised when a sample cannot safely enter the active sample library."""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Brute-force cosine similarity used by the SQLite fallback path."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _embedding_to_pg_str(vec: list[float]) -> str:
    """Convert a Python list[float] to pgvector text literal: ``[0.1,0.2,...]``."""
    return "[" + ",".join(repr(x) for x in vec) + "]"


def _orm_to_domain(record: SampleORM) -> SampleRecord:
    """Convert a SampleORM row to the domain SampleRecord model."""
    quality = None
    if (
        record.quality_disposition is not None
        and record.retrieval_fingerprint is not None
        and record.retrieval_policy_version is not None
    ):
        quality = SampleQualityAssessment.model_validate(
            {
                "disposition": record.quality_disposition,
                "reasons": record.quality_reasons or [],
                "fingerprint": record.retrieval_fingerprint,
                "taxonomy_schema_version": record.taxonomy_schema_version or "flat-v1",
                "retrieval_policy_version": record.retrieval_policy_version,
            }
        )
    return SampleRecord(
        id=UUID(record.id),
        mail_hash=record.mail_hash,
        subject_raw=record.subject_raw,
        subject_clean=record.subject_clean,
        sender=record.sender,
        sender_domain=record.sender_domain,
        body=record.body,
        label_l1=record.label_l1,
        label_l2=record.label_l2,
        label_l3=record.label_l3,
        confidence=record.confidence,
        source=record.source,  # type: ignore[arg-type]
        reviewed=record.reviewed,
        thread_parsed=record.thread_parsed,
        created_at=record.created_at,  # type: ignore[arg-type]
        batch_confirmed_at=record.batch_confirmed_at,
        taxonomy_schema_version=record.taxonomy_schema_version or "legacy",
        retrieval_document=record.retrieval_document,
        retrieval_fingerprint=record.retrieval_fingerprint,
        retrieval_policy_version=record.retrieval_policy_version,
        quality=quality,
        review_override_reason=record.review_override_reason,
    )


class VectorStore:
    """Manages labeled samples and similarity search for Path B."""

    def __init__(self, settings: VectorStoreSettings, engine: AsyncEngine) -> None:
        self.settings = settings
        self.engine = engine
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._dialect_name = engine.dialect.name

        # Fail-fast: if PostgreSQL is configured but pgvector is unavailable,
        # refuse to construct. Silently falling back to brute-force would hide
        # a critical production misconfiguration (kNN latency degrades from
        # <50ms to minutes once samples grow). Operators must install
        # `pgvector` (pip) and `CREATE EXTENSION vector` in the database.
        if self._dialect_name == "postgresql" and not HAS_PGVECTOR:
            raise RuntimeError(
                "PostgreSQL dialect requires the pgvector extension. "
                "Install with `pip install pgvector` and run "
                "`CREATE EXTENSION IF NOT EXISTS vector` on the target database. "
                "For local dev, use a SQLite database URL instead."
            )

    # ------------------------------------------------------------------
    # Insert / delete / update
    # ------------------------------------------------------------------

    async def _validate_admission(self, sample: SampleRecord) -> None:
        """Reject unsafe flat samples before a database write is attempted."""

        if sample.taxonomy_schema_version == "flat-v1":
            if sample.quality is None or sample.quality.disposition != "accepted":
                raise SampleAdmissionError("quality_not_accepted")
            if sample.label_l2 is not None or sample.label_l3 is not None:
                raise SampleAdmissionError("non_flat_taxonomy")
        async with self.sessions() as session:
            existing_mail_hash = await session.scalar(
                select(SampleORM.id).where(SampleORM.mail_hash == sample.mail_hash)
            )
            if existing_mail_hash is not None:
                raise SampleAdmissionError("duplicate_mail_hash")
            if sample.retrieval_fingerprint is not None:
                existing_fingerprint = await session.scalar(
                    select(SampleORM.id).where(
                        SampleORM.retrieval_fingerprint == sample.retrieval_fingerprint
                    )
                )
                if existing_fingerprint is not None:
                    raise SampleAdmissionError("duplicate_retrieval_fingerprint")

    async def insert_sample(
        self,
        sample: SampleRecord,
        embedding_thread: list[float],
        embedding_segment_0: list[float],
    ) -> None:
        """Insert one labeled sample with its thread and segment-0 embeddings."""
        await self._validate_admission(sample)
        record = SampleORM(
            id=str(sample.id),
            mail_hash=sample.mail_hash,
            subject_raw=sample.subject_raw,
            subject_clean=sample.subject_clean,
            sender=sample.sender,
            sender_domain=sample.sender_domain,
            body=sample.body,
            label_l1=sample.label_l1,
            label_l2=sample.label_l2,
            label_l3=sample.label_l3,
            confidence=sample.confidence,
            source=sample.source,
            reviewed=sample.reviewed,
            thread_parsed=sample.thread_parsed,
            embedding_thread=embedding_thread,
            embedding_segment_0=embedding_segment_0,
            created_at=sample.created_at,
            batch_confirmed_at=sample.batch_confirmed_at,
            taxonomy_schema_version=sample.taxonomy_schema_version,
            retrieval_document=sample.retrieval_document,
            retrieval_fingerprint=sample.retrieval_fingerprint,
            retrieval_policy_version=sample.retrieval_policy_version,
            quality_disposition=sample.quality.disposition if sample.quality else None,
            quality_reasons=sample.quality.reasons if sample.quality else None,
            review_override_reason=sample.review_override_reason,
        )
        async with self.sessions() as session:
            session.add(record)
            await session.commit()

    async def delete_sample(self, id: UUID) -> None:
        """Delete a sample by id (no-op if the id does not exist)."""
        async with self.sessions() as session:
            await session.execute(delete(SampleORM).where(SampleORM.id == str(id)))
            await session.commit()

    async def update_sample_label(
        self,
        id: UUID,
        label: str,
        source: str = "human_fix",
        confidence: float = 1.0,
    ) -> None:
        """Update a sample's flat label, source, confidence, and review state."""
        async with self.sessions() as session:
            record = await session.get(SampleORM, str(id))
            if record is None:
                return
            record.label_l1 = label
            record.label_l2 = None
            record.label_l3 = None
            record.taxonomy_schema_version = "flat-v1"
            record.source = source
            record.confidence = confidence
            record.reviewed = True
            await session.commit()

    async def backfill_samples_label(
        self,
        new_code: str,
        sample_ids: list[UUID],
    ) -> None:
        """Idempotently rewrite the flat label of the given samples to ``new_code``."""
        if not sample_ids:
            return
        ids = [str(sid) for sid in sample_ids]
        async with self.sessions() as session:
            stmt = select(SampleORM).where(SampleORM.id.in_(ids))
            result = await session.scalars(stmt)
            for rec in result.all():
                rec.label_l1 = new_code
                rec.label_l2 = None
                rec.label_l3 = None
                rec.taxonomy_schema_version = "flat-v1"
            await session.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def knn_search(
        self,
        query: list[float],
        top_k: int = 5,
        threshold: float = 0.85,
        label_scope: list[str] | None = None,
    ) -> list[PathBCandidate]:
        """KNN search over accepted flat samples, aggregated by ``label_l1``.

        Returns up to ``top_k`` ``PathBCandidate`` entries, sorted by
        ``max_similarity`` descending. Each candidate aggregates the neighbors
        whose similarity >= ``threshold`` for a given label.

        When ``label_scope`` is provided, only samples whose ``label_l1`` is in
        the scope list are considered (pushed down to SQL ``WHERE`` clause).
        When ``label_scope`` is None, the search behaves identically to the
        pre-change implementation (global scan).
        """
        if self._dialect_name == "postgresql" and HAS_PGVECTOR:
            rows = await self._knn_pg(query, top_k, threshold, label_scope)
        else:
            rows = await self._knn_bruteforce(query, top_k, threshold, label_scope)

        # Group by label_l1 and aggregate max / count / mean similarity.
        groups: dict[str, list[float]] = {}
        for label, sim in rows:
            groups.setdefault(label, []).append(sim)

        candidates: list[PathBCandidate] = []
        for label, sims in groups.items():
            if not sims:
                continue
            max_sim = max(sims)
            mean_sim = sum(sims) / len(sims)
            # Clamp to [0, 1] to guard against floating-point drift.
            max_sim = min(max(max_sim, 0.0), 1.0)
            mean_sim = min(max(mean_sim, 0.0), 1.0)
            candidates.append(
                PathBCandidate(
                    label=label,
                    max_similarity=max_sim,
                    count=len(sims),
                    mean_similarity=mean_sim,
                    confidence=max_sim,
                )
            )
        candidates.sort(key=lambda c: c.max_similarity, reverse=True)
        return candidates[:top_k]

    async def _knn_bruteforce(
        self,
        query: list[float],
        top_k: int,
        threshold: float,
        label_scope: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """SQLite fallback: scan all embeddings and compute cosine similarity in Python."""
        async with self.sessions() as session:
            stmt = select(SampleORM.label_l1, SampleORM.embedding_thread).where(
                SampleORM.embedding_thread.is_not(None),
                SampleORM.label_l1.is_not(None),
                SampleORM.label_l2.is_(None),
                SampleORM.label_l3.is_(None),
                SampleORM.taxonomy_schema_version == "flat-v1",
                SampleORM.quality_disposition == "accepted",
                SampleORM.reviewed.is_(True),
            )
            if label_scope:
                stmt = stmt.where(SampleORM.label_l1.in_(label_scope))
            result = await session.execute(stmt)
            rows = result.all()
        scored: list[tuple[str, float]] = []
        for label, emb in rows:
            if emb is None:
                continue
            sim = _cosine_similarity(query, emb)
            if sim >= threshold:
                scored.append((label, sim))
        # Over-fetch to leave room for per-label aggregation.
        scored.sort(key=lambda x: x[1], reverse=True)
        fetch = max(top_k * 5, top_k)
        return scored[:fetch]

    async def _knn_pg(
        self,
        query: list[float],
        top_k: int,
        threshold: float,
        label_scope: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """PostgreSQL path: use pgvector cosine distance operator ``<=>`` with HNSW index."""
        q_str = _embedding_to_pg_str(query)
        scope_clause = "AND label_l1 = ANY(:scope) " if label_scope else ""
        params: dict[str, Any] = {
            "q": q_str,
            "threshold": threshold,
            "limit": max(top_k * 5, top_k),
        }
        if label_scope:
            params["scope"] = list(label_scope)
        sql = text(
            "SELECT label_l1, 1.0 - (embedding_thread <=> CAST(:q AS vector)) AS sim "
            "FROM samples "
            "WHERE embedding_thread IS NOT NULL "
            "AND label_l1 IS NOT NULL "
            "AND label_l2 IS NULL "
            "AND label_l3 IS NULL "
            "AND taxonomy_schema_version = 'flat-v1' "
            "AND quality_disposition = 'accepted' "
            "AND reviewed = TRUE "
            f"{scope_clause}"
            "AND 1.0 - (embedding_thread <=> CAST(:q AS vector)) >= :threshold "
            "ORDER BY embedding_thread <=> CAST(:q AS vector) "
            "LIMIT :limit"
        )
        async with self.sessions() as session:
            result = await session.execute(sql, params)
            rows = result.all()
        return [(label, float(sim)) for label, sim in rows]

    # ------------------------------------------------------------------
    # Query / count
    # ------------------------------------------------------------------

    async def get_samples(
        self,
        label: str | None = None,
        source: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> list[SampleRecord]:
        """Paginated sample listing with optional flat label_l1 / source filters."""
        async with self.sessions() as session:
            stmt = select(SampleORM)
            if label is not None:
                stmt = stmt.where(SampleORM.label_l1 == label)
            if source is not None:
                stmt = stmt.where(SampleORM.source == source)
            offset = (page - 1) * page_size
            stmt = stmt.order_by(SampleORM.created_at.desc()).offset(offset).limit(page_size)
            result = await session.scalars(stmt)
            records = result.all()
        return [_orm_to_domain(r) for r in records]

    async def get_sample(self, id: UUID) -> SampleRecord | None:
        """Return a single sample by id, or None if not found."""
        async with self.sessions() as session:
            record = await session.get(SampleORM, str(id))
            if record is None:
                return None
            return _orm_to_domain(record)

    async def count_samples(self, days: int | None = None) -> int:
        """Count samples; optionally restricted to the last ``days`` days."""
        async with self.sessions() as session:
            stmt = select(func.count()).select_from(SampleORM)
            if days is not None:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                stmt = stmt.where(SampleORM.created_at >= cutoff)
            result = await session.scalar(stmt)
            return int(result or 0)

    async def get_quality_stats(self) -> dict[str, Any]:
        """Aggregate quality / taxonomy / policy-version distribution.

        Used by ``mailagent samples stats`` and the re-embedding dry-run
        report. Returns counts keyed by disposition, taxonomy schema version,
        retrieval policy version, label_l1, and duplicate fingerprint.
        """
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(
                        SampleORM.quality_disposition,
                        SampleORM.taxonomy_schema_version,
                        SampleORM.retrieval_policy_version,
                        SampleORM.label_l1,
                        func.count().label("n"),
                    ).group_by(
                        SampleORM.quality_disposition,
                        SampleORM.taxonomy_schema_version,
                        SampleORM.retrieval_policy_version,
                        SampleORM.label_l1,
                    )
                )
            ).all()

        by_disposition: dict[str, int] = {}
        by_taxonomy: dict[str, int] = {}
        by_policy: dict[str, int] = {}
        by_label: dict[str, int] = {}
        duplicate_fingerprints = 0
        for (
            disposition,
            taxonomy_version,
            policy_version,
            label_l1,
            n,
        ) in rows:
            disp = disposition or "unassessed"
            by_disposition[disp] = by_disposition.get(disp, 0) + n
            if taxonomy_version:
                by_taxonomy[taxonomy_version] = by_taxonomy.get(taxonomy_version, 0) + n
            if policy_version:
                by_policy[policy_version] = by_policy.get(policy_version, 0) + n
            if label_l1:
                by_label[label_l1] = by_label.get(label_l1, 0) + n

        # Duplicate fingerprint count (samples sharing a fingerprint with
        # at least one other sample). Computed separately because the
        # admission gate normally prevents duplicates, but legacy rows or
        # manual SQL edits can still introduce them.
        async with self.sessions() as session:
            dup_rows = (
                await session.execute(
                    select(
                        SampleORM.retrieval_fingerprint, func.count().label("n")
                    )
                    .where(SampleORM.retrieval_fingerprint.is_not(None))
                    .group_by(SampleORM.retrieval_fingerprint)
                    .having(func.count() > 1)
                )
            ).all()
            for _fp, n in dup_rows:
                duplicate_fingerprints += n

        return {
            "by_disposition": by_disposition,
            "by_taxonomy_schema": by_taxonomy,
            "by_retrieval_policy": by_policy,
            "by_label_l1": by_label,
            "duplicate_fingerprint_rows": duplicate_fingerprints,
        }

    async def get_reembed_candidates(
        self,
        target_policy_version: str,
        batch_size: int = 100,
    ) -> list[UUID]:
        """Return sample ids whose retrieval_policy_version != target.

        Used by the resumable re-embedding workflow: each batch marks
        completed samples with the target policy version, so a subsequent
        run after a crash skips them automatically.
        """
        async with self.sessions() as session:
            stmt = (
                select(SampleORM.id)
                .where(
                    SampleORM.taxonomy_schema_version == "flat-v1",
                    SampleORM.quality_disposition == "accepted",
                    SampleORM.reviewed.is_(True),
                    SampleORM.retrieval_policy_version != target_policy_version,
                )
                .order_by(SampleORM.created_at.asc())
                .limit(batch_size)
            )
            result = await session.scalars(stmt)
            return [UUID(sid) for sid in result.all()]

    async def mark_reembed_complete(
        self,
        sample_id: UUID,
        embedding_thread: list[float],
        embedding_segment_0: list[float],
        retrieval_policy_version: str,
    ) -> None:
        """Update a sample's embeddings and policy version after re-embedding."""
        async with self.sessions() as session:
            record = await session.get(SampleORM, str(sample_id))
            if record is None:
                return
            record.embedding_thread = embedding_thread
            record.embedding_segment_0 = embedding_segment_0
            record.retrieval_policy_version = retrieval_policy_version
            await session.commit()

    # ------------------------------------------------------------------
    # Embedding extraction (for clustering / centroid computation)
    # ------------------------------------------------------------------

    async def get_embeddings(
        self,
        days: int | None = None,
    ) -> list[tuple[UUID, list[float], str]]:
        """Return ``(id, embedding_thread, label_l1)`` for active flat samples."""
        async with self.sessions() as session:
            stmt = (
                select(SampleORM.id, SampleORM.embedding_thread, SampleORM.label_l1)
                .where(
                    SampleORM.embedding_thread.is_not(None),
                    SampleORM.label_l1.is_not(None),
                    SampleORM.label_l2.is_(None),
                    SampleORM.label_l3.is_(None),
                    SampleORM.taxonomy_schema_version == "flat-v1",
                    SampleORM.quality_disposition == "accepted",
                    SampleORM.reviewed.is_(True),
                )
            )
            if days is not None:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                stmt = stmt.where(SampleORM.created_at >= cutoff)
            result = await session.execute(stmt)
            rows = result.all()
        return [
            (UUID(sample_id), emb, label)
            for sample_id, emb, label in rows
            if emb is not None
        ]

    async def stratified_sample(
        self,
        days: int,
        max_per_label: int,
    ) -> list[tuple[UUID, list[float], str]]:
        """Return at most ``max_per_label`` embeddings per label within the time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with self.sessions() as session:
            stmt = (
                select(SampleORM.id, SampleORM.embedding_thread, SampleORM.label_l1)
                .where(
                    SampleORM.embedding_thread.is_not(None),
                    SampleORM.label_l1.is_not(None),
                    SampleORM.label_l2.is_(None),
                    SampleORM.label_l3.is_(None),
                    SampleORM.taxonomy_schema_version == "flat-v1",
                    SampleORM.quality_disposition == "accepted",
                    SampleORM.reviewed.is_(True),
                    SampleORM.created_at >= cutoff,
                )
                .order_by(SampleORM.created_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.all()
        per_label: dict[str, list[tuple[UUID, list[float], str]]] = {}
        for sample_id, emb, label in rows:
            if emb is None or label is None:
                continue
            per_label.setdefault(label, []).append((UUID(sample_id), emb, label))
        out: list[tuple[UUID, list[float], str]] = []
        for items in per_label.values():
            out.extend(items[:max_per_label])
        return out

    async def get_centroids(self) -> dict[str, list[float]]:
        """Compute the mean embedding (centroid) per active flat label_l1."""
        async with self.sessions() as session:
            stmt = select(SampleORM.label_l1, SampleORM.embedding_thread).where(
                SampleORM.embedding_thread.is_not(None),
                SampleORM.label_l1.is_not(None),
                SampleORM.label_l2.is_(None),
                SampleORM.label_l3.is_(None),
                SampleORM.taxonomy_schema_version == "flat-v1",
                SampleORM.quality_disposition == "accepted",
                SampleORM.reviewed.is_(True),
            )
            result = await session.execute(stmt)
            rows = result.all()
        per_label: dict[str, list[list[float]]] = {}
        for label, emb in rows:
            if emb is None or label is None:
                continue
            per_label.setdefault(label, []).append(emb)
        centroids: dict[str, list[float]] = {}
        for label, embs in per_label.items():
            dim = len(embs[0])
            centroid = [0.0] * dim
            for emb in embs:
                for i, value in enumerate(emb):
                    centroid[i] += value
            n = len(embs)
            centroids[label] = [value / n for value in centroid]
        return centroids

    # ------------------------------------------------------------------
    # Archival
    # ------------------------------------------------------------------

    async def archive_old_samples(self, months: int = 12) -> int:
        """Move samples older than ``months`` to ``samples_archive`` and delete from active table."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months)
        async with self.sessions() as session:
            stmt = select(SampleORM).where(SampleORM.created_at < cutoff)
            result = await session.scalars(stmt)
            old_records = result.all()
            count = 0
            for rec in old_records:
                archive = SampleArchiveORM(
                    id=rec.id,
                    mail_hash=rec.mail_hash,
                    subject_raw=rec.subject_raw,
                    subject_clean=rec.subject_clean,
                    sender=rec.sender,
                    sender_domain=rec.sender_domain,
                    body=rec.body,
                    label_l1=rec.label_l1,
                    label_l2=rec.label_l2,
                    label_l3=rec.label_l3,
                    confidence=rec.confidence,
                    source=rec.source,
                    reviewed=rec.reviewed,
                    thread_parsed=rec.thread_parsed,
                    embedding_thread=rec.embedding_thread,
                    embedding_segment_0=rec.embedding_segment_0,
                    created_at=rec.created_at,
                    batch_confirmed_at=rec.batch_confirmed_at,
                    taxonomy_schema_version=rec.taxonomy_schema_version,
                    retrieval_document=rec.retrieval_document,
                    retrieval_fingerprint=rec.retrieval_fingerprint,
                    retrieval_policy_version=rec.retrieval_policy_version,
                    quality_disposition=rec.quality_disposition,
                    quality_reasons=rec.quality_reasons,
                    review_override_reason=rec.review_override_reason,
                )
                session.add(archive)
                await session.delete(rec)
                count += 1
            if count:
                await session.commit()
            return count
