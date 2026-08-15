from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
    select,
    text,
    update,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeEngine

from ..domain.models import (
    ClassificationFeedback,
    ClassificationFeedbackErrorReason,
    ClassificationResponse,
    ClassificationVersions,
    FusionMeta,
    RunResponse,
    SkillDefinition,
    SkillVersion,
)

if TYPE_CHECKING:
    from ..domain.models import RunStatus

# Conditional pgvector import — dev uses SQLite (JSON columns); prod may use
# pgvector Vector type for native cosine distance and HNSW indexing.
try:
    from pgvector.sqlalchemy import Vector  # type: ignore[import-not-found]

    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None  # type: ignore[assignment, misc]

EMBEDDING_DIMENSION = 4096


class Embedding(TypeDecorator):
    """Cross-dialect embedding column type.

    Uses pgvector ``Vector(dim)`` on PostgreSQL (when pgvector is installed);
    falls back to ``JSON`` (list[float]) on SQLite for dev and tests.
    """

    impl = JSON
    cache_ok = True

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        super().__init__()
        self.dimension = dimension

    def load_dialect_impl(self, dialect: Any) -> TypeEngine[Any]:
        if dialect.name == "postgresql" and HAS_PGVECTOR and Vector is not None:
            return dialect.type_descriptor(Vector(self.dimension))
        return dialect.type_descriptor(JSON())


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "processing_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    classification: Mapped[str | None] = mapped_column(Text, nullable=True)
    calibration_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    # fusion_meta 单独存储（便于 SQL 查询融合策略审计），同时也在 classification JSON 内嵌一份
    fusion_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[object] = mapped_column(DateTime(timezone=True))


class JobOutboxRecord(Base):
    """Durable intent to enqueue a background job.

    The row is committed in the same transaction as the run state change, so a
    Redis outage cannot leave a run permanently pending without a retryable job.
    """

    __tablename__ = "job_outbox"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    dispatched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


@dataclass(frozen=True)
class PendingJob:
    id: UUID
    job_name: str
    payload: dict[str, Any]
    attempts: int


class ClassificationFeedbackRecord(Base):
    __tablename__ = "classification_feedback"
    __table_args__ = (
        UniqueConstraint("run_id", "revision", name="uq_feedback_run_revision"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("processing_runs.id"),
        nullable=False,
        index=True,
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    final_labels: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    error_reasons: Mapped[list[ClassificationFeedbackErrorReason]] = mapped_column(
        JSON,
        nullable=False,
    )
    reviewer_id: Mapped[str] = mapped_column(String(255), nullable=False)
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    versions: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    eligible_for_sample_proposal: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )


class SkillRecord(Base):
    __tablename__ = "skill_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    skill_id: Mapped[str] = mapped_column(String(36), index=True)
    version: Mapped[int]
    published: Mapped[bool] = mapped_column(default=False)
    payload: Mapped[dict] = mapped_column(JSON)


class SampleORM(Base):
    """ORM model for labeled email samples stored for vector similarity search.

    Corresponds 1:1 to the domain ``SampleRecord``. Embeddings are stored as
    pgvector ``Vector`` on PostgreSQL (with HNSW index for cosine search) or as
    JSON ``list[float]`` on SQLite for dev and tests.
    """

    __tablename__ = "samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mail_hash: Mapped[str] = mapped_column(String(64), index=True)
    subject_raw: Mapped[str] = mapped_column(Text)
    subject_clean: Mapped[str] = mapped_column(Text)
    sender: Mapped[str] = mapped_column(Text)
    sender_domain: Mapped[str] = mapped_column(String(255), index=True)
    body: Mapped[str] = mapped_column(Text)
    label_l1: Mapped[str] = mapped_column(String(64), index=True)
    label_l2: Mapped[str | None] = mapped_column(String(64), nullable=True)
    label_l3: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), index=True)
    reviewed: Mapped[bool] = mapped_column(default=False)
    thread_parsed: Mapped[bool] = mapped_column(default=True)
    embedding_thread: Mapped[list[float] | None] = mapped_column(
        Embedding(), nullable=True
    )
    embedding_segment_0: Mapped[list[float] | None] = mapped_column(
        Embedding(), nullable=True
    )
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), index=True)
    batch_confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    taxonomy_schema_version: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    retrieval_document: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retrieval_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    retrieval_policy_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    quality_disposition: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )
    quality_reasons: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    review_override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class SampleArchiveORM(Base):
    """Archive table for old samples — same schema as ``SampleORM`` but no HNSW index.

    ``archive_old_samples()`` moves rows past the retention window here to keep
    the active ``samples`` table small for fast KNN search.
    """

    __tablename__ = "samples_archive"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mail_hash: Mapped[str] = mapped_column(String(64))
    subject_raw: Mapped[str] = mapped_column(Text)
    subject_clean: Mapped[str] = mapped_column(Text)
    sender: Mapped[str] = mapped_column(Text)
    sender_domain: Mapped[str] = mapped_column(String(255))
    body: Mapped[str] = mapped_column(Text)
    label_l1: Mapped[str] = mapped_column(String(64))
    label_l2: Mapped[str | None] = mapped_column(String(64), nullable=True)
    label_l3: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))
    reviewed: Mapped[bool] = mapped_column(default=False)
    thread_parsed: Mapped[bool] = mapped_column(default=True)
    embedding_thread: Mapped[list[float] | None] = mapped_column(
        Embedding(), nullable=True
    )
    embedding_segment_0: Mapped[list[float] | None] = mapped_column(
        Embedding(), nullable=True
    )
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True))
    batch_confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    taxonomy_schema_version: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    retrieval_document: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retrieval_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieval_policy_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    quality_disposition: Mapped[str | None] = mapped_column(String(16), nullable=True)
    quality_reasons: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    review_override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class SqlStore:
    def __init__(self, database_url: str) -> None:
        if database_url.startswith("sqlite"):
            Path("data").mkdir(exist_ok=True)
        self.engine = create_async_engine(database_url)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def save_run(self, run: RunResponse) -> None:
        payload = run.model_dump(mode="json")
        classification_json = (
            run.classification.model_dump_json() if run.classification else None
        )
        # 注：calibration_log 嵌入 classification JSON 内（ClassificationResponse.calibration_log）
        async with self.sessions() as session:
            record = await session.get(RunRecord, str(run.id))
            if record is None:
                session.add(
                    RunRecord(
                        id=str(run.id),
                        status=run.status.value,
                        payload=payload,
                        classification=classification_json,
                        created_at=run.created_at,
                        updated_at=run.updated_at,
                    )
                )
            else:
                record.status = run.status.value
                record.payload = payload
                if classification_json is not None:
                    record.classification = classification_json
                record.updated_at = run.updated_at
            await session.commit()

    @staticmethod
    def _new_outbox_record(job_name: str, payload: dict[str, Any]) -> JobOutboxRecord:
        return JobOutboxRecord(
            id=str(uuid4()),
            job_name=job_name,
            payload=payload,
            created_at=datetime.now(timezone.utc),
            dispatched_at=None,
            attempts=0,
            last_error=None,
        )

    async def save_run_with_outbox(
        self,
        run: RunResponse,
        *,
        job_name: str,
        job_payload: dict[str, Any],
    ) -> PendingJob:
        """Atomically persist a new run and its enqueue intent."""

        classification_json = (
            run.classification.model_dump_json() if run.classification else None
        )
        outbox = self._new_outbox_record(job_name, job_payload)
        async with self.sessions() as session:
            session.add(
                RunRecord(
                    id=str(run.id),
                    status=run.status.value,
                    payload=run.model_dump(mode="json"),
                    classification=classification_json,
                    created_at=run.created_at,
                    updated_at=run.updated_at,
                )
            )
            session.add(outbox)
            await session.commit()
        return PendingJob(
            UUID(outbox.id), outbox.job_name, outbox.payload, outbox.attempts
        )

    async def list_pending_outbox(self, *, limit: int = 100) -> list[PendingJob]:
        async with self.sessions() as session:
            records = (
                await session.scalars(
                    select(JobOutboxRecord)
                    .where(JobOutboxRecord.dispatched_at.is_(None))
                    .order_by(JobOutboxRecord.created_at.asc())
                    .limit(limit)
                )
            ).all()
            return [
                PendingJob(
                    UUID(record.id), record.job_name, record.payload, record.attempts
                )
                for record in records
            ]

    async def mark_outbox_dispatched(self, outbox_id: UUID) -> None:
        async with self.sessions() as session:
            await session.execute(
                update(JobOutboxRecord)
                .where(JobOutboxRecord.id == str(outbox_id))
                .values(
                    dispatched_at=datetime.now(timezone.utc),
                    attempts=JobOutboxRecord.attempts + 1,
                    last_error=None,
                )
            )
            await session.commit()

    async def record_outbox_failure(self, outbox_id: UUID, error: str) -> None:
        async with self.sessions() as session:
            await session.execute(
                update(JobOutboxRecord)
                .where(JobOutboxRecord.id == str(outbox_id))
                .values(
                    attempts=JobOutboxRecord.attempts + 1,
                    last_error=error[:4000],
                )
            )
            await session.commit()

    async def get_run(self, run_id: UUID) -> RunResponse | None:
        async with self.sessions() as session:
            record = await session.get(RunRecord, str(run_id))
            if not record:
                return None
            run = RunResponse.model_validate(record.payload)
            # status 以 DB 为准（Worker 更新）
            from ..domain.models import RunStatus

            try:
                run = run.model_copy(update={"status": RunStatus(record.status)})
            except ValueError:
                pass  # 未知 status 保留原值
            # 加载 classification（如有）
            if record.classification:
                try:
                    run = run.model_copy(
                        update={
                            "classification": ClassificationResponse.model_validate_json(
                                record.classification
                            )
                        }
                    )
                except Exception:
                    pass  # 解析失败保留 None
            return run

    @staticmethod
    def _feedback_from_record(
        record: ClassificationFeedbackRecord,
    ) -> ClassificationFeedback:
        reviewed_at = record.reviewed_at
        if reviewed_at.tzinfo is None:
            # SQLite discards timezone offsets even for DateTime(timezone=True).
            # Application-written timestamps are UTC, so restore that contract
            # explicitly when reconstructing the domain model.
            reviewed_at = reviewed_at.replace(tzinfo=timezone.utc)
        return ClassificationFeedback(
            id=UUID(record.id),
            run_id=UUID(record.run_id),
            revision=record.revision,
            predicted_labels=record.predicted_labels,
            final_labels=record.final_labels,
            error_reasons=record.error_reasons,
            reviewer_id=record.reviewer_id,
            reviewed_at=reviewed_at,
            versions=(
                ClassificationVersions.model_validate(record.versions)
                if record.versions is not None
                else None
            ),
            eligible_for_sample_proposal=record.eligible_for_sample_proposal,
        )

    async def append_classification_feedback(
        self,
        *,
        run_id: UUID,
        predicted_labels: list[str],
        final_labels: list[str],
        error_reasons: list[ClassificationFeedbackErrorReason],
        reviewer_id: str,
        versions: ClassificationVersions | None,
    ) -> ClassificationFeedback:
        """Append one immutable correction using the next revision for a run."""

        async with self.sessions() as session:
            try:
                if self.engine.dialect.name == "sqlite":
                    # SQLite ignores SELECT FOR UPDATE. An immediate transaction
                    # takes its database write lock before revision allocation,
                    # deterministically serializing concurrent appenders.
                    await session.execute(text("BEGIN IMMEDIATE"))

                run_record = await session.scalar(
                    select(RunRecord)
                    .where(RunRecord.id == str(run_id))
                    .with_for_update()
                )
                if run_record is None:
                    raise ValueError("run not found")

                latest_revision = await session.scalar(
                    select(func.max(ClassificationFeedbackRecord.revision)).where(
                        ClassificationFeedbackRecord.run_id == str(run_id)
                    )
                )
                record = ClassificationFeedbackRecord(
                    id=str(uuid4()),
                    run_id=str(run_id),
                    revision=(latest_revision or 0) + 1,
                    predicted_labels=list(predicted_labels),
                    final_labels=list(final_labels),
                    error_reasons=list(error_reasons),
                    reviewer_id=reviewer_id,
                    reviewed_at=datetime.now(timezone.utc),
                    versions=versions.model_dump(mode="json") if versions else None,
                    # Feedback never enters the sample proposal path implicitly.
                    # A future trusted promotion workflow may update this flag.
                    eligible_for_sample_proposal=False,
                )
                session.add(record)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            return self._feedback_from_record(record)

    async def list_classification_feedback(
        self,
        run_id: UUID,
    ) -> list[ClassificationFeedback]:
        async with self.sessions() as session:
            records = (
                await session.scalars(
                    select(ClassificationFeedbackRecord)
                    .where(ClassificationFeedbackRecord.run_id == str(run_id))
                    .order_by(ClassificationFeedbackRecord.revision.asc())
                )
            ).all()
            return [self._feedback_from_record(record) for record in records]

    async def update_run_status(self, run_id: UUID, status: "RunStatus") -> None:
        """Worker 调用：更新 run 状态（PENDING → PROCESSING → COMPLETED/FAILED）"""

        async with self.sessions() as session:
            record = await session.get(RunRecord, str(run_id))
            if record is None:
                return
            record.status = status.value
            record.updated_at = datetime.now(timezone.utc)
            await session.commit()

    async def transition_run_status(
        self,
        run_id: UUID,
        *,
        expected: set["RunStatus"],
        target: "RunStatus",
    ) -> bool:
        """Compare-and-set a run status, returning whether this caller won."""

        async with self.sessions() as session:
            result = await session.execute(
                update(RunRecord)
                .where(
                    RunRecord.id == str(run_id),
                    RunRecord.status.in_([status.value for status in expected]),
                )
                .values(status=target.value, updated_at=datetime.now(timezone.utc))
            )
            await session.commit()
            return getattr(result, "rowcount", 0) == 1

    async def replace_run_if_status(
        self,
        run: RunResponse,
        *,
        expected: set["RunStatus"],
    ) -> bool:
        """Persist a full run only when its current DB status is expected."""

        classification_json = (
            run.classification.model_dump_json() if run.classification else None
        )
        values: dict[str, Any] = {
            "status": run.status.value,
            "payload": run.model_dump(mode="json"),
            "updated_at": run.updated_at,
        }
        if classification_json is not None:
            values["classification"] = classification_json
        async with self.sessions() as session:
            result = await session.execute(
                update(RunRecord)
                .where(
                    RunRecord.id == str(run.id),
                    RunRecord.status.in_([status.value for status in expected]),
                )
                .values(**values)
            )
            await session.commit()
            return getattr(result, "rowcount", 0) == 1

    async def replace_run_with_outbox_if_status(
        self,
        run: RunResponse,
        *,
        expected: set["RunStatus"],
        job_name: str,
        job_payload: dict[str, Any],
    ) -> PendingJob | None:
        """CAS a run and create its enqueue intent in one transaction."""

        outbox = self._new_outbox_record(job_name, job_payload)
        async with self.sessions() as session:
            result = await session.execute(
                update(RunRecord)
                .where(
                    RunRecord.id == str(run.id),
                    RunRecord.status.in_([status.value for status in expected]),
                )
                .values(
                    status=run.status.value,
                    payload=run.model_dump(mode="json"),
                    updated_at=run.updated_at,
                )
            )
            if getattr(result, "rowcount", 0) != 1:
                await session.rollback()
                return None
            session.add(outbox)
            await session.commit()
        return PendingJob(
            UUID(outbox.id), outbox.job_name, outbox.payload, outbox.attempts
        )

    async def update_run_classification(
        self,
        run_id: UUID,
        classification: ClassificationResponse,
        status: "RunStatus | None" = None,
    ) -> None:
        """Worker 调用：更新 run 的 classification 与最终状态。"""

        from ..domain.models import RunStatus

        final_status = status or RunStatus.COMPLETED

        classification_json = classification.model_dump_json()
        classification_payload = classification.model_dump(mode="json")
        # fusion_meta 单独提取（便于 SQL 查询融合策略审计；None 时存 NULL）
        fusion_meta_dict = (
            classification.fusion_meta.model_dump(mode="json")
            if classification.fusion_meta
            else None
        )
        async with self.sessions() as session:
            record = await session.get(RunRecord, str(run_id))
            if record is None:
                return
            record.classification = classification_json
            record.calibration_log = (
                classification.calibration_log.model_dump_json()
                if classification.calibration_log
                else None
            )
            record.fusion_meta = fusion_meta_dict
            # 同步更新 status + payload 中的 classification 字段
            record.status = final_status.value
            payload = dict(record.payload or {})
            payload["classification"] = classification_payload
            payload["status"] = final_status.value
            record.payload = payload
            record.updated_at = datetime.now(timezone.utc)
            await session.commit()

    async def complete_run_classification(
        self,
        run_id: UUID,
        classification: ClassificationResponse,
        *,
        status: "RunStatus",
    ) -> bool:
        """Store a worker result only while the run is still PROCESSING."""

        from ..domain.models import RunStatus

        classification_payload = classification.model_dump(mode="json")
        fusion_meta = (
            classification.fusion_meta.model_dump(mode="json")
            if classification.fusion_meta
            else None
        )
        async with self.sessions() as session:
            record = await session.scalar(
                select(RunRecord).where(
                    RunRecord.id == str(run_id),
                    RunRecord.status == RunStatus.PROCESSING.value,
                )
            )
            if record is None:
                return False
            payload = dict(record.payload or {})
            payload["classification"] = classification_payload
            payload["status"] = status.value
            record.classification = classification.model_dump_json()
            record.calibration_log = (
                classification.calibration_log.model_dump_json()
                if classification.calibration_log
                else None
            )
            record.fusion_meta = fusion_meta
            record.status = status.value
            record.payload = payload
            record.updated_at = datetime.now(timezone.utc)
            await session.commit()
            return True

    async def save_fusion_meta(self, run_id: UUID, fusion_meta: FusionMeta) -> None:
        """Persist fusion audit metadata to the ``fusion_meta`` JSON column.

        Allows the orchestrator to update fusion metadata independently of the
        full classification envelope (e.g. when LLM fallback completes later).
        """

        async with self.sessions() as session:
            record = await session.get(RunRecord, str(run_id))
            if record is None:
                return
            record.fusion_meta = fusion_meta.model_dump(mode="json")
            record.updated_at = datetime.now(timezone.utc)
            await session.commit()

    async def get_fusion_meta(self, run_id: UUID) -> FusionMeta | None:
        """Read the ``fusion_meta`` column; returns ``None`` if absent or unparseable."""

        async with self.sessions() as session:
            record = await session.get(RunRecord, str(run_id))
            if not record or not record.fusion_meta:
                return None
            try:
                return FusionMeta.model_validate(record.fusion_meta)
            except Exception:
                return None

    async def create_skill(self, definition: SkillDefinition) -> SkillVersion:
        async with self.sessions() as session:
            result = await session.scalar(
                select(SkillRecord.version)
                .where(SkillRecord.skill_id == str(definition.id))
                .order_by(SkillRecord.version.desc())
            )
            skill = SkillVersion(
                skill_id=definition.id, version=(result or 0) + 1, definition=definition
            )
            session.add(
                SkillRecord(
                    id=str(skill.id),
                    skill_id=str(skill.skill_id),
                    version=skill.version,
                    published=False,
                    payload=skill.model_dump(mode="json"),
                )
            )
            await session.commit()
            return skill

    async def list_skills(self) -> list[SkillVersion]:
        async with self.sessions() as session:
            records = (
                await session.scalars(
                    select(SkillRecord).order_by(
                        SkillRecord.skill_id, SkillRecord.version.desc()
                    )
                )
            ).all()
            return [SkillVersion.model_validate(record.payload) for record in records]

    async def get_published(self, skill_id: UUID | None) -> SkillVersion | None:
        async with self.sessions() as session:
            query = select(SkillRecord).where(SkillRecord.published.is_(True))
            if skill_id:
                query = query.where(SkillRecord.skill_id == str(skill_id))
            record = await session.scalar(query.order_by(SkillRecord.version.desc()))
            return SkillVersion.model_validate(record.payload) if record else None

    async def get_skill_version(self, version_id: UUID) -> SkillVersion | None:
        async with self.sessions() as session:
            record = await session.get(SkillRecord, str(version_id))
            return SkillVersion.model_validate(record.payload) if record else None

    async def publish_skill(self, skill_id: UUID) -> SkillVersion | None:
        async with self.sessions() as session:
            record = await session.scalar(
                select(SkillRecord).where(SkillRecord.id == str(skill_id))
            )
            if not record:
                return None
            record.published = True
            skill = SkillVersion.model_validate(record.payload).model_copy(
                update={"published_at": datetime.now(timezone.utc)}
            )
            record.payload = skill.model_dump(mode="json")
            await session.commit()
            return skill

    async def cleanup_expired_data(
        self,
        *,
        runs_retention_days: int = 90,
        archive_retention_days: int = 365,
        feedback_retention_days: int = 365,
        ledger_retention_days: int = 90,
        backfill_audit_retention_days: int = 365,
    ) -> dict[str, int]:
        """清理过期数据，控制表增长。

        清理顺序保证先删子表（feedback）再删父表（runs），避免 FK 冲突。
        runs 清理时排除仍有 feedback 关联的行。
        ledger 仅清理非 'claimed' 状态的终态行。
        设为 0 表示跳过对应表。
        """
        now = datetime.now(timezone.utc)
        deleted: dict[str, int] = {}

        async with self.engine.begin() as conn:
            # 1. classification_feedback（先删，避免 FK 冲突）
            if feedback_retention_days > 0:
                cutoff = now - timedelta(days=feedback_retention_days)
                result = await conn.execute(
                    text(
                        "DELETE FROM classification_feedback WHERE reviewed_at < :cutoff"
                    ),
                    {"cutoff": cutoff},
                )
                deleted["classification_feedback"] = result.rowcount

            # 2. processing_runs（排除仍有 feedback 关联的行）
            if runs_retention_days > 0:
                cutoff = now - timedelta(days=runs_retention_days)
                result = await conn.execute(
                    text(
                        "DELETE FROM processing_runs "
                        "WHERE created_at < :cutoff "
                        "AND NOT EXISTS ("
                        "  SELECT 1 FROM classification_feedback "
                        "  WHERE classification_feedback.run_id = processing_runs.id"
                        ")"
                    ),
                    {"cutoff": cutoff},
                )
                deleted["processing_runs"] = result.rowcount

            # 3. samples_archive
            if archive_retention_days > 0:
                cutoff = now - timedelta(days=archive_retention_days)
                result = await conn.execute(
                    text("DELETE FROM samples_archive WHERE created_at < :cutoff"),
                    {"cutoff": cutoff},
                )
                deleted["samples_archive"] = result.rowcount

            # 4. mail_gateway_ingest_ledger（仅清理终态行，保留 claimed）
            if ledger_retention_days > 0:
                cutoff = now - timedelta(days=ledger_retention_days)
                result = await conn.execute(
                    text(
                        "DELETE FROM mail_gateway_ingest_ledger "
                        "WHERE created_at < :cutoff AND status != 'claimed'"
                    ),
                    {"cutoff": cutoff},
                )
                deleted["mail_gateway_ingest_ledger"] = result.rowcount

            # 5. mail_gateway_backfill_audit
            if backfill_audit_retention_days > 0:
                cutoff = now - timedelta(days=backfill_audit_retention_days)
                result = await conn.execute(
                    text(
                        "DELETE FROM mail_gateway_backfill_audit WHERE created_at < :cutoff"
                    ),
                    {"cutoff": cutoff},
                )
                deleted["mail_gateway_backfill_audit"] = result.rowcount

        return deleted
