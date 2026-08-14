"""store 模块 classification 持久化测试。

验证：
- save_run 写入 classification JSON 列
- get_run 反序列化 classification
- update_run_status 更新 status
- update_run_classification 更新 classification + status=COMPLETED
- 无 classification 的 run 仍可正常读取（向后兼容）
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4
import warnings

import pytest

from mailagent.domain import (
    CalibrationLog,
    ClassificationMeta,
    ClassificationResponse,
    ClassificationVersions,
    FusionMeta,
    MailEvent,
    PathBResult,
    RuleResult,
    RunResponse,
    RunStatus,
    TaxonomyLabel,
)
from mailagent.infra.store import SqlStore


@pytest.fixture
async def store(tmp_path) -> SqlStore:
    db_path = tmp_path / "test_store.db"
    s = SqlStore(f"sqlite+aiosqlite:///{db_path}")
    await s.create_tables()
    yield s
    await s.close()


@pytest.fixture
def sample_mail() -> MailEvent:
    return MailEvent(
        message_id="test-store-1",
        sender="ops@example.com",
        subject="Berlin Example STATUS update",
        body="Entity Berlin Example STATUS Shanghai 2026-08-01 08:00 UTC.",
    )


@pytest.fixture
def sample_classification() -> ClassificationResponse:
    return ClassificationResponse(
        labels=[
            TaxonomyLabel(
                l1_code="schedule",
                l1_label="船期",
                l2_code=None,
                l2_label=None,
                l3_code=None,
                l3_label=None,
                confidence=0.87,
                reasoning="STATUS update email",
            ),
            TaxonomyLabel(
                l1_code="operation",
                l1_label="实体作业",
                l2_code=None,
                l2_label=None,
                l3_code=None,
                l3_label=None,
                confidence=0.70,
                reasoning="Directional secondary match",
            ),
        ],
        meta=ClassificationMeta(
            urgency="medium",
            language="en",
            sentiment="neutral",
            has_attachments=False,
            overall_confidence=0.87,
            needs_human_review=False,
            fallback=False,
            model_used="test-model",
            latency_ms=120,
        ),
        calibration_log=CalibrationLog(raw=0.92, calibrated=0.87, anchor="fairly certain"),
        versions=ClassificationVersions(
            taxonomy="sha256:taxonomy",
            rules="sha256:rules",
            prompt="llm-classifier-v1",
            model="test-model",
            embedding="test-embedding",
            preprocessing="sha256:preprocessing",
        ),
        autonomy_level="L0",
        vertical_id="example-triage",
        data={},
    )


def _make_run(mail: MailEvent, status: RunStatus = RunStatus.PENDING) -> RunResponse:
    now = datetime.now(timezone.utc)
    return RunResponse(
        id=uuid4(),
        status=status,
        email=mail,
        skill_version_id=None,
        decision=None,
        actions=[],
        trace=["test:created"],
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
async def completed_run(
    store: SqlStore,
    sample_mail: MailEvent,
    sample_classification: ClassificationResponse,
) -> RunResponse:
    run = _make_run(sample_mail, RunStatus.COMPLETED)
    run.classification = sample_classification
    await store.save_run(run)
    return run


async def test_feedback_appends_revisions_without_overwrite(
    store: SqlStore,
    completed_run: RunResponse,
) -> None:
    first = await store.append_classification_feedback(
        run_id=completed_run.id,
        predicted_labels=["schedule"],
        final_labels=["operation"],
        error_reasons=["wrong_label"],
        reviewer_id="reviewer-1",
        versions=completed_run.classification.versions,
    )
    second = await store.append_classification_feedback(
        run_id=completed_run.id,
        predicted_labels=["schedule"],
        final_labels=["document"],
        error_reasons=["wrong_label"],
        reviewer_id="reviewer-2",
        versions=completed_run.classification.versions,
    )

    assert first.revision == 1
    assert second.revision == 2
    listed = await store.list_classification_feedback(completed_run.id)
    assert [item.final_labels for item in listed] == [["operation"], ["document"]]
    assert listed[0].reviewed_at.tzinfo is timezone.utc
    assert listed[0].reviewed_at.utcoffset() == timezone.utc.utcoffset(None)


async def test_feedback_concurrent_appends_receive_distinct_ordered_revisions(
    store: SqlStore,
    completed_run: RunResponse,
) -> None:
    async def append(index: int):
        return await store.append_classification_feedback(
            run_id=completed_run.id,
            predicted_labels=["schedule"],
            final_labels=["operation"],
            error_reasons=["wrong_label"],
            reviewer_id=f"reviewer-{index}",
            versions=completed_run.classification.versions,
        )

    appended = await asyncio.gather(*(append(index) for index in range(5)))
    listed = await store.list_classification_feedback(completed_run.id)

    assert sorted(item.revision for item in appended) == [1, 2, 3, 4, 5]
    assert [item.revision for item in listed] == [1, 2, 3, 4, 5]
    assert {item.reviewer_id for item in listed} == {
        "reviewer-0",
        "reviewer-1",
        "reviewer-2",
        "reviewer-3",
        "reviewer-4",
    }


class TestSaveAndGetClassification:
    async def test_save_run_with_classification_roundtrip(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
        sample_classification: ClassificationResponse,
    ) -> None:
        """save_run 写入 classification JSON，get_run 应能完整反序列化"""

        run = _make_run(sample_mail)
        run.classification = sample_classification
        await store.save_run(run)

        loaded = await store.get_run(run.id)
        assert loaded is not None
        assert loaded.classification is not None
        assert len(loaded.classification.labels) == 2
        assert loaded.classification.labels[0].l1_code == "schedule"
        assert loaded.classification.labels[0].l3_code is None
        assert loaded.classification.meta.model_used == "test-model"
        assert loaded.classification.meta.fallback is False
        assert loaded.classification.calibration_log is not None
        assert loaded.classification.calibration_log.calibrated == 0.87
        assert loaded.classification.versions == sample_classification.versions
        assert loaded.classification.data == {}

    async def test_classification_envelope_survives_store_restart(
        self,
        tmp_path,
        sample_mail: MailEvent,
        sample_classification: ClassificationResponse,
    ) -> None:
        db_path = tmp_path / "restart.db"
        first_store = SqlStore(f"sqlite+aiosqlite:///{db_path}")
        await first_store.create_tables()
        run = _make_run(sample_mail)
        await first_store.save_run(run)
        await first_store.update_run_classification(run.id, sample_classification)
        await first_store.close()

        restarted_store = SqlStore(f"sqlite+aiosqlite:///{db_path}")
        loaded = await restarted_store.get_run(run.id)
        await restarted_store.close()

        assert loaded is not None
        assert loaded.classification is not None
        assert loaded.classification.vertical_id == "example-triage"
        assert loaded.classification.data_schema_version == "1"
        assert loaded.classification.data == sample_classification.data

    async def test_save_run_without_classification_returns_none(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
    ) -> None:
        """无 classification 的 run（如 PENDING 状态）读取时 classification=None"""

        run = _make_run(sample_mail)
        await store.save_run(run)

        loaded = await store.get_run(run.id)
        assert loaded is not None
        assert loaded.classification is None

    async def test_save_run_overwrites_classification(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
        sample_classification: ClassificationResponse,
    ) -> None:
        """先保存无 classification 的 run，再 save_run 同一 run（带 classification）应覆盖"""

        run = _make_run(sample_mail)
        await store.save_run(run)

        # 模拟 Worker 完成分类后保存
        run.classification = sample_classification
        run.status = RunStatus.COMPLETED
        await store.save_run(run)

        loaded = await store.get_run(run.id)
        assert loaded is not None
        assert loaded.classification is not None
        assert loaded.classification.labels[0].l3_code is None
        assert loaded.status == RunStatus.COMPLETED


class TestUpdateRunStatus:
    async def test_update_run_status_pending_to_processing(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
    ) -> None:
        run = _make_run(sample_mail, RunStatus.PENDING)
        await store.save_run(run)

        await store.update_run_status(run.id, RunStatus.PROCESSING)
        loaded = await store.get_run(run.id)
        assert loaded is not None
        assert loaded.status == RunStatus.PROCESSING

    async def test_update_run_status_to_failed(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
    ) -> None:
        run = _make_run(sample_mail, RunStatus.PROCESSING)
        await store.save_run(run)

        await store.update_run_status(run.id, RunStatus.FAILED)
        loaded = await store.get_run(run.id)
        assert loaded is not None
        assert loaded.status == RunStatus.FAILED

    async def test_update_run_status_unknown_run_id_is_noop(
        self,
        store: SqlStore,
    ) -> None:
        """不存在的 run_id 不应抛异常"""

        await store.update_run_status(uuid4(), RunStatus.FAILED)


class TestUpdateRunClassification:
    async def test_update_classification_does_not_read_deprecated_api_field(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
        sample_classification: ClassificationResponse,
    ) -> None:
        run = _make_run(sample_mail)
        await store.save_run(run)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await store.update_run_classification(run.id, sample_classification)

    async def test_update_classification_sets_completed_status(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
        sample_classification: ClassificationResponse,
    ) -> None:
        """update_run_classification 应同时更新 status=COMPLETED + classification JSON"""

        run = _make_run(sample_mail, RunStatus.PROCESSING)
        await store.save_run(run)

        await store.update_run_classification(run.id, sample_classification)

        loaded = await store.get_run(run.id)
        assert loaded is not None
        assert loaded.status == RunStatus.COMPLETED
        assert loaded.classification is not None
        assert loaded.classification.meta.overall_confidence == 0.87

    async def test_update_classification_persists_calibraion_log_separately(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
        sample_classification: ClassificationResponse,
    ) -> None:
        """calibration_log 列独立存储（用于回放评估）"""

        run = _make_run(sample_mail)
        await store.save_run(run)

        await store.update_run_classification(run.id, sample_classification)

        # 直接读 DB 验证 calibration_log 列
        from mailagent.infra.store import RunRecord

        async with store.sessions() as session:
            record = await session.get(RunRecord, str(run.id))
            assert record is not None
            assert record.calibration_log is not None
            assert "fairly certain" in record.calibration_log

    async def test_update_classification_unknown_run_id_is_noop(
        self,
        store: SqlStore,
        sample_classification: ClassificationResponse,
    ) -> None:
        await store.update_run_classification(uuid4(), sample_classification)


class TestEmptyClassification:
    async def test_run_with_empty_labels_persists(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
    ) -> None:
        """空 labels + needs_human_review=true（ambiguous 邮件）应能正确持久化"""

        empty_classification = ClassificationResponse(
            labels=[],
            meta=ClassificationMeta(
                urgency="low",
                language="en",
                sentiment="neutral",
                has_attachments=False,
                overall_confidence=0.0,
                needs_human_review=True,
                fallback=False,
                model_used="test-model",
                latency_ms=50,
            ),
            calibration_log=None,
            autonomy_level="L0",
        )

        run = _make_run(sample_mail)
        run.classification = empty_classification
        await store.save_run(run)

        loaded = await store.get_run(run.id)
        assert loaded is not None
        assert loaded.classification is not None
        assert loaded.classification.labels == []
        assert loaded.classification.meta.needs_human_review is True


class TestLegacyCompatibility:
    async def test_get_run_with_legacy_payload_no_classification_column(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
    ) -> None:
        """旧 run（payload 中无 classification 字段）仍可读取，classification=None"""

        run = _make_run(sample_mail)
        await store.save_run(run)

        loaded = await store.get_run(run.id)
        assert loaded is not None
        assert loaded.classification is None
        # 其他字段正常
        assert loaded.email.message_id == sample_mail.message_id
        assert loaded.status == RunStatus.PENDING


def _make_fusion_meta() -> FusionMeta:
    """Build a sample FusionMeta for testing."""
    return FusionMeta(
        fusion_strategy="rule_vector_confirmed",
        source="rule",
        confidence=0.9,
        rule_result=RuleResult(),
        vector_result=PathBResult(),
        vector_confirmed=True,
    )


class TestFusionMeta:
    """fusion_meta JSON column read/write roundtrip and backward compatibility."""

    async def test_save_fusion_meta_and_get_roundtrip(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
    ) -> None:
        """save_fusion_meta writes the JSON column; get_fusion_meta reads it back."""
        run = _make_run(sample_mail)
        await store.save_run(run)

        fusion = _make_fusion_meta()
        await store.save_fusion_meta(run.id, fusion)

        loaded = await store.get_fusion_meta(run.id)
        assert loaded is not None
        assert loaded.fusion_strategy == "rule_vector_confirmed"
        assert loaded.source == "rule"
        assert loaded.confidence == 0.9
        assert loaded.vector_confirmed is True

    async def test_get_fusion_meta_returns_none_when_absent(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
    ) -> None:
        """A run without fusion_meta should return None (backward compat)."""
        run = _make_run(sample_mail)
        await store.save_run(run)

        loaded = await store.get_fusion_meta(run.id)
        assert loaded is None

    async def test_get_fusion_meta_unknown_run_id_returns_none(
        self,
        store: SqlStore,
    ) -> None:
        """Reading fusion_meta for a non-existent run should not raise."""
        loaded = await store.get_fusion_meta(uuid4())
        assert loaded is None

    async def test_update_run_classification_persists_fusion_meta(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
        sample_classification: ClassificationResponse,
    ) -> None:
        """update_run_classification should persist fusion_meta to its dedicated column."""
        sample_classification.fusion_meta = _make_fusion_meta()

        run = _make_run(sample_mail)
        await store.save_run(run)
        await store.update_run_classification(run.id, sample_classification)

        loaded = await store.get_fusion_meta(run.id)
        assert loaded is not None
        assert loaded.fusion_strategy == "rule_vector_confirmed"
        assert loaded.vector_confirmed is True

    async def test_update_run_classification_without_fusion_meta_stores_null(
        self,
        store: SqlStore,
        sample_mail: MailEvent,
        sample_classification: ClassificationResponse,
    ) -> None:
        """When classification has no fusion_meta, the column should be NULL."""
        # sample_classification.fusion_meta defaults to None
        assert sample_classification.fusion_meta is None

        run = _make_run(sample_mail)
        await store.save_run(run)
        await store.update_run_classification(run.id, sample_classification)

        loaded = await store.get_fusion_meta(run.id)
        assert loaded is None

    async def test_save_fusion_meta_unknown_run_id_is_noop(
        self,
        store: SqlStore,
    ) -> None:
        """Saving fusion_meta for a non-existent run should not raise."""
        await store.save_fusion_meta(uuid4(), _make_fusion_meta())
