from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from ..domain.models import (
    ClassificationFeedback,
    ClassificationFeedbackRequest,
    ClassificationResponse,
    CreateRunRequest,
    RunResponse,
    RunStatus,
)
from ..domain.policy import DefaultPolicyEngine
from ..infra.store import SqlStore

logger = logging.getLogger(__name__)


class MailProcessingService:
    """邮件处理服务

    create_run 立即创建 PENDING 状态的 run，入队 arq 任务后返回 run_id。
    实际 LLM 分类由 Worker 后台执行，Worker 完成后调用 update_classification 更新 store。
    """

    def __init__(
        self,
        store: SqlStore,
        policy: DefaultPolicyEngine,
        enqueue_fn=None,
    ) -> None:
        self.store = store
        self.policy = policy
        # enqueue_fn 是 async callable(redis_pool, run_id) -> job_id
        # 测试时可传 None，API 创建 run 后 status=PENDING，需要手动调用 update_classification
        self._enqueue_fn = enqueue_fn
        self._redis_pool = None

    async def set_redis_pool(self, redis_pool) -> None:
        """设置 Redis 连接池（main.py lifespan 调用）"""

        self._redis_pool = redis_pool

    async def create_run(
        self, request: CreateRunRequest, *, enqueue: bool = True
    ) -> RunResponse:
        """创建 PENDING 状态的 run，入队 arq 任务后立即返回"""

        skill = await self.store.get_published(request.skill_id)
        now = datetime.now(timezone.utc)
        run = RunResponse(
            id=uuid4(),
            status=RunStatus.PENDING,
            email=request.email,
            skill_version_id=skill.id if skill else None,
            decision=None,
            actions=[],
            trace=["create_run:enqueued"],
            created_at=now,
            updated_at=now,
        )
        await self.store.save_run(run)

        # 入队 arq 任务（如配置了 enqueue_fn + redis pool）
        if enqueue and self._enqueue_fn and self._redis_pool is not None:
            try:
                job_id = await self.enqueue_run(run.id)
                logger.info("enqueued classify job: run_id=%s job_id=%s", run.id, job_id)
            except Exception as exc:
                logger.error("failed to enqueue classify job: %s (run stays PENDING)", exc)
        else:
            logger.debug("no enqueue_fn configured, run stays PENDING: %s", run.id)

        return run

    async def enqueue_run(self, run_id: UUID) -> str:
        """Schedule an already-persisted run, raising on retryable failures."""

        if self._enqueue_fn is None or self._redis_pool is None:
            raise RuntimeError("classification enqueue is not configured")
        job_id = await self._enqueue_fn(self._redis_pool, str(run_id))
        if not job_id:
            raise RuntimeError("classification enqueue returned no job id")
        return job_id

    async def update_classification(
        self,
        run_id: UUID,
        classification: ClassificationResponse,
        status: RunStatus = RunStatus.COMPLETED,
    ) -> RunResponse | None:
        """Worker 调用：更新分类结果与最终状态。"""

        await self.store.update_run_classification(run_id, classification, status=status)
        return await self.store.get_run(run_id)

    async def get_run(self, run_id: UUID) -> RunResponse | None:
        return await self.store.get_run(run_id)

    async def record_classification_feedback(
        self,
        run_id: UUID,
        feedback: ClassificationFeedbackRequest,
        *,
        valid_labels: set[str],
        reviewer_id: str,
    ) -> ClassificationFeedback | None:
        run = await self.store.get_run(run_id)
        if run is None:
            return None
        if run.classification is None:
            raise ValueError("run has no classification")

        final_label_set = set(feedback.final_labels)
        invalid_labels = sorted(final_label_set - valid_labels)
        if invalid_labels:
            raise ValueError(
                "labels are absent from active taxonomy: "
                + ", ".join(invalid_labels)
            )
        if "noise" in final_label_set and len(final_label_set) > 1:
            raise ValueError("noise cannot be combined with a business label")

        return await self.store.append_classification_feedback(
            run_id=run_id,
            predicted_labels=[label.l1_code for label in run.classification.labels],
            final_labels=feedback.final_labels,
            error_reasons=feedback.error_reasons,
            reviewer_id=reviewer_id,
            versions=run.classification.versions,
        )

    async def list_classification_feedback(
        self,
        run_id: UUID,
    ) -> list[ClassificationFeedback] | None:
        if await self.store.get_run(run_id) is None:
            return None
        return await self.store.list_classification_feedback(run_id)

    async def approve(self, run_id: UUID) -> RunResponse | None:
        run = await self.get_run(run_id)
        if not run:
            return None
        actions = [action.model_copy(update={"status": "approved" if action.status == "proposed" else action.status}) for action in run.actions]
        updated = run.model_copy(update={"status": RunStatus.COMPLETED, "actions": actions, "trace": [*run.trace, "approval:recorded_no_send"], "updated_at": datetime.now(timezone.utc)})
        await self.store.save_run(updated)
        return updated

    async def reject(self, run_id: UUID) -> RunResponse | None:
        run = await self.get_run(run_id)
        if not run:
            return None
        actions = [action.model_copy(update={"status": "rejected" if action.status in {"proposed", "blocked"} else action.status}) for action in run.actions]
        updated = run.model_copy(update={"status": RunStatus.REJECTED, "actions": actions, "trace": [*run.trace, "approval:rejected"], "updated_at": datetime.now(timezone.utc)})
        await self.store.save_run(updated)
        return updated

    async def retry(self, run_id: UUID) -> RunResponse | None:
        """重新入队分类（不立即处理，状态保持 PENDING 等待 Worker）"""

        previous = await self.get_run(run_id)
        if not previous:
            return None
        # 重置状态为 PENDING
        updated = previous.model_copy(
            update={
                "status": RunStatus.PENDING,
                "trace": [*previous.trace, "retry:re-enqueued"],
                "updated_at": datetime.now(timezone.utc),
            }
        )
        await self.store.save_run(updated)

        # 重新入队
        if self._enqueue_fn and self._redis_pool is not None:
            try:
                await self._enqueue_fn(self._redis_pool, str(run_id))
            except Exception as exc:
                logger.error("failed to re-enqueue: %s", exc)

        return updated
