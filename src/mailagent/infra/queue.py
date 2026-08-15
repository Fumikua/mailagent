"""arq + Redis 异步队列：classify_job 入队/执行。

API 立即返回 PENDING + run_id，Worker 后台调用 ClassifyAgent，更新 status=COMPLETED。
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from arq import ArqRedis, create_pool, cron
from arq.connections import RedisSettings

from ..domain.models import RunStatus

logger = logging.getLogger(__name__)

CLASSIFY_JOB_NAME = "classify_job"
CLUSTERING_JOB_NAME = "clustering_job"
RULE_LEARN_JOB_NAME = "rule_learn_job"
ARCHIVE_JOB_NAME = "archive_job"
MAIL_POLL_JOB_NAME = "mail_poll_job"
CLEANUP_JOB_NAME = "cleanup_job"
OUTBOX_DISPATCH_JOB_NAME = "outbox_dispatch_job"
WORKER_HEARTBEAT_JOB_NAME = "worker_heartbeat_job"
WORKER_HEARTBEAT_KEY = "mailagent:worker:heartbeat"

_DEFAULT_BODY_MAX_CHARS = 16_000


def build_mail_understanding_pipeline(
    settings: Any,
    llm_client: Any,
    vertical_runtime: Any,
    embedding_client: Any = None,
    vector_store: Any = None,
    runtime_components: dict[str, Any] | None = None,
    loaded_vertical: Any = None,
) -> Any:
    """Build the selected vertical's pipeline during Worker startup.

    When ``settings.fusion.enabled`` is True, a :class:`FusionOrchestrator`
    coordinates the rule / vector / LLM cascade (Path B); otherwise the
    original :class:`CascadeClassificationOrchestrator` is used for backward
    compatibility (Path A).
    """

    from ..classification.llm_classifier import LLMClassifier
    from ..classification.rule_classifier import RuleClassifier
    from ..classification.taxonomy import TaxonomyLoader
    from ..classification.vector_classifier import VectorClassifier
    from ..core import (
        CascadeClassificationOrchestrator,
        FusionOrchestrator,
        MailUnderstandingPipeline,
        TargetProfileLoader,
        ClassificationVersionProvider,
    )
    from ..verticals import load_selected_vertical

    loaded = loaded_vertical
    if loaded is None:
        loaded = load_selected_vertical(settings.vertical).assets
    taxonomy_loader = TaxonomyLoader(loaded.taxonomy_path)
    body_max_chars = (
        loaded.manifest.llm.body_max_chars
        if loaded.manifest.llm is not None
        else _DEFAULT_BODY_MAX_CHARS
    )
    llm_classifier = LLMClassifier(
        llm_client,
        taxonomy_loader,
        settings.model.model_name,
        body_max_chars=body_max_chars,
    )
    actual_enricher_ids = {enricher.id for enricher in vertical_runtime.enrichers}
    declared_enricher_ids = set(loaded.manifest.enrichers)
    if actual_enricher_ids != declared_enricher_ids:
        raise ValueError(
            f"vertical runtime enrichers {sorted(actual_enricher_ids)} do not match "
            f"manifest {sorted(declared_enricher_ids)}"
        )

    # Orchestrator selection: FusionOrchestrator (Path B) vs Cascade (Path A).
    # classify_job consumes the pipeline opaquely, so the feature flag is
    # resolved entirely here at construction time.
    rule_classifier = None
    vector_classifier = None
    target_profile_loader: TargetProfileLoader | None = None
    if settings.fusion.enabled:
        fusion_classifiers: list[Any] = [llm_classifier]
        if loaded.rules is not None:
            rule_classifier = RuleClassifier(
                rules_dir=loaded.rules.path,
                taxonomy_loader=taxonomy_loader,
            )
            fusion_classifiers.insert(0, rule_classifier)
        if (
            loaded.rag is not None
            and embedding_client is not None
            and vector_store is not None
        ):
            vector_classifier = VectorClassifier(
                vector_store,
                embedding_client,
                settings.vector_store,
                cleaning_policy=vertical_runtime.retrieval_cleaning_policy,
                preprocessing_extension=vertical_runtime.preprocessing_extension,
                taxonomy_loader=taxonomy_loader,
            )
            fusion_classifiers.insert(-1, vector_classifier)
        elif loaded.rag is not None:
            logger.warning(
                "fusion.enabled=true but embedding_client/vector_store missing; "
                "FusionOrchestrator degrades to rule + llm only"
            )
        # Target profile loader: enables label-scoped vector retrieval when a
        # rule candidate matches a declared target. Missing target_profiles.yaml
        # → loader returns empty target list (feature off, no error).
        if loaded.target_profiles is not None:
            target_profile_loader = TargetProfileLoader(
                config_path=loaded.target_profiles.path,
                taxonomy_loader=taxonomy_loader,
            )
        orchestrator: Any = FusionOrchestrator(
            classifiers=fusion_classifiers,
            settings=settings.fusion,
            target_profile_loader=target_profile_loader,
        )
        if runtime_components is not None:
            runtime_components.update(
                {
                    "taxonomy_loader": taxonomy_loader,
                    "llm_classifier": llm_classifier,
                }
            )
            if rule_classifier is not None:
                runtime_components["rule_classifier"] = rule_classifier
    else:
        orchestrator = CascadeClassificationOrchestrator(
            classifiers=[llm_classifier],
            acceptance_thresholds={"llm": settings.classification.confidence_threshold},
        )

    version_provider = ClassificationVersionProvider(
        taxonomy_loader=taxonomy_loader,
        rule_classifier=rule_classifier,
        preprocessing_extension=(
            vertical_runtime.preprocessing_extension
            if vector_classifier is not None
            else None
        ),
        retrieval_cleaning_policy=(
            vertical_runtime.retrieval_cleaning_policy
            if vector_classifier is not None
            else None
        ),
        target_profile_loader=target_profile_loader,
        prompt_version=llm_classifier.prompt_version,
        model_version=settings.model.model_name,
        embedding_version=(
            settings.embedding.model_name if vector_classifier is not None else None
        ),
    )

    return MailUnderstandingPipeline(
        orchestrator=orchestrator,
        vertical_id=loaded.manifest.id,
        data_schema_version=loaded.manifest.data_schema_version,
        vertical_namespace=loaded.manifest.namespace,
        enrichers=vertical_runtime.enrichers,
        data_schema=loaded.data_schema,
        version_provider=version_provider,
        auto_accept_enabled=settings.classification.auto_accept_enabled,
    )


def redis_settings_from_url(url: str) -> RedisSettings:
    """从 redis URL 解析为 arq RedisSettings。

    支持形如 redis://localhost:6379/0 或 redis://:password@host:port/db
    """

    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        database=int(parsed.path.lstrip("/") or "0") if parsed.path else 0,
    )


async def enqueue_classify(
    redis: ArqRedis,
    run_id: str,
    *,
    job_id: str | None = None,
) -> str:
    """入队 classify 任务，返回 job_id。

    Args:
        redis: ArqRedis 实例
        run_id: 待分类的 run UUID 字符串

    Returns:
        str: arq job_id（入队失败返回空字符串）
    """

    if job_id is None:
        job = await redis.enqueue_job(CLASSIFY_JOB_NAME, run_id)
    else:
        job = await redis.enqueue_job(CLASSIFY_JOB_NAME, run_id, _job_id=job_id)
    return job.job_id if job else ""


async def dispatch_outbox_item(redis: ArqRedis, store: Any, item: Any) -> str:
    """Dispatch one durable outbox row with a deterministic Redis job id."""

    if item.job_name != CLASSIFY_JOB_NAME:
        error = f"unsupported outbox job: {item.job_name}"
        await store.record_outbox_failure(item.id, error)
        raise ValueError(error)
    try:
        run_id = str(item.payload["run_id"])
        deterministic_id = f"outbox:{item.id}"
        job_id = await enqueue_classify(redis, run_id, job_id=deterministic_id)
        # arq returns None when this deterministic job already exists. That is
        # a successful idempotent dispatch, not a reason to retry forever.
        await store.mark_outbox_dispatched(item.id)
        return job_id or deterministic_id
    except Exception as exc:
        await store.record_outbox_failure(item.id, str(exc))
        raise


async def classify_job(ctx: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Worker 处理函数：加载 run → 调用 Pipeline（或兼容 Agent）→ 更新 store。

    Args:
        ctx: arq worker context，包含 'store' / 'mail_understanding_pipeline' / 'service'
        run_id: run UUID 字符串

    Returns:
        dict: 处理结果摘要
    """

    from uuid import UUID

    store = ctx["store"]
    pipeline = ctx.get("mail_understanding_pipeline")
    classify_agent = ctx.get("classify_agent")
    run_uuid = UUID(run_id)

    # 1. 只有一个 worker 能认领 PENDING run；重复投递直接安全退出。
    claimed = await store.transition_run_status(
        run_uuid,
        expected={RunStatus.PENDING},
        target=RunStatus.PROCESSING,
    )
    if not claimed:
        existing = await store.get_run(run_uuid)
        if existing is None:
            logger.error("run not found: %s", run_id)
            return {"run_id": run_id, "status": "failed", "reason": "run not found"}
        return {
            "run_id": run_id,
            "status": "ignored",
            "reason": f"run is already {existing.status.value}",
        }

    # 2. 加载 run
    run = await store.get_run(run_uuid)
    if run is None:
        logger.error("run not found: %s", run_id)
        return {"run_id": run_id, "status": "failed", "reason": "run not found"}

    # 3. 调用新 Pipeline；旧 ClassifyAgent 仅保留给尚未迁移的测试/部署兼容。
    try:
        if pipeline is not None:
            classification = await pipeline.process(run.email)
            compatibility_projector = ctx.get("classification_compatibility_projector")
            if compatibility_projector is not None:
                classification = compatibility_projector(classification)
        elif classify_agent is not None:
            classification = await classify_agent.classify(run.email)
        else:
            raise RuntimeError("mail understanding pipeline is not configured")
    except Exception as exc:
        logger.exception("classify_job failed for run %s: %s", run_id, exc)
        await store.transition_run_status(
            run_uuid,
            expected={RunStatus.PROCESSING},
            target=RunStatus.FAILED,
        )
        return {"run_id": run_id, "status": "failed", "reason": str(exc)}

    # 4. 不可信输入与模型兜底结果都只能进入人工复核，不能自动完成。
    review_required = (
        classification.meta.needs_human_review or classification.meta.fallback
    )
    suspicious_instruction = _contains_prompt_injection(run.email.body)
    if suspicious_instruction:
        logger.warning(
            "prompt injection detected for run %s; holding for review", run_id
        )
        review_required = True

    if review_required and not classification.meta.needs_human_review:
        classification = classification.model_copy(
            update={
                "meta": classification.meta.model_copy(
                    update={"needs_human_review": True}
                )
            }
        )

    final_status = (
        RunStatus.WAITING_APPROVAL if review_required else RunStatus.COMPLETED
    )
    updated = await store.complete_run_classification(
        run_uuid,
        classification,
        status=final_status,
    )
    if not updated:
        return {
            "run_id": run_id,
            "status": "ignored",
            "reason": "run state changed before classification completed",
        }
    return {
        "run_id": run_id,
        "status": final_status.value,
        "labels_count": len(classification.labels),
        "fallback": classification.meta.fallback,
    }


def _contains_prompt_injection(body: str) -> bool:
    """检测明确的越权指令模式；命中后始终交由人工复核。"""

    normalized = body.casefold()
    patterns = (
        "ignore previous instructions",
        "ignore all previous instructions",
        "忽略之前指令",
        "忽略前面的指令",
        "忽略先前指令",
    )
    return any(pattern in normalized for pattern in patterns)


async def worker_on_startup(ctx: dict[str, Any]) -> None:
    """Worker 启动时初始化通用服务与所选 vertical runtime。"""

    from ..llm.client import LLMClient
    from ..domain.policy import DefaultPolicyEngine
    from ..api.service import MailProcessingService
    from ..infra.store import SqlStore
    from ..infra.config import Settings
    from ..verticals import (
        VerticalRuntimeDependencies,
        build_vertical_runtime,
        load_selected_vertical,
    )
    import os

    settings = Settings.from_yaml()
    from ..infra.migrations import upgrade_database

    await upgrade_database(settings.database.url)
    store = SqlStore(settings.database.url)

    api_key = os.getenv(settings.model.api_key_env, "")
    llm_client = LLMClient(
        base_url=settings.model.base_url,
        api_key=api_key,
        model=settings.model.model_name,
    )

    # Path B components: only initialized when fusion is enabled. These are
    # shared with cron jobs (clustering / rule-learn / archive) and must be
    # closed on shutdown.
    embedding_client = None
    vector_store = None
    if settings.fusion.enabled:
        from ..llm.embedding import EmbeddingClient
        from .vector_store import VectorStore

        embedding_client = EmbeddingClient(settings.embedding)
        vector_store = VectorStore(settings.vector_store, store.engine)
        logger.info(
            "fusion path enabled: embedding_api=%s, vector_store_dialect=%s",
            settings.embedding.api_base,
            store.engine.dialect.name,
        )

    selected_vertical = load_selected_vertical(settings.vertical)
    loaded_vertical = selected_vertical.assets
    vertical_runtime = await build_vertical_runtime(
        selected_vertical.plugin.build_runtime,
        VerticalRuntimeDependencies(settings, llm_client, loaded_vertical),
    )
    fusion_components: dict[str, Any] = {}
    pipeline = build_mail_understanding_pipeline(
        settings,
        llm_client,
        vertical_runtime,
        embedding_client,
        vector_store,
        fusion_components,
        loaded_vertical,
    )
    if (
        settings.fusion.enabled
        and embedding_client is not None
        and vector_store is not None
    ):
        from .bootstrap import BootstrapPipeline
        from .clustering import ClusteringEngine
        from .rule_learner import RuleLearner

        ctx["clustering_engine"] = ClusteringEngine(
            vector_store,
            fusion_components["taxonomy_loader"],
            llm_client,
            settings.clustering,
        )
        rule_classifier = fusion_components.get("rule_classifier")
        if rule_classifier is not None and loaded_vertical.rules is not None:
            rules_settings = settings.rules.model_copy(
                update={"rules_dir": str(loaded_vertical.rules.path)}
            )
            ctx["rule_learner"] = RuleLearner(vector_store, rules_settings)
            ctx["bootstrap_pipeline"] = BootstrapPipeline(
                rule_classifier,
                fusion_components["llm_classifier"],
                vector_store,
                embedding_client,
                settings.bootstrap,
                cleaning_policy=vertical_runtime.retrieval_cleaning_policy,
                preprocessing_extension=vertical_runtime.preprocessing_extension,
                taxonomy_loader=fusion_components["taxonomy_loader"],
            )
    policy = DefaultPolicyEngine(settings.policy)
    service = MailProcessingService(store, policy, enqueue_fn=enqueue_classify)
    if redis_pool := ctx.get("redis"):
        await service.set_redis_pool(redis_pool)

    ctx["store"] = store
    ctx["mail_understanding_pipeline"] = pipeline
    ctx["llm_client"] = llm_client
    ctx["embedding_client"] = embedding_client
    ctx["vector_store"] = vector_store
    ctx["vertical_runtime"] = vertical_runtime
    ctx.update(fusion_components)
    if vertical_runtime.compatibility_projector is not None:
        ctx["classification_compatibility_projector"] = (
            vertical_runtime.compatibility_projector
        )
    ctx["service"] = service
    ctx["settings"] = settings
    if any(gw.enabled for gw in settings.mail_gateways):
        from ..gateway.state import MailGatewayStateStore

        ctx["mail_gateway_state"] = MailGatewayStateStore(store.engine)
    if redis := ctx.get("redis"):
        await redis.set(WORKER_HEARTBEAT_KEY, "1", ex=90)
    logger.info(
        "arq worker started: vertical=%s, enrichers=%s, fusion=%s",
        settings.vertical.id,
        sorted(enricher.id for enricher in vertical_runtime.enrichers),
        "enabled" if settings.fusion.enabled else "disabled",
    )


async def worker_on_shutdown(ctx: dict[str, Any]) -> None:
    """Worker 关闭时释放资源。"""

    store = ctx.get("store")
    if store:
        await store.close()
    llm_client = ctx.get("llm_client")
    if llm_client:
        await llm_client.close()
    embedding_client = ctx.get("embedding_client")
    if embedding_client:
        await embedding_client.close()
    vertical_runtime = ctx.get("vertical_runtime")
    if vertical_runtime:
        await vertical_runtime.close()
    logger.info("arq worker shutdown")


# ---------------------------------------------------------------------------
# Path B cron jobs
# ---------------------------------------------------------------------------
async def clustering_job(ctx: dict[str, Any]) -> str:
    """Weekly HDBSCAN clustering job (arq cron, every Sunday 02:00).

    Detects new intents and taxonomy drift via ``ClusteringEngine``.
    """
    engine = ctx.get("clustering_engine")
    if engine is None:
        return "skipped: fusion disabled"
    return await engine.run_weekly_clustering()


async def rule_learn_job(ctx: dict[str, Any]) -> str:
    """Weekly rule auto-learn scan (arq cron, every Sunday 03:00).

    Proposes new sender-domain rules via ``RuleLearner``.
    """
    learner = ctx.get("rule_learner")
    if learner is None:
        return "skipped: fusion disabled"
    return await learner.run_weekly_scan()


async def archive_job(ctx: dict[str, Any]) -> str:
    """Monthly sample archival job (arq cron, 1st of each month 02:00).

    Moves stale samples to the archive table via ``BootstrapPipeline``.
    """
    pipeline = ctx.get("bootstrap_pipeline")
    if pipeline is None:
        return "skipped: fusion disabled"
    settings = ctx["settings"]
    archived = await pipeline.archive_old_samples(
        settings.vector_store.archive_window_months
    )
    return f"archived: {archived}"


async def cleanup_job(ctx: dict[str, Any]) -> str:
    """Daily data retention cleanup job (arq cron, every day 03:30).

    Purges expired rows from processing_runs, samples_archive,
    classification_feedback, mail_gateway_ingest_ledger, and
    mail_gateway_backfill_audit per ``settings.retention``.
    """
    store = ctx.get("store")
    if store is None:
        return "skipped: store unavailable"
    settings = ctx["settings"]
    retention = settings.retention
    deleted = await store.cleanup_expired_data(
        runs_retention_days=retention.runs_retention_days,
        archive_retention_days=retention.archive_retention_days,
        feedback_retention_days=retention.feedback_retention_days,
        ledger_retention_days=retention.ledger_retention_days,
        backfill_audit_retention_days=retention.backfill_audit_retention_days,
    )
    summary = ", ".join(f"{k}={v}" for k, v in deleted.items()) or "nothing expired"
    logger.info("cleanup_job completed: %s", summary)
    return f"cleanup: {summary}"


async def outbox_dispatch_job(ctx: dict[str, Any]) -> str:
    """Retry durable enqueue intents left by API/Redis partial failures."""

    store = ctx.get("store")
    redis = ctx.get("redis")
    if store is None or redis is None:
        return "skipped: store or redis unavailable"
    pending = await store.list_pending_outbox(limit=100)
    dispatched = 0
    failed = 0
    for item in pending:
        try:
            await dispatch_outbox_item(redis, store, item)
            dispatched += 1
        except Exception:
            failed += 1
            logger.exception("outbox dispatch failed: outbox_id=%s", item.id)
    return f"outbox: dispatched={dispatched}, failed={failed}"


async def worker_heartbeat_job(ctx: dict[str, Any]) -> str:
    """Publish a short-lived marker used by API readiness checks."""

    redis = ctx.get("redis")
    if redis is None:
        return "skipped: redis unavailable"
    await redis.set(WORKER_HEARTBEAT_KEY, "1", ex=90)
    return "worker heartbeat refreshed"


cron_jobs = [
    cron(worker_heartbeat_job, name=WORKER_HEARTBEAT_JOB_NAME, minute=set(range(60))),
    cron(outbox_dispatch_job, name=OUTBOX_DISPATCH_JOB_NAME, minute=set(range(60))),
    cron(clustering_job, name=CLUSTERING_JOB_NAME, weekday="sun", hour=2, minute=0),
    cron(rule_learn_job, name=RULE_LEARN_JOB_NAME, weekday="sun", hour=3, minute=0),
    cron(archive_job, name=ARCHIVE_JOB_NAME, day=1, hour=2, minute=0),
    cron(cleanup_job, name=CLEANUP_JOB_NAME, hour=3, minute=30),
]


async def create_redis_pool(redis_url: str) -> ArqRedis:
    """创建 Redis 连接池（API 侧入队用）。"""

    return await create_pool(redis_settings_from_url(redis_url))
