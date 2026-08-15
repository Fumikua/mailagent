from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import text

from ..infra.config import Settings
from ..domain.models import (
    ApprovalRequest,
    ClassificationFeedback,
    ClassificationFeedbackRequest,
    CreateRunRequest,
    RunResponse,
    SampleRecord,
    SkillDefinition,
    SkillVersion,
    normalize_reviewer_identity,
)
from ..domain.policy import DefaultPolicyEngine
from ..infra.queue import WORKER_HEARTBEAT_KEY, create_redis_pool, enqueue_classify
from ..infra.migrations import upgrade_database
from ..infra.vector_store import VectorStore
from ..llm.taxonomy import TaxonomyLoader
from ..verticals import load_selected_vertical
from .auth import (
    ApiPrincipal,
    require_admin,
    require_operator,
    require_reviewer,
    require_submitter,
    validate_api_auth_secrets,
)
from .service import InvalidRunTransition, MailProcessingService
from ..infra.store import SqlStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_yaml()
    validate_api_auth_secrets(settings.api_auth)
    await upgrade_database(settings.database.url)
    store = SqlStore(settings.database.url)
    app.state.settings = settings
    app.state.store = store

    loaded_vertical = load_selected_vertical(settings.vertical).assets
    app.state.taxonomy_loader = TaxonomyLoader(loaded_vertical.taxonomy_path)

    # 尝试连接 Redis（失败时不阻断启动，API 创建 run 会保持 PENDING 状态等待手动处理）
    redis_pool = None
    try:
        redis_pool = await create_redis_pool(settings.redis.url)
        logger.info("Redis pool connected: %s", settings.redis.url)
    except Exception as exc:
        logger.warning(
            "Redis connection failed (%s); runs will be created PENDING without enqueue. "
            "Start Redis and the worker to process them.",
            exc,
        )

    service = MailProcessingService(
        store,
        DefaultPolicyEngine(settings.policy),
        enqueue_fn=enqueue_classify if redis_pool is not None else None,
    )
    if redis_pool is not None:
        await service.set_redis_pool(redis_pool)
    app.state.service = service
    app.state.redis_pool = redis_pool
    # VectorStore shares the SqlStore engine so sample CRUD endpoints can query
    # the same database without a second connection pool.
    app.state.vector_store = VectorStore(settings.vector_store, store.engine)

    yield

    if redis_pool is not None:
        await redis_pool.close()
    await store.close()


app = FastAPI(title="MailAgent API", version="0.1.0", lifespan=lifespan)

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a safe correlation ID and emit one request completion record."""

    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def service(request: Request) -> MailProcessingService:
    return request.app.state.service


def store(request: Request) -> SqlStore:
    return request.app.state.store


def get_vector_store(request: Request) -> VectorStore | None:
    """Return the application-level VectorStore, or None if not configured.

    Tests override this via ``app.dependency_overrides`` to inject a mock.
    """

    return getattr(request.app.state, "vector_store", None)


def _require_feedback_enabled(request: Request) -> None:
    settings = request.app.state.settings.classification_feedback
    if settings.mode != "trusted_internal":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="classification feedback is disabled",
        )


def _trusted_reviewer_id(request: Request, principal: ApiPrincipal) -> str:
    if request.app.state.settings.api_auth.mode == "api_key":
        return principal.subject
    settings = request.app.state.settings.classification_feedback
    raw_identity = request.headers.get(settings.reviewer_identity_header)
    try:
        return normalize_reviewer_identity(raw_identity)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="valid trusted reviewer identity is required",
        ) from exc


# ---------------------------------------------------------------------------
# Request body models for the Section 17 API extensions
# ---------------------------------------------------------------------------


class SampleLabelUpdate(BaseModel):
    label: str


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict[str, Any]:
    checks: dict[str, str] = {}
    try:
        async with request.app.state.store.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error:{type(exc).__name__}"

    redis_pool = request.app.state.redis_pool
    if redis_pool is None:
        checks["redis"] = "unavailable"
        checks["worker"] = "unknown"
    else:
        try:
            await redis_pool.ping()
            checks["redis"] = "ok"
            checks["worker"] = (
                "ok" if await redis_pool.get(WORKER_HEARTBEAT_KEY) else "stale"
            )
        except Exception as exc:
            checks["redis"] = f"error:{type(exc).__name__}"
            checks["worker"] = "unknown"

    try:
        taxonomy = request.app.state.taxonomy_loader.get_tree()
        checks["vertical"] = "ok" if taxonomy.node_count() else "empty"
    except Exception as exc:
        checks["vertical"] = f"error:{type(exc).__name__}"

    ready = all(value == "ok" for value in checks.values())
    payload: dict[str, Any] = {
        "status": "ready" if ready else "not_ready",
        "environment": request.app.state.settings.environment,
        "checks": checks,
    }
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=payload
        )
    return payload


@app.post(
    "/api/v1/runs",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_run(
    payload: CreateRunRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_submitter),
) -> RunResponse:
    """提交邮件 → 立即创建 PENDING run + 入队 arq 任务 → 返回 run_id。

    Worker 后台调用 ClassifyAgent，完成后 status=COMPLETED + classification 字段写入。
    客户端轮询 GET /api/v1/runs/{run_id} 检查状态。
    """

    return await service(request).create_run(payload, actor_id=principal.subject)


@app.get("/api/v1/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: UUID,
    request: Request,
    _: ApiPrincipal = Depends(require_submitter),
) -> RunResponse:
    """查询 run：含完整 classification 字段（如 Worker 已处理完成）。

    status 可能值：pending / processing / completed / waiting_approval / rejected / failed
    """

    run = await service(request).get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.post(
    "/api/v1/runs/{run_id}/classification-feedback",
    response_model=ClassificationFeedback,
    status_code=status.HTTP_201_CREATED,
)
async def record_classification_feedback(
    run_id: UUID,
    payload: ClassificationFeedbackRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_reviewer),
) -> ClassificationFeedback:
    _require_feedback_enabled(request)
    reviewer_id = _trusted_reviewer_id(request, principal)
    taxonomy = request.app.state.taxonomy_loader.get_tree()
    valid_labels = taxonomy.all_codes()
    exclusive_labels = {node.code for node in taxonomy.nodes if node.exclusive}
    try:
        feedback = await service(request).record_classification_feedback(
            run_id,
            payload,
            valid_labels=valid_labels,
            exclusive_labels=exclusive_labels,
            reviewer_id=reviewer_id,
        )
    except ValueError as exc:
        if str(exc) == "run has no classification":
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if feedback is None:
        raise HTTPException(status_code=404, detail="run not found")
    return feedback


@app.get(
    "/api/v1/runs/{run_id}/classification-feedback",
    response_model=list[ClassificationFeedback],
)
async def list_classification_feedback(
    run_id: UUID,
    request: Request,
    _: ApiPrincipal = Depends(require_reviewer),
) -> list[ClassificationFeedback]:
    _require_feedback_enabled(request)
    feedback = await service(request).list_classification_feedback(run_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail="run not found")
    return feedback


@app.post("/api/v1/runs/{run_id}/approve", response_model=RunResponse)
async def approve_run(
    run_id: UUID,
    _: ApprovalRequest,
    request: Request,
    principal: ApiPrincipal = Depends(require_reviewer),
) -> RunResponse:
    try:
        run = await service(request).approve(run_id, actor_id=principal.subject)
    except InvalidRunTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.post("/api/v1/runs/{run_id}/reject", response_model=RunResponse)
async def reject_run(
    run_id: UUID,
    request: Request,
    principal: ApiPrincipal = Depends(require_reviewer),
) -> RunResponse:
    try:
        run = await service(request).reject(run_id, actor_id=principal.subject)
    except InvalidRunTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.post("/api/v1/runs/{run_id}/retry", response_model=RunResponse)
async def retry_run(
    run_id: UUID,
    request: Request,
    principal: ApiPrincipal = Depends(require_operator),
) -> RunResponse:
    try:
        run = await service(request).retry(run_id, actor_id=principal.subject)
    except InvalidRunTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/api/v1/skills", response_model=list[SkillVersion])
async def list_skills(
    request: Request,
    _: ApiPrincipal = Depends(require_submitter),
) -> list[SkillVersion]:
    return await store(request).list_skills()


@app.post(
    "/api/v1/skills", response_model=SkillVersion, status_code=status.HTTP_201_CREATED
)
async def create_skill(
    payload: SkillDefinition,
    request: Request,
    _: ApiPrincipal = Depends(require_admin),
) -> SkillVersion:
    return await store(request).create_skill(payload)


@app.post("/api/v1/skills/{skill_version_id}/publish", response_model=SkillVersion)
async def publish_skill(
    skill_version_id: UUID,
    request: Request,
    _: ApiPrincipal = Depends(require_admin),
) -> SkillVersion:
    skill = await store(request).publish_skill(skill_version_id)
    if not skill:
        raise HTTPException(status_code=404, detail="skill version not found")
    return skill


# ---------------------------------------------------------------------------
# Section 17: bootstrap / samples / clustering API extensions
# ---------------------------------------------------------------------------


@app.get("/api/v1/bootstrap/report/{report_id}")
async def bootstrap_report(
    report_id: str,
    request: Request,
    _: ApiPrincipal = Depends(require_reviewer),
) -> dict[str, Any]:
    """Return the markdown content and pending sample list for a bootstrap report.

    Reads ``bootstrap_{report_id}.md`` and ``bootstrap_{report_id}.json`` from
    the configured reports directory.
    """

    settings: Settings = request.app.state.settings
    reports_dir = Path(settings.bootstrap.reports_dir)
    md_path = reports_dir / f"bootstrap_{report_id}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    markdown = md_path.read_text(encoding="utf-8")

    pending_samples: list[dict[str, Any]] = []
    json_path = reports_dir / f"bootstrap_{report_id}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        pending_samples = data.get("samples", [])

    return {
        "report_id": report_id,
        "markdown": markdown,
        "pending_samples": pending_samples,
    }


@app.get("/api/v1/samples", response_model=list[SampleRecord])
async def list_samples(
    label: str | None = None,
    source: str | None = None,
    page: int = 1,
    vs: VectorStore | None = Depends(get_vector_store),
    _: ApiPrincipal = Depends(require_reviewer),
) -> list[SampleRecord]:
    """List labeled samples with optional label/source filters and pagination."""

    if vs is None:
        return []
    return await vs.get_samples(label=label, source=source, page=page)


@app.get("/api/v1/samples/{sample_id}", response_model=SampleRecord)
async def get_sample(
    sample_id: UUID,
    vs: VectorStore | None = Depends(get_vector_store),
    _: ApiPrincipal = Depends(require_reviewer),
) -> SampleRecord:
    """Return a single sample by id."""

    if vs is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="vector store not configured",
        )
    sample = await vs.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="sample not found")
    return sample


@app.delete("/api/v1/samples/{sample_id}")
async def delete_sample(
    sample_id: UUID,
    vs: VectorStore | None = Depends(get_vector_store),
    _: ApiPrincipal = Depends(require_admin),
) -> dict[str, str]:
    """Delete a sample by id."""

    if vs is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="vector store not configured",
        )
    await vs.delete_sample(sample_id)
    return {"deleted": str(sample_id)}


@app.patch("/api/v1/samples/{sample_id}", response_model=SampleRecord)
async def update_sample_label(
    sample_id: UUID,
    payload: SampleLabelUpdate,
    vs: VectorStore | None = Depends(get_vector_store),
    _: ApiPrincipal = Depends(require_admin),
) -> SampleRecord:
    """Update a sample's leaf label (label_l3) and mark it reviewed."""

    if vs is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="vector store not configured",
        )
    await vs.update_sample_label(sample_id, payload.label)
    sample = await vs.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="sample not found")
    return sample


@app.get("/api/v1/clustering/report")
async def clustering_report(
    request: Request,
    _: ApiPrincipal = Depends(require_reviewer),
) -> dict[str, str]:
    """Return the most recent intent-discovery clustering report.

    Scans the reports directory for ``intent_discovery_*.md`` files and
    returns the latest one (sorted by filename descending).
    """

    settings: Settings = request.app.state.settings
    reports_dir = Path(settings.bootstrap.reports_dir)
    if not reports_dir.exists():
        raise HTTPException(status_code=404, detail="no clustering reports found")
    candidates = sorted(reports_dir.glob("intent_discovery_*.md"), reverse=True)
    if not candidates:
        raise HTTPException(status_code=404, detail="no clustering reports found")
    latest = candidates[0]
    markdown = latest.read_text(encoding="utf-8")
    return {"report_path": str(latest), "markdown": markdown}
