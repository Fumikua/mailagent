"""Worker entry point — arq Worker 处理邮件分类任务。

启动方式：
    python -m mailagent.infra.worker
或：
    arq mailagent.infra.worker.WorkerSettings

Worker 启动时：
1. 加载 Settings.from_yaml()
2. 创建 SqlStore + ClassifyAgent + MailProcessingService
3. 订阅 classify 队列，处理 LLM 调用
4. 完成后更新 store status=COMPLETED + classification 字段
"""
from __future__ import annotations

import logging

from .config import Settings
from .queue import (
    classify_job,
    cron_jobs,
    MAIL_POLL_JOB_NAME,
    redis_settings_from_url,
    worker_on_shutdown,
    worker_on_startup,
)
from arq import cron
from ..gateway.runner import mail_poll_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_settings() -> Settings:
    return Settings.from_yaml()


def _gateway_worker_configuration(settings: Settings):
    """Return Worker functions, cron jobs and a timeout for one poll cycle."""

    enabled_gateways = [gw for gw in settings.mail_gateways if gw.enabled]
    if not enabled_gateways:
        return [classify_job], cron_jobs, 60
    # Use the shortest interval among enabled gateways as the cron frequency
    min_interval = min(gw.poll_interval_seconds for gw in enabled_gateways)
    interval_minutes = min_interval // 60
    poll_cron = cron(
        mail_poll_job,
        name=MAIL_POLL_JOB_NAME,
        minute=set(range(0, 60, interval_minutes)),
    )
    return (
        [classify_job, mail_poll_job],
        [*cron_jobs, poll_cron],
        min_interval + 60,
    )


class WorkerSettings:
    """arq Worker 配置（模块级，供 arq CLI 加载）。

    注：redis_settings 在 import 时根据 config.yml 解析。
    """

    _settings = _load_settings()
    functions, cron_jobs, job_timeout = _gateway_worker_configuration(_settings)
    max_retries = 1
    on_startup = worker_on_startup
    on_shutdown = worker_on_shutdown
    redis_settings = redis_settings_from_url(_load_settings().redis.url)


def main() -> None:
    """命令行入口：从 config.yml 加载 redis 配置后启动 arq Worker。"""

    from arq import run_worker

    # 重新读取 settings，确保最新（CLI 启动场景）
    WorkerSettings.redis_settings = redis_settings_from_url(_load_settings().redis.url)
    run_worker(WorkerSettings)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
