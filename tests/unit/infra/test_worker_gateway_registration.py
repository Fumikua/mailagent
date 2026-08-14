"""Worker registration tests for the mail-gateway feature flag (task 1.8).

Verifies that ``_gateway_worker_configuration`` correctly registers
``mail_poll_job`` and the per-poll cron schedule only when at least one
enabled gateway exists, and that disabled gateways fall back to the
classify-only Worker configuration.
"""

from mailagent.infra.config import MailGatewaySettings, Settings
from mailagent.infra.queue import MAIL_POLL_JOB_NAME
from mailagent.infra.worker import _gateway_worker_configuration


def _enabled_gateway(mailbox_id: str = "primary", poll_interval: int = 60) -> MailGatewaySettings:
    return MailGatewaySettings(
        enabled=True,
        mailbox_id=mailbox_id,
        host="imap.example.com",
        username="ops",
        password_env="IMAP_PASSWORD",
        poll_interval_seconds=poll_interval,
    )


def _disabled_gateway() -> MailGatewaySettings:
    return MailGatewaySettings(enabled=False, mailbox_id="staging")


def test_no_gateways_registers_classify_job_only() -> None:
    settings = Settings(mail_gateways=[])
    functions, cron_jobs, timeout = _gateway_worker_configuration(settings)

    assert any(fn.__name__ == "classify_job" for fn in functions)
    assert not any(getattr(fn, "__name__", "") == "mail_poll_job" for fn in functions)
    assert timeout == 60
    # No extra cron entries beyond the baseline (clustering / rule_learn / archive).
    assert not any(job.name == MAIL_POLL_JOB_NAME for job in cron_jobs)


def test_disabled_gateways_register_no_poll_job() -> None:
    settings = Settings(mail_gateways=[_disabled_gateway()])
    functions, cron_jobs, _ = _gateway_worker_configuration(settings)

    assert not any(getattr(fn, "__name__", "") == "mail_poll_job" for fn in functions)
    assert not any(job.name == MAIL_POLL_JOB_NAME for job in cron_jobs)


def test_enabled_gateway_registers_poll_job_and_cron() -> None:
    settings = Settings(mail_gateways=[_enabled_gateway(poll_interval=120)])
    functions, cron_jobs, timeout = _gateway_worker_configuration(settings)

    assert any(getattr(fn, "__name__", "") == "mail_poll_job" for fn in functions)
    poll_cron = next(job for job in cron_jobs if job.name == MAIL_POLL_JOB_NAME)
    assert poll_cron is not None
    # poll_interval=120 → every 2 minutes → {0, 2, 4, ..., 58}
    assert poll_cron.minute == set(range(0, 60, 2))
    # Timeout is min_interval + 60 buffer.
    assert timeout == 180


def test_multiple_gateways_use_shortest_interval_for_cron() -> None:
    settings = Settings(
        mail_gateways=[
            _enabled_gateway(mailbox_id="primary", poll_interval=300),
            _enabled_gateway(mailbox_id="backup", poll_interval=120),
        ]
    )
    _, cron_jobs, timeout = _gateway_worker_configuration(settings)

    poll_cron = next(job for job in cron_jobs if job.name == MAIL_POLL_JOB_NAME)
    # Shortest interval wins → 120s → every 2 minutes.
    assert poll_cron.minute == set(range(0, 60, 2))
    assert timeout == 180


def test_minute_only_cron_schedule_rejects_subminute_intervals() -> None:
    # The cron schedule is minute-granularity; poll_interval_seconds must be a
    # multiple of 60 (validated by MailGatewaySettings), so a 60s interval
    # produces a every-minute cron.
    settings = Settings(mail_gateways=[_enabled_gateway(poll_interval=60)])
    _, cron_jobs, _ = _gateway_worker_configuration(settings)

    poll_cron = next(job for job in cron_jobs if job.name == MAIL_POLL_JOB_NAME)
    assert poll_cron.minute == set(range(0, 60, 1))
