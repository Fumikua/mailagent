"""Explicit, bounded operational entry point for IMAP historical backfill."""
from __future__ import annotations

import logging
from typing import Any

from .runner import mail_poll_job

logger = logging.getLogger(__name__)


async def run_confirmed_backfill(
    ctx: dict[str, Any],
    *,
    confirmed: bool,
    since_days: int,
    max_messages: int,
    mailbox_id: str | None = None,
) -> str:
    """Run one human-confirmed bounded backfill poll.

    This function is deliberately not scheduled.  Callers must explicitly
    provide confirmation; the normal configuration caps protect routine polls.
    """

    if not confirmed:
        raise ValueError("manual IMAP backfill requires confirmed=True")
    if since_days < 1 or max_messages < 1:
        raise ValueError("backfill range and message limit must be positive")

    imap_gateways = [
        gateway
        for gateway in ctx["settings"].mail_gateways
        if gateway.enabled and gateway.adapter == "imap"
    ]
    if mailbox_id is not None:
        matching = [
            gateway for gateway in imap_gateways if gateway.mailbox_id == mailbox_id
        ]
        if not matching:
            raise ValueError(f"enabled IMAP gateway not found: {mailbox_id}")
        settings = matching[0]
    elif len(imap_gateways) == 1:
        settings = imap_gateways[0]
    else:
        raise ValueError(
            "manual IMAP backfill requires mailbox_id when there is not exactly one enabled IMAP gateway"
        )

    override = settings.model_copy(
        update={
            "initial_sync_mode": "bounded_backfill",
            "initial_backfill_confirmed": True,
            "backfill_since_days": since_days,
            "backfill_max_messages": max_messages,
        }
    )
    cloned = dict(ctx)
    cloned["settings"] = ctx["settings"].model_copy(
        update={"mail_gateways": [override]}
    )
    await ctx["mail_gateway_state"].record_backfill_audit(
        settings.mailbox_id, since_days=since_days, max_messages=max_messages
    )
    logger.info(
        "manual IMAP backfill requested: mailbox_id=%s since_days=%s max_messages=%s",
        settings.mailbox_id,
        since_days,
        max_messages,
    )
    return await mail_poll_job(cloned)
