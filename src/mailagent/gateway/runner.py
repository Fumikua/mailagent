from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from ..domain.mail_parser import dedup_key_for_message, parse_email_message
from ..domain.models import CreateRunRequest
from .base import FetchCursor

logger = logging.getLogger(__name__)


def _eligible(event, settings) -> bool:
    sender = event.sender.rsplit("@", 1)[-1].lower()
    recipients = [item.rsplit("@", 1)[-1].lower() for item in event.recipients]
    if settings.sender_domain_allowlist and sender not in settings.sender_domain_allowlist:
        return False
    if settings.recipient_domain_allowlist and not set(recipients).intersection(settings.recipient_domain_allowlist):
        return False
    return not settings.subject_patterns or any(pattern in event.subject for pattern in settings.subject_patterns)


def _build_adapter(settings):
    """Construct the appropriate gateway adapter for the given settings."""
    if settings.adapter == "pop3":
        from .pop3_adapter import Pop3Adapter

        return Pop3Adapter(settings)
    from .imap_adapter import ImapAdapter

    return ImapAdapter(settings)


async def _pop3_initial_sync(settings, gateway, state) -> str:
    """POP3 first poll: LIST all, RETR top N, mark skipped_initial, advance cursor.

    Builds the dedup baseline so subsequent polls only process new mail.
    """
    cursor = FetchCursor(
        uidvalidity=None,
        after_uid=None,
        since=None,
        batch_size=settings.initial_sync_max_messages,
        protocol="pop3",
    )
    count = 0
    last_uid = 0
    async for item in gateway.fetch(cursor):
        if item.server_id is None:
            raise RuntimeError("POP3 message missing stable UIDL server_id")
        event = parse_email_message(item.raw_bytes, settings.mailbox_id)
        key = dedup_key_for_message(item.raw_bytes, event.message_id)
        inserted = await state.insert_skipped_initial(
            settings.mailbox_id,
            key,
            item.uid,
            protocol="pop3",
            server_id=item.server_id,
        )
        if inserted:
            count += 1
        if item.uid > last_uid:
            last_uid = item.uid
    # Persist an explicit zero cursor for an empty mailbox. Otherwise the first
    # future message would be mistaken for baseline mail and skipped.
    await state.advance_cursor(
        settings.mailbox_id, None, last_uid, protocol="pop3"
    )
    logger.info(
        "pop3 incremental baseline: mailbox_id=%s, baseline=%d, cursor_uid=%d",
        settings.mailbox_id,
        count,
        last_uid,
    )
    return f"initialized: incremental ({count} messages)"


async def _poll_single_gateway(settings, state, service, redis) -> str:
    """Poll one gateway: acquire lock, connect, fetch, ingest, advance cursor."""
    gateway = _build_adapter(settings)
    protocol = settings.adapter  # "imap" or "pop3"
    lock_key = f"mail_gateway:{settings.mailbox_id}"
    lock_acquired = False
    if redis is not None:
        lock_acquired = bool(
            await redis.set(
                lock_key,
                "1",
                nx=True,
                ex=max(settings.poll_interval_seconds * 2, 60),
            )
        )
        if not lock_acquired:
            return "skipped_locked"
    try:
        await gateway.connect()
        previous = await state.get_cursor(settings.mailbox_id)

        if protocol == "pop3":
            # POP3: no UIDVALIDITY check; first sync builds incremental baseline
            if previous is None:
                return await _pop3_initial_sync(settings, gateway, state)
            seen_server_ids = await state.get_server_ids(
                settings.mailbox_id, protocol="pop3"
            )
            cursor = FetchCursor(
                uidvalidity=None,
                after_uid=previous.high_water_uid,
                since=None,
                batch_size=settings.fetch_batch_size,
                protocol="pop3",
                seen_server_ids=frozenset(seen_server_ids),
            )
        else:
            # IMAP path: UIDVALIDITY change detection + from_now / backfill
            if previous is not None and previous.uidvalidity != gateway.uidvalidity:
                logger.info(
                    "mail gateway UIDVALIDITY changed: mailbox_id=%s old=%s new=%s",
                    settings.mailbox_id,
                    previous.uidvalidity,
                    gateway.uidvalidity,
                )
                await state.reset_cursor(settings.mailbox_id)
                previous = None
            if previous is None and settings.initial_sync_mode == "from_now":
                await state.advance_cursor(settings.mailbox_id, gateway.uidvalidity, await gateway.highest_uid())
                return "initialized: from_now"
            cursor = FetchCursor(
                uidvalidity=previous.uidvalidity if previous else None,
                after_uid=previous.high_water_uid if previous else None,
                since=(datetime.now(timezone.utc) - timedelta(days=settings.backfill_since_days)) if previous is None else None,
                batch_size=min(
                    settings.fetch_batch_size,
                    settings.backfill_max_messages
                    if previous is None and settings.initial_sync_mode == "bounded_backfill"
                    else settings.fetch_batch_size,
                ),
                protocol="imap",
            )
        accepted = 0
        async for item in gateway.fetch(cursor):
            server_id = getattr(item, "server_id", None)
            if protocol == "pop3" and server_id is None:
                raise RuntimeError("POP3 message missing stable UIDL server_id")
            if len(item.raw_bytes) > settings.max_message_bytes:
                key = dedup_key_for_message(item.raw_bytes, None)
                await state.claim(
                    settings.mailbox_id,
                    key,
                    item.uidvalidity,
                    item.uid,
                    protocol=protocol,
                    server_id=server_id,
                )
                await state.terminal(settings.mailbox_id, key, "skipped_oversize")
                await state.advance_cursor(settings.mailbox_id, item.uidvalidity, item.uid, protocol=protocol)
                continue
            event = parse_email_message(item.raw_bytes, settings.mailbox_id)
            key = dedup_key_for_message(item.raw_bytes, event.message_id)
            claimed = await state.claim(
                settings.mailbox_id,
                key,
                item.uidvalidity,
                item.uid,
                protocol=protocol,
                server_id=server_id,
            )
            existing = None if claimed else await state.get_claim(settings.mailbox_id, key)
            if not claimed and (existing is None or existing.status != "claimed"):
                await state.advance_cursor(settings.mailbox_id, item.uidvalidity, item.uid, protocol=protocol)
                continue
            if not _eligible(event, settings):
                await state.terminal(settings.mailbox_id, key, "skipped_filtered")
                await state.advance_cursor(settings.mailbox_id, item.uidvalidity, item.uid, protocol=protocol)
                continue
            if existing is not None and existing.run_id:
                run_id = UUID(existing.run_id)
            else:
                run = await service.create_run(CreateRunRequest(email=event), enqueue=False)
                run_id = run.id
                await state.attach_run(settings.mailbox_id, key, run_id)
            # A failure here deliberately leaves a claimed row with run_id and
            # the old cursor.  The next poll re-enqueues that exact run.
            await service.enqueue_run(run_id)
            await state.accept(settings.mailbox_id, key, run_id)
            await state.advance_cursor(settings.mailbox_id, item.uidvalidity, item.uid, protocol=protocol)
            accepted += 1
        return f"accepted: {accepted}"
    finally:
        try:
            await gateway.close()
        finally:
            if lock_acquired and redis is not None:
                await redis.delete(lock_key)


async def mail_poll_job(ctx: dict) -> str:
    """Poll all enabled mail gateways configured in ``settings.mail_gateways``.

    Each gateway is polled sequentially with its own Redis lock and cursor.
    A failure in one gateway does not block the others — the error is logged
    and the remaining gateways are still polled.
    """
    settings_list = ctx["settings"].mail_gateways
    state = ctx["mail_gateway_state"]
    service = ctx["service"]
    redis = ctx.get("redis")
    results: list[str] = []
    for gw_settings in settings_list:
        if not gw_settings.enabled:
            continue
        try:
            result = await _poll_single_gateway(gw_settings, state, service, redis)
            results.append(f"{gw_settings.mailbox_id}: {result}")
        except Exception:
            logger.exception(
                "mail gateway poll failed: mailbox_id=%s, adapter=%s",
                gw_settings.mailbox_id,
                gw_settings.adapter,
            )
            results.append(f"{gw_settings.mailbox_id}: error")
    return "; ".join(results) if results else "no_enabled_gateways"
