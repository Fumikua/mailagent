from __future__ import annotations

import asyncio
import logging
import os
import poplib
from collections.abc import AsyncIterator

from .base import FetchCursor, FetchedMessage

logger = logging.getLogger(__name__)


class Pop3Adapter:
    """POP3 over TLS adapter implementing the MailGateway Protocol.

    Uses stdlib ``poplib.POP3_SSL`` wrapped with ``asyncio.to_thread`` so the
    synchronous POP3 calls don't block the event loop.  Designed for PoC:
    no ``DELE`` (messages stay on server), single INBOX, no IDLE.
    """

    def __init__(self, settings) -> None:
        self.settings = settings
        self.client: poplib.POP3_SSL | None = None
        self.uidvalidity: int | None = None  # POP3 has no UIDVALIDITY

    async def connect(self) -> None:
        password = os.environ[self.settings.password_env]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                client = await asyncio.to_thread(
                    poplib.POP3_SSL, self.settings.host, self.settings.port
                )
                self.client = client
                await asyncio.to_thread(client.user, self.settings.username)
                await asyncio.to_thread(client.pass_, password)
                return
            except Exception as exc:
                last_error = exc
                self.client = None
                if attempt < 2:
                    await asyncio.sleep(0.1 * (attempt + 1))
        raise RuntimeError(
            f"POP3 connection failed: host={self.settings.host}, "
            f"password_env={self.settings.password_env}: {last_error}"
        ) from last_error

    async def fetch(self, cursor: FetchCursor) -> AsyncIterator[FetchedMessage]:
        assert self.client is not None
        # LIST returns (response, [b'1 1234', b'2 5678', ...], octets)
        resp = await asyncio.to_thread(self.client.list)
        msg_numbers: list[int] = []
        for line in resp[1]:
            if isinstance(line, bytes):
                parts = line.split()
                if parts and parts[0].isdigit():
                    msg_numbers.append(int(parts[0]))
        msg_numbers.sort()

        uidl_resp = await asyncio.to_thread(self.client.uidl)
        server_ids: dict[int, str] = {}
        for line in uidl_resp[1]:
            if not isinstance(line, bytes):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            server_ids[int(parts[0])] = parts[1].decode("utf-8", errors="strict")
        missing_uidls = [number for number in msg_numbers if number not in server_ids]
        if missing_uidls:
            raise RuntimeError(
                f"POP3 UIDL response missing message numbers: {missing_uidls[:5]}"
            )

        if cursor.after_uid is None:
            # First sync: take the top N messages by message number
            selected = msg_numbers[-cursor.batch_size :] if msg_numbers else []
            logger.info(
                "pop3 first-sync: total=%d, selecting top %d",
                len(msg_numbers),
                len(selected),
            )
        else:
            # Message numbers can be reassigned after server-side deletion.
            # Stable UIDLs, not positions, determine which messages are new.
            selected = [
                number
                for number in msg_numbers
                if server_ids[number] not in cursor.seen_server_ids
            ][: cursor.batch_size]

        for msg_no in selected:
            retr_resp = await asyncio.to_thread(self.client.retr, msg_no)
            # retr_resp[1] is a list of bytes lines without CRLF; reconstruct raw RFC822
            raw = b"\r\n".join(retr_resp[1]) + b"\r\n"
            yield FetchedMessage(
                uid=msg_no,
                uidvalidity=None,
                raw_bytes=raw,
                server_id=server_ids[msg_no],
            )

    async def close(self) -> None:
        if self.client is not None:
            try:
                await asyncio.to_thread(self.client.quit)
            except Exception:
                logger.debug("pop3 close error (ignored)", exc_info=True)
            finally:
                self.client = None
