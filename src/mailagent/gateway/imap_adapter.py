from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os
import re

from .base import FetchCursor, FetchedMessage


class ImapAdapter:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.client = None
        self.uidvalidity = 0

    async def connect(self) -> None:
        import aioimaplib  # type: ignore[import-untyped]

        password = os.environ[self.settings.password_env]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                client = aioimaplib.IMAP4_SSL(self.settings.host, self.settings.port)
                self.client = client
                await client.wait_hello_from_server()
                await client.login(self.settings.username, password)
                response = await client.examine(self.settings.mailbox)
                match = next(
                    (re.search(rb"UIDVALIDITY\s+(\d+)", line) for line in response.lines if b"UIDVALIDITY" in line),
                    None,
                )
                if match is None:
                    raise RuntimeError("IMAP SELECT response did not include UIDVALIDITY")
                self.uidvalidity = int(match.group(1))
                return
            except Exception as exc:
                last_error = exc
                self.client = None
                if attempt < 2:
                    await asyncio.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"IMAP connection failed: {last_error}") from last_error

    async def highest_uid(self) -> int:
        assert self.client is not None
        response = await self.client.uid("search", "ALL")
        values = [int(value) for line in response.lines for value in line.split() if value.isdigit()]
        return max(values, default=0)

    async def fetch(self, cursor: FetchCursor) -> AsyncIterator[FetchedMessage]:
        assert self.client is not None
        if cursor.after_uid is not None:
            criteria = [f"UID {cursor.after_uid + 1}:*"]
        elif cursor.since is not None:
            criteria = ["SINCE " + cursor.since.strftime("%d-%b-%Y")]
        else:
            criteria = ["ALL"]
        if self.settings.seen_filter == "unseen":
            criteria.append("UNSEEN")
        response = await self.client.uid("search", *criteria)
        uids = [int(value) for line in response.lines for value in line.split() if value.isdigit()]
        for uid in uids[: cursor.batch_size]:
            fetched = await self.client.uid("fetch", str(uid), "(UID FLAGS BODY.PEEK[])")
            raw = next((line for line in fetched.lines if isinstance(line, bytes) and b"FETCH" not in line), b"")
            yield FetchedMessage(uid=uid, uidvalidity=self.uidvalidity, raw_bytes=raw)

    async def close(self) -> None:
        if self.client is not None:
            try:
                unselect = getattr(self.client, "unselect", None)
                if unselect is not None:
                    await unselect()
                await self.client.logout()
            finally:
                self.client = None
