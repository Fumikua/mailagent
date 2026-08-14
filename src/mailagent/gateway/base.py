from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol


@dataclass(frozen=True)
class FetchCursor:
    uidvalidity: int | None
    after_uid: int | None
    since: datetime | None
    batch_size: int
    protocol: Literal["imap", "pop3"] = "imap"
    seen_server_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FetchedMessage:
    uid: int
    uidvalidity: int | None
    raw_bytes: bytes
    flags: tuple[str, ...] = ()
    server_id: str | None = None


class MailGateway(Protocol):
    async def connect(self) -> None: ...
    async def fetch(self, cursor: FetchCursor) -> AsyncIterator[FetchedMessage]: ...
    async def close(self) -> None: ...
