from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass(frozen=True)
class MailboxCursor:
    mailbox_id: str
    uidvalidity: int | None
    high_water_uid: int
    protocol: str = "imap"


@dataclass(frozen=True)
class IngestClaim:
    mailbox_id: str
    dedup_key: str
    uidvalidity: int | None
    uid: int
    run_id: str | None
    status: str
    protocol: str = "imap"
    server_id: str | None = None


class MailGatewayStateStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def get_cursor(self, mailbox_id: str) -> MailboxCursor | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(text("SELECT mailbox_id, uidvalidity, high_water_uid, protocol FROM mail_gateway_cursor WHERE mailbox_id=:id"), {"id": mailbox_id})).mappings().first()
        return MailboxCursor(**row) if row else None

    async def reset_cursor(self, mailbox_id: str) -> None:
        """Forget only the mailbox position after a UIDVALIDITY change."""

        async with self.engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM mail_gateway_cursor WHERE mailbox_id=:id"),
                {"id": mailbox_id},
            )

    async def get_claim(self, mailbox_id: str, dedup_key: str) -> IngestClaim | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT mailbox_id,dedup_key,uidvalidity,uid,run_id,status,protocol,server_id "
                        "FROM mail_gateway_ingest_ledger "
                        "WHERE mailbox_id=:m AND dedup_key=:d"
                    ),
                    {"m": mailbox_id, "d": dedup_key},
                )
            ).mappings().first()
        return IngestClaim(**row) if row else None

    async def record_backfill_audit(
        self, mailbox_id: str, *, since_days: int, max_messages: int
    ) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO mail_gateway_backfill_audit "
                    "(mailbox_id,since_days,max_messages,created_at) "
                    "VALUES (:m,:d,:n,:created_at)"
                ),
                {
                    "m": mailbox_id,
                    "d": since_days,
                    "n": max_messages,
                    "created_at": datetime.now(timezone.utc),
                },
            )

    async def claim(
        self,
        mailbox_id: str,
        dedup_key: str,
        uidvalidity: int | None,
        uid: int,
        protocol: str = "imap",
        server_id: str | None = None,
    ) -> bool:
        now = datetime.now(timezone.utc)
        async with self.engine.begin() as conn:
            result = await conn.execute(
                text(
                    "INSERT INTO mail_gateway_ingest_ledger "
                    "(mailbox_id,dedup_key,uidvalidity,uid,status,protocol,server_id,created_at,updated_at) "
                    "VALUES (:m,:d,:v,:u,'claimed',:p,:server_id,:n,:n) "
                    "ON CONFLICT (mailbox_id,dedup_key) DO NOTHING"
                ),
                {
                    "m": mailbox_id,
                    "d": dedup_key,
                    "v": uidvalidity,
                    "u": uid,
                    "p": protocol,
                    "server_id": server_id,
                    "n": now,
                },
            )
        return result.rowcount == 1

    async def insert_skipped_initial(
        self,
        mailbox_id: str,
        dedup_key: str,
        uid: int,
        protocol: str = "pop3",
        server_id: str | None = None,
    ) -> bool:
        """Insert a ledger row with status='skipped_initial' for POP3 first-sync baseline.

        Returns True if inserted, False if already existed (dedup_key conflict).
        """
        now = datetime.now(timezone.utc)
        async with self.engine.begin() as conn:
            result = await conn.execute(
                text(
                    "INSERT INTO mail_gateway_ingest_ledger "
                    "(mailbox_id,dedup_key,uidvalidity,uid,run_id,status,protocol,server_id,created_at,updated_at) "
                    "VALUES (:m,:d,NULL,:u,NULL,'skipped_initial',:p,:server_id,:n,:n) "
                    "ON CONFLICT (mailbox_id,dedup_key) DO NOTHING"
                ),
                {
                    "m": mailbox_id,
                    "d": dedup_key,
                    "u": uid,
                    "p": protocol,
                    "server_id": server_id,
                    "n": now,
                },
            )
        return result.rowcount == 1

    async def get_server_ids(self, mailbox_id: str, protocol: str) -> set[str]:
        """Return durable server identities already observed for one mailbox."""

        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT server_id FROM mail_gateway_ingest_ledger "
                        "WHERE mailbox_id=:m AND protocol=:p AND server_id IS NOT NULL"
                    ),
                    {"m": mailbox_id, "p": protocol},
                )
            ).scalars()
            return set(rows)

    async def advance_cursor(self, mailbox_id: str, uidvalidity: int | None, uid: int, protocol: str = "imap") -> None:
        now = datetime.now(timezone.utc)
        async with self.engine.begin() as conn:
            await conn.execute(text("INSERT INTO mail_gateway_cursor (mailbox_id,uidvalidity,high_water_uid,protocol,updated_at) VALUES (:m,:v,:u,:p,:n) ON CONFLICT (mailbox_id) DO UPDATE SET uidvalidity=excluded.uidvalidity, high_water_uid=excluded.high_water_uid, protocol=excluded.protocol, updated_at=excluded.updated_at"), {"m": mailbox_id, "v": uidvalidity, "u": uid, "p": protocol, "n": now})

    async def accept(self, mailbox_id: str, dedup_key: str, run_id: UUID | str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("UPDATE mail_gateway_ingest_ledger SET run_id=:r,status='accepted',updated_at=:n WHERE mailbox_id=:m AND dedup_key=:d"), {"m": mailbox_id, "d": dedup_key, "r": str(run_id), "n": datetime.now(timezone.utc)})

    async def attach_run(self, mailbox_id: str, dedup_key: str, run_id: UUID) -> None:
        """Persist the run id before enqueueing so a failed enqueue is resumable."""

        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE mail_gateway_ingest_ledger SET run_id=:r,updated_at=:n "
                    "WHERE mailbox_id=:m AND dedup_key=:d AND status='claimed'"
                ),
                {"m": mailbox_id, "d": dedup_key, "r": str(run_id), "n": datetime.now(timezone.utc)},
            )

    async def terminal(self, mailbox_id: str, dedup_key: str, status: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("UPDATE mail_gateway_ingest_ledger SET status=:s,updated_at=:n WHERE mailbox_id=:m AND dedup_key=:d"), {"m": mailbox_id, "d": dedup_key, "s": status, "n": datetime.now(timezone.utc)})
