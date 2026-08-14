"""End-to-end IMAP gateway integration tests (mail-gateway-imap tasks 6.1-6.3).

Drives the real ``mail_poll_job`` → ``MailGatewayStateStore`` (SQLite) →
``RunService`` path with a stateful in-memory gateway stub. The stub
mimics the IMAP adapter contract (connect / fetch / highest_uid / close)
without touching the network; ``ImapAdapter`` itself is covered by the
unit tests in ``tests/unit/gateway/test_imap_adapter.py``.

The stub is stateful across polls so the tests can simulate new-mail
arrival, UIDVALIDITY changes, and redelivery between polls.
"""

from __future__ import annotations

from email.message import EmailMessage
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from mailagent.gateway import runner as gateway_runner
from mailagent.gateway.base import FetchCursor
from mailagent.gateway.runner import mail_poll_job
from mailagent.gateway.state import MailGatewayStateStore
from mailagent.infra.config import MailGatewaySettings, Settings


def _rfc822(
    *,
    message_id: str,
    sender: str = "ops@example.com",
    to: str = "agent@exampleco.com.cn",
    subject: str = "STATUS update",
    body: str = "Entity STATUS 2026-08-01 08:00 UTC.",
) -> bytes:
    msg = EmailMessage()
    msg["Message-ID"] = message_id
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg["Date"] = "Mon, 1 Jul 2026 08:00:00 +0000"
    msg.set_content(body)
    return bytes(msg)


class _StatefulImapGateway:
    """Stateful in-memory IMAP adapter stub.

    Holds a UID → raw-bytes map and a UIDVALIDITY value. Each poll sees the
    current state, so tests can mutate the map between polls to simulate
    new mail, redelivery, or UIDVALIDITY changes.
    """

    def __init__(self) -> None:
        self.messages: dict[int, bytes] = {}
        self.uidvalidity: int = 100
        self.connected: bool = False

    def add_message(self, uid: int, raw: bytes) -> None:
        self.messages[uid] = raw

    def change_uidvalidity(self, new_value: int) -> None:
        self.uidvalidity = new_value

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def highest_uid(self) -> int:
        return max(self.messages, default=0)

    async def fetch(self, cursor: FetchCursor):
        # IMAP-style fetch: respects after_uid and batch_size.
        if cursor.after_uid is not None:
            selected = sorted(uid for uid in self.messages if uid > cursor.after_uid)
        else:
            selected = sorted(self.messages)
        for uid in selected[: cursor.batch_size]:
            yield SimpleNamespace(
                uid=uid,
                uidvalidity=self.uidvalidity,
                raw_bytes=self.messages[uid],
            )


class _FakeRedis:
    def __init__(self) -> None:
        self._locks: set[str] = set()

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int = 0) -> bool:
        if nx and key in self._locks:
            return False
        self._locks.add(key)
        return True

    async def delete(self, key: str) -> None:
        self._locks.discard(key)


class _RecordingService:
    """Minimal RunService stub: records every create_run / enqueue_run call."""

    def __init__(self) -> None:
        self.created: list = []
        self.enqueued: list[UUID] = []

    async def create_run(self, request, *, enqueue: bool):
        assert enqueue is False
        from uuid import uuid4
        run_id = uuid4()
        self.created.append(request.email)
        return SimpleNamespace(id=run_id)

    async def enqueue_run(self, run_id: UUID):
        self.enqueued.append(run_id)
        return "job"


_LEDGER_DDL = (
    "CREATE TABLE mail_gateway_ingest_ledger "
    "(id INTEGER PRIMARY KEY, mailbox_id TEXT NOT NULL, dedup_key TEXT NOT NULL, "
    "uidvalidity INTEGER, uid INTEGER NOT NULL, run_id TEXT, status TEXT NOT NULL, "
    "protocol TEXT NOT NULL DEFAULT 'imap', server_id TEXT, "
    "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
    "UNIQUE(mailbox_id, dedup_key))"
)
_CURSOR_DDL = (
    "CREATE TABLE mail_gateway_cursor "
    "(mailbox_id TEXT PRIMARY KEY, uidvalidity INTEGER, high_water_uid INTEGER NOT NULL, "
    "protocol TEXT NOT NULL DEFAULT 'imap', updated_at DATETIME NOT NULL)"
)


async def _build_state() -> tuple[MailGatewayStateStore, object]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
        await conn.exec_driver_sql(_CURSOR_DDL)
    return MailGatewayStateStore(engine), engine


def _imap_settings(**overrides) -> MailGatewaySettings:
    base = {
        "enabled": True,
        "adapter": "imap",
        "mailbox_id": "primary",
        "host": "imap.example.com",
        "username": "ops",
        "password_env": "IMAP_PASSWORD",
        "initial_sync_mode": "from_now",
        "poll_interval_seconds": 60,
        "fetch_batch_size": 50,
        "max_message_bytes": 26_214_400,
    }
    base.update(overrides)
    return MailGatewaySettings(**base)


def _ctx(settings: Settings, state: MailGatewayStateStore, service, redis) -> dict:
    return {
        "settings": settings,
        "mail_gateway_state": state,
        "service": service,
        "redis": redis,
    }


def _wire_gateway(monkeypatch: pytest.MonkeyPatch, gateway: _StatefulImapGateway) -> None:
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _s: gateway)


# ---------------------------------------------------------------------------
# Task 6.1: fake fixture verification (smoke test for the test harness itself)
# ---------------------------------------------------------------------------


async def test_stateful_gateway_fixture_yields_canned_messages() -> None:
    """6.1 smoke test: the in-memory gateway stub yields the canned RFC822
    messages through the gateway contract (connect / fetch / close)."""

    gateway = _StatefulImapGateway()
    gateway.add_message(7, _rfc822(message_id="<canned-1>"))
    gateway.add_message(11, _rfc822(message_id="<canned-2>"))

    await gateway.connect()
    assert gateway.connected is True
    assert await gateway.highest_uid() == 11

    messages = [m async for m in gateway.fetch(FetchCursor(100, None, None, 10))]
    assert sorted(m.uid for m in messages) == [7, 11]
    assert all(b"Message-ID: <canned-" in m.raw_bytes for m in messages)
    assert all(m.uidvalidity == 100 for m in messages)

    await gateway.close()
    assert gateway.connected is False


# ---------------------------------------------------------------------------
# Task 6.2: end-to-end from_now + new mail + redelivery
# ---------------------------------------------------------------------------


async def test_e2e_from_now_then_new_mail_then_redelivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """6.2: first from_now poll creates zero historical runs; later eligible
    mail creates one PENDING run + ledger entry; repeat delivery creates no
    second run."""

    gateway = _StatefulImapGateway()
    # Mailbox starts with 3 historical messages (UIDs 1, 2, 3).
    gateway.add_message(1, _rfc822(message_id="<hist-1>"))
    gateway.add_message(2, _rfc822(message_id="<hist-2>"))
    gateway.add_message(3, _rfc822(message_id="<hist-3>"))
    _wire_gateway(monkeypatch, gateway)

    state, engine = await _build_state()
    service = _RecordingService()
    redis = _FakeRedis()
    settings = Settings(mail_gateways=[_imap_settings()])

    # --- First poll: from_now → cursor jumps to highest UID, zero runs ---
    result1 = await mail_poll_job(_ctx(settings, state, service, redis))

    assert result1 == "primary: initialized: from_now"
    assert service.created == []  # No runs for historical mail.
    cursor = await state.get_cursor("primary")
    assert cursor is not None
    assert cursor.high_water_uid == 3
    assert cursor.uidvalidity == 100

    # No ledger entries yet — from_now doesn't claim historical messages.
    async with engine.connect() as conn:
        ledger_count = (await conn.exec_driver_sql("SELECT COUNT(*) FROM mail_gateway_ingest_ledger")).scalar_one()
    assert ledger_count == 0

    # --- Second poll: one new eligible message arrives (UID 4) ---
    gateway.add_message(4, _rfc822(message_id="<new-1>", subject="STATUS update"))
    result2 = await mail_poll_job(_ctx(settings, state, service, redis))

    assert result2 == "primary: accepted: 1"
    assert len(service.created) == 1
    assert service.created[0].message_id == "<new-1>"
    assert len(service.enqueued) == 1

    # Ledger entry recorded as accepted.
    claim = await state.get_claim("primary", "<new-1>")
    assert claim is not None
    assert claim.status == "accepted"
    assert claim.uid == 4
    assert claim.uidvalidity == 100
    assert claim.run_id is not None

    # Cursor advanced past UID 4.
    cursor = await state.get_cursor("primary")
    assert cursor is not None
    assert cursor.high_water_uid == 4

    # --- Third poll: same message redelivered (UID 5, same Message-ID) ---
    gateway.add_message(5, _rfc822(message_id="<new-1>", subject="STATUS update"))
    result3 = await mail_poll_job(_ctx(settings, state, service, redis))

    assert result3 == "primary: accepted: 0"
    assert len(service.created) == 1  # No new run created.
    assert len(service.enqueued) == 1  # No re-enqueue.

    # Cursor still advances past the duplicate UID (no reprocessing).
    cursor = await state.get_cursor("primary")
    assert cursor is not None
    assert cursor.high_water_uid == 5

    await engine.dispose()


# ---------------------------------------------------------------------------
# Task 6.3: UIDVALIDITY change preserves ledger dedup, resets only cursor
# ---------------------------------------------------------------------------


async def test_e2e_uidvalidity_change_preserves_ledger_and_reinitializes_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """6.3: UIDVALIDITY change preserves ledger duplicate protection and
    reinitializes only cursor behavior."""

    gateway = _StatefulImapGateway()
    gateway.add_message(1, _rfc822(message_id="<mail-a>"))
    gateway.add_message(2, _rfc822(message_id="<mail-b>"))
    _wire_gateway(monkeypatch, gateway)

    state, engine = await _build_state()
    service = _RecordingService()
    redis = _FakeRedis()
    # Start in bounded_backfill so the first poll actually ingests mail.
    settings = Settings(
        mail_gateways=[
            _imap_settings(
                initial_sync_mode="bounded_backfill",
                initial_backfill_confirmed=True,
                backfill_since_days=7,
                backfill_max_messages=100,
            )
        ]
    )

    # --- First poll: ingest the 2 existing messages ---
    result1 = await mail_poll_job(_ctx(settings, state, service, redis))
    assert result1 == "primary: accepted: 2"
    assert len(service.created) == 2
    cursor = await state.get_cursor("primary")
    assert cursor is not None
    assert cursor.uidvalidity == 100
    assert cursor.high_water_uid == 2

    # --- UIDVALIDITY changes from 100 → 200, same messages still present ---
    gateway.change_uidvalidity(200)
    # Add one genuinely new message at UID 3 (post-reset numbering).
    gateway.add_message(3, _rfc822(message_id="<mail-c>"))

    result2 = await mail_poll_job(_ctx(settings, state, service, redis))

    # Cursor was reset → from_now-like behavior re-initializes at highest UID.
    # The 2 already-ingested messages (now at UIDs 1, 2 under new UIDVALIDITY)
    # are NOT re-ingested because the ledger dedup_key protection survives.
    # Only <mail-c> is new and gets accepted.
    assert "accepted: 1" in result2
    assert len(service.created) == 3  # 2 from before + 1 new.

    # Ledger still has the original 2 entries + 1 new = 3 total.
    async with engine.connect() as conn:
        ledger_count = (await conn.exec_driver_sql("SELECT COUNT(*) FROM mail_gateway_ingest_ledger")).scalar_one()
    assert ledger_count == 3

    # Cursor reinitialized with the new UIDVALIDITY.
    cursor = await state.get_cursor("primary")
    assert cursor is not None
    assert cursor.uidvalidity == 200
    assert cursor.high_water_uid == 3

    # The previously-accepted mail-a/mail-b are still in the ledger with
    # the ORIGINAL uidvalidity=100, proving dedup protection persisted.
    claim_a = await state.get_claim("primary", "<mail-a>")
    assert claim_a is not None
    assert claim_a.uidvalidity == 100
    assert claim_a.status == "accepted"
    claim_b = await state.get_claim("primary", "<mail-b>")
    assert claim_b is not None
    assert claim_b.uidvalidity == 100

    # The new mail-c is recorded with the new uidvalidity=200.
    claim_c = await state.get_claim("primary", "<mail-c>")
    assert claim_c is not None
    assert claim_c.uidvalidity == 200
    assert claim_c.status == "accepted"

    await engine.dispose()
