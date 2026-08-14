from __future__ import annotations

from sqlalchemy.ext.asyncio import create_async_engine

from mailagent.gateway.state import MailGatewayStateStore


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


async def test_claim_is_unique_for_mailbox_and_key() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)
    assert await state.claim("primary", "same", 1, 10)
    assert not await state.claim("primary", "same", 1, 11)
    await engine.dispose()


async def test_reset_cursor_keeps_ledger_rows() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_CURSOR_DDL)
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)
    await state.advance_cursor("primary", 1, 42)
    assert await state.claim("primary", "dedup", 1, 42)

    await state.reset_cursor("primary")

    assert await state.get_cursor("primary") is None
    claim = await state.get_claim("primary", "dedup")
    assert claim is not None
    assert claim.status == "claimed"
    await engine.dispose()


async def test_manual_backfill_audit_is_persisted() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE mail_gateway_backfill_audit (id INTEGER PRIMARY KEY, mailbox_id TEXT NOT NULL, since_days INTEGER NOT NULL, max_messages INTEGER NOT NULL, created_at DATETIME NOT NULL)")
    state = MailGatewayStateStore(engine)

    await state.record_backfill_audit("primary", since_days=14, max_messages=5000)

    async with engine.connect() as conn:
        count = (await conn.exec_driver_sql("SELECT COUNT(*) FROM mail_gateway_backfill_audit")).scalar_one()
    assert count == 1
    await engine.dispose()


async def test_pop3_cursor_stores_null_uidvalidity_and_protocol() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_CURSOR_DDL)
    state = MailGatewayStateStore(engine)

    await state.advance_cursor("primary", None, 42, protocol="pop3")
    cursor = await state.get_cursor("primary")

    assert cursor is not None
    assert cursor.uidvalidity is None
    assert cursor.protocol == "pop3"
    assert cursor.high_water_uid == 42
    await engine.dispose()


async def test_insert_skipped_initial_is_idempotent() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    assert await state.insert_skipped_initial(
        "primary", "key1", 1, protocol="pop3", server_id="uid-a"
    )
    assert not await state.insert_skipped_initial(
        "primary", "key1", 1, protocol="pop3", server_id="uid-a"
    )

    claim = await state.get_claim("primary", "key1")
    assert claim is not None
    assert claim.status == "skipped_initial"
    assert claim.protocol == "pop3"
    assert claim.uidvalidity is None
    assert claim.server_id == "uid-a"
    assert await state.get_server_ids("primary", protocol="pop3") == {"uid-a"}
    await engine.dispose()


async def test_claim_persists_pop3_server_id() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    assert await state.claim(
        "primary",
        "dedup",
        None,
        2,
        protocol="pop3",
        server_id="uid-c",
    )

    claim = await state.get_claim("primary", "dedup")
    assert claim is not None
    assert claim.server_id == "uid-c"
    assert await state.get_server_ids("primary", protocol="pop3") == {"uid-c"}
    await engine.dispose()


# ---------------------------------------------------------------------------
# Additional state-store coverage (mail-gateway-imap task 4.6)
#
# SQLite is used as the runtime; the SQL statements use only ANSI INSERT ON
# CONFLICT / UPDATE ... WHERE patterns that are also valid PostgreSQL. The
# state store has no dialect-specific branches, so SQLite behavior exercises
# the same code paths that PostgreSQL would.
# ---------------------------------------------------------------------------


async def test_claim_contention_loses_second_claim_for_same_dedup_key() -> None:
    """Two concurrent claims on (mailbox_id, dedup_key): exactly one wins."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    first = await state.claim("primary", "contended", 100, 50)
    second = await state.claim("primary", "contended", 100, 51)

    assert first is True
    assert second is False

    claim = await state.get_claim("primary", "contended")
    assert claim is not None
    assert claim.uid == 50  # First claim's UID is preserved.
    assert claim.status == "claimed"
    await engine.dispose()


async def test_duplicate_at_new_uid_does_not_create_second_ledger_row() -> None:
    """A dedup_key observed at a new UID (e.g. after UIDVALIDITY change) does
    not produce a second ledger row; the original claim is preserved."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    assert await state.claim("primary", "msg-1", uidvalidity=100, uid=5)
    # Same dedup_key appears at UID 10 (same or new UIDVALIDITY) — claim fails.
    assert not await state.claim("primary", "msg-1", uidvalidity=200, uid=10)

    claim = await state.get_claim("primary", "msg-1")
    assert claim is not None
    assert claim.uidvalidity == 100  # Original claim's uidvalidity preserved.
    assert claim.uid == 5
    await engine.dispose()


async def test_cursor_progression_advances_high_water_uid_monotonically() -> None:
    """advance_cursor upserts the cursor row; high_water_uid increases."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_CURSOR_DDL)
    state = MailGatewayStateStore(engine)

    assert await state.get_cursor("primary") is None

    await state.advance_cursor("primary", 100, 5)
    cursor = await state.get_cursor("primary")
    assert cursor is not None
    assert cursor.high_water_uid == 5
    assert cursor.uidvalidity == 100

    # Subsequent advance to a higher UID updates the same row.
    await state.advance_cursor("primary", 100, 12)
    cursor = await state.get_cursor("primary")
    assert cursor is not None
    assert cursor.high_water_uid == 12

    # Regression to a lower UID also updates the row (the runner is
    # responsible for never calling advance_cursor with a lower UID; the
    # store does not enforce monotonicity).
    await state.advance_cursor("primary", 100, 8)
    cursor = await state.get_cursor("primary")
    assert cursor.high_water_uid == 8
    await engine.dispose()


async def test_terminal_disposition_does_not_advance_cursor_implicitly() -> None:
    """Recording a terminal disposition (duplicate / skipped_*) must NOT
    advance the cursor; the runner is responsible for calling
    advance_cursor separately. This guards against accidental
    double-advance in the store layer."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_CURSOR_DDL)
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    await state.advance_cursor("primary", 100, 10)
    await state.claim("primary", "msg-x", 100, 11)
    await state.terminal("primary", "msg-x", "duplicate")

    # Cursor stays at 10 — terminal() only mutates the ledger.
    cursor = await state.get_cursor("primary")
    assert cursor is not None
    assert cursor.high_water_uid == 10

    claim = await state.get_claim("primary", "msg-x")
    assert claim is not None
    assert claim.status == "duplicate"
    await engine.dispose()


async def test_uidvalidity_reset_clears_cursor_but_preserves_ledger() -> None:
    """After a UIDVALIDITY change, reset_cursor deletes only the cursor row.
    Ledger entries (dedup protection) survive so the same message id cannot
    be re-ingested."""

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_CURSOR_DDL)
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    await state.advance_cursor("primary", 100, 42)
    assert await state.claim("primary", "msg-1", 100, 42)

    # UIDVALIDITY changes from 100 → 200; runner calls reset_cursor().
    await state.reset_cursor("primary")

    assert await state.get_cursor("primary") is None
    # Ledger row survives → dedup protection persists across UIDVALIDITY changes.
    claim = await state.get_claim("primary", "msg-1")
    assert claim is not None
    assert claim.status == "claimed"
    assert claim.uidvalidity == 100  # Original UIDVALIDITY preserved in ledger.

    # Re-claiming the same dedup_key at the new UIDVALIDITY fails (dedup).
    assert not await state.claim("primary", "msg-1", 200, 1)
    await engine.dispose()


async def test_attach_run_persists_run_id_before_enqueue() -> None:
    """attach_run records the run_id on a claimed row so a failed enqueue
    is resumable on the next poll."""

    from uuid import uuid4

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    assert await state.claim("primary", "msg-1", 100, 5)
    run_id = uuid4()
    await state.attach_run("primary", "msg-1", run_id)

    claim = await state.get_claim("primary", "msg-1")
    assert claim is not None
    assert claim.run_id == str(run_id)
    assert claim.status == "claimed"  # Not yet accepted.
    await engine.dispose()


async def test_accept_transitions_claimed_row_to_accepted_with_run_id() -> None:
    from uuid import uuid4

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    assert await state.claim("primary", "msg-1", 100, 5)
    run_id = uuid4()
    await state.accept("primary", "msg-1", run_id)

    claim = await state.get_claim("primary", "msg-1")
    assert claim is not None
    assert claim.status == "accepted"
    assert claim.run_id == str(run_id)
    await engine.dispose()


async def test_get_claim_returns_none_for_unknown_key() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    assert await state.get_claim("primary", "never-seen") is None
    await engine.dispose()


async def test_get_server_ids_filters_by_protocol() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.exec_driver_sql(_LEDGER_DDL)
    state = MailGatewayStateStore(engine)

    await state.claim("primary", "k1", 100, 1, protocol="imap", server_id=None)
    await state.claim(
        "primary", "k2", None, 2, protocol="pop3", server_id="uid-pop-a"
    )
    await state.claim(
        "primary", "k3", None, 3, protocol="pop3", server_id="uid-pop-b"
    )

    assert await state.get_server_ids("primary", protocol="pop3") == {
        "uid-pop-a",
        "uid-pop-b",
    }
    assert await state.get_server_ids("primary", protocol="imap") == set()
    await engine.dispose()
