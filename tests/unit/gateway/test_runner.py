from types import SimpleNamespace
from uuid import uuid4

import pytest

from mailagent.gateway import runner as gateway_runner
from mailagent.gateway.runner import _poll_single_gateway, mail_poll_job


class _Gateway:
    uidvalidity = 1

    def __init__(self) -> None:
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def highest_uid(self) -> int:
        return 4

    async def fetch(self, _cursor):
        if False:
            yield None


class _LockedRedis:
    async def set(self, *_args, **_kwargs) -> bool:
        return False


class _Redis:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def set(self, *_args, **_kwargs) -> bool:
        return True

    async def delete(self, key: str) -> None:
        self.deleted.append(key)


class _FailingGateway(_Gateway):
    async def connect(self) -> None:
        raise RuntimeError("connection failed")


class _State:
    def __init__(self) -> None:
        self.cursor = None
        self.claims: dict[str, SimpleNamespace] = {}
        self.advanced: list[int] = []

    async def get_cursor(self, _mailbox_id):
        return self.cursor

    async def reset_cursor(self, _mailbox_id):
        self.cursor = None

    async def advance_cursor(self, _mailbox_id, uidvalidity, uid, protocol="imap"):
        self.cursor = SimpleNamespace(uidvalidity=uidvalidity, high_water_uid=uid, protocol=protocol)
        self.advanced.append(uid)

    async def claim(
        self,
        _mailbox_id,
        key,
        uidvalidity,
        uid,
        protocol="imap",
        server_id=None,
    ):
        if key in self.claims:
            return False
        self.claims[key] = SimpleNamespace(
            status="claimed",
            run_id=None,
            uidvalidity=uidvalidity,
            uid=uid,
            protocol=protocol,
            server_id=server_id,
        )
        return True

    async def get_claim(self, _mailbox_id, key):
        return self.claims.get(key)

    async def attach_run(self, _mailbox_id, key, run_id):
        self.claims[key].run_id = str(run_id)

    async def accept(self, _mailbox_id, key, run_id):
        self.claims[key].status = "accepted"
        self.claims[key].run_id = str(run_id)

    async def terminal(self, _mailbox_id, key, status):
        self.claims[key].status = status

    async def insert_skipped_initial(
        self, _mailbox_id, key, uid, protocol="pop3", server_id=None
    ):
        if key in self.claims:
            return False
        self.claims[key] = SimpleNamespace(
            status="skipped_initial",
            run_id=None,
            uidvalidity=None,
            uid=uid,
            protocol=protocol,
            server_id=server_id,
        )
        return True

    async def get_server_ids(self, _mailbox_id, protocol):
        return {
            claim.server_id
            for claim in self.claims.values()
            if claim.protocol == protocol and getattr(claim, "server_id", None)
        }


class _MessageGateway(_Gateway):
    def __init__(self, raw: bytes) -> None:
        super().__init__()
        self.raw = raw

    async def fetch(self, _cursor):
        yield SimpleNamespace(uid=5, uidvalidity=1, raw_bytes=self.raw)


class _Service:
    def __init__(self, fail_enqueue: bool = False) -> None:
        self.created = 0
        self.enqueued: list[object] = []
        self.fail_enqueue = fail_enqueue

    async def create_run(self, _request, *, enqueue: bool):
        assert enqueue is False
        self.created += 1
        return SimpleNamespace(id=uuid4())

    async def enqueue_run(self, run_id):
        if self.fail_enqueue:
            raise RuntimeError("redis unavailable")
        self.enqueued.append(run_id)
        return "job"


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        adapter="imap",
        mailbox_id="primary", poll_interval_seconds=60,
        initial_sync_mode="bounded_backfill", backfill_since_days=1,
        backfill_max_messages=10, fetch_batch_size=10, max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[], recipient_domain_allowlist=[], subject_patterns=[],
    )


async def test_poll_returns_skipped_locked_without_connecting_to_imap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _Gateway()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="from_now",
        backfill_since_days=1,
        fetch_batch_size=10,
        max_message_bytes=1024,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    context = {
        "settings": SimpleNamespace(mail_gateways=[settings]),
        "mail_gateway_state": object(),
        "service": object(),
        "redis": _LockedRedis(),
    }

    assert await mail_poll_job(context) == "primary: skipped_locked"
    assert gateway.connected is False


async def test_enqueue_failure_reuses_persisted_run_without_advancing_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _MessageGateway(b"From: sender@example.com\nMessage-ID: <one>\n\nhello")
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    failed = _Service(fail_enqueue=True)

    with pytest.raises(RuntimeError, match="redis unavailable"):
        await _poll_single_gateway(_settings(), state, failed, redis)

    assert failed.created == 1
    assert state.advanced == []
    saved = next(iter(state.claims.values()))
    assert saved.status == "claimed"
    assert saved.run_id is not None

    recovered = _Service()
    assert await _poll_single_gateway(_settings(), state, recovered, redis) == "accepted: 1"
    assert recovered.created == 0
    assert len(recovered.enqueued) == 1
    assert state.advanced == [5]


async def test_one_gateway_failure_does_not_block_other_mailboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken_values = vars(_settings()).copy()
    broken_values.update(enabled=True, mailbox_id="broken")
    broken = SimpleNamespace(**broken_values)
    broken.initial_sync_mode = "from_now"
    healthy_values = vars(_settings()).copy()
    healthy_values.update(enabled=True, mailbox_id="healthy")
    healthy = SimpleNamespace(**healthy_values)
    healthy.initial_sync_mode = "from_now"
    adapters = {"broken": _FailingGateway(), "healthy": _Gateway()}
    monkeypatch.setattr(
        gateway_runner,
        "_build_adapter",
        lambda settings: adapters[settings.mailbox_id],
    )
    context = {
        "settings": SimpleNamespace(mail_gateways=[broken, healthy]),
        "mail_gateway_state": _State(),
        "service": _Service(),
        "redis": _Redis(),
    }

    result = await mail_poll_job(context)

    assert result == "broken: error; healthy: initialized: from_now"
    assert adapters["healthy"].connected is False


# ---------------------------------------------------------------------------
# POP3 runner tests
# ---------------------------------------------------------------------------


class _Pop3Gateway:
    """Mock POP3 gateway yielding N messages on first sync, new ones after."""

    uidvalidity = None

    def __init__(self, messages: list[bytes] | list[tuple[str, bytes]]) -> None:
        self.messages = [
            item if isinstance(item, tuple) else (f"uid-{index}", item)
            for index, item in enumerate(messages, start=1)
        ]
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def fetch(self, cursor):
        # First sync (after_uid is None): return top batch_size messages
        if cursor.after_uid is None:
            selected = self.messages[-cursor.batch_size :]
        else:
            selected = [
                item
                for item in self.messages
                if item[0] not in cursor.seen_server_ids
            ][: cursor.batch_size]
        for position, (server_id, raw) in enumerate(self.messages, start=1):
            if (server_id, raw) in selected:
                yield SimpleNamespace(
                    uid=position,
                    uidvalidity=None,
                    raw_bytes=raw,
                    server_id=server_id,
                )


def _pop3_settings() -> SimpleNamespace:
    return SimpleNamespace(
        adapter="pop3",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="incremental",
        initial_sync_max_messages=3,
        fetch_batch_size=10,
        max_message_bytes=1024,
        backfill_since_days=1,
        backfill_max_messages=10,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )


async def test_pop3_first_sync_marks_skipped_initial_and_advances_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msgs = [b"From: a@x.com\nMessage-ID: <1>\n\nbody1", b"From: b@x.com\nMessage-ID: <2>\n\nbody2"]
    gateway = _Pop3Gateway(msgs)
    state = _State()
    redis = _Redis()
    service = _Service()
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)

    result = await _poll_single_gateway(_pop3_settings(), state, service, redis)

    assert result == "initialized: incremental (2 messages)"
    assert service.created == 0  # No runs created
    assert len(state.claims) == 2
    for claim in state.claims.values():
        assert claim.status == "skipped_initial"
    assert state.cursor.high_water_uid == 2


async def test_pop3_normal_poll_creates_run_for_new_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msgs = [
        b"From: a@x.com\nMessage-ID: <1>\n\nbody1",
        b"From: b@x.com\nMessage-ID: <2>\n\nbody2",
        b"From: c@x.com\nMessage-ID: <3>\n\nbody3",
    ]
    gateway = _Pop3Gateway(msgs)
    state = _State()
    state.cursor = SimpleNamespace(uidvalidity=None, high_water_uid=2, protocol="pop3")
    state.claims["<1>"] = SimpleNamespace(
        status="skipped_initial",
        run_id=None,
        uidvalidity=None,
        uid=1,
        protocol="pop3",
        server_id="uid-1",
    )
    state.claims["<2>"] = SimpleNamespace(
        status="skipped_initial",
        run_id=None,
        uidvalidity=None,
        uid=2,
        protocol="pop3",
        server_id="uid-2",
    )
    redis = _Redis()
    service = _Service()
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)

    result = await _poll_single_gateway(_pop3_settings(), state, service, redis)

    assert result == "accepted: 1"
    assert service.created == 1
    assert len(service.enqueued) == 1


async def test_pop3_duplicate_message_does_not_create_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"From: a@x.com\nMessage-ID: <dup>\n\nbody"
    gateway = _Pop3Gateway([raw])
    state = _State()
    # Simulate that message with uid=1 was already processed
    state.cursor = SimpleNamespace(uidvalidity=None, high_water_uid=0, protocol="pop3")
    state.claims["<dup>"] = SimpleNamespace(
        status="accepted",
        run_id="abc",
        uidvalidity=None,
        uid=1,
        protocol="pop3",
        server_id="uid-1",
    )
    redis = _Redis()
    service = _Service()
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)

    result = await _poll_single_gateway(_pop3_settings(), state, service, redis)

    assert result == "accepted: 0"
    assert service.created == 0


async def test_pop3_renumbering_does_not_hide_new_uidl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_a = b"From: a@x.com\nMessage-ID: <a>\n\nbody-a"
    raw_b = b"From: b@x.com\nMessage-ID: <b>\n\nbody-b"
    raw_c = b"From: c@x.com\nMessage-ID: <c>\n\nbody-c"
    state = _State()
    redis = _Redis()
    service = _Service()

    first_gateway = _Pop3Gateway([("uid-a", raw_a), ("uid-b", raw_b)])
    monkeypatch.setattr(
        gateway_runner, "_build_adapter", lambda _settings: first_gateway
    )
    assert (
        await _poll_single_gateway(_pop3_settings(), state, service, redis)
        == "initialized: incremental (2 messages)"
    )

    # The server deletes uid-a. uid-b is renumbered from 2 to 1, while the
    # genuinely new uid-c receives message number 2.
    second_gateway = _Pop3Gateway([("uid-b", raw_b), ("uid-c", raw_c)])
    monkeypatch.setattr(
        gateway_runner, "_build_adapter", lambda _settings: second_gateway
    )

    result = await _poll_single_gateway(_pop3_settings(), state, service, redis)

    assert result == "accepted: 1"
    assert service.created == 1
    assert state.claims["<c>"].server_id == "uid-c"


# ---------------------------------------------------------------------------
# Additional IMAP runner coverage (mail-gateway-imap task 5.10)
# ---------------------------------------------------------------------------


class _HighestUidGateway(_Gateway):
    """IMAP gateway for from_now first-sync: returns no messages, reports a
    fixed highest UID so the runner can record the starting position."""

    def __init__(self, highest: int = 99) -> None:
        super().__init__()
        self._highest = highest

    async def highest_uid(self) -> int:
        return self._highest

    async def fetch(self, _cursor):
        if False:
            yield None  # empty mailbox — no messages to yield


async def test_from_now_first_poll_records_highest_uid_without_creating_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _HighestUidGateway(highest=99)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="from_now",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )

    result = await _poll_single_gateway(settings, state, service, redis)

    assert result == "initialized: from_now"
    assert service.created == 0  # No runs created on first from_now poll.
    assert state.cursor is not None
    assert state.cursor.high_water_uid == 99  # Cursor positioned at highest UID.
    assert state.cursor.uidvalidity == 1


async def test_bounded_backfill_uses_since_filter_and_creates_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"From: ops@example.com\nMessage-ID: <backfill-1>\n\nold mail"
    gateway = _MessageGateway(raw)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="bounded_backfill",
        backfill_since_days=7,
        backfill_max_messages=500,
        fetch_batch_size=50,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )

    result = await _poll_single_gateway(settings, state, service, redis)

    assert result == "accepted: 1"
    assert service.created == 1
    assert state.cursor.high_water_uid == 5


async def test_oversize_message_is_skipped_with_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big_body = b"x" * 2048  # Exceeds max_message_bytes=1024.
    raw = b"From: ops@x.com\nMessage-ID: <big>\n\n" + big_body
    gateway = _MessageGateway(raw)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    service = _Service()

    result = await _poll_single_gateway(_settings(), state, service, redis)

    assert result == "accepted: 0"  # No run created.
    assert service.created == 0
    saved = next(iter(state.claims.values()))
    assert saved.status == "skipped_oversize"
    assert state.cursor.high_water_uid == 5  # Cursor still advances.


async def test_duplicate_message_at_new_uid_advances_cursor_without_new_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"From: ops@x.com\nMessage-ID: <dup>\n\nbody"
    gateway = _MessageGateway(raw)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    # Pretend the same dedup_key was already accepted in a previous poll.
    state.claims["<dup>"] = SimpleNamespace(
        status="accepted",
        run_id="previous-run",
        uidvalidity=1,
        uid=4,
        protocol="imap",
        server_id=None,
    )
    redis = _Redis()
    service = _Service()

    result = await _poll_single_gateway(_settings(), state, service, redis)

    assert result == "accepted: 0"
    assert service.created == 0
    # Cursor advances past the duplicate UID.
    assert state.cursor.high_water_uid == 5


async def test_sender_domain_filter_skips_disallowed_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"From: spam@evil.com\nMessage-ID: <spam-1>\n\nspam body"
    gateway = _MessageGateway(raw)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="bounded_backfill",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=["example.com", "partner.example"],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )

    result = await _poll_single_gateway(settings, state, service, redis)

    assert result == "accepted: 0"
    assert service.created == 0
    saved = next(iter(state.claims.values()))
    assert saved.status == "skipped_filtered"


async def test_recipient_domain_filter_skips_disallowed_recipient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"From: ops@example.com\nTo: wrong@other.com\nMessage-ID: <r-1>\n\nbody"
    gateway = _MessageGateway(raw)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="bounded_backfill",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=["exampleco.com.cn"],
        subject_patterns=[],
    )

    result = await _poll_single_gateway(settings, state, service, redis)

    assert result == "accepted: 0"
    saved = next(iter(state.claims.values()))
    assert saved.status == "skipped_filtered"


async def test_subject_pattern_filter_skips_non_matching_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"From: ops@example.com\nMessage-ID: <subj-1>\nSubject: Random chat\n\nbody"
    gateway = _MessageGateway(raw)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="bounded_backfill",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=["STATUS", "Status Report"],
    )

    result = await _poll_single_gateway(settings, state, service, redis)

    assert result == "accepted: 0"
    saved = next(iter(state.claims.values()))
    assert saved.status == "skipped_filtered"


async def test_lock_contention_returns_skipped_locked_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis lock already held → poll returns skipped_locked, gateway never
    connects."""

    gateway = _Gateway()
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="from_now",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )
    state = _State()
    service = _Service()

    result = await _poll_single_gateway(settings, state, service, _LockedRedis())

    assert result == "skipped_locked"
    assert gateway.connected is False
    assert state.cursor is None  # Cursor untouched.


async def test_uidvalidity_change_resets_cursor_then_reinitializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When UIDVALIDITY changes, the runner resets the cursor and re-runs
    first-sync behavior."""

    gateway = _HighestUidGateway(highest=200)
    gateway.uidvalidity = 999  # New UIDVALIDITY.
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    # Existing cursor has an old UIDVALIDITY.
    state.cursor = SimpleNamespace(
        uidvalidity=1, high_water_uid=99, protocol="imap"
    )
    redis = _Redis()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="from_now",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )

    result = await _poll_single_gateway(settings, state, service, redis)

    # After reset, from_now reinitializes at the new highest UID.
    assert result == "initialized: from_now"
    assert state.cursor.uidvalidity == 999
    assert state.cursor.high_water_uid == 200
    assert service.created == 0


async def test_no_redis_lock_skips_contention_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When redis is None (e.g. tests / single-instance dev), the poll still
    proceeds without acquiring a lock."""

    gateway = _HighestUidGateway(highest=50)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="from_now",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )

    result = await _poll_single_gateway(settings, state, service, redis=None)

    assert result == "initialized: from_now"
    # from_now path connects, records highest UID, then closes in finally.
    # The cursor being set proves the connect → highest_uid path executed.
    assert state.cursor is not None
    assert state.cursor.high_water_uid == 50


async def test_lock_is_released_after_successful_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _HighestUidGateway(highest=10)
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="from_now",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )

    await _poll_single_gateway(settings, state, service, redis)

    # Lock key format: mail_gateway:{mailbox_id}
    assert redis.deleted == ["mail_gateway:primary"]


async def test_lock_is_released_even_when_poll_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = _FailingGateway()
    monkeypatch.setattr(gateway_runner, "_build_adapter", lambda _settings: gateway)
    state = _State()
    redis = _Redis()
    service = _Service()
    settings = SimpleNamespace(
        enabled=True,
        adapter="imap",
        mailbox_id="primary",
        poll_interval_seconds=60,
        initial_sync_mode="from_now",
        backfill_since_days=1,
        backfill_max_messages=10,
        fetch_batch_size=10,
        max_message_bytes=1024,
        initial_sync_max_messages=1000,
        sender_domain_allowlist=[],
        recipient_domain_allowlist=[],
        subject_patterns=[],
    )

    with pytest.raises(RuntimeError, match="connection failed"):
        await _poll_single_gateway(settings, state, service, redis)

    # Lock must still be released in the finally block.
    assert redis.deleted == ["mail_gateway:primary"]
