"""Mocked-adapter tests for ImapAdapter (mail-gateway-imap task 3.6).

Covers TLS connection, missing password variable, UID search construction,
batching, ``BODY.PEEK[]`` usage, and safe shutdown.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mailagent.gateway.base import FetchCursor
from mailagent.gateway.imap_adapter import ImapAdapter


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "host": "imap.example.com",
        "port": 993,
        "username": "ops@example.com",
        "password_env": "IMAP_PASSWORD",
        "mailbox": "INBOX",
        "seen_filter": "all",
        "fetch_batch_size": 50,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _imap_response(*lines: bytes) -> SimpleNamespace:
    return SimpleNamespace(lines=list(lines))


def _make_connected_adapter(settings=None) -> tuple[ImapAdapter, AsyncMock]:
    """Build an adapter with a mocked aioimaplib client already connected."""

    adapter = ImapAdapter(settings or _settings())
    adapter.uidvalidity = 12345
    mock_client = AsyncMock()
    mock_client.unselect = AsyncMock()
    mock_client.logout = AsyncMock()
    adapter.client = mock_client
    return adapter, mock_client


async def test_fetch_uses_uid_range_unseen_and_body_peek() -> None:
    settings = _settings(seen_filter="unseen")
    adapter = ImapAdapter(settings)
    adapter.uidvalidity = 9
    adapter.client = SimpleNamespace(
        uid=AsyncMock(
            side_effect=[
                _imap_response(b"12 13"),
                _imap_response(b"* 12 FETCH (UID 12 FLAGS () {5}", b"hello", b")"),
                _imap_response(b"* 13 FETCH (UID 13 FLAGS () {5}", b"world", b")"),
            ]
        )
    )

    messages = [item async for item in adapter.fetch(FetchCursor(9, 11, None, 2))]

    assert [message.uid for message in messages] == [12, 13]
    assert [message.raw_bytes for message in messages] == [b"hello", b"world"]
    assert adapter.client.uid.await_args_list[0].args == ("search", "UID 12:*", "UNSEEN")
    assert adapter.client.uid.await_args_list[1].args == ("fetch", "12", "(UID FLAGS BODY.PEEK[])")
    # BODY.PEEK (not BODY) is used so mail is never marked as read.
    assert all(
        "BODY.PEEK[]" in call.args[2]
        for call in adapter.client.uid.await_args_list[1:]
    )


async def test_fetch_uses_since_criteria_when_after_uid_is_none() -> None:
    adapter, _ = _make_connected_adapter()
    adapter.client.uid = AsyncMock(
        side_effect=[
            _imap_response(b"5"),
            _imap_response(b"* 5 FETCH (UID 5 FLAGS () {5}", b"body5", b")"),
        ]
    )

    from datetime import datetime, timezone, timedelta
    since = datetime.now(timezone.utc) - timedelta(days=1)
    messages = [item async for item in adapter.fetch(FetchCursor(12345, None, since, 10))]

    assert [m.uid for m in messages] == [5]
    # First call should be a SINCE search (no UID range, no UNSEEN with "all").
    search_call = adapter.client.uid.await_args_list[0]
    assert search_call.args[0] == "search"
    assert search_call.args[1].startswith("SINCE ")
    assert "UNSEEN" not in search_call.args[1:]


async def test_fetch_falls_back_to_all_when_no_cursor_constraints() -> None:
    adapter, _ = _make_connected_adapter()
    adapter.client.uid = AsyncMock(
        side_effect=[
            _imap_response(b""),
        ]
    )

    messages = [item async for item in adapter.fetch(FetchCursor(12345, None, None, 10))]

    assert messages == []
    search_call = adapter.client.uid.await_args_list[0]
    assert search_call.args == ("search", "ALL")


async def test_fetch_respects_batch_size_limit() -> None:
    adapter, _ = _make_connected_adapter()
    # Server returns 5 UIDs but cursor.batch_size=2 → only first 2 fetched.
    adapter.client.uid = AsyncMock(
        side_effect=[
            _imap_response(b"1 2 3 4 5"),
            _imap_response(b"* 1 FETCH (UID 1 FLAGS () {4}", b"body", b")"),
            _imap_response(b"* 2 FETCH (UID 2 FLAGS () {4}", b"body", b")"),
        ]
    )

    messages = [item async for item in adapter.fetch(FetchCursor(12345, 0, None, 2))]

    assert [m.uid for m in messages] == [1, 2]
    # One search + two fetches (not five).
    assert adapter.client.uid.await_count == 3


async def test_fetch_propagates_uidvalidity_into_fetched_messages() -> None:
    adapter, _ = _make_connected_adapter()
    adapter.uidvalidity = 999
    adapter.client.uid = AsyncMock(
        side_effect=[
            _imap_response(b"7"),
            _imap_response(b"* 7 FETCH (UID 7 FLAGS () {4}", b"body", b")"),
        ]
    )

    messages = [item async for item in adapter.fetch(FetchCursor(999, 6, None, 1))]

    assert messages[0].uidvalidity == 999
    assert messages[0].uid == 7


async def test_connect_raises_when_password_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the env var is unset.
    monkeypatch.delenv("IMAP_PASSWORD", raising=False)
    adapter = ImapAdapter(_settings())

    with pytest.raises(KeyError, match="IMAP_PASSWORD"):
        await adapter.connect()


async def test_connect_retries_on_failure_and_raises_after_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAP_PASSWORD", "secret")
    adapter = ImapAdapter(_settings())

    call_count = 0

    class _MockClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def wait_hello_from_server(self) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("server unavailable")

        async def login(self, *_args) -> None:
            pass

        async def examine(self, *_args) -> None:
            pass

    with patch("aioimaplib.IMAP4_SSL", _MockClient):
        with pytest.raises(RuntimeError, match="IMAP connection failed"):
            await adapter.connect()

    assert call_count == 3  # Three retries before giving up.
    assert adapter.client is None


async def test_connect_succeeds_and_extracts_uidvalidity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAP_PASSWORD", "secret")
    adapter = ImapAdapter(_settings())

    class _MockClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.calls = []

        async def wait_hello_from_server(self) -> None:
            pass

        async def login(self, *_args) -> None:
            pass

        async def examine(self, *_args) -> SimpleNamespace:
            return SimpleNamespace(lines=[b"* OK [UIDVALIDITY 4242] Inbox"])

    with patch("aioimaplib.IMAP4_SSL", _MockClient):
        await adapter.connect()

    assert adapter.uidvalidity == 4242
    assert adapter.client is not None


async def test_close_calls_unselect_then_logout() -> None:
    adapter, mock_client = _make_connected_adapter()

    await adapter.close()

    mock_client.unselect.assert_awaited_once()
    mock_client.logout.assert_awaited_once()
    assert adapter.client is None


async def test_close_falls_back_to_logout_when_unselect_unavailable() -> None:
    adapter = ImapAdapter(_settings())
    adapter.uidvalidity = 1
    # Client without unselect method (older aioimaplib).
    mock_client = AsyncMock()
    del mock_client.unselect
    adapter.client = mock_client

    await adapter.close()

    mock_client.logout.assert_awaited_once()
    assert adapter.client is None


async def test_close_is_idempotent_when_already_closed() -> None:
    adapter = ImapAdapter(_settings())
    assert adapter.client is None

    # Should not raise.
    await adapter.close()
    await adapter.close()
    assert adapter.client is None


async def test_close_releases_client_even_when_logout_raises() -> None:
    adapter, mock_client = _make_connected_adapter()
    mock_client.logout = AsyncMock(side_effect=RuntimeError("network gone"))

    # The finally block must still clear self.client.
    with pytest.raises(RuntimeError, match="network gone"):
        await adapter.close()
    assert adapter.client is None


async def test_highest_uid_returns_max_uid_or_zero_when_empty() -> None:
    adapter, mock_client = _make_connected_adapter()

    # Non-empty search.
    mock_client.uid = AsyncMock(return_value=_imap_response(b"5 17 3"))
    assert await adapter.highest_uid() == 17

    # Empty search.
    mock_client.uid = AsyncMock(return_value=_imap_response(b""))
    assert await adapter.highest_uid() == 0
