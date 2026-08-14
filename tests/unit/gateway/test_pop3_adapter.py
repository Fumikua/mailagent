from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mailagent.gateway.base import FetchCursor
from mailagent.gateway.pop3_adapter import Pop3Adapter


def _settings(**overrides) -> SimpleNamespace:
    defaults = dict(
        host="pop.example.com",
        port=995,
        username="ops@example.com",
        password_env="POP3_PASSWORD",
        mailbox="INBOX",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _FakePop3:
    """Minimal poplib.POP3_SSL mock for adapter testing."""

    def __init__(self, messages: list[bytes], uidls: list[str] | None = None) -> None:
        self.messages = messages
        self.uidls = uidls or [f"uid-{i + 1}" for i in range(len(messages))]
        self._user: str = ""
        self._dele_called = False
        self._quit_called = False

    def user(self, username: str) -> None:
        self._user = username

    def pass_(self, _password: str) -> None:
        pass

    def list(self):
        listing = [f"{i + 1} {len(m)}".encode() for i, m in enumerate(self.messages)]
        return (b"+OK", listing, sum(len(m) for m in self.messages))

    def uidl(self):
        listing = [
            f"{i + 1} {uidl}".encode() for i, uidl in enumerate(self.uidls)
        ]
        return (b"+OK", listing, 0)

    def retr(self, msg_no: int):
        # poplib returns lines without trailing CRLF
        raw = self.messages[msg_no - 1]
        lines = raw.split(b"\r\n")
        if lines and lines[-1] == b"":
            lines.pop()  # strip trailing empty from final \r\n
        return (b"+OK", lines, len(raw))

    def quit(self) -> None:
        self._quit_called = True


async def test_connect_uses_pop3_ssl_and_authenticates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POP3_PASSWORD", "secret")
    fake = _FakePop3([])

    def fake_ssl(_host: str, _port: int) -> _FakePop3:
        return fake

    with patch("poplib.POP3_SSL", side_effect=fake_ssl):
        adapter = Pop3Adapter(_settings())
        await adapter.connect()

    assert adapter.client is fake
    assert fake._user == "ops@example.com"
    await adapter.close()


async def test_connect_raises_on_missing_password_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POP3_PASSWORD", raising=False)
    adapter = Pop3Adapter(_settings())

    with pytest.raises(KeyError):
        await adapter.connect()


async def test_fetch_first_sync_returns_top_n_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POP3_PASSWORD", "secret")
    msgs = [b"msg1\r\n", b"msg2\r\n", b"msg3\r\n", b"msg4\r\n", b"msg5\r\n"]
    fake = _FakePop3(msgs)

    with patch("poplib.POP3_SSL", side_effect=lambda *_a: fake):
        adapter = Pop3Adapter(_settings())
        await adapter.connect()

    cursor = FetchCursor(
        uidvalidity=None, after_uid=None, since=None, batch_size=3, protocol="pop3"
    )
    result = [item async for item in adapter.fetch(cursor)]

    # Top 3 messages: msg3, msg4, msg5 (by message number 3, 4, 5)
    assert len(result) == 3
    assert [m.uid for m in result] == [3, 4, 5]
    assert [m.server_id for m in result] == ["uid-3", "uid-4", "uid-5"]
    assert all(m.uidvalidity is None for m in result)
    assert result[0].raw_bytes == b"msg3\r\n"
    await adapter.close()


async def test_fetch_normal_poll_returns_only_new_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POP3_PASSWORD", "secret")
    msgs = [b"msg1\r\n", b"msg2\r\n", b"msg3\r\n", b"msg4\r\n", b"msg5\r\n"]
    fake = _FakePop3(msgs)

    with patch("poplib.POP3_SSL", side_effect=lambda *_a: fake):
        adapter = Pop3Adapter(_settings())
        await adapter.connect()

    cursor = FetchCursor(
        uidvalidity=None,
        after_uid=99,
        since=None,
        batch_size=50,
        protocol="pop3",
        seen_server_ids=frozenset({"uid-1", "uid-2"}),
    )
    result = [item async for item in adapter.fetch(cursor)]

    # UIDLs, not renumberable message positions, determine novelty.
    assert len(result) == 3
    assert [m.uid for m in result] == [3, 4, 5]
    assert [m.server_id for m in result] == ["uid-3", "uid-4", "uid-5"]
    await adapter.close()


async def test_close_issues_quit_and_does_not_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POP3_PASSWORD", "secret")
    fake = _FakePop3([b"msg1\r\n"])

    with patch("poplib.POP3_SSL", side_effect=lambda *_a: fake):
        adapter = Pop3Adapter(_settings())
        await adapter.connect()
        await adapter.close()

    assert fake._quit_called is True
    assert adapter.client is None


async def test_close_is_idempotent() -> None:
    adapter = Pop3Adapter(_settings())
    # close before connect should not raise
    await adapter.close()
    await adapter.close()
    assert adapter.client is None
