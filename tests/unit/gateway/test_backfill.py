from __future__ import annotations

from types import SimpleNamespace

import pytest

from mailagent.gateway import backfill
from mailagent.infra.config import MailGatewaySettings, Settings


class _State:
    def __init__(self) -> None:
        self.audits: list[tuple[str, int, int]] = []

    async def record_backfill_audit(
        self, mailbox_id: str, *, since_days: int, max_messages: int
    ) -> None:
        self.audits.append((mailbox_id, since_days, max_messages))


def _imap(mailbox_id: str) -> MailGatewaySettings:
    return MailGatewaySettings(
        enabled=True,
        mailbox_id=mailbox_id,
        host=f"{mailbox_id}.example.com",
        username="ops",
        password_env="IMAP_PASSWORD",
    )


async def test_confirmed_backfill_selects_requested_imap_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    captured: dict[str, object] = {}

    async def fake_poll(ctx: dict) -> str:
        captured["settings"] = ctx["settings"]
        return "accepted: 0"

    monkeypatch.setattr(backfill, "mail_poll_job", fake_poll)
    ctx = {
        "settings": Settings(mail_gateways=[_imap("primary"), _imap("secondary")]),
        "mail_gateway_state": state,
    }

    result = await backfill.run_confirmed_backfill(
        ctx,
        confirmed=True,
        since_days=2,
        max_messages=25,
        mailbox_id="secondary",
    )

    assert result == "accepted: 0"
    selected = captured["settings"]
    assert isinstance(selected, Settings)
    assert [gateway.mailbox_id for gateway in selected.mail_gateways] == ["secondary"]
    assert selected.mail_gateways[0].initial_sync_mode == "bounded_backfill"
    assert state.audits == [("secondary", 2, 25)]


async def test_confirmed_backfill_rejects_pop3_gateway() -> None:
    pop3 = MailGatewaySettings(
        enabled=True,
        adapter="pop3",
        mailbox_id="pop",
        host="pop.example.com",
        username="ops",
        password_env="POP_PASSWORD",
        initial_sync_mode="incremental",
    )
    ctx = {
        "settings": Settings(mail_gateways=[pop3]),
        "mail_gateway_state": SimpleNamespace(),
    }

    with pytest.raises(ValueError, match="IMAP"):
        await backfill.run_confirmed_backfill(
            ctx,
            confirmed=True,
            since_days=1,
            max_messages=10,
        )
