from pydantic import ValidationError
import pytest

from mailagent.infra.config import MailGatewaySettings, Settings


def test_gateway_defaults_to_safe_from_now_polling() -> None:
    settings = MailGatewaySettings()
    assert settings.initial_sync_mode == "from_now"
    assert settings.seen_filter == "all"
    assert settings.max_message_bytes == 26_214_400
    assert settings.adapter == "imap"
    assert settings.initial_sync_max_messages == 1000


def test_enabled_gateway_rejects_unconfirmed_backfill() -> None:
    with pytest.raises(ValidationError):
        MailGatewaySettings(
            enabled=True, host="imap.example.com", username="ops", password_env="IMAP_PASSWORD",
            initial_sync_mode="bounded_backfill",
        )


def test_yaml_gateway_settings_are_overridden_by_nested_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yml"
    config.write_text(
        "mail_gateway:\n  enabled: false\n  host: yaml.example.com\n  fetch_batch_size: 10\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MAILAGENT_MAIL_GATEWAY__ENABLED", "true")
    monkeypatch.setenv("MAILAGENT_MAIL_GATEWAY__HOST", "env.example.com")
    monkeypatch.setenv("MAILAGENT_MAIL_GATEWAY__FETCH_BATCH_SIZE", "25")
    monkeypatch.setenv("MAILAGENT_MAIL_GATEWAY__USERNAME", "ops")
    monkeypatch.setenv("MAILAGENT_MAIL_GATEWAY__PASSWORD_ENV", "IMAP_PASSWORD")

    settings = Settings.from_yaml(config)

    assert len(settings.mail_gateways) == 1
    gateway = settings.mail_gateways[0]
    assert gateway.enabled is True
    assert gateway.host == "env.example.com"
    assert gateway.fetch_batch_size == 25


def test_yaml_mail_gateways_list_loads_canonical_configuration(tmp_path) -> None:
    config = tmp_path / "config.yml"
    config.write_text(
        "mail_gateways:\n"
        "  - enabled: true\n"
        "    mailbox_id: primary\n"
        "    host: imap.example.com\n"
        "    username: ops\n"
        "    password_env: IMAP_PASSWORD\n",
        encoding="utf-8",
    )

    settings = Settings.from_yaml(config)

    assert [gateway.mailbox_id for gateway in settings.mail_gateways] == ["primary"]


def test_yaml_rejects_mixed_legacy_and_canonical_gateway_forms(tmp_path) -> None:
    config = tmp_path / "config.yml"
    config.write_text(
        "mail_gateway:\n"
        "  enabled: false\n"
        "mail_gateways:\n"
        "  - enabled: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="either mail_gateway or mail_gateways"):
        Settings.from_yaml(config)


def test_duplicate_enabled_mailbox_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate mailbox_id"):
        Settings(
            mail_gateways=[
                MailGatewaySettings(
                    enabled=True,
                    mailbox_id="primary",
                    host="imap-a.example.com",
                    username="ops",
                    password_env="IMAP_PASSWORD",
                ),
                MailGatewaySettings(
                    enabled=True,
                    mailbox_id="primary",
                    host="imap-b.example.com",
                    username="ops",
                    password_env="IMAP_PASSWORD",
                ),
            ]
        )


# ---------------------------------------------------------------------------
# POP3 settings validation
# ---------------------------------------------------------------------------


def test_pop3_defaults_port_to_995() -> None:
    settings = MailGatewaySettings(
        adapter="pop3",
        enabled=True,
        host="pop.example.com",
        username="ops",
        password_env="POP3_PASSWORD",
        initial_sync_mode="incremental",
    )
    assert settings.port == 995


def test_pop3_requires_incremental_sync_mode() -> None:
    with pytest.raises(ValidationError, match="incremental"):
        MailGatewaySettings(
            adapter="pop3",
            initial_sync_mode="from_now",
        )


def test_imap_rejects_incremental_sync_mode() -> None:
    with pytest.raises(ValidationError, match="pop3-only"):
        MailGatewaySettings(
            adapter="imap",
            initial_sync_mode="incremental",
        )


def test_initial_sync_max_messages_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        MailGatewaySettings(initial_sync_max_messages=0)


# ---------------------------------------------------------------------------
# Additional settings + Worker-registration coverage (mail-gateway-imap task 1.8)
# ---------------------------------------------------------------------------


def test_poll_interval_must_be_at_least_60_seconds() -> None:
    with pytest.raises(ValidationError):
        MailGatewaySettings(poll_interval_seconds=30)


def test_poll_interval_must_be_a_multiple_of_60() -> None:
    with pytest.raises(ValidationError, match="multiple of 60"):
        MailGatewaySettings(poll_interval_seconds=90)


def test_fetch_batch_size_upper_bound_is_enforced() -> None:
    with pytest.raises(ValidationError):
        MailGatewaySettings(fetch_batch_size=101)


def test_max_message_bytes_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        MailGatewaySettings(max_message_bytes=0)


def test_backfill_since_days_upper_bound_is_enforced() -> None:
    with pytest.raises(ValidationError):
        MailGatewaySettings(backfill_since_days=8)


def test_disabled_gateway_does_not_require_connection_fields() -> None:
    # Disabled gateways may carry only mailbox_id; the validator must not
    # reject a half-populated entry used as a placeholder.
    settings = MailGatewaySettings(enabled=False, mailbox_id="staging")
    assert settings.enabled is False
    assert settings.host == ""


def test_enabled_gateway_requires_all_connection_fields() -> None:
    with pytest.raises(ValidationError, match="requires host, username, password_env, and mailbox_id"):
        MailGatewaySettings(
            enabled=True,
            host="imap.example.com",
            username="ops",
            # password_env intentionally missing
        )


def test_non_tls_use_ssl_is_rejected() -> None:
    # use_ssl is Literal[True]; assigning False is a pydantic type error.
    with pytest.raises(ValidationError):
        MailGatewaySettings(use_ssl=False)  # type: ignore[arg-type]


def test_enabled_gateway_with_complete_fields_is_accepted() -> None:
    settings = MailGatewaySettings(
        enabled=True,
        mailbox_id="primary",
        host="imap.example.com",
        username="ops",
        password_env="IMAP_PASSWORD",
    )
    assert settings.enabled is True
    assert settings.use_ssl is True
