"""Unit tests for the shared RFC822 mail parser (mail-gateway-imap task 2.5).

Covers plain text, HTML fallback, multipart attachments, missing Message-ID
stability, empty body, and backward-compatible API payloads.
"""

from email.message import EmailMessage

from mailagent.domain.mail_parser import dedup_key_for_message, parse_email_message
from mailagent.domain.models import MailEvent


def _build_message(
    *,
    message_id: str | None = "<test@example.com>",
    sender: str = "ops@example.com",
    subject: str = "STATUS change",
    to: str = "agent@exampleco.com",
    date: str | None = "Mon, 1 Jul 2026 08:00:00 +0000",
    body_plain: str | None = "Entity STATUS 2026-08-01 08:00 UTC.",
    body_html: str | None = None,
    attachments: list[tuple[str, str, bytes]] | None = None,
) -> bytes:
    msg = EmailMessage()
    if message_id is not None:
        msg["Message-ID"] = message_id
    msg["From"] = sender
    msg["Subject"] = subject
    msg["To"] = to
    if date is not None:
        msg["Date"] = date

    if body_plain is not None and body_html is not None:
        msg.set_content(body_plain)
        msg.add_alternative(body_html, subtype="html")
    elif body_html is not None:
        msg.set_content(body_html, subtype="html")
    elif body_plain is not None:
        msg.set_content(body_plain)
    # else: leave body empty.

    for filename, content_type, payload in attachments or []:
        msg.add_attachment(payload, maintype=content_type.split("/")[0], subtype=content_type.split("/")[1], filename=filename)
    return bytes(msg)


def test_parses_plain_text_body_and_headers() -> None:
    raw = _build_message(body_plain="Hello ship agent.")

    event = parse_email_message(raw, "primary")

    assert event.message_id == "<test@example.com>"
    assert event.sender == "ops@example.com"
    assert event.subject == "STATUS change"
    assert event.body.strip() == "Hello ship agent."
    assert event.recipients == ["agent@exampleco.com"]
    assert event.mailbox_id == "primary"
    assert event.attachments == []
    assert event.attachment_meta is None


def test_html_only_falls_back_to_html_body() -> None:
    raw = _build_message(body_plain=None, body_html="<html><body><p>HTML body</p></body></html>")

    event = parse_email_message(raw, "primary")

    assert "HTML body" in event.body
    # Plain text takes precedence when both are present.
    raw_both = _build_message(body_plain="PLAIN", body_html="<p>HTML</p>")
    event_both = parse_email_message(raw_both, "primary")
    assert event_both.body.strip() == "PLAIN"


def test_multipart_attachments_are_extracted_with_metadata() -> None:
    payload_a = b"DG list content"
    payload_b = b"\x89PNG fake image bytes"
    raw = _build_message(
        body_plain="See attached.",
        attachments=[("dg_list.pdf", "application/pdf", payload_a), ("logo.png", "image/png", payload_b)],
    )

    event = parse_email_message(raw, "primary")

    assert event.attachments == ["dg_list.pdf", "logo.png"]
    assert event.attachment_meta is not None
    assert len(event.attachment_meta) == 2
    meta_a = next(m for m in event.attachment_meta if m.filename == "dg_list.pdf")
    assert meta_a.content_type == "application/pdf"
    assert meta_a.size == len(payload_a)
    meta_b = next(m for m in event.attachment_meta if m.filename == "logo.png")
    assert meta_b.content_type == "image/png"
    assert meta_b.size == len(payload_b)
    # Attachments are not part of the body.
    assert event.body.strip() == "See attached."


def test_missing_message_id_uses_stable_sha256_dedup_key() -> None:
    raw = _build_message(message_id=None, body_plain="No id here.")
    # Strip the Message-ID header if email library still injects one.
    raw_no_id = raw.replace(b"Message-ID:", b"X-Removed:")

    event = parse_email_message(raw_no_id, "primary")

    # The parser falls back to a SHA-256 of the raw bytes; it must be stable
    # across repeated calls and look like a 64-char hex digest.
    assert event.message_id != ""
    assert len(event.message_id) == 64
    assert all(c in "0123456789abcdef" for c in event.message_id)
    # Re-parsing the same bytes yields the same fallback id.
    event2 = parse_email_message(raw_no_id, "primary")
    assert event2.message_id == event.message_id


def test_empty_body_is_returned_as_empty_string() -> None:
    raw = _build_message(body_plain=None, body_html=None)

    event = parse_email_message(raw, "primary")

    assert event.body == ""
    # Headers are still parsed normally.
    assert event.sender == "ops@example.com"


def test_invalid_date_falls_back_to_now_with_timezone() -> None:
    raw = _build_message(date="not-a-real-date")

    event = parse_email_message(raw, "primary")

    assert event.received_at.tzinfo is not None
    # Year should be a recent year (the fallback uses datetime.now).
    assert event.received_at.year >= 2026


def test_dedup_key_uses_message_id_when_present() -> None:
    raw = _build_message(message_id="<unique@example.com>")

    event = parse_email_message(raw, "primary")
    key = dedup_key_for_message(raw, event.message_id)

    assert key == "<unique@example.com>".casefold()


def test_dedup_key_uses_sha256_when_message_id_missing() -> None:
    raw = _build_message(message_id=None)
    raw_no_id = raw.replace(b"Message-ID:", b"X-Removed:")

    event = parse_email_message(raw_no_id, "primary")
    key = dedup_key_for_message(raw_no_id, event.message_id)

    # Same fallback as the parser: SHA-256 hex digest.
    assert key == event.message_id


def test_dedup_key_is_casefolded_for_message_id() -> None:
    # Mixed-case Message-ID should produce the same dedup key as lowercase.
    key = dedup_key_for_message(b"raw", "<Mixed@Case.COM>")
    assert key == "<mixed@case.com>"


def test_backward_compatible_with_api_payload_mail_event() -> None:
    # The API payload constructs MailEvent directly from JSON; ensure the
    # parser output is structurally compatible (same fields/types) so that
    # gateway-created runs and API-created runs are interchangeable.
    raw = _build_message()
    parsed = parse_email_message(raw, "primary")

    api_event = MailEvent(
        message_id="<api@example.com>",
        sender="api@example.com",
        subject="API submission",
        body="Hello",
        recipients=["agent@exampleco.com"],
        mailbox_id="primary",
    )

    # Field set must match (no parser-only or api-only required fields).
    parser_fields = set(type(parsed).model_fields.keys())
    api_fields = set(type(api_event).model_fields.keys())
    assert parser_fields == api_fields
    # Both validate against the same model.
    assert isinstance(parsed, MailEvent)
    assert isinstance(api_event, MailEvent)
