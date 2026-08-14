from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from typing import Iterable

from .models import AttachmentMeta, MailEvent


def dedup_key_for_message(raw: bytes, message_id: str | None) -> str:
    value = (message_id or "").strip()
    return value.casefold() if value else hashlib.sha256(raw).hexdigest()


def _recipients(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _body(message) -> str:
    html = ""
    for part in message.walk() if message.is_multipart() else (message,):
        if part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type not in {"text/plain", "text/html"}:
            continue
        try:
            value = part.get_content()
        except Exception:
            value = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
        if content_type == "text/plain":
            return str(value)
        html = str(value)
    return html


def _attachments(message) -> Iterable[AttachmentMeta]:
    for part in message.walk():
        filename = part.get_filename()
        if not filename or part.get_content_disposition() != "attachment":
            continue
        yield AttachmentMeta(
            filename=filename,
            content_type=part.get_content_type(),
            size=len(part.get_payload(decode=True) or b""),
        )


def parse_email_message(raw: bytes, mailbox_id: str) -> MailEvent:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    date = message.get("date")
    try:
        received_at = parsedate_to_datetime(date) if date else datetime.now(timezone.utc)
    except (TypeError, ValueError):
        received_at = datetime.now(timezone.utc)
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=timezone.utc)
    metadata = list(_attachments(message))
    message_id = str(message.get("message-id") or "") or dedup_key_for_message(raw, None)
    return MailEvent(
        message_id=message_id,
        sender=str(message.get("from") or ""),
        subject=str(message.get("subject") or ""),
        body=_body(message),
        recipients=_recipients(str(message.get("to") or "")),
        mailbox_id=mailbox_id,
        received_at=received_at,
        attachments=[item.filename for item in metadata],
        attachment_meta=metadata or None,
    )
