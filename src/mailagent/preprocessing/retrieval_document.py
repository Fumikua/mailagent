"""Build the same clean, bounded email representation for query and samples."""
from __future__ import annotations

from collections.abc import Mapping
import re

from mailagent.domain.models import MailEvent
from mailagent.preprocessing.html_to_text import html_to_text
from mailagent.preprocessing.footer_cleaner import clean_message_section
from mailagent.preprocessing.retrieval_models import RetrievalCleaningPolicy, RetrievalDocument
from mailagent.preprocessing.subject_normalizer import normalize_subject
from mailagent.preprocessing.thread_parser import parse_thread_with_flag


_HTML_RE = re.compile(r"^\s*<(?:html|!doctype)", re.IGNORECASE)
_ATTACHMENT_ONLY_RE = re.compile(r"\b(?:see|please find)\s+(?:the\s+)?attached\b", re.IGNORECASE)


async def build_retrieval_document(
    mail: MailEvent,
    policy: RetrievalCleaningPolicy,
    *,
    extracted_fields: Mapping[str, str] | None = None,
) -> RetrievalDocument:
    flags: list[str] = []
    subject = normalize_subject(mail.subject)
    if subject.clean != mail.subject:
        flags.append("reply_prefix_removed")
    body = mail.body
    if _HTML_RE.match(body):
        body = html_to_text(body)
        flags.append("html_to_text")
    segments, parsed = parse_thread_with_flag(body)
    if parsed:
        flags.append("quoted_history_removed")
    latest_raw = segments[0].text if segments else ""
    latest, latest_flags = clean_message_section(latest_raw, policy)
    flags.extend(latest_flags)
    # Pair the latest reply with the immediately preceding quoted segment
    # (the "question" it answers) so embeddings carry the ask+answer context.
    # For reply chains with no quote block, fall back to single-segment
    # behavior (only the latest message is embedded).
    context_raw = segments[1].text if parsed and len(segments) >= 2 else ""
    context, context_flags = clean_message_section(context_raw, policy)
    flags.extend(flag for flag in context_flags if flag not in flags)
    if len(latest) > policy.latest_max_chars:
        latest = latest[: policy.latest_max_chars].strip()
        flags.append("latest_truncated")
    if len(context) > policy.context_max_chars:
        context = context[: policy.context_max_chars].strip()
        flags.append("context_truncated")
    sections = [f"Subject: {subject.clean}", "Latest message:", latest]
    if context:
        sections.extend(["Context:", context])
    # Attachment filenames often carry operational intent (DG list, crew list,
    # B/L copy) that the body alone cannot express.  Append a deduplicated,
    # length-bounded comma-separated list so embeddings can surface them
    # without overwhelming the embedding input window.
    attachment_list = ""
    if mail.attachments:
        seen: set[str] = set()
        unique_names: list[str] = []
        for name in mail.attachments:
            stripped = name.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique_names.append(stripped)
        attachment_list = ", ".join(unique_names)
        if len(attachment_list) > policy.attachments_max_chars:
            attachment_list = attachment_list[: policy.attachments_max_chars].strip()
            flags.append("attachments_truncated")
        if attachment_list:
            sections.append(f"Attachments: {attachment_list}")
            flags.append("attachments_listed")
    text = "\n".join(section for section in sections if section).strip()
    attachment_only = bool(_ATTACHMENT_ONLY_RE.search(latest))
    # A concise body can still be a useful request when its normalized subject
    # carries the operational intent, so assess the exact primary view rather
    # than body text in isolation.
    meaningful = len(re.sub(r"\W", "", f"{subject.clean} {latest}"))
    if attachment_only:
        # body 只是"请查收附件"类模板，语义价值集中在附件文件名。
        # 不再判定为 ineligible，改用"主题+附件名"作为 embedding 输入，
        # 让这类靠附件的邮件也能入库参与向量检索。
        if attachment_list:
            name_text = f"Subject: {subject.clean}\nAttachments: {attachment_list}".strip()
            flags.append("attachment_name_embedded")
            return RetrievalDocument(text=name_text, primary_text=name_text, context_text="", policy_version=policy.version, flags=flags, eligible=True, extracted_fields=dict(extracted_fields or {}), latest_char_count=len(latest), context_char_count=0)
        return RetrievalDocument(text=text, primary_text=latest, context_text=context, policy_version=policy.version, flags=flags, eligible=False, ineligible_reason="attachment_dependent", extracted_fields=dict(extracted_fields or {}), latest_char_count=len(latest), context_char_count=len(context))
    if not latest:
        return RetrievalDocument(text=text, primary_text=latest, context_text=context, policy_version=policy.version, flags=flags, eligible=False, ineligible_reason="insufficient_meaningful_content", extracted_fields=dict(extracted_fields or {}), latest_char_count=len(latest), context_char_count=len(context))
    if meaningful < policy.min_meaningful_chars:
        return RetrievalDocument(text=text, primary_text=latest, context_text=context, policy_version=policy.version, flags=flags, eligible=False, ineligible_reason="insufficient_meaningful_content", extracted_fields=dict(extracted_fields or {}), latest_char_count=len(latest), context_char_count=len(context))
    return RetrievalDocument(text=text, primary_text=latest, context_text=context, policy_version=policy.version, flags=flags, eligible=True, extracted_fields=dict(extracted_fields or {}), latest_char_count=len(latest), context_char_count=len(context))
