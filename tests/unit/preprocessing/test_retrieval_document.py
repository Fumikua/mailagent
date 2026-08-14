from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mailagent.domain.models import MailEvent
from mailagent.preprocessing.retrieval_document import build_retrieval_document
from mailagent.preprocessing.retrieval_models import (
    RetrievalCleaningPolicy,
    load_retrieval_cleaning_policy,
    validated_policy_version,
)


def _policy() -> RetrievalCleaningPolicy:
    return RetrievalCleaningPolicy(
        version="test-v1",
        latest_max_chars=120,
        context_max_chars=40,
        min_meaningful_chars=4,
        signature_delimiters=("Kind regards", "-- ", "此致"),
        disclaimer_patterns=(r"This email may contain confidential information.*",),
    )


def test_loaded_cleaning_policy_version_ignores_later_disk_changes(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "retrieval_cleaning.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": "clean-v1",
                "latest_max_chars": 120,
                "min_meaningful_chars": 4,
            }
        ),
        encoding="utf-8",
    )
    policy = load_retrieval_cleaning_policy(policy_path)
    before = validated_policy_version(policy)

    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": "clean-v2",
                "latest_max_chars": 999,
                "min_meaningful_chars": 4,
            }
        ),
        encoding="utf-8",
    )

    assert validated_policy_version(policy) == before
    assert policy.version == "clean-v1"
    assert policy.latest_max_chars == 120


@pytest.mark.asyncio
async def test_builds_auditable_document_from_subject_and_latest_message() -> None:
    mail = MailEvent(
        message_id="clean-1",
        sender="ops@example.com",
        subject="Re: STATUS update",
        body=(
            "STATUS revised to 14:00 tomorrow. Please arrange location.\n\n"
            "Kind regards,\nAlice\n-- \n"
            "This email may contain confidential information.\n\n"
            "On yesterday wrote:\n> Previous STATUS was 10:00."
        ),
    )

    result = await build_retrieval_document(mail, _policy())

    # The immediately preceding quoted segment is retained as context
    # (ask+answer pairing). With context_max_chars=40 the quote is truncated.
    assert result.text == (
        "Subject: STATUS update\n"
        "Latest message:\n"
        "STATUS revised to 14:00 tomorrow. Please arrange location.\n"
        "Context:\n"
        "On yesterday wrote:\n"
        "Previous STATUS was"
    )
    assert result.eligible is True
    assert set(result.flags) >= {"reply_prefix_removed", "signature_removed", "context_truncated"}
    assert result.latest_char_count == len(result.primary_text)
    assert result.context_char_count == 39
    assert mail.body.startswith("STATUS revised")


@pytest.mark.asyncio
async def test_marks_attachment_only_or_footer_only_mail_ineligible() -> None:
    mail = MailEvent(
        message_id="clean-2",
        sender="ops@example.com",
        subject="Document attached",
        body="Please see attached document.\n\nKind regards,\nAlice",
    )

    result = await build_retrieval_document(mail, _policy())

    assert result.eligible is False
    assert result.ineligible_reason == "attachment_dependent"


@pytest.mark.asyncio
async def test_attachment_only_mail_with_filenames_uses_name_embedding() -> None:
    """attachment_only 邮件有附件名时改用"主题+附件名"做 embedding，不再排除。"""
    mail = MailEvent(
        message_id="attach-only-names",
        sender="ops@example.com",
        subject="Pre-event documents",
        body="Please find the attached documents.\nRegards",
        attachments=["Crew_List.pdf", "DG_Declaration.xlsx"],
    )

    result = await build_retrieval_document(mail, _policy())

    assert result.eligible is True
    assert result.ineligible_reason is None
    assert "attachment_name_embedded" in result.flags
    assert "Subject: Pre-event documents" in result.text
    assert "Attachments: Crew_List.pdf, DG_Declaration.xlsx" in result.text
    # body 模板内容不应进入 embedding 输入
    assert "Regards" not in result.text


@pytest.mark.asyncio
async def test_converts_html_without_embedding_markup() -> None:
    mail = MailEvent(
        message_id="clean-3",
        sender="ops@example.com",
        subject="Schedule",
        body="<html><body><p>STATUS <b>14:00</b></p><script>alert(1)</script></body></html>",
    )

    result = await build_retrieval_document(mail, _policy())

    assert "STATUS 14:00" in result.text
    assert "<" not in result.text
    assert "html_to_text" in result.flags


@pytest.mark.asyncio
async def test_cleans_chinese_footer_and_control_characters_without_removing_request() -> None:
    mail = MailEvent(
        message_id="clean-4",
        sender="ops@example.com",
        subject="回复: 靠泊安排",
        body="\u200b请安排明日 08:00 靠泊。\x00\n\n此致\n船代部",
    )

    result = await build_retrieval_document(mail, _policy())

    assert result.primary_text == "请安排明日 08:00 靠泊。"
    assert result.eligible is True
    assert "signature_removed" in result.flags


@pytest.mark.asyncio
async def test_caps_content_and_marks_signature_only_message_ineligible() -> None:
    policy = _policy().model_copy(update={"latest_max_chars": 10})
    mail = MailEvent(
        message_id="clean-5",
        sender="ops@example.com",
        subject="Update",
        body="Kind regards,\nAlice",
    )

    result = await build_retrieval_document(mail, policy)

    assert result.primary_text == ""
    assert result.eligible is False
    assert result.ineligible_reason == "insufficient_meaningful_content"


@pytest.mark.asyncio
async def test_caps_oversize_latest_content_and_records_the_change() -> None:
    policy = _policy().model_copy(update={"latest_max_chars": 10})
    mail = MailEvent(
        message_id="clean-oversize",
        sender="ops@example.com",
        subject="STATUS",
        body="STATUS revised to 14:00 tomorrow; please arrange the location immediately.",
    )

    result = await build_retrieval_document(mail, policy)

    assert result.primary_text == "STATUS rev"
    assert result.latest_char_count == 10
    assert "latest_truncated" in result.flags


@pytest.mark.asyncio
async def test_redacts_obvious_credentials_from_derived_text_only() -> None:
    mail = MailEvent(
        message_id="clean-6",
        sender="ops@example.com",
        subject="System access",
        body="Please use API_KEY=abc123secret to retrieve the schedule update.",
    )

    result = await build_retrieval_document(mail, _policy())

    assert "abc123secret" not in result.text
    assert "credential redacted" in result.text
    assert "abc123secret" in mail.body
    assert "credential_redacted" in result.flags


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "subject", "body", "protected_fact", "removed_noise"),
    [
        (
            "schedule",
            "Re: Location schedule",
            "MV PACIFIC DAWN STATUS 2026-07-28 08:00; please reserve location 3.\n\nKind regards,\nOps",
            "MV PACIFIC DAWN STATUS 2026-07-28 08:00; please reserve location 3.",
            "Kind regards",
        ),
        (
            "report",
            "FW: Status report",
            "STATUS REPORT: REFERENCE ITEM position 31.2304N / 121.4737E, fuel ROB 420 MT.\n\nKind regards,\nMaster",
            "REFERENCE ITEM position 31.2304N / 121.4737E, fuel ROB 420 MT.",
            "Kind regards",
        ),
        (
            "document",
            "Shipping documents",
            "Original B/L number MAEU123456789 and manifest are ready for collection.\n\nKind regards,\nDocumentation",
            "B/L number MAEU123456789 and manifest are ready for collection.",
            "Kind regards",
        ),
        (
            "service_request",
            "回复: Crew service request",
            "请为 MV HAI XING 安排 3 名人员在 2026-07-29 09:00 换班。\n\n此致\n船代部",
            "请为 MV HAI XING 安排 3 名人员在 2026-07-29 09:00 换班。",
            "船代部",
        ),
        (
            "broadcast",
            "Port schedule broadcast",
            "All agents: typhoon warning closes Yangshan terminal from 18:00 today.\n\nKind regards,\nPort control",
            "typhoon warning closes Yangshan terminal from 18:00 today.",
            "Kind regards",
        ),
    ],
    ids=lambda scenario: scenario,
)
async def test_business_scenarios_preserve_protected_facts_after_cleaning(
    scenario: str,
    subject: str,
    body: str,
    protected_fact: str,
    removed_noise: str,
) -> None:
    """Representative mail types retain operational facts in the derived view."""

    mail = MailEvent(
        message_id=f"scenario-{scenario}",
        sender="ops@example.com",
        subject=subject,
        body=body,
    )

    result = await build_retrieval_document(mail, _policy())

    assert result.eligible is True
    assert protected_fact in result.text
    assert removed_noise not in result.text
    assert mail.body == body


@pytest.mark.asyncio
async def test_attachments_filenames_appended_to_retrieval_text() -> None:
    """Attachment filenames carrying operational intent are surfaced in text."""
    mail = MailEvent(
        message_id="attach-1",
        sender="ops@example.com",
        subject="Re: DG list",
        body="Please find the DG list for tomorrow's locationing.",
        attachments=["DG_List_MV_PACIFIC.xlsx", "Crew_List.pdf"],
    )

    result = await build_retrieval_document(mail, _policy())

    assert result.eligible is True
    assert "Attachments: DG_List_MV_PACIFIC.xlsx, Crew_List.pdf" in result.text
    assert "attachments_listed" in result.flags


@pytest.mark.asyncio
async def test_attachments_deduplicated_and_truncated() -> None:
    """Duplicate names collapse and overly long lists are truncated with a flag."""
    duplicate_name = "Manifest.pdf"
    long_names = [f"Document_{i:03d}.pdf" for i in range(20)]
    mail = MailEvent(
        message_id="attach-2",
        sender="ops@example.com",
        subject="Documents",
        body="Please review the attached documents for the journey.",
        attachments=[duplicate_name, duplicate_name] + long_names,
    )

    result = await build_retrieval_document(mail, _policy())

    assert result.eligible is True
    # Duplicate collapsed: the second Manifest.pdf must not appear twice.
    assert result.text.count("Manifest.pdf") == 1
    assert "attachments_truncated" in result.flags
    assert len(result.text.split("Attachments: ", 1)[1]) <= _policy().attachments_max_chars


@pytest.mark.asyncio
async def test_no_attachments_section_when_list_empty() -> None:
    """Mails without attachments must not emit an Attachments: section."""
    mail = MailEvent(
        message_id="attach-3",
        sender="ops@example.com",
        subject="Plain",
        body="A plain message with no attachments and enough meaningful content.",
    )

    result = await build_retrieval_document(mail, _policy())

    assert result.eligible is True
    assert "Attachments:" not in result.text
    assert "attachments_listed" not in result.flags
