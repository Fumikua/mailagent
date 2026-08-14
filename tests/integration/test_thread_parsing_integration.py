"""End-to-end integration tests for thread parsing (Section 18.2).

Validates ``parse_thread`` and the preprocessing pipeline's 0.6/0.4 fusion on
synthetic thread emails that mirror real Outlook / RFC 3676 quote structures.

Covers:
  - Two-segment thread split + quote prefix stripping
  - Three-segment nested quote chain
  - HTML-only degradation (single segment, thread_parsed=False)
  - ``preprocess_mail`` 0.6/0.4 fusion with a mocked EmbeddingClient
  - Single-segment collapse: embedding_thread == embedding_segment_0
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mailagent.domain.models import MailEvent
from mailagent.llm.embedding import EmbeddingClient
from mailagent.preprocessing.pipeline import preprocess_mail
from mailagent.preprocessing.thread_parser import parse_thread, parse_thread_with_flag

_DIM = 4


def _embedding(seed: float) -> list[float]:
    """Deterministic embedding vector for the mock client."""
    return [seed + i * 0.01 for i in range(_DIM)]


def _mock_embedding_client(
    e0: list[float] | None = None, e_ctx: list[float] | None = None
) -> MagicMock:
    """Build a mock EmbeddingClient returning [e0, e_ctx] from embed_batch."""
    client = MagicMock(spec=EmbeddingClient)
    e0 = e0 if e0 is not None else _embedding(0.1)
    e_ctx = e_ctx if e_ctx is not None else _embedding(0.2)
    # embed_batch is awaited inside preprocess_mail, so it must be an AsyncMock.
    client.embed_batch = AsyncMock(return_value=[e0, e_ctx])
    return client


# ---------------------------------------------------------------------------
# Tests: parse_thread on real-world thread structures
# ---------------------------------------------------------------------------


class TestThreadSplitIntegration:
    """Synthetic thread emails: segment splitting + quote prefix stripping."""

    def test_outlook_two_segment_split(self) -> None:
        """Chinese Outlook '发件人:' header splits a thread into two segments."""
        body = (
            "Hi team,\n\nPlease find the updated STATUS below.\n\n"
            "发件人: Alice <alice@example.com>\nSent: 2026-07-20\n"
            "Subject: Original STATUS\n\nOriginal message body here."
        )
        segments = parse_thread(body)
        assert len(segments) == 2
        assert "Please find the updated STATUS" in segments[0].text
        assert segments[0].is_latest is True
        assert segments[0].position == 0
        assert "发件人: Alice" in segments[1].text
        assert segments[1].is_latest is False

    def test_rfc3676_quoted_reply_chain(self) -> None:
        """RFC 3676 '>' quote chain splits into multiple segments."""
        body = (
            "Latest reply text.\n\n"
            "> First-level quote.\n>> Deeper quote line.\n"
        )
        segments = parse_thread(body)
        assert len(segments) >= 2
        assert segments[0].text == "Latest reply text."
        assert all(not seg.is_latest for seg in segments[1:])

    def test_chinese_outlook_reply_split(self) -> None:
        """Chinese Outlook '发件人:' header splits a thread."""
        body = (
            "请确认STATUS变更。\n\n"
            "发件人: bob@example.com\n"
            "主题: 原始邮件\n\n原始内容。"
        )
        segments = parse_thread(body)
        assert len(segments) == 2
        assert "请确认STATUS变更" in segments[0].text
        assert "发件人: bob@example.com" in segments[1].text

    def test_html_body_degradation(self) -> None:
        """HTML-only body returns a single segment with thread_parsed=False."""
        body = "<html><body><p>Reply text</p></body></html>"
        segments, flag = parse_thread_with_flag(body)
        assert len(segments) == 1
        assert flag is False
        assert segments[0].is_latest is True

    def test_no_quote_single_segment(self) -> None:
        """Body without quotes returns a single segment with thread_parsed=False."""
        body = "Just a plain message body."
        segments, flag = parse_thread_with_flag(body)
        assert len(segments) == 1
        assert flag is False


# ---------------------------------------------------------------------------
# Tests: preprocess_mail 0.6/0.4 fusion
# ---------------------------------------------------------------------------


class TestPreprocessMailFusion:
    """End-to-end preprocess_mail: 0.6/0.4 fusion of seg0 + context embeddings."""

    async def test_two_segment_fusion_weights(self) -> None:
        """Two-segment thread: embedding_thread = 0.6*e0 + 0.4*e_ctx."""
        e0 = _embedding(0.1)
        e_ctx = _embedding(0.5)
        client = _mock_embedding_client(e0=e0, e_ctx=e_ctx)

        mail = MailEvent(
            message_id="thread-1",
            sender="ops@example.com",
            subject="STATUS update",
            body="Latest reply\n> Original message",
        )
        segments, embedding_thread, embedding_segment_0, _normalized = await preprocess_mail(
            mail, client
        )

        assert len(segments) == 2
        assert embedding_segment_0 == e0
        # Fusion: 0.6 * e0 + 0.4 * e_ctx, per-element.
        expected = [0.6 * a + 0.4 * b for a, b in zip(e0, e_ctx, strict=True)]
        assert embedding_thread == pytest.approx(expected, rel=1e-9)

    async def test_single_segment_collapses_to_e0(self) -> None:
        """Single-segment thread: embedding_thread == embedding_segment_0 == e0."""
        e0 = _embedding(0.7)
        client = _mock_embedding_client(e0=e0, e_ctx=e0)

        mail = MailEvent(
            message_id="single",
            sender="ops@example.com",
            subject="Plain subject",
            body="Plain body without quotes.",
        )
        segments, embedding_thread, embedding_segment_0, _normalized = await preprocess_mail(
            mail, client
        )

        assert len(segments) == 1
        assert embedding_segment_0 == e0
        assert embedding_thread == pytest.approx(e0, rel=1e-9)

    async def test_embed_batch_called_once(self) -> None:
        """preprocess_mail issues exactly one batch embedding call."""
        client = _mock_embedding_client()
        mail = MailEvent(
            message_id="once",
            sender="ops@example.com",
            subject="Subject",
            body="Please find the updated STATUS below.",
        )
        await preprocess_mail(mail, client)
        client.embed_batch.assert_called_once()

    async def test_three_segment_fusion_uses_mean_of_context(self) -> None:
        """Three-segment thread: context = seg1 + seg2 (joined by space)."""
        e0 = _embedding(0.1)
        e_ctx = _embedding(0.9)
        client = _mock_embedding_client(e0=e0, e_ctx=e_ctx)

        mail = MailEvent(
            message_id="three",
            sender="ops@example.com",
            subject="Re: thread",
            body=(
                "Latest segment.\n"
                "> First quoted segment.\n"
                ">> Second quoted segment."
            ),
        )
        segments, embedding_thread, embedding_segment_0, _normalized = await preprocess_mail(
            mail, client
        )

        assert len(segments) >= 2
        assert embedding_segment_0 == e0
        expected = [0.6 * a + 0.4 * b for a, b in zip(e0, e_ctx, strict=True)]
        assert embedding_thread == pytest.approx(expected, rel=1e-9)
