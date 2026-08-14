"""Unit tests for thread parsing.

Covers standard quote detection (RFC 3676 ``>`` prefix), Chinese Outlook
``发件人:`` header detection, no-quote single-segment, two-segment split
with prefix stripping, HTML email degradation, and ThreadSegment field integrity.
"""
from __future__ import annotations

from mailagent.domain.models import ThreadSegment
from mailagent.preprocessing.thread_parser import parse_thread, parse_thread_with_flag


class TestQuoteDetection:
    """Standard quote block detection scenarios."""

    def test_gt_prefix_detected(self) -> None:
        """Lines starting with '>' are detected as quote blocks."""
        body = "Latest reply\n> Original message"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[0].text == "Latest reply"
        assert segments[1].text == "Original message"

    def test_pipe_prefix_detected(self) -> None:
        """Lines starting with '|' are detected as quote blocks."""
        body = "My reply\n| Original text"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[0].text == "My reply"
        assert segments[1].text == "Original text"

    def test_on_wrote_detected(self) -> None:
        """'On ... wrote:' English reply header is detected as quote start."""
        body = "My reply\nOn Monday, Alice wrote:\nOriginal text here"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[0].text == "My reply"
        assert "On Monday, Alice wrote:" in segments[1].text

    def test_separator_detected(self) -> None:
        """'---' separator line is detected as quote start."""
        body = "Latest content\n---\nOriginal content"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[0].text == "Latest content"

    def test_underscore_separator_detected(self) -> None:
        """Underscore separator (____) is detected as quote start."""
        body = "Latest content\n______\nOriginal content"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[0].text == "Latest content"


class TestChineseOutlookDetection:
    """Chinese Outlook / reply header detection scenarios."""

    def test_chinese_from_header_detected(self) -> None:
        """'发件人:' Chinese Outlook From header is detected as quote start."""
        body = "This is the latest reply.\n发件人: alice@example.com\nSubject: Original\nOriginal message body"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[0].text == "This is the latest reply."
        assert "发件人: alice@example.com" in segments[1].text

    def test_chinese_reply_header_detected(self) -> None:
        """'在 ...写道:' Chinese reply header is detected as quote start."""
        body = "My reply\n在 2026年7月20日写道:\nOriginal content"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[0].text == "My reply"
        assert "在 2026年7月20日写道:" in segments[1].text


class TestNoQuoteSingleSegment:
    """No quote pattern found — single segment returned."""

    def test_no_quote_returns_single_segment(self) -> None:
        """Body without any quote patterns returns a single segment."""
        body = "Just a plain email body with no quotes."
        segments, flag = parse_thread_with_flag(body)
        assert len(segments) == 1
        assert segments[0].text == body
        assert flag is False

    def test_no_quote_multiline_single_segment(self) -> None:
        """Multi-line body without quote patterns returns a single segment."""
        body = "Line one\nLine two\nLine three"
        segments, flag = parse_thread_with_flag(body)
        assert len(segments) == 1
        assert flag is False


class TestTwoSegmentSplit:
    """Two-segment thread split with quote prefix stripping."""

    def test_two_segment_split_strips_gt_prefix(self) -> None:
        """Two-segment split: '>' prefix is stripped from quoted segment."""
        body = "Latest reply\n> Original message"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[0].text == "Latest reply"
        assert segments[1].text == "Original message"

    def test_two_segment_split_strips_multiple_gt(self) -> None:
        """Multiple '>' levels are stripped from quoted segment."""
        body = "Latest reply\n>> Deeply quoted message"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[1].text == "Deeply quoted message"

    def test_multiline_quoted_segment(self) -> None:
        """Multi-line quoted segment has prefixes stripped from each line."""
        body = "My reply\n> line one\n> line two\n> line three"
        segments = parse_thread(body)
        assert len(segments) == 2
        assert segments[1].text == "line one\nline two\nline three"

    def test_thread_parsed_flag_true_for_split(self) -> None:
        """thread_parsed flag is True when quotes are detected and split occurs."""
        body = "Latest reply\n> Original message"
        _, flag = parse_thread_with_flag(body)
        assert flag is True


class TestHtmlDegradation:
    """HTML-only email degradation — thread_parsed=False."""

    def test_html_body_returns_single_segment(self) -> None:
        """HTML body returns a single segment with thread_parsed=False."""
        body = "<html><body><p>Hello world</p></body></html>"
        segments, flag = parse_thread_with_flag(body)
        assert len(segments) == 1
        assert segments[0].text == body
        assert flag is False

    def test_doctype_html_returns_single_segment(self) -> None:
        """DOCTYPE-prefixed body returns a single segment with thread_parsed=False."""
        body = "<!DOCTYPE html><html><body>Hello</body></html>"
        segments, flag = parse_thread_with_flag(body)
        assert len(segments) == 1
        assert flag is False

    def test_html_with_leading_whitespace(self) -> None:
        """HTML body with leading whitespace is still detected as HTML."""
        body = "  <html><body>Hello</body></html>"
        segments, flag = parse_thread_with_flag(body)
        assert len(segments) == 1
        assert flag is False


class TestThreadSegmentFields:
    """ThreadSegment position and is_latest field integrity."""

    def test_segment_position_zero_is_latest(self) -> None:
        """Segment at position 0 has is_latest=True."""
        body = "Latest reply\n> Original message"
        segments = parse_thread(body)
        assert segments[0].position == 0
        assert segments[0].is_latest is True

    def test_segment_position_one_not_latest(self) -> None:
        """Segment at position 1 has is_latest=False."""
        body = "Latest reply\n> Original message"
        segments = parse_thread(body)
        assert segments[1].position == 1
        assert segments[1].is_latest is False

    def test_single_segment_is_latest(self) -> None:
        """Single segment (no quotes) has position=0 and is_latest=True."""
        body = "Just a plain message"
        segments = parse_thread(body)
        assert len(segments) == 1
        assert segments[0].position == 0
        assert segments[0].is_latest is True

    def test_returns_thread_segment_instances(self) -> None:
        """parse_thread returns ThreadSegment instances."""
        body = "Latest reply\n> Original message"
        segments = parse_thread(body)
        assert all(isinstance(s, ThreadSegment) for s in segments)

    def test_three_segment_positions(self) -> None:
        """Three-segment thread has positions 0, 1, 2 with correct is_latest."""
        body = "Latest reply\n> First quote\n>> Deeper quote"
        segments = parse_thread(body)
        assert len(segments) >= 2
        assert segments[0].position == 0
        assert segments[0].is_latest is True
        for seg in segments[1:]:
            assert seg.is_latest is False
            assert seg.position > 0
