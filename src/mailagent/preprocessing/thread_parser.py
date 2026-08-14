"""Thread parsing: quote block detection and segment splitting.

The parser detects quote blocks in email body using multiple patterns
(RFC 3676 ``>`` prefixes, pipe prefixes, ``On ... wrote:`` headers,
Chinese Outlook ``发件人:`` headers, ``---`` / ``____`` separators, etc.)
and splits the body into ordered segments (newest first).

HTML-only emails are not parsed; the whole body is treated as a single
segment with ``thread_parsed=False``.
"""
from __future__ import annotations

import re

from mailagent.domain.models import ThreadSegment

# ---------------------------------------------------------------------------
# Quote block start patterns. A line matching any of these marks the beginning
# of a quoted / reply block. Detection is performed line by line.
# ---------------------------------------------------------------------------
_QUOTE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^>+"),  # RFC 3676 quote prefix (one or more '>')
    re.compile(r"^\|"),  # Pipe-prefixed quote
    re.compile(r"^On .+ wrote:", re.IGNORECASE),  # English reply header
    re.compile(r"^---"),  # Separator line (e.g. --- Original Message ---)
    re.compile(r"^在 .+写道:"),  # Chinese reply header
    re.compile(r"^发件人"),  # Chinese Outlook From header
    re.compile(r"^_{2,}"),  # Underscore separator (2+ underscores)
]

# ---------------------------------------------------------------------------
# Patterns for cleaning quote prefixes from individual lines within a segment.
# Applied per line to strip leading ``>`` / ``|`` markers and trailing whitespace.
# ---------------------------------------------------------------------------
_QUOTE_PREFIX_CLEANERS: list[re.Pattern[str]] = [
    re.compile(r"^>+\s*"),  # > prefix(es) with trailing whitespace
    re.compile(r"^\|+\s*"),  # | prefix(es) with trailing whitespace
]

# HTML detection: body starting with <html or <!DOCTYPE (case-insensitive, allows leading whitespace)
_HTML_PREFIX_RE = re.compile(r"^\s*(?:<html|<!DOCTYPE)", re.IGNORECASE)


def _is_html_body(body: str) -> bool:
    """Detect HTML-only email body by checking for HTML/DOCTYPE prefix."""
    return bool(_HTML_PREFIX_RE.match(body))


def _clean_quote_prefixes(text: str) -> str:
    """Remove quote prefixes (``>`` / ``|``) from each line of *text*."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line
        for pattern in _QUOTE_PREFIX_CLEANERS:
            stripped = pattern.sub("", stripped)
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def _find_quote_starts(lines: list[str]) -> list[int]:
    """Return line indices where quote blocks begin.

    A quote block begins at the first line of a consecutive run of
    quote-matching lines. Consecutive quote lines are grouped into the
    same block so that multi-line quoted segments are not split apart.
    """
    quote_starts: list[int] = []
    prev_was_quote = False
    for i, line in enumerate(lines):
        is_quote = any(pattern.search(line) for pattern in _QUOTE_PATTERNS)
        if is_quote and not prev_was_quote:
            quote_starts.append(i)
        prev_was_quote = is_quote
    return quote_starts


def parse_thread_with_flag(body: str) -> tuple[list[ThreadSegment], bool]:
    """Parse email body into ordered thread segments (latest first).

    Returns a tuple of ``(segments, thread_parsed)``.

    * ``thread_parsed=True`` — quote blocks were detected and the body was split
      into multiple segments.
    * ``thread_parsed=False`` — HTML-only email (not parsed) or no quote patterns
      found (single segment, nothing to split).

    Args:
        body: The raw email body text.

    Returns:
        A tuple of (list of ThreadSegment ordered newest-first, thread_parsed flag).
    """
    # HTML-only degradation: return single segment, not parsed.
    if _is_html_body(body):
        return [ThreadSegment(text=body, position=0, is_latest=True)], False

    lines = body.split("\n")
    quote_starts = _find_quote_starts(lines)

    # No quote patterns found: single segment, not parsed.
    if not quote_starts:
        return [ThreadSegment(text=body.strip(), position=0, is_latest=True)], False

    # Split at quote boundaries.
    segments_raw: list[str] = []

    # Latest segment: everything before the first quote block.
    first_quote_idx = quote_starts[0]
    latest = "\n".join(lines[:first_quote_idx]).strip()
    if latest:
        segments_raw.append(latest)

    # Quoted segments: each runs from its quote-start line until the next
    # quote-start (or end of body). Quote prefixes are cleaned from each segment.
    boundaries = quote_starts + [len(lines)]
    for i in range(len(quote_starts)):
        start = quote_starts[i]
        end = boundaries[i + 1]
        segment = "\n".join(lines[start:end])
        cleaned = _clean_quote_prefixes(segment)
        if cleaned:
            segments_raw.append(cleaned)

    segments = [
        ThreadSegment(text=text, position=i, is_latest=(i == 0))
        for i, text in enumerate(segments_raw)
    ]
    return segments, True


def parse_thread(body: str) -> list[ThreadSegment]:
    """Parse email body into ordered thread segments (latest first).

    Convenience wrapper around :func:`parse_thread_with_flag` that discards
    the ``thread_parsed`` flag.
    """
    segments, _ = parse_thread_with_flag(body)
    return segments
