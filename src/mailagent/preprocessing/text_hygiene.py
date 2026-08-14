"""Generic, deterministic text normalization used before retrieval embedding."""
from __future__ import annotations

import re


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CREDENTIAL_RE = re.compile(
    r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Remove invisible controls and normalize line endings without changing meaning."""

    return (
        _CONTROL_RE.sub("", text)
        .replace("\u200b", "")
        .replace("\xa0", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def collapse_blank_lines(text: str) -> str:
    """Trim trailing whitespace and reduce runs of blank lines to one."""

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def redact_credential_assignments(text: str) -> tuple[str, bool]:
    """Replace obvious credential assignments in derived retrieval text only."""

    redacted = _CREDENTIAL_RE.sub("[credential redacted]", text)
    return redacted, redacted != text
