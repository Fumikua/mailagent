"""Generic subject normalization.

Vertical-specific field extraction is deliberately performed by an injected
``MailPreprocessingExtension`` rather than this generic module.
"""
from __future__ import annotations

import re

from mailagent.domain.models import NormalizedSubject


_PREFIX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^re\[\d+\][\s:_]+", re.IGNORECASE),
    re.compile(r"^re[\s:_]+", re.IGNORECASE),
    re.compile(r"^回复[\s:_]+", re.IGNORECASE),
    re.compile(r"^转发[\s:_]+", re.IGNORECASE),
    re.compile(r"^fwd[\s:_]+", re.IGNORECASE),
    re.compile(r"^【外部邮件】\s*", re.IGNORECASE),
    re.compile(r"^\[re\]\s*", re.IGNORECASE),
]


def _strip_prefixes(text: str) -> str:
    prev: str | None = None
    current = text
    while prev != current:
        prev = current
        for pattern in _PREFIX_PATTERNS:
            current = pattern.sub("", current, count=1)
    return current


def _fold_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def normalize_subject(raw: str) -> NormalizedSubject:
    """Strip generic reply prefixes and normalize whitespace."""

    return NormalizedSubject(raw=raw, clean=_fold_whitespace(_strip_prefixes(raw)))
