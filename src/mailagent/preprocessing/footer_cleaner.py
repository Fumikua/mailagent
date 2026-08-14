"""Conservative, policy-driven signature and disclaimer removal."""
from __future__ import annotations

import re

from mailagent.preprocessing.retrieval_models import RetrievalCleaningPolicy
from mailagent.preprocessing.text_hygiene import (
    collapse_blank_lines,
    normalize_text,
    redact_credential_assignments,
)


def clean_message_section(
    text: str, policy: RetrievalCleaningPolicy
) -> tuple[str, list[str]]:
    """Return readable content after configured mechanical footer cleanup.

    A signature is removed only from its explicit delimiter to the end of the
    current section.  This intentionally leaves unfamiliar text untouched.
    """

    flags: list[str] = []
    lines = normalize_text(text).split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if any(
            stripped.casefold().startswith(delimiter.casefold())
            for delimiter in policy.signature_delimiters
        ):
            lines = lines[:index]
            flags.append("signature_removed")
            break
    cleaned = collapse_blank_lines("\n".join(lines))
    cleaned, credential_redacted = redact_credential_assignments(cleaned)
    if credential_redacted:
        flags.append("credential_redacted")
    for pattern in policy.disclaimer_patterns:
        replaced = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if replaced != cleaned:
            flags.append("disclaimer_removed")
            cleaned = collapse_blank_lines(replaced)
    return cleaned, flags
