"""Generic extension contracts for selected-vertical mail preprocessing."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from mailagent.domain.models import MailEvent, NormalizedSubject


class MailPreprocessingExtension(Protocol):
    """Optionally enrich a normalized mail with vertical-owned string fields."""

    async def enrich(
        self,
        mail: MailEvent,
        normalized_subject: NormalizedSubject,
        *,
        snapshot: Any | None = None,
    ) -> Mapping[str, str]: ...

    def get_snapshot(self) -> Any: ...
