from __future__ import annotations

from ..infra.config import PolicySettings
from .models import ProposedAction


class DefaultPolicyEngine:
    """Agent output is only a proposal; policy owns execution eligibility."""

    def __init__(self, settings: PolicySettings) -> None:
        self.settings = settings

    def apply(self, actions: list[ProposedAction]) -> list[ProposedAction]:
        checked: list[ProposedAction] = []
        for action in actions:
            if action.type in self.settings.blocked_actions:
                checked.append(action.model_copy(update={"risk": "high", "requires_approval": True, "status": "blocked"}))
            elif action.type in self.settings.approval_actions:
                checked.append(action.model_copy(update={"risk": "high", "requires_approval": True}))
            else:
                checked.append(action)
        return checked
