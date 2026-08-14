from mailagent.infra.config import PolicySettings
from mailagent.domain import ProposedAction
from mailagent.domain.policy import DefaultPolicyEngine


def test_high_risk_actions_never_execute() -> None:
    action = ProposedAction(type="send_email", risk="high", requires_approval=False, preview="send it")
    checked = DefaultPolicyEngine(PolicySettings()).apply([action])[0]
    assert checked.status == "blocked"
    assert checked.requires_approval is True
