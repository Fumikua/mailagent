"""Mail-understanding orchestration above the classification foundation.

Classification re-exports remain for one compatibility release. New code
should import classifiers and contracts from :mod:`mailagent.classification`.
"""

from mailagent.classification.contracts import (  # noqa: F401
    AttemptStatus,
    ClassificationAttempt,
    ClassificationCoreResult,
    ClassificationOrchestrator,
    ClassificationRequest,
    Classifier,
    Enricher,
    EnrichmentPatch,
)
from .orchestration import CascadeClassificationOrchestrator  # noqa: F401
from mailagent.classification.llm_classifier import LLMClassifier  # noqa: F401
from .pipeline import MailUnderstandingPipeline  # noqa: F401
from mailagent.classification.rule_classifier import RuleClassifier  # noqa: F401
from mailagent.classification.vector_classifier import VectorClassifier  # noqa: F401
from .fusion_orchestrator import FusionOrchestrator  # noqa: F401
from .target_profile import TargetProfile, TargetProfileLoader  # noqa: F401
from .versioning import ClassificationVersionProvider  # noqa: F401
