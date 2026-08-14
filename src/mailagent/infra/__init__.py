"""Infrastructure — config, store, queue, worker.

Note: BootstrapPipeline and RuleLearner are NOT re-exported here to avoid a
circular import (rule_learner/bootstrap → core.rule_classifier → preprocessing
→ llm.embedding → infra). Import them directly from their modules instead.
"""
from mailagent.infra.clustering import ClusteringEngine  # noqa: F401
from mailagent.infra.vector_store import VectorStore  # noqa: F401
