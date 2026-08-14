from mailagent.classification.contracts import ClassificationAttempt
from mailagent.classification.llm_classifier import LLMClassifier
from mailagent.classification.rule_classifier import RuleClassifier
from mailagent.classification.taxonomy import TaxonomyLoader
from mailagent.classification.vector_classifier import VectorClassifier
from mailagent.core.classification import ClassificationAttempt as LegacyClassificationAttempt
from mailagent.core.llm_classifier import LLMClassifier as LegacyLLMClassifier
from mailagent.core.rule_classifier import RuleClassifier as LegacyRuleClassifier
from mailagent.core.vector_classifier import VectorClassifier as LegacyVectorClassifier
from mailagent.llm.taxonomy import TaxonomyLoader as LegacyTaxonomyLoader


def test_legacy_import_facades_preserve_object_identity() -> None:
    assert LegacyClassificationAttempt is ClassificationAttempt
    assert LegacyLLMClassifier is LLMClassifier
    assert LegacyRuleClassifier is RuleClassifier
    assert LegacyVectorClassifier is VectorClassifier
    assert LegacyTaxonomyLoader is TaxonomyLoader
