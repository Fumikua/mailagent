"""Deprecated compatibility facade for :mod:`mailagent.classification`."""

from mailagent.classification.rule_classifier import (
    BodyKeywordRule,
    RuleClassifier,
    SenderDomainRule,
    StructuralRule,
    SubjectPatternRule,
    validate_rule_labels,
)

__all__ = [
    "BodyKeywordRule",
    "RuleClassifier",
    "SenderDomainRule",
    "StructuralRule",
    "SubjectPatternRule",
    "validate_rule_labels",
]
