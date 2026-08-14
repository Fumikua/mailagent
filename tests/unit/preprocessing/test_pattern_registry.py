"""Regression guard: generic preprocessing must not export vertical pattern APIs."""
from __future__ import annotations

import mailagent.preprocessing as preprocessing


def test_preprocessing_public_api_has_no_shipping_pattern_registry() -> None:
    assert not hasattr(preprocessing, "load_pattern_registry")
    assert not hasattr(preprocessing, "SubjectPatternEntry")
