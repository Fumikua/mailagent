"""Deprecated compatibility facade for classification taxonomy configuration."""

from mailagent.classification.taxonomy import (
    TaxonomyLoader,
    TaxonomyNode,
    TaxonomyTree,
    load_taxonomy,
    serialize_for_prompt,
)

__all__ = [
    "TaxonomyLoader",
    "TaxonomyNode",
    "TaxonomyTree",
    "load_taxonomy",
    "serialize_for_prompt",
]
