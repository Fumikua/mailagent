"""Preprocessing package: subject normalization, thread parsing, and embedding fusion.

Re-exports the public API for the mail preprocessing pipeline.
"""
from __future__ import annotations

from mailagent.preprocessing.pipeline import preprocess_mail
from mailagent.preprocessing.contracts import MailPreprocessingExtension
from mailagent.preprocessing.subject_normalizer import normalize_subject
from mailagent.preprocessing.thread_parser import parse_thread, parse_thread_with_flag
from mailagent.preprocessing.retrieval_document import build_retrieval_document
from mailagent.preprocessing.retrieval_models import (
    RetrievalCleaningPolicy,
    RetrievalDocument,
    load_retrieval_cleaning_policy,
)

__all__ = [
    "normalize_subject",
    "parse_thread",
    "parse_thread_with_flag",
    "preprocess_mail",
    "MailPreprocessingExtension",
    "RetrievalCleaningPolicy",
    "RetrievalDocument",
    "build_retrieval_document",
    "load_retrieval_cleaning_policy",
]
