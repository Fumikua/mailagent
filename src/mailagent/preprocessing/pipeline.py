"""Preprocessing pipeline: deterministic cleaning + vertical enrichment + embedding.

The :func:`preprocess_mail` function orchestrates the three preprocessing steps
into a single batch-optimized call:

1. ``build_retrieval_document`` — normalize subject, remove quoted history and
   presentation noise, then assess whether the mail is eligible for retrieval.
2. An optional vertical extension supplies namespaced extracted fields.
3. ``embed_batch`` — one TEI batch API call producing two vectors:
   ``embedding_thread`` (recency-weighted fusion for coarse retrieval) and
   ``embedding_segment_0`` (latest segment alone for fine-grained re-ranking).

When the body has only one segment, both embeddings collapse to ``e0``.

The raw :class:`MailEvent` is never modified.  ``RetrievalDocument`` is the
auditable derived representation used by vector retrieval and sample import.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from mailagent.domain.models import MailEvent, NormalizedSubject, ThreadSegment
from mailagent.llm.embedding import EmbeddingClient
from mailagent.preprocessing.subject_normalizer import normalize_subject
from mailagent.preprocessing.thread_parser import parse_thread_with_flag
from mailagent.preprocessing.contracts import MailPreprocessingExtension
from mailagent.preprocessing.retrieval_document import build_retrieval_document
from mailagent.preprocessing.retrieval_models import RetrievalCleaningPolicy, RetrievalDocument

# Recency-weighted fusion weights: latest segment gets 0.6, context gets 0.4.
_LATEST_WEIGHT = 0.6
_CONTEXT_WEIGHT = 0.4


_DEFAULT_RETRIEVAL_CLEANING_POLICY = RetrievalCleaningPolicy(
    version="generic-v1",
    min_meaningful_chars=10,
    signature_delimiters=("-- ", "Kind regards", "Best regards", "Regards", "谢谢", "此致"),
    disclaimer_patterns=(r"This email (?:and any attachments )?may contain confidential information.*",),
)


@dataclass(slots=True)
class PreprocessResult:
    """Output of :func:`preprocess_mail`.

    Attributes:
        segments: Ordered thread segments (latest first).
        embedding_thread: Recency-weighted fusion vector for coarse retrieval.
        embedding_segment_0: Latest segment embedding for fine-grained re-ranking.
        normalized_subject: Generic normalized subject, with no vertical fields.
        retrieval_document: Canonical derived text and its quality assessment.
    """

    segments: list[ThreadSegment]
    embedding_thread: list[float]
    embedding_segment_0: list[float]
    normalized_subject: NormalizedSubject
    retrieval_document: RetrievalDocument

    def __iter__(self) -> Iterator[object]:
        """Keep the historic four-value unpacking contract during migration."""

        yield self.segments
        yield self.embedding_thread
        yield self.embedding_segment_0
        yield self.normalized_subject


async def preprocess_mail(
    mail_event: MailEvent,
    embedding_client: EmbeddingClient,
    *,
    cleaning_policy: RetrievalCleaningPolicy | None = None,
    extension: MailPreprocessingExtension | None = None,
    extension_snapshot: Any | None = None,
    embed: bool = True,
) -> PreprocessResult:
    """Preprocess one mail into thread segments plus fused embeddings.

    Steps:
        1. Normalize the generic subject and execute the optional vertical
           extension, which returns namespaced fields (e.g.
           ``<vertical_namespace>.<field>``).
        2. Build the deterministic retrieval document from the same cleaning
           policy used for runtime queries and bootstrap samples.
        3. Parse raw body segments for compatibility consumers.
        4. Call ``embed_batch([query_text, context_text])`` — a single batch API
           request producing two embedding vectors.
        5. Fuse: ``embedding_thread = 0.6*e0 + 0.4*e_ctx`` and
           ``embedding_segment_0 = e0``.

    When only one segment exists, ``ctx_text == seg0_text`` so both embeddings
    are identical to ``e0``.

    Args:
        mail_event: The raw incoming mail event.
        embedding_client: TEI embedding client used for the batch embedding call.

    Returns:
        A :class:`PreprocessResult`. Historic four-value unpacking remains
        supported while callers migrate to ``result.retrieval_document``.
    """
    # Step 1: generic subject normalization; vertical-specific extraction is
    # injected at runtime instead of living in this generic module.
    normalized_subject = normalize_subject(mail_event.subject)
    extracted_fields = (
        dict(
            await extension.enrich(
                mail_event,
                normalized_subject,
                **(
                    {"snapshot": extension_snapshot}
                    if extension_snapshot is not None
                    else {}
                ),
            )
        )
        if extension is not None
        else {}
    )
    retrieval_document = await build_retrieval_document(
        mail_event,
        cleaning_policy or _DEFAULT_RETRIEVAL_CLEANING_POLICY,
        extracted_fields=extracted_fields,
    )

    # Step 2: thread parsing remains available to existing callers; retrieval
    # itself embeds the clean document below.
    segments, _thread_parsed = parse_thread_with_flag(mail_event.body)

    # Defensive guard: parse_thread_with_flag always returns at least one
    # segment, but keep the type checker satisfied for an empty-list edge case.
    if not segments:
        segments = [ThreadSegment(text=mail_event.body, position=0, is_latest=True)]

    # An attachment-only or footer-only mail has no trustworthy semantic text.
    # Do not spend an embedding request on it; callers receive the documented
    # eligibility result and can use their normal fallback path.
    if not retrieval_document.eligible or not embed:
        return PreprocessResult(
            segments=segments,
            embedding_thread=[],
            embedding_segment_0=[],
            normalized_subject=normalized_subject,
            retrieval_document=retrieval_document,
        )

    embedding_thread, embedding_segment_0 = await embed_retrieval_document(
        retrieval_document, embedding_client
    )

    return PreprocessResult(
        segments=segments,
        embedding_thread=embedding_thread,
        embedding_segment_0=embedding_segment_0,
        normalized_subject=normalized_subject,
        retrieval_document=retrieval_document,
    )


async def embed_retrieval_document(
    retrieval_document: RetrievalDocument,
    embedding_client: EmbeddingClient,
) -> tuple[list[float], list[float]]:
    """Embed an already-approved retrieval document using the standard fusion."""

    if not retrieval_document.eligible:
        return [], []
    query_text = retrieval_document.text
    ctx_text = retrieval_document.context_text or query_text

    # Single batch API call for both vectors (query + context).
    embeddings = await embedding_client.embed_batch([query_text, ctx_text])
    e0 = embeddings[0]
    e_ctx = embeddings[1]

    # Step 4: recency-weighted fusion.
    embedding_thread = [
        _LATEST_WEIGHT * a + _CONTEXT_WEIGHT * b for a, b in zip(e0, e_ctx)
    ]
    return embedding_thread, e0
