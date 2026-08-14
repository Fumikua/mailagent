"""Unit tests for preprocess_mail: thread parsing + recency-weighted embedding fusion.

Covers two-segment fusion (0.6/0.4), single-segment collapse, three-segment
context concatenation, and the single-batch-API-call invariant. All embedding
calls are mocked via a fake EmbeddingClient.
"""
from __future__ import annotations

import pytest

from mailagent.domain.models import MailEvent
from mailagent.preprocessing.pipeline import preprocess_mail


class _FakeEmbeddingClient:
    """Fake EmbeddingClient that records embed_batch calls and returns canned vectors."""

    def __init__(self, return_vectors: list[list[float]]) -> None:
        self._return_vectors = return_vectors
        self.calls: list[list[str]] = []

    async def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return self._return_vectors


def _mail(body: str, subject: str = "Test subject") -> MailEvent:
    return MailEvent(
        message_id="test-1",
        sender="ops@example.com",
        subject=subject,
        body=body,
    )


class TestTwoSegmentFusion:
    """Two-segment thread: embedding_thread = 0.6*e0 + 0.4*e_ctx."""

    async def test_two_segment_fusion_weights(self) -> None:
        e0 = [1.0, 2.0, 3.0]
        e_ctx = [4.0, 5.0, 6.0]
        client = _FakeEmbeddingClient([e0, e_ctx])
        body = "Latest reply\n> Original message"
        mail = _mail(body)

        segments, embedding_thread, embedding_segment_0, _normalized = await preprocess_mail(
            mail, client  # type: ignore[arg-type]
        )

        assert len(segments) == 2
        assert embedding_segment_0 == e0
        expected_thread = [0.6 * a + 0.4 * b for a, b in zip(e0, e_ctx)]
        assert embedding_thread == pytest.approx(expected_thread)


class TestSingleSegment:
    """Single-segment body: embedding_thread == embedding_segment_0."""

    async def test_single_segment_thread_equals_segment_0(self) -> None:
        e0 = [1.0, 2.0, 3.0]
        # ctx_text == seg0_text, so the mock returns the same vector for both.
        client = _FakeEmbeddingClient([e0, e0])
        body = "Just a plain message with no quotes."
        mail = _mail(body)

        segments, embedding_thread, embedding_segment_0, _normalized = await preprocess_mail(
            mail, client  # type: ignore[arg-type]
        )

        assert len(segments) == 1
        assert embedding_segment_0 == e0
        assert embedding_thread == pytest.approx(e0)

    async def test_single_segment_ctx_equals_seg0_text(self) -> None:
        """When only one segment exists, embed_batch receives [seg0, seg0]."""
        e0 = [1.0, 2.0]
        client = _FakeEmbeddingClient([e0, e0])
        mail = _mail("Plain body")

        await preprocess_mail(mail, client)  # type: ignore[arg-type]

        assert len(client.calls) == 1
        assert client.calls[0][0] == client.calls[0][1]


class TestThreeSegmentFusion:
    """Latest reply paired with the immediately preceding quoted segment."""

    async def test_three_segment_context_concatenation(self) -> None:
        e0 = [1.0, 0.0]
        e_ctx = [0.0, 1.0]
        client = _FakeEmbeddingClient([e0, e_ctx])
        body = "Latest reply\n> First quote\n>> Deeper quote"
        mail = _mail(body)

        segments, embedding_thread, embedding_segment_0, _normalized = await preprocess_mail(
            mail, client  # type: ignore[arg-type]
        )

        assert len(segments) >= 2
        # The retrieval query pairs subject + latest message with the immediately
        # preceding quoted segment (ask+answer). ctx_text is the quoted "question".
        assert len(client.calls) == 1
        assert client.calls[0][0] == (
            "Subject: Test subject\nLatest message:\nLatest reply\n"
            "Context:\nFirst quote\nDeeper quote"
        )
        assert client.calls[0][1] == "First quote\nDeeper quote"
        # Verify fusion formula.
        expected_thread = [0.6 * a + 0.4 * b for a, b in zip(e0, e_ctx)]
        assert embedding_thread == pytest.approx(expected_thread)
        assert embedding_segment_0 == e0


class TestEmbedBatchCallCount:
    """embed_batch must be invoked exactly once (single batch API call)."""

    async def test_embed_batch_called_once_single_segment(self) -> None:
        client = _FakeEmbeddingClient([[1.0, 2.0], [1.0, 2.0]])
        mail = _mail("Plain body")

        await preprocess_mail(mail, client)  # type: ignore[arg-type]

        assert len(client.calls) == 1
        assert len(client.calls[0]) == 2

    async def test_embed_batch_called_once_multi_segment(self) -> None:
        client = _FakeEmbeddingClient([[1.0, 2.0], [3.0, 4.0]])
        body = "Latest\n> Quote 1\n>> Quote 2"
        mail = _mail(body)

        await preprocess_mail(mail, client)  # type: ignore[arg-type]

        assert len(client.calls) == 1
        assert len(client.calls[0]) == 2


class TestSegmentIntegrity:
    """Returned segments are the raw parse_thread_with_flag output."""

    async def test_segments_are_thread_segments(self) -> None:
        from mailagent.domain.models import ThreadSegment

        client = _FakeEmbeddingClient([[1.0], [2.0]])
        body = "Latest reply\n> Original message"
        mail = _mail(body)

        segments, _, _, _ = await preprocess_mail(mail, client)  # type: ignore[arg-type]

        assert all(isinstance(s, ThreadSegment) for s in segments)
        assert segments[0].position == 0
        assert segments[0].is_latest is True


class TestNormalizedSubjectExposure:
    """``preprocess_mail`` exposes only generic normalized subject data."""

    async def test_normalized_subject_has_no_vertical_fields(self) -> None:
        from mailagent.domain.models import NormalizedSubject

        client = _FakeEmbeddingClient([[1.0, 0.0], [0.0, 1.0]])
        mail = _mail(
            "Body",
            subject="Long Term STATUS: AE7/123/BM40/Berlin Example/East - STATUS STN Jul 25, 14:00 LT",
        )

        _, _, _, normalized = await preprocess_mail(mail, client)  # type: ignore[arg-type]

        assert isinstance(normalized, NormalizedSubject)
        assert normalized.model_dump() == {"raw": mail.subject, "clean": mail.subject}


class TestVerticalPreprocessingInjection:
    async def test_vertical_fields_are_auditable_but_not_added_to_raw_mail(self) -> None:
        class _Extension:
            async def enrich(self, mail: MailEvent, normalized: object) -> dict[str, str]:
                assert mail.subject == "STATUS update"
                return {"example.reference": "STATUS-42"}

        client = _FakeEmbeddingClient([[1.0], [1.0]])
        mail = _mail("STATUS revised to 14:00 tomorrow.", subject="STATUS update")

        result = await preprocess_mail(
            mail,
            client,  # type: ignore[arg-type]
            extension=_Extension(),  # type: ignore[arg-type]
        )

        assert result.retrieval_document.extracted_fields == {"example.reference": "STATUS-42"}
        assert mail.subject == "STATUS update"


class TestIneligibleDocument:
    async def test_skips_embedding_for_attachment_only_message(self) -> None:
        client = _FakeEmbeddingClient([[1.0], [1.0]])
        mail = _mail("Please see attached document.", subject="Document")

        result = await preprocess_mail(mail, client)  # type: ignore[arg-type]

        assert result.retrieval_document.eligible is False
        assert result.embedding_thread == []
        assert client.calls == []

    async def test_normalized_subject_returned_for_plain_subject(self) -> None:
        """Plain subjects also remain generic."""
        from mailagent.domain.models import NormalizedSubject

        client = _FakeEmbeddingClient([[1.0], [1.0]])
        mail = _mail("Body", subject="Quick question about schedule")

        _, _, _, normalized = await preprocess_mail(mail, client)  # type: ignore[arg-type]

        assert isinstance(normalized, NormalizedSubject)
        assert normalized.model_dump() == {"raw": mail.subject, "clean": mail.subject}
