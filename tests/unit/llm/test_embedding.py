"""Unit tests for the TEI embedding client.

Covers batch API payload format, 32/batch splitting logic, single-text fallback,
timeout retry, and 404 / 500 error handling. All HTTP calls are mocked.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from mailagent.infra.config import EmbeddingSettings
from mailagent.llm.embedding import (
    EmbeddingClient,
    EmbeddingResponseError,
    EmbeddingTimeoutError,
)


def _make_embeddings(n: int, dim: int = 4) -> list[list[float]]:
    """Generate *n* mock embedding vectors of given *dim*."""
    return [[float(i * dim + j) for j in range(dim)] for i in range(n)]


def _mock_response(status_code: int = 200, data: Any = None) -> httpx.Response:
    """Build an httpx.Response with a JSON body."""
    body = json.dumps(data if data is not None else []).encode()
    return httpx.Response(
        status_code=status_code,
        content=body,
        headers={"content-type": "application/json"},
    )


@pytest.fixture
def embedding_client() -> EmbeddingClient:
    """Create an EmbeddingClient with test settings."""
    return EmbeddingClient(EmbeddingSettings(api_base="http://localhost:8080", timeout=1))


class TestBatchPayloadFormat:
    """Verify the TEI batch API request payload format."""

    async def test_batch_payload_format(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_batch sends {'inputs': [text1, text2]} to /embed."""
        captured: dict[str, Any] = {}

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            captured["url"] = url
            captured["payload"] = json
            inputs = json["inputs"] if json else []
            return _mock_response(200, _make_embeddings(len(inputs)))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        texts = ["hello world", "status report"]
        result = await embedding_client.embed_batch(texts)

        assert captured["url"] == "/embed"
        assert captured["payload"] == {"inputs": texts}
        assert len(result) == 2
        assert result[0] == _make_embeddings(2)[0]
        assert embedding_client.embedding_fallback is False
        await embedding_client.close()

    async def test_batch_response_parsed_correctly(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TEI native response (list of list of float) is parsed correctly."""
        expected = _make_embeddings(3, dim=8)

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            inputs = json["inputs"] if json else []
            return _mock_response(200, _make_embeddings(len(inputs), dim=8))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await embedding_client.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result == expected
        await embedding_client.close()


class TestBatchSplitting:
    """Verify 32/batch splitting logic."""

    async def test_35_texts_split_into_2_batches(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """35 texts with batch_size=32 produce 2 batches (32 + 3)."""
        captured_sizes: list[int] = []

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            inputs = json["inputs"] if json else []
            captured_sizes.append(len(inputs))
            return _mock_response(200, _make_embeddings(len(inputs)))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        texts = [f"text_{i}" for i in range(35)]
        result = await embedding_client.embed_batch(texts, batch_size=32)

        assert len(captured_sizes) == 2
        assert captured_sizes[0] == 32
        assert captured_sizes[1] == 3
        assert len(result) == 35
        await embedding_client.close()

    async def test_exact_batch_size_no_extra_batch(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exactly 32 texts produce 1 batch (no empty second batch)."""
        call_count = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            inputs = json["inputs"] if json else []
            return _mock_response(200, _make_embeddings(len(inputs)))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        texts = [f"text_{i}" for i in range(32)]
        result = await embedding_client.embed_batch(texts, batch_size=32)

        assert call_count == 1
        assert len(result) == 32
        await embedding_client.close()

    async def test_empty_texts_returns_empty(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty text list returns empty result without any HTTP calls."""
        call_count = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _mock_response(200, [])

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await embedding_client.embed_batch([])
        assert result == []
        assert call_count == 0
        await embedding_client.close()


class TestSingleFallback:
    """Single-text fallback when batch API is unavailable."""

    async def test_500_triggers_fallback_to_single(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Batch 500 → retry 3 times → fallback to single (embedding_fallback=True)."""
        call_counts = {"batch": 0, "single": 0}

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            inputs = json["inputs"] if json else []
            if isinstance(inputs, list):
                call_counts["batch"] += 1
                return _mock_response(500, {"error": "server error"})
            call_counts["single"] += 1
            return _mock_response(200, _make_embeddings(1))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await embedding_client.embed_batch(["text1", "text2"])

        assert call_counts["batch"] == 3  # 3 retries on 500
        assert call_counts["single"] == 2  # one per text
        assert embedding_client.embedding_fallback is True
        assert len(result) == 2
        await embedding_client.close()

    async def test_404_triggers_immediate_fallback(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Batch 404 (4xx, not retried) → immediate fallback to single."""
        call_counts = {"batch": 0, "single": 0}

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            inputs = json["inputs"] if json else []
            if isinstance(inputs, list):
                call_counts["batch"] += 1
                return _mock_response(404, {"error": "not found"})
            call_counts["single"] += 1
            return _mock_response(200, _make_embeddings(1))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await embedding_client.embed_batch(["text1", "text2"])

        assert call_counts["batch"] == 1  # 4xx not retried
        assert call_counts["single"] == 2
        assert embedding_client.embedding_fallback is True
        assert len(result) == 2
        await embedding_client.close()

    async def test_fallback_results_correct(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback single-text embeddings are returned in correct order."""
        expected_embeddings = _make_embeddings(2, dim=4)
        single_idx = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            inputs = json["inputs"] if json else []
            if isinstance(inputs, list):
                return _mock_response(500, {"error": "server error"})
            nonlocal single_idx
            emb = expected_embeddings[single_idx]
            single_idx += 1
            return _mock_response(200, [emb])

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await embedding_client.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == expected_embeddings[0]
        assert result[1] == expected_embeddings[1]
        await embedding_client.close()


class TestRetryAndTimeout:
    """Retry and timeout exception handling."""

    async def test_embed_single_timeout_retries_3_times(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_single retries 3 times on timeout, then raises EmbeddingTimeoutError."""
        call_count = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("simulated timeout")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with pytest.raises(EmbeddingTimeoutError):
            await embedding_client.embed_single("test text")

        assert call_count == 3  # 3 retry attempts
        await embedding_client.close()

    async def test_embed_batch_timeout_falls_back_then_raises(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_batch: batch timeout → 3 retries → fallback → single timeout → raise."""
        call_count = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("simulated timeout")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with pytest.raises(EmbeddingTimeoutError):
            await embedding_client.embed_batch(["text1"])

        # 3 batch retries + 3 single retries = 6 total
        assert call_count == 6
        assert embedding_client.embedding_fallback is True
        await embedding_client.close()

    async def test_embed_single_network_error_retries(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_single retries 3 times on network error, then raises."""
        from mailagent.llm.embedding import EmbeddingNetworkError

        call_count = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.NetworkError("simulated network error")

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with pytest.raises(EmbeddingNetworkError):
            await embedding_client.embed_single("test")

        assert call_count == 3
        await embedding_client.close()


class TestErrorResponseHandling:
    """TEI 404 / 500 error response handling."""

    async def test_embed_single_500_retries_then_raises_server_error(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_single: 5xx retries 3 times, then raises EmbeddingServerError."""
        from mailagent.llm.embedding import EmbeddingServerError

        call_count = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _mock_response(500, {"error": "internal server error"})

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with pytest.raises(EmbeddingServerError, match="500"):
            await embedding_client.embed_single("test")

        assert call_count == 3
        await embedding_client.close()

    async def test_embed_single_404_raises_response_error(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_single: 4xx is not retried, raises EmbeddingResponseError."""
        call_count = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _mock_response(404, {"error": "not found"})

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with pytest.raises(EmbeddingResponseError, match="404"):
            await embedding_client.embed_single("test")

        assert call_count == 1  # 4xx not retried
        await embedding_client.close()

    async def test_embed_single_400_raises_response_error(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_single: 400 raises EmbeddingResponseError without retry."""
        call_count = 0

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return _mock_response(400, {"error": "bad request"})

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        with pytest.raises(EmbeddingResponseError, match="400"):
            await embedding_client.embed_single("test")

        assert call_count == 1
        await embedding_client.close()


class TestEmbedSingle:
    """embed_single direct usage."""

    async def test_embed_single_returns_vector(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_single returns a single embedding vector."""
        expected = [0.1, 0.2, 0.3, 0.4]

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            return _mock_response(200, [expected])

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await embedding_client.embed_single("test text")
        assert result == expected
        await embedding_client.close()

    async def test_embed_single_payload_format(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_single sends {'inputs': 'text'} (string, not list)."""
        captured: dict[str, Any] = {}

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            captured["payload"] = json
            return _mock_response(200, _make_embeddings(1))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        await embedding_client.embed_single("test text")
        assert captured["payload"] == {"inputs": "test text"}
        await embedding_client.close()


class TestClose:
    """Client lifecycle management."""

    async def test_close_releases_client(
        self, embedding_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """close() releases the underlying httpx client."""

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            inputs = json["inputs"] if json else []
            return _mock_response(200, _make_embeddings(len(inputs)))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        await embedding_client.embed_batch(["test"])
        await embedding_client.close()
        assert embedding_client._client is None


def _openai_response(n: int, dim: int = 4) -> dict[str, Any]:
    """Build an OpenAI-style /v1/embeddings response body."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [float(i * dim + j) for j in range(dim)]}
            for i in range(n)
        ],
        "model": "Qwen3-Embedding-8B",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch) -> EmbeddingClient:
    """Create an EmbeddingClient configured for an OpenAI-compatible provider."""
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-key")
    settings = EmbeddingSettings(
        provider="openai",
        model_name="Qwen3-Embedding-8B",
        api_base="https://api.example.com",
        dimension=4096,
        timeout=1,
        api_key_env="EMBEDDING_API_KEY",
    )
    return EmbeddingClient(settings)


class TestOpenAIProvider:
    """Verify OpenAI-compatible /v1/embeddings request/response handling."""

    async def test_batch_uses_v1_embeddings_endpoint(
        self, openai_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_batch sends POST /v1/embeddings with {input, model} payload."""
        captured: dict[str, Any] = {}

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            captured["url"] = url
            captured["payload"] = json
            n = len(json["input"]) if json and isinstance(json["input"], list) else 1
            return _mock_response(200, _openai_response(n))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        texts = ["hello", "status report"]
        result = await openai_client.embed_batch(texts)

        assert captured["url"] == "/v1/embeddings"
        assert captured["payload"]["input"] == texts
        assert captured["payload"]["model"] == "Qwen3-Embedding-8B"
        assert len(result) == 2
        assert openai_client.embedding_fallback is False
        await openai_client.close()

    async def test_single_uses_v1_embeddings_endpoint(
        self, openai_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """embed_single sends a string input (not list) to /v1/embeddings."""
        captured: dict[str, Any] = {}

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            captured["url"] = url
            captured["payload"] = json
            return _mock_response(200, _openai_response(1))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await openai_client.embed_single("test text")

        assert captured["url"] == "/v1/embeddings"
        assert captured["payload"]["input"] == "test text"
        assert captured["payload"]["model"] == "Qwen3-Embedding-8B"
        assert len(result) == 4
        await openai_client.close()

    async def test_authorization_header_set(
        self, openai_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI provider sets Authorization: Bearer <key> from env var."""

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            # httpx client-level headers are accessible on self.headers
            auth = self.headers.get("Authorization")
            assert auth == "Bearer sk-test-key"
            n = len(json["input"]) if json and isinstance(json["input"], list) else 1
            return _mock_response(200, _openai_response(n))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        await openai_client.embed_batch(["test"])
        await openai_client.close()

    async def test_missing_api_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI provider without EMBEDDING_API_KEY raises EmbeddingResponseError."""
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        settings = EmbeddingSettings(
            provider="openai",
            api_base="https://api.example.com",
            api_key_env="EMBEDDING_API_KEY",
        )
        client = EmbeddingClient(settings)

        with pytest.raises(EmbeddingResponseError, match="EMBEDDING_API_KEY"):
            await client.embed_batch(["test"])
        await client.close()

    async def test_openai_response_parsed_correctly(
        self, openai_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI {data: [{embedding: [...]}]} response is parsed into vectors."""

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            n = len(json["input"]) if json and isinstance(json["input"], list) else 1
            return _mock_response(200, _openai_response(n, dim=8))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await openai_client.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(vec) == 8 for vec in result)
        await openai_client.close()

    async def test_404_triggers_single_fallback(
        self, openai_client: EmbeddingClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """4xx on batch falls back to embed_single per text (OpenAI provider)."""
        call_counts = {"batch": 0, "single": 0}

        async def fake_post(
            self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None
        ) -> httpx.Response:
            inp = json["input"] if json else None
            if isinstance(inp, list):
                call_counts["batch"] += 1
                return _mock_response(404, {"error": "not found"})
            call_counts["single"] += 1
            return _mock_response(200, _openai_response(1))

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        result = await openai_client.embed_batch(["text1", "text2"])

        assert call_counts["batch"] == 1
        assert call_counts["single"] == 2
        assert openai_client.embedding_fallback is True
        assert len(result) == 2
        await openai_client.close()
