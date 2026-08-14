"""Embedding client supporting TEI and OpenAI-compatible APIs.

Provides batch and single-text embedding via either:
- TEI (Text Embeddings Inference) local service (``POST /embed``)
- OpenAI-compatible API (``POST /v1/embeddings``) — e.g. SiliconFlow,
  Azure OpenAI, vLLM OpenAI server.

Uses tenacity for retry (3 attempts, exponential backoff) and falls back
to single-text embedding when the batch endpoint is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mailagent.infra.config import EmbeddingSettings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Embedding client base error."""


class EmbeddingServerError(EmbeddingError):
    """TEI returned 5xx — retryable."""


class EmbeddingTimeoutError(EmbeddingError):
    """Embedding call timed out after retries."""


class EmbeddingNetworkError(EmbeddingError):
    """Embedding network error after retries."""


class EmbeddingResponseError(EmbeddingError):
    """Embedding response invalid or 4xx client error — not retryable."""


def _parse_embeddings(data: Any) -> list[list[float]]:
    """Parse TEI ``/embed`` response into a list of embedding vectors.

    Handles two response formats:
    - TEI native: ``[[0.1, 0.2, ...], [0.3, 0.4, ...]]`` (list of list of float)
    - HF-style: ``{"data": [{"embedding": [0.1, 0.2, ...]}, ...]}``

    Also handles a single bare embedding vector ``[0.1, 0.2, ...]`` by wrapping
    it in a list.
    """
    if isinstance(data, list):
        if len(data) == 0:
            return []
        first = data[0]
        if isinstance(first, list):
            return data  # Already a list of embedding vectors
        if isinstance(first, (int, float)):
            return [data]  # Single bare embedding — wrap it
        raise EmbeddingResponseError(f"Unexpected TEI response element type: {type(first)}")
    if isinstance(data, dict) and "data" in data:
        return [item["embedding"] for item in data["data"]]
    raise EmbeddingResponseError(f"Unexpected TEI response format: {type(data)}")


class EmbeddingClient:
    """Async TEI embedding client with batch API, retry, and single-text fallback.

    - Batch path: ``POST /embed`` with ``{"inputs": [text1, text2, ...]}``
    - Single path: ``POST /embed`` with ``{"inputs": "text"}``
    - tenacity retry: 3 attempts, exponential backoff 2-10s
    - Retryable: TimeoutException / NetworkError / 5xx (EmbeddingServerError)
    - Non-retryable: 4xx (EmbeddingResponseError) — triggers immediate fallback
    - When batch fails after retries or on 4xx, falls back to embed_single loop
      and sets ``embedding_fallback = True``
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self.provider = settings.provider.lower()
        self.api_base = settings.api_base.rstrip("/")
        self.timeout = settings.timeout
        self.embedding_fallback: bool = False
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.provider == "openai":
                api_key = os.getenv(self.settings.api_key_env, "")
                if not api_key:
                    raise EmbeddingResponseError(
                        f"OpenAI embedding provider requires env var "
                        f"{self.settings.api_key_env} to be set"
                    )
                headers["Authorization"] = f"Bearer {api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
            )
        return self._client

    async def _do_call(self, inputs: list[str] | str) -> list[list[float]]:
        """Low-level HTTP call to the embedding endpoint. No retry logic.

        Routes to the correct endpoint based on ``provider``:
        - ``tei``: ``POST /embed`` with ``{"inputs": [...]}``
        - ``openai``: ``POST /v1/embeddings`` with ``{"input": [...], "model": "..."}``

        Raises:
            EmbeddingServerError: 5xx (retryable)
            EmbeddingResponseError: 4xx or invalid JSON (not retryable)
            httpx.TimeoutException: timeout (retryable, let tenacity catch)
            httpx.NetworkError: network error (retryable, let tenacity catch)
        """
        client = await self._get_client()
        if self.provider == "openai":
            url = "/v1/embeddings"
            payload: dict[str, Any] = {
                "input": inputs,
                "model": self.settings.model_name,
            }
        else:
            url = "/embed"
            payload = {"inputs": inputs}
        try:
            response = await client.post(url, json=payload)
        except (httpx.TimeoutException, httpx.NetworkError):
            raise  # Let tenacity retry

        if response.status_code >= 500:
            raise EmbeddingServerError(
                f"Embedding server error {response.status_code}: {response.text}"
            )
        if response.status_code >= 400:
            raise EmbeddingResponseError(
                f"Embedding client error {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise EmbeddingResponseError(f"Embedding response not JSON: {exc}") from exc

        return _parse_embeddings(data)

    def _build_retrying(self) -> AsyncRetrying:
        """Build a tenacity AsyncRetrying instance with standard config."""
        return AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.NetworkError, EmbeddingServerError)
            ),
            reraise=True,
        )

    async def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed multiple texts via TEI batch API.

        Splits *texts* into batches of *batch_size* and calls ``/embed`` once
        per batch. When a batch call fails (after retries or on 4xx), falls back
        to :meth:`embed_single` for each text in that batch and sets
        ``self.embedding_fallback = True``.

        Args:
            texts: List of text strings to embed.
            batch_size: Maximum number of texts per batch API call.

        Returns:
            List of embedding vectors (one per input text), in the same order.
        """
        self.embedding_fallback = False
        results: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                retrying = self._build_retrying()
                embeddings: list[list[float]] = []
                async for attempt in retrying:
                    with attempt:
                        embeddings = await self._do_call(batch)
                results.extend(embeddings)
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                EmbeddingServerError,
                EmbeddingResponseError,
            ):
                # Fallback: embed each text individually.
                self.embedding_fallback = True
                logger.warning(
                    "TEI batch embed failed, falling back to single-text embed"
                )
                for text in batch:
                    embedding = await self.embed_single(text)
                    results.append(embedding)

        return results

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text via TEI API.

        Used as the normal single-text path and as the fallback when batch
        API is unavailable. Retries 3 times on timeout / network / 5xx.

        Raises:
            EmbeddingTimeoutError: Timeout after all retries.
            EmbeddingNetworkError: Network error after all retries.
            EmbeddingServerError: 5xx after all retries.
            EmbeddingResponseError: 4xx or invalid response (not retried).
        """
        retrying = self._build_retrying()
        try:
            embeddings: list[list[float]] = []
            async for attempt in retrying:
                with attempt:
                    embeddings = await self._do_call(text)
            return embeddings[0]
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError(
                f"Embedding call timed out after {self.timeout}s"
            ) from exc
        except httpx.NetworkError as exc:
            raise EmbeddingNetworkError(f"Embedding network error: {exc}") from exc
        except RetryError as exc:  # Unreachable with reraise=True
            raise EmbeddingError(f"Retry exhausted: {exc}") from exc
        return []  # Unreachable

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
