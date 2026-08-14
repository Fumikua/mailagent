"""OpenAI-compatible LLM 客户端，基于 httpx.AsyncClient + tenacity 重试。

支持 DeepSeek/Qwen/OpenAI/Ollama 等兼容 OpenAI API 的服务。
429 Rate limit 不重试，直接抛出 RateLimitError 让上层 fallback。
超时、网络错误、5xx 服务端错误均重试 3 次（指数退避 2-10s）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用基础异常"""


class RateLimitError(LLMError):
    """429 Rate limit — 不重试，直接抛出"""


class LLMTimeoutError(LLMError):
    """LLM 调用超时"""


class LLMNetworkError(LLMError):
    """LLM 网络错误"""


class LLMResponseError(LLMError):
    """LLM 返回内容无效（malformed JSON / function call 缺失）"""


class LLMClient:
    """OpenAI-compatible LLM 客户端

    - 超时 30s（生产可用）
    - tenacity 重试：3 次，exponential backoff 2-10s
    - 针对 TimeoutException/NetworkError/5xx 服务端错误重试
    - 429 Rate limit 不重试，直接抛 RateLimitError
    - 支持 function calling（tools 参数）
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """调用 /v1/chat/completions，支持 function calling

        Args:
            messages: OpenAI messages 数组
            tools: function calling 工具定义
            tool_choice: "auto" / "none" / {"type": "function", "function": {"name": "..."}}
            temperature: 0.0 表示确定性输出

        Returns:
            dict: OpenAI chat completion 响应

        Raises:
            RateLimitError: 429
            LLMTimeoutError: 超时
            LLMNetworkError: 网络错误
            LLMResponseError: 响应无效
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.NetworkError, LLMNetworkError)
            ),
            reraise=True,
        )

        try:
            async for attempt in retrying:
                with attempt:
                    return await self._do_call(payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"LLM call timed out after {self.timeout}s") from exc
        except httpx.NetworkError as exc:
            raise LLMNetworkError(f"LLM network error: {exc}") from exc
        except RetryError as exc:  # 理论不可达，reraise=True
            raise LLMError(f"retry exhausted: {exc}") from exc
        return {}  # 不可达

    async def _do_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        try:
            response = await client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException:
            raise  # 让 tenacity 重试
        except httpx.NetworkError:
            raise  # 让 tenacity 重试

        if response.status_code == 429:
            raise RateLimitError(f"LLM rate limited (429): {response.text}")

        if response.status_code >= 500:
            raise LLMNetworkError(f"LLM server error {response.status_code}: {response.text}")

        if response.status_code >= 400:
            raise LLMResponseError(f"LLM client error {response.status_code}: {response.text}")

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"LLM response not JSON: {exc}") from exc

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
