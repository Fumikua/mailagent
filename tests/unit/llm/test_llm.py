"""LLM 客户端测试：mock httpx 响应，验证 function calling、重试、超时、429。"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from mailagent.llm.client import (
    LLMClient,
    LLMNetworkError,
    LLMResponseError,
    LLMTimeoutError,
    RateLimitError,
)


@pytest.fixture
def llm_client() -> LLMClient:
    return LLMClient(
        base_url="https://api.example.com",
        api_key="test-key",
        model="gpt-4.1-mini",
        timeout=1.0,
    )


def _mock_response(
    status_code: int = 200,
    body: dict[str, Any] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(body or {"choices": [{"message": {"content": "ok"}}]}).encode(),
        headers={"content-type": "application/json"},
    )


async def test_chat_completion_returns_response(llm_client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None) -> httpx.Response:
        captured["url"] = url
        captured["payload"] = json
        return _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await llm_client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "classify", "parameters": {}}}],
    )
    assert result["choices"][0]["message"]["content"] == "ok"
    assert captured["payload"]["model"] == "gpt-4.1-mini"
    assert captured["payload"]["tools"] is not None
    assert captured["payload"]["tool_choice"] == "auto"

    await llm_client.close()


async def test_chat_completion_429_no_retry(llm_client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code=429, content=b'{"error":"rate limited"}')

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(RateLimitError):
        await llm_client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert call_count == 1  # 429 不重试
    await llm_client.close()


async def test_chat_completion_timeout_retries(llm_client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException("simulated timeout")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(LLMTimeoutError):
        await llm_client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert call_count == 3  # 3 次重试
    await llm_client.close()


async def test_chat_completion_network_error_retries(llm_client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.NetworkError("simulated network error")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(LLMNetworkError):
        await llm_client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert call_count == 3
    await llm_client.close()


async def test_chat_completion_500_retries_then_raises_network_error(
    llm_client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_count = 0

    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code=500, content=b"server error")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(LLMNetworkError, match="500"):
        await llm_client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert call_count == 3  # 5xx 重试 3 次
    await llm_client.close()


async def test_chat_completion_400_raises_response_error(
    llm_client: LLMClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None) -> httpx.Response:
        return httpx.Response(status_code=400, content=b'{"error":"bad request"}')

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(LLMResponseError, match="400"):
        await llm_client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    await llm_client.close()


async def test_chat_completion_function_call_payload(llm_client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None) -> httpx.Response:
        captured["payload"] = json
        return _mock_response(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "classify",
                                        "arguments": '{"labels":[]}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify",
                "description": "Classify email",
                "parameters": {"type": "object", "properties": {"labels": {"type": "array"}}},
            },
        }
    ]
    result = await llm_client.chat_completion(
        messages=[{"role": "user", "content": "test"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "classify"}},
        temperature=0.0,
    )
    assert captured["payload"]["tool_choice"]["function"]["name"] == "classify"
    assert captured["payload"]["temperature"] == 0.0
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "classify"

    await llm_client.close()


async def test_close_releases_client(llm_client: LLMClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(self: httpx.AsyncClient, url: str, json: dict[str, Any] | None = None) -> httpx.Response:
        return _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    await llm_client.chat_completion(messages=[{"role": "user", "content": "hi"}])
    await llm_client.close()
    assert llm_client._client is None
