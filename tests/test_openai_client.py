"""Tests for the OpenAI-compatible HTTP LLM client."""

from __future__ import annotations

import json

import httpx
import pytest

from src.config import AppConfig
from src.llm.client import LLMUnavailableError
from src.llm.openai_client import OpenAIHTTPClient


def _make_config(**overrides) -> AppConfig:
    llm = {
        "backend": "openai",
        "model": "test-model",
        "openai_base_url": "http://localhost:11434/v1",
        "openai_api_key": "",
        "openai_model": "",
        **overrides,
    }
    return AppConfig.model_validate({"llm": llm})


class TestOpenAIHTTPClientInit:
    def test_creates_client(self):
        config = _make_config()
        client = OpenAIHTTPClient(config)
        assert client.current_model == "test-model"

    def test_openai_model_override(self):
        config = _make_config(openai_model="gpt-4")
        client = OpenAIHTTPClient(config)
        assert client.current_model == "gpt-4"


class TestSwitchModel:
    @pytest.mark.asyncio
    async def test_switch_model(self):
        config = _make_config()
        client = OpenAIHTTPClient(config)
        await client.switch_model("new-model")
        assert client.current_model == "new-model"

    @pytest.mark.asyncio
    async def test_switch_same_model_noop(self):
        config = _make_config()
        client = OpenAIHTTPClient(config)
        await client.switch_model("test-model")
        assert client.current_model == "test-model"


class TestStreamChat:
    @pytest.mark.asyncio
    async def test_parses_sse_stream(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        sse_lines = []
        for chunk in chunks:
            sse_lines.append(f"data: {json.dumps(chunk)}")
        sse_lines.append("data: [DONE]")
        sse_body = "\n".join(sse_lines)

        async def _mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=sse_body)

        config = _make_config()
        client = OpenAIHTTPClient(config)
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(_mock_handler),
            base_url="http://localhost:11434/v1",
        )

        tokens = []
        async for token in client.stream_chat([{"role": "user", "content": "hi"}]):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        async def _mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        config = _make_config()
        client = OpenAIHTTPClient(config)
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(_mock_handler),
            base_url="http://localhost:11434/v1",
        )

        with pytest.raises(LLMUnavailableError, match="500"):
            async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
                pass


class TestEmbed:
    @pytest.mark.asyncio
    async def test_returns_embedding(self):
        embedding = [0.1, 0.2, 0.3]

        async def _mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"embedding": embedding}]})

        config = _make_config()
        client = OpenAIHTTPClient(config)
        client._http = httpx.AsyncClient(
            transport=httpx.MockTransport(_mock_handler),
            base_url="http://localhost:11434/v1",
        )

        result = await client.embed("test text")
        assert result == embedding


class TestFactory:
    def test_auto_falls_back_to_openai(self):
        """On non-MLX platforms, auto should fall back to OpenAIHTTPClient."""
        from src.llm import create_llm_client

        config = _make_config(backend="auto")
        client = create_llm_client(config)
        assert isinstance(client, OpenAIHTTPClient)

    def test_explicit_openai_backend(self):
        from src.llm import create_llm_client

        config = _make_config(backend="openai")
        client = create_llm_client(config)
        assert isinstance(client, OpenAIHTTPClient)

    def test_unknown_backend_raises(self):
        from src.llm import create_llm_client

        config = _make_config(backend="unknown")
        with pytest.raises(LLMUnavailableError, match="Unknown"):
            create_llm_client(config)
