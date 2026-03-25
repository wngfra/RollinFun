"""OpenAI-compatible HTTP LLM client for non-MLX platforms."""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import httpx

from src.config import AppConfig
from src.llm.client import LLMUnavailableError

logger = logging.getLogger(__name__)


class OpenAIHTTPClient:
    """LLM client that talks to any OpenAI-compatible API (ollama, vllm, OpenAI, etc.)."""

    def __init__(self, config: AppConfig) -> None:
        self._model_id = config.llm.openai_model or config.llm.model
        self._embed_model = config.llm.embed_model
        self._max_tokens = config.llm.max_tokens
        self._temperature = config.llm.temperature
        self._top_p = config.llm.top_p

        headers: dict[str, str] = {}
        if config.llm.openai_api_key:
            headers["Authorization"] = f"Bearer {config.llm.openai_api_key}"

        self._http = httpx.AsyncClient(
            base_url=config.llm.openai_base_url,
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        logger.info(
            "OpenAI HTTP client ready — model=%s base_url=%s",
            self._model_id,
            config.llm.openai_base_url,
        )

    @property
    def current_model(self) -> str:
        return self._model_id

    async def switch_model(self, model_id: str) -> None:
        if model_id == self._model_id:
            return
        self._model_id = model_id
        logger.info("OpenAI HTTP client switched to model: %s", model_id)

    async def close(self) -> None:
        await self._http.aclose()

    async def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": self._model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }
        try:
            async with self._http.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise LLMUnavailableError(
                        f"LLM API returned {resp.status_code}: {body.decode(errors='replace')}"
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot connect to LLM API at {self._http.base_url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"LLM API request failed: {exc}") from exc

    async def embed(self, text: str) -> list[float]:
        payload = {
            "model": self._embed_model,
            "input": text,
        }
        try:
            resp = await self._http.post("/embeddings", json=payload)
            if resp.status_code != 200:
                raise LLMUnavailableError(
                    f"Embeddings API returned {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Embeddings API request failed: {exc}") from exc
