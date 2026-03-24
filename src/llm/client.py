"""Async streaming client for Ollama HTTP API."""

from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

from src.config import AppConfig


class LLMUnavailableError(Exception):
    """Raised when the LLM backend cannot be reached."""


class OllamaClient:
    """Thin async wrapper around Ollama's ``/api/chat`` and ``/api/embed``."""

    def __init__(self, config: AppConfig) -> None:
        self._base_url = config.llm.base_url.rstrip("/")
        self._model = config.llm.model
        self._embed_model = config.llm.embed_model
        self._temperature = config.llm.temperature
        self._max_tokens = config.llm.max_tokens
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream token strings from Ollama /api/chat."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        return
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc

    async def embed(self, text: str) -> list[float]:
        """Get an embedding vector via Ollama /api/embed."""
        payload = {"model": self._embed_model, "input": text}
        try:
            resp = await self._client.post(
                f"{self._base_url}/api/embed", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"][0]
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(
                f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc
