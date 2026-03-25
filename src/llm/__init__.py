"""LLM client factory — selects MLX or OpenAI-compatible HTTP backend."""

from __future__ import annotations

import logging

from src.config import AppConfig
from src.llm.client import LLMUnavailableError

logger = logging.getLogger(__name__)


def create_llm_client(config: AppConfig):
    """Create an LLM client based on config.llm.backend.

    With ``"auto"`` (default), tries MLX first and falls back to the
    OpenAI-compatible HTTP client.
    """
    backend = config.llm.backend

    if backend in ("mlx", "auto"):
        try:
            from src.llm.client import MLXLMClient

            return MLXLMClient(config)
        except LLMUnavailableError:
            if backend == "mlx":
                raise
            logger.info("MLX backend unavailable, falling back to OpenAI HTTP client")

    if backend in ("openai", "auto"):
        from src.llm.openai_client import OpenAIHTTPClient

        return OpenAIHTTPClient(config)

    raise LLMUnavailableError(f"Unknown LLM backend: {backend}")
