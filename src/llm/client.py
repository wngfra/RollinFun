"""In-process LLM client using mlx-lm on Apple Silicon."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import numpy as np

from src.config import AppConfig

logger = logging.getLogger(__name__)

try:
    from mlx_lm import load as _mlx_lm_load
    from mlx_lm import stream_generate as _mlx_stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    MLX_LM_AVAILABLE = True
except Exception:
    logger.warning("mlx-lm import failed — LLM generation will be disabled", exc_info=True)
    MLX_LM_AVAILABLE = False

try:
    from mlx_embeddings import load as _mlx_embed_load

    MLX_EMBED_AVAILABLE = True
except Exception:
    logger.warning("mlx-embeddings import failed — embeddings will be disabled", exc_info=True)
    MLX_EMBED_AVAILABLE = False


class LLMUnavailableError(Exception):
    """Raised when the LLM backend is not available."""


class MLXLMClient:
    """In-process LLM inference via mlx-lm with streaming generation.

    Loads the model into Apple Silicon unified memory on init.
    On non-MLX platforms, raises :class:`LLMUnavailableError`.
    """

    def __init__(self, config: AppConfig) -> None:
        if not MLX_LM_AVAILABLE:
            raise LLMUnavailableError(
                "mlx-lm is not available. Install with: pip install mlx-lm "
                "(requires Apple Silicon)."
            )

        self._max_tokens = config.llm.max_tokens
        self._model_id = config.llm.model

        # Load LLM
        logger.info("Loading LLM: %s", config.llm.model)
        try:
            self._model, self._tokenizer = _mlx_lm_load(config.llm.model)
        except Exception as exc:
            raise LLMUnavailableError(
                f"Failed to load model {config.llm.model}: {exc}"
            ) from exc

        # Build sampler and logits processors
        self._sampler = make_sampler(
            temperature=config.llm.temperature,
            top_p=config.llm.top_p,
        )
        self._logits_processors = make_logits_processors(
            repetition_penalty=config.llm.repetition_penalty,
        )

        # Load embedding model
        self._embed_model = None
        self._embed_tokenizer = None
        if MLX_EMBED_AVAILABLE:
            logger.info("Loading embedding model: %s", config.llm.embed_model)
            try:
                self._embed_model, self._embed_tokenizer = _mlx_embed_load(
                    config.llm.embed_model
                )
            except Exception:
                logger.warning(
                    "Failed to load embedding model %s — embeddings disabled",
                    config.llm.embed_model,
                    exc_info=True,
                )
        else:
            logger.warning("mlx-embeddings not available — embeddings disabled")

        logger.info("MLX LLM client ready")

    @property
    def current_model(self) -> str:
        """Return the currently loaded model ID."""
        return self._model_id

    async def switch_model(self, model_id: str) -> None:
        """Load a new model in a background thread. Auto-downloads if not cached."""
        if model_id == self._model_id:
            return
        logger.info("Switching LLM model to: %s", model_id)
        loop = asyncio.get_event_loop()
        try:
            model, tokenizer = await loop.run_in_executor(None, _mlx_lm_load, model_id)
        except Exception as exc:
            raise LLMUnavailableError(
                f"Failed to load model {model_id}: {exc}"
            ) from exc
        # Atomic swap
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        logger.info("LLM model switched to: %s", model_id)

    async def close(self) -> None:
        """No-op — kept for interface compatibility."""

    async def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream token strings from the local MLX model.

        Applies the tokenizer's chat template, then streams generation.
        The synchronous ``stream_generate`` call is offloaded to a thread
        to avoid blocking the event loop.
        """
        # Apply chat template
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            # Fallback: concatenate messages
            prompt = "\n".join(
                f"{'### ' if m['role'] == 'system' else ''}{m['content']}"
                for m in messages
            )

        # Run synchronous stream_generate in a thread and yield tokens
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _generate() -> None:
            try:
                for response in _mlx_stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt,
                    max_tokens=self._max_tokens,
                    sampler=self._sampler,
                    logits_processors=self._logits_processors,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, response.text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_event_loop().run_in_executor(None, _generate)

        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise LLMUnavailableError(f"Generation error: {item}") from item
            yield item

    async def embed(self, text: str) -> list[float]:
        """Get an embedding vector using the MLX embedding model."""
        if self._embed_model is None or self._embed_tokenizer is None:
            raise LLMUnavailableError("Embedding model not loaded")

        def _encode() -> list[float]:
            inputs = self._embed_tokenizer(
                text, return_tensors="np", padding=True, truncation=True
            )
            outputs = self._embed_model(**{k: v for k, v in inputs.items()})
            # Extract the [CLS] token embedding or mean pooling
            embeddings = outputs.last_hidden_state
            # Mean pooling over token dimension
            mask = inputs["attention_mask"]
            masked = embeddings * np.expand_dims(mask, -1)
            pooled = masked.sum(axis=1) / mask.sum(axis=1, keepdims=True)
            return pooled[0].tolist()

        return await asyncio.get_event_loop().run_in_executor(None, _encode)
