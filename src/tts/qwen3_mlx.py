"""Qwen3 TTS engine wrapper for mlx-audio."""

from __future__ import annotations

import numpy as np

from src.tts.engine import MLXTTSEngine


class Qwen3Engine(MLXTTSEngine):
    """Qwen3-TTS-specific defaults on top of :class:`MLXTTSEngine`."""

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        *,
        voice: str = "Chelsie",
        language: str = "English",
    ) -> None:
        super().__init__(model_id)
        self._voice = voice
        self._language = language

    async def synthesize(self, text: str, **kwargs: object) -> tuple[np.ndarray, int]:
        defaults: dict[str, object] = {
            "voice": self._voice,
            "language": self._language,
        }
        defaults.update(kwargs)  # type: ignore[arg-type]
        return await super().synthesize(text, **defaults)
