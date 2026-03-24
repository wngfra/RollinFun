"""Kokoro TTS engine wrapper for mlx-audio."""

from __future__ import annotations

import numpy as np

from src.tts.engine import MLXTTSEngine


class KokoroEngine(MLXTTSEngine):
    """Kokoro-specific defaults on top of :class:`MLXTTSEngine`."""

    def __init__(
        self,
        model_id: str = "mlx-community/Kokoro-82M-bf16",
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> None:
        super().__init__(model_id)
        self._voice = voice
        self._speed = speed
        self._lang_code = lang_code

    async def synthesize(self, text: str, **kwargs: object) -> tuple[np.ndarray, int]:
        defaults = {
            "voice": self._voice,
            "speed": self._speed,
            "lang_code": self._lang_code,
        }
        defaults.update(kwargs)  # type: ignore[arg-type]
        return await super().synthesize(text, **defaults)
