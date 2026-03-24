"""Chatterbox TTS engine wrapper for mlx-audio."""

from __future__ import annotations

import numpy as np

from src.tts.engine import MLXTTSEngine


class ChatterboxEngine(MLXTTSEngine):
    """Chatterbox-specific defaults on top of :class:`MLXTTSEngine`."""

    def __init__(
        self,
        model_id: str = "mlx-community/chatterbox-turbo-fp16",
        *,
        ref_audio: str | None = None,
        exaggeration: float = 0.5,
        cfg: float = 0.5,
    ) -> None:
        super().__init__(model_id)
        self._ref_audio = ref_audio
        self._exaggeration = exaggeration
        self._cfg = cfg

    async def synthesize(self, text: str, **kwargs: object) -> tuple[np.ndarray, int]:
        defaults: dict[str, object] = {
            "exaggeration": self._exaggeration,
            "cfg": self._cfg,
        }
        if self._ref_audio:
            defaults["ref_audio"] = self._ref_audio
        defaults.update(kwargs)  # type: ignore[arg-type]
        return await super().synthesize(text, **defaults)
