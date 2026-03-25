"""TTS engine abstraction with MLX and dummy backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import logging as _logging

try:
    from mlx_audio.tts.utils import load_model as _mlx_load_model

    MLX_AVAILABLE = True
except Exception:
    _logging.getLogger(__name__).warning("mlx-audio import failed — TTS will use silent fallback", exc_info=True)
    MLX_AVAILABLE = False
    _mlx_load_model = None


class TTSEngine(ABC):
    """Abstract base for text-to-speech engines."""

    @abstractmethod
    async def synthesize(self, text: str, **kwargs: object) -> tuple[np.ndarray, int]:
        """Return ``(pcm_float32, sample_rate)``."""


class MLXTTSEngine(TTSEngine):
    """Wraps an ``mlx-audio`` model for on-device TTS."""

    def __init__(self, model_id: str) -> None:
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "mlx-audio is not available on this platform. "
                "Install with: pip install mlx-audio (requires Apple Silicon)."
            )
        self._model_id = model_id
        self._model = _mlx_load_model(model_id)  # type: ignore[misc]

    async def synthesize(self, text: str, **kwargs: object) -> tuple[np.ndarray, int]:
        results = list(self._model.generate(text=text, **kwargs))
        audio = np.array(results[0].audio, dtype=np.float32)
        return audio, 24000


class DummyTTSEngine(TTSEngine):
    """Returns silence — for testing on non-MLX platforms."""

    async def synthesize(self, text: str, **kwargs: object) -> tuple[np.ndarray, int]:
        word_count = max(len(text.split()), 1)
        duration_s = word_count * 0.3
        samples = np.zeros(int(24000 * duration_s), dtype=np.float32)
        return samples, 24000
