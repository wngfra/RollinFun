"""Voice routing: character_id → (TTSEngine, kwargs).

Lazy-loads engine instances on first use. Falls back to DummyTTSEngine
when MLX is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import AppConfig
from src.tts.engine import MLX_AVAILABLE, DummyTTSEngine, TTSEngine

logger = logging.getLogger(__name__)

# Engine class registry — import guarded so the module loads on any platform.
_ENGINE_CLASSES: dict[str, type] = {}

if MLX_AVAILABLE:
    try:
        from src.tts.kokoro_mlx import KokoroEngine

        _ENGINE_CLASSES["kokoro"] = KokoroEngine
    except Exception:
        logger.warning("Failed to import KokoroEngine", exc_info=True)
    try:
        from src.tts.chatterbox_mlx import ChatterboxEngine

        _ENGINE_CLASSES["chatterbox"] = ChatterboxEngine
    except Exception:
        logger.warning("Failed to import ChatterboxEngine", exc_info=True)
    try:
        from src.tts.qwen3_mlx import Qwen3Engine

        _ENGINE_CLASSES["qwen3"] = Qwen3Engine
    except Exception:
        logger.warning("Failed to import Qwen3Engine", exc_info=True)


class VoiceRouter:
    """Map character IDs to TTS engines and synthesis kwargs."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._engine_cache: dict[str, TTSEngine] = {}
        self._char_map: dict[str, tuple[str, dict[str, object]]] = {}
        self._voice_pool = self._scan_voice_pool()
        self._pool_index = 0

        if not MLX_AVAILABLE:
            logger.warning(
                "mlx-audio not available — all TTS will use DummyTTSEngine (silence)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_character(
        self,
        character_id: str,
        engine_name: str,
        **kwargs: object,
    ) -> None:
        """Explicitly assign a voice configuration to a character."""
        self._char_map[character_id] = (engine_name, kwargs)

    async def get_engine(
        self, character_id: str
    ) -> tuple[TTSEngine, dict[str, object]]:
        """Return ``(engine_instance, synthesis_kwargs)`` for *character_id*."""
        if character_id == "narrator":
            return self._get_narrator()

        if character_id in self._char_map:
            engine_name, kwargs = self._char_map[character_id]
            engine = self._resolve_engine(engine_name)
            return engine, kwargs

        # Auto-assign from voice pool (round-robin)
        return self._auto_assign(character_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_narrator(self) -> tuple[TTSEngine, dict[str, object]]:
        cfg = self._config.tts
        engine = self._resolve_engine(cfg.narrator_engine)
        kwargs: dict[str, object] = {
            "voice": cfg.narrator_voice,
            "speed": cfg.narrator_speed,
        }
        return engine, kwargs

    def _resolve_engine(self, name: str) -> TTSEngine:
        if name in self._engine_cache:
            return self._engine_cache[name]

        if not MLX_AVAILABLE or name not in _ENGINE_CLASSES:
            engine: TTSEngine = DummyTTSEngine()
            self._engine_cache[name] = engine
            return engine

        engine_cfg = self._config.tts.engines.get(name)
        model_id = engine_cfg.model if engine_cfg else ""
        try:
            engine = _ENGINE_CLASSES[name](model_id)
        except Exception:
            logger.exception("Failed to load %s engine, falling back to dummy", name)
            engine = DummyTTSEngine()
        self._engine_cache[name] = engine
        return engine

    def _auto_assign(self, character_id: str) -> tuple[TTSEngine, dict[str, object]]:
        engine = self._resolve_engine(self._config.tts.narrator_engine)
        kwargs: dict[str, object] = {}

        if self._voice_pool:
            ref = self._voice_pool[self._pool_index % len(self._voice_pool)]
            self._pool_index += 1
            kwargs["ref_audio"] = str(ref)

        # Cache assignment so the same character keeps the same voice
        engine_name = self._config.tts.narrator_engine
        self._char_map[character_id] = (engine_name, kwargs)
        return engine, kwargs

    def _scan_voice_pool(self) -> list[Path]:
        pool_dir = Path("voices/pool")
        if not pool_dir.is_dir():
            return []
        return sorted(pool_dir.glob("*.wav"))
