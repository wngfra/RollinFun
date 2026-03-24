"""Application configuration loaded from YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "MN-Violet-Lotus-12B:Q4_K_M"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.8
    max_tokens: int = 1024


class MemoryConfig(BaseModel):
    summary_interval: int = 15
    summary_max_tokens: int = 2048
    rag_top_k: int = 5
    recent_turns: int = 12


class TTSEngineEntry(BaseModel):
    model: str
    enabled: bool = True


class TTSConfig(BaseModel):
    engines: dict[str, TTSEngineEntry] = {}
    narrator_engine: str = "kokoro"
    narrator_voice: str = "bf_emma"
    narrator_speed: float = 0.95
    sample_rate: int = 24000


class AudioConfig(BaseModel):
    inter_speaker_silence_ms: int = 300
    intra_speaker_silence_ms: int = 80


class AvatarConfig(BaseModel):
    default_glb: str = "./avatars/default.glb"
    camera_view: str = "upper"


class StoriesConfig(BaseModel):
    directory: str = "./stories"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    tts: TTSConfig = TTSConfig()
    audio: AudioConfig = AudioConfig()
    avatar: AvatarConfig = AvatarConfig()
    stories: StoriesConfig = StoriesConfig()
    server: ServerConfig = ServerConfig()


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> AppConfig:
    """Load and cache application config from YAML.

    Override the default path via the ``RPTTS_CONFIG`` environment variable
    or by passing *path* directly.
    """
    config_path = Path(path or os.environ.get("RPTTS_CONFIG", str(_DEFAULT_CONFIG_PATH)))
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return AppConfig.model_validate(data)
    return AppConfig()
