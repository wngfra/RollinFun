# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-24

### Changed

- Replaced Ollama HTTP backend with mlx-lm for in-process LLM inference on Apple Silicon unified memory.
- Replaced Ollama embeddings with mlx-embeddings for MLX-native vector generation.
- Default LLM model changed to `mlx-community/Mistral-7B-Instruct-v0.2`.
- Default embedding model changed to `mlx-community/all-MiniLM-L6-v2`.
- Moved mlx-audio from optional to core dependency (entire stack is now MLX-native).
- LLM config: removed `base_url`, added `top_p` and `repetition_penalty` sampler parameters.

### Added

- `launch.sh` script using `uv` for automated virtual environment creation and server startup.
- Sampling control via mlx-lm: temperature, top\_p, repetition\_penalty via `make_sampler` / `make_logits_processors`.
- Async-safe LLM streaming: synchronous `stream_generate` offloaded to thread pool to avoid blocking the event loop.
- Graceful degradation: server starts even when MLX LLM is unavailable (e.g., on Linux).

### Removed

- Ollama dependency -- no external LLM server required.
- `httpx`-based LLM client (HTTP streaming replaced by direct in-process calls).

## [0.1.0] - 2026-03-24

### Added

- FastAPI application with WebSocket endpoint at `/ws` and static file serving.
- Pydantic configuration layer loading from `config/default.yaml`, overridable via `RPTTS_CONFIG` environment variable.
- LLM client with streaming generation and graceful `LLMUnavailableError` handling.
- Per-turn context assembly injecting story progress, memory summary, RAG results, and recent turns into the chat prompt.
- Streaming regex segment parser for tagged LLM output: `[role mood="x" gesture="y"]...[/role]`, `[choices]...[/choices]`, `[scene location="id"]`. Supports both complete-text and incremental streaming modes.
- TTS engine abstraction (`TTSEngine` ABC) with `MLXTTSEngine` backend wrapping mlx-audio and `DummyTTSEngine` fallback producing silence on non-MLX platforms.
- Kokoro, Chatterbox, and Qwen3 engine-specific wrappers with per-engine default kwargs.
- Voice router mapping character IDs to TTS engines with lazy model loading, instance caching, and round-robin assignment from `voices/pool/` for unregistered characters.
- Audio post-processing pipeline: peak normalization, inter/intra-speaker silence insertion, float32-to-int16 conversion, and proportional word timestamp estimation.
- WebSocket turn-cycle handler: message dispatch, LLM streaming, segment parsing, TTS synthesis, audio encoding, and base64 delivery in a single async pipeline.
- Project scaffolding with directories for stories, voices, avatars, web UI, tests, and runtime data.
