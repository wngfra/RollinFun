# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-24

### Added

- FastAPI application with WebSocket endpoint at `/ws` and static file serving.
- Pydantic configuration layer loading from `config/default.yaml`, overridable via `RPTTS_CONFIG` environment variable.
- Async streaming Ollama LLM client (`/api/chat` and `/api/embed`) with graceful `LLMUnavailableError` handling.
- Per-turn context assembly injecting story progress, memory summary, RAG results, and recent turns into the chat prompt.
- Streaming regex segment parser for tagged LLM output: `[role mood="x" gesture="y"]...[/role]`, `[choices]...[/choices]`, `[scene location="id"]`. Supports both complete-text and incremental streaming modes.
- TTS engine abstraction (`TTSEngine` ABC) with `MLXTTSEngine` backend wrapping mlx-audio and `DummyTTSEngine` fallback producing silence on non-MLX platforms.
- Kokoro, Chatterbox, and Qwen3 engine-specific wrappers with per-engine default kwargs.
- Voice router mapping character IDs to TTS engines with lazy model loading, instance caching, and round-robin assignment from `voices/pool/` for unregistered characters.
- Audio post-processing pipeline: peak normalization, inter/intra-speaker silence insertion, float32-to-int16 conversion, and proportional word timestamp estimation.
- WebSocket turn-cycle handler: message dispatch, LLM streaming, segment parsing, TTS synthesis, audio encoding, and base64 delivery in a single async pipeline.
- Project scaffolding with directories for stories, voices, avatars, web UI, tests, and runtime data.
