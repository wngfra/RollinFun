# RP-TTS Engine

**Real-time role-play narration with multi-voice text-to-speech, powered by local LLMs and MLX.**

RP-TTS Engine is a single-process Python server that streams interactive fiction through a WebSocket, with each character voiced by a distinct TTS model running on Apple Silicon unified memory. A browser-based UI renders a cinematic dialogue experience with lip-synced 3D avatars, ambient particles, and BG3-inspired visual design.

---

## Architecture

```
Browser <--WebSocket--> FastAPI
                          |-- StoryManager     -> .story.yaml CRUD + compile to system prompt
                          |-- MemoryManager    -> ChromaDB + SQLite + rolling summary
                          |-- OllamaClient     -> streaming HTTP to localhost:11434
                          |-- SegmentParser    -> regex stream -> typed Segment objects
                          |-- VoiceRouter      -> character_id -> mlx-audio engine + kwargs
                          '-- AudioPipeline    -> PCM normalize + silence + base64 -> WS
```

**Memory budget:** ~11 GB unified (8 GB LLM Q4\_K\_M + 2 GB Chatterbox fp16 + 500 MB Kokoro bf16 + 300 MB embeddings + 200 MB Python).

---

## Features

| Phase | Scope | Status |
|-------|-------|--------|
| P0 | Core pipeline: config, LLM streaming, segment parser, TTS, audio, WebSocket | Done |
| P1 | Story scripts: schema, CRUD, compiler, validator, beat tracker | Planned |
| P2 | Web UI: avatar, dialogue, choices, audio playback | Planned |
| P3 | Story library, editor, origin selection | Planned |
| P4 | Visual polish: particles, scene manager, SFX | Planned |
| P5 | Multi-voice: Chatterbox and Qwen3 engine wrappers | Planned |
| P6 | Long-term memory: ChromaDB, summarizer, session log | Planned |
| P7 | Settings, history, beat progress UI | Planned |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) with a loaded model (default: `MN-Violet-Lotus-12B:Q4_K_M`)
- macOS with Apple Silicon (for MLX TTS; Linux runs with silent dummy engine)

### Install

```bash
git clone https://github.com/wngfra/RollinFun.git
cd RollinFun

# Core dependencies
pip install -e .

# With MLX TTS support (Apple Silicon only)
pip install -e ".[mlx]"

# Development tools
pip install -e ".[dev]"
```

### Run

```bash
# Pull the LLM and embedding models
ollama pull MN-Violet-Lotus-12B:Q4_K_M
ollama pull nomic-embed-text

# Start the server
python -m src.main
# or
rp-tts
```

The server starts at `http://127.0.0.1:8000`. Connect via WebSocket at `ws://127.0.0.1:8000/ws`.

---

## Project Structure

```
RollinFun/
├── pyproject.toml              # Project metadata and dependencies
├── config/
│   └── default.yaml            # Runtime configuration
├── src/
│   ├── main.py                 # FastAPI app, static mount, WS endpoint
│   ├── config.py               # Pydantic settings from YAML
│   ├── llm/
│   │   ├── client.py           # Async Ollama HTTP streaming
│   │   └── prompt.py           # Context assembly for chat API
│   ├── parser/
│   │   └── segment_parser.py   # Streaming regex parser for tagged output
│   ├── tts/
│   │   ├── engine.py           # TTSEngine ABC, MLX backend, Dummy fallback
│   │   ├── kokoro_mlx.py       # Kokoro-specific wrapper
│   │   ├── chatterbox_mlx.py   # Chatterbox-specific wrapper
│   │   ├── qwen3_mlx.py        # Qwen3-specific wrapper
│   │   └── router.py           # Character -> voice mapping, lazy loading
│   ├── audio/
│   │   └── pipeline.py         # Normalize, silence, int16, word timestamps
│   ├── ws/
│   │   └── handler.py          # WebSocket dispatch, turn-cycle orchestration
│   ├── story/                  # (P1) Story script management
│   └── memory/                 # (P6) Long-term memory
├── stories/                    # .story.yaml files
├── voices/pool/                # Reference audio for voice cloning
├── avatars/                    # 3D avatar models (.glb)
├── web/                        # Browser UI (P2+)
├── tests/
└── data/
    ├── chroma/                 # ChromaDB vector store
    └── sessions/               # SQLite session logs
```

---

## Configuration

All settings live in `config/default.yaml`. Override with the `RPTTS_CONFIG` environment variable:

```bash
RPTTS_CONFIG=./config/custom.yaml rp-tts
```

Key sections:

| Section | Controls |
|---------|----------|
| `llm` | Ollama base URL, model name, temperature, max tokens |
| `tts.engines` | Model IDs and enabled state for Kokoro, Chatterbox, Qwen3 |
| `tts.narrator_*` | Default narrator voice, speed, engine |
| `audio` | Inter/intra speaker silence durations |
| `memory` | Summary interval, RAG top-K, recent turn window |
| `server` | Host and port |

---

## TTS Engines

| Engine | Model | Use Case | Platform |
|--------|-------|----------|----------|
| Kokoro | `mlx-community/Kokoro-82M-bf16` | Fast narration, preset voices | Apple Silicon |
| Chatterbox | `mlx-community/chatterbox-turbo-fp16` | Voice cloning from reference audio | Apple Silicon |
| Qwen3 | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | Natural language voice description | Apple Silicon |
| Dummy | (built-in) | Silent fallback for development | Any |

Engines are lazy-loaded on first use. Unknown characters are auto-assigned voices from the `voices/pool/` directory via round-robin.

---

## WebSocket Protocol

### Server to Client

| Type | Description |
|------|-------------|
| `speech` | Voiced segment with base64 PCM audio, word timestamps, mood, gesture |
| `choices` | Dialogue choice options |
| `scene_change` | Location transition with background and lighting |
| `beat_progress` | Current act/beat and completion state |
| `origin_select` | Player origin selection options |
| `status` | Engine state: `generating`, `synthesizing`, `ready`, `error` |
| `story_list` | Available stories metadata |
| `story_data` | Full story YAML |
| `validation` | Story validation errors and warnings |

### Client to Server

| Type | Description |
|------|-------------|
| `message` | Player free-text input |
| `start_story` | Begin a story session |
| `story_action` | CRUD operations on stories |
| `voice_config` | Set character reference audio |
| `voice_preview` | Preview a character voice |
| `control` | Session save/load/new/resummary |

---

## Story Script Format

Stories are defined in `.story.yaml` files with structured sections:

```yaml
meta:       # Title, author, tags, content rating
world:      # Setting, era, geography, rules, atmosphere
characters: # NPCs and player character with voice config
style:      # Tone, pacing, dialogue density, forbidden content
plot:       # Acts, beats, transitions, secrets
dynamic:    # Random encounters, ambient NPCs, flavor
scenes:     # Location backgrounds and ambient audio
```

Each character defines a `voice` block specifying the TTS engine, reference audio or preset, and speech parameters.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Start server in development
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

---

## Tech Stack

| Component | Library | License |
|-----------|---------|---------|
| Web framework | FastAPI | MIT |
| ASGI server | Uvicorn | BSD-3 |
| HTTP client | httpx | BSD-3 |
| Config / validation | Pydantic | MIT |
| TTS inference | mlx-audio | MIT |
| Vector store | ChromaDB | Apache-2.0 |
| LLM backend | Ollama | MIT |
| Numerical | NumPy | BSD-3 |

---

## License

[Apache License 2.0](LICENSE)
