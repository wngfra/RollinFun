# RP-TTS Engine

**Real-time role-play narration with multi-voice text-to-speech, running entirely on Apple Silicon via MLX.**

RP-TTS Engine is a single-process Python server that streams interactive fiction through a WebSocket. Both the LLM and TTS models run in-process on Apple Silicon unified memory via MLX -- no external servers required. A browser-based UI renders a cinematic dialogue experience with lip-synced 3D avatars, ambient particles, and BG3-inspired visual design.

---

## Architecture

```
Browser <--WebSocket--> FastAPI
                          |-- StoryManager     -> .story.yaml CRUD + compile to system prompt
                          |-- MemoryManager    -> ChromaDB + SQLite + rolling summary
                          |-- MLXLMClient      -> in-process mlx-lm inference
                          |-- SegmentParser    -> regex stream -> typed Segment objects
                          |-- VoiceRouter      -> character_id -> mlx-audio engine + kwargs
                          '-- AudioPipeline    -> PCM normalize + silence + base64 -> WS
```

Everything runs in a single Python process on unified memory:

| Component | Model | Memory |
|-----------|-------|--------|
| LLM | `mlx-community/Mistral-7B-Instruct-v0.2` | ~4 GB |
| TTS (Chatterbox) | `mlx-community/chatterbox-turbo-fp16` | ~2 GB |
| TTS (Kokoro) | `mlx-community/Kokoro-82M-bf16` | ~500 MB |
| Embeddings | `mlx-community/all-MiniLM-L6-v2` | ~100 MB |
| Python + ChromaDB | -- | ~200 MB |

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

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (auto-installed by `launch.sh` if missing)

### One-Command Launch

```bash
git clone https://github.com/wngfra/RollinFun.git
cd RollinFun
./launch.sh
```

The launch script will:
1. Install `uv` if not present
2. Create a `.venv` with Python 3.11
3. Install all dependencies
4. Download the LLM and TTS models on first run (cached by HuggingFace Hub)
5. Start the server

### Manual Install

```bash
pip install -e .

# Or with dev tools
pip install -e ".[dev]"
```

### Run

```bash
python -m src.main
# or
rp-tts
```

The server starts at `http://127.0.0.1:8000`. Connect via WebSocket at `ws://127.0.0.1:8000/ws`.

Models are downloaded automatically from HuggingFace Hub on first use and cached locally.

---

## Project Structure

```
RollinFun/
├── pyproject.toml              # Project metadata and dependencies
├── launch.sh                   # One-command setup and run via uv
├── config/
│   └── default.yaml            # Runtime configuration
├── src/
│   ├── main.py                 # FastAPI app, static mount, WS endpoint
│   ├── config.py               # Pydantic settings from YAML
│   ├── llm/
│   │   ├── client.py           # In-process MLX LLM streaming via mlx-lm
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
| `llm` | MLX model ID, temperature, top\_p, max tokens, repetition penalty |
| `tts.engines` | Model IDs and enabled state for Kokoro, Chatterbox, Qwen3 |
| `tts.narrator_*` | Default narrator voice, speed, engine |
| `audio` | Inter/intra speaker silence durations |
| `memory` | Summary interval, RAG top-K, recent turn window |
| `server` | Host and port |

### Changing the LLM

Edit `config/default.yaml`:

```yaml
llm:
  model: "mlx-community/Mistral-7B-Instruct-v0.2"   # any mlx-community model
  temperature: 0.8
  top_p: 0.9
  max_tokens: 1024
  repetition_penalty: 1.05
```

Any model from the [mlx-community](https://huggingface.co/mlx-community) HuggingFace organization works. The model is downloaded and cached on first use.

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
| LLM inference | mlx-lm | MIT |
| TTS inference | mlx-audio | MIT |
| Embeddings | mlx-embeddings | MIT |
| Web framework | FastAPI | MIT |
| ASGI server | Uvicorn | BSD-3 |
| Config / validation | Pydantic | MIT |
| Vector store | ChromaDB | Apache-2.0 |
| Numerical | NumPy | BSD-3 |

All ML inference runs locally via [MLX](https://github.com/ml-explore/mlx) on Apple Silicon unified memory. No cloud APIs, no external servers.

---

## License

[Apache License 2.0](LICENSE)
