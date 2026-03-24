"""WebSocket message dispatch and turn-cycle orchestration."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field

from starlette.websockets import WebSocket

from src.audio.pipeline import estimate_word_timestamps, process_segment
from src.config import AppConfig
from src.llm.client import LLMUnavailableError, MLXLMClient
from src.llm.prompt import assemble_context
from src.parser.segment_parser import (
    Choices,
    SceneChange,
    Segment,
    StreamingParser,
)
from src.tts.router import VoiceRouter

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Lightweight per-connection session state."""

    story_id: str | None = None
    player_name: str = "Player"
    system_prompt: str = "You are a narrator."
    recent_turns: list[dict[str, str]] = field(default_factory=list)
    current_location: str | None = None
    act_title: str | None = None
    beat_id: str | None = None
    objectives: list[str] = field(default_factory=list)
    completed_beats: list[str] = field(default_factory=list)
    turn_count: int = 0


class TurnHandler:
    """Orchestrates the full request → LLM → TTS → WS response cycle."""

    def __init__(
        self,
        config: AppConfig,
        llm_client: MLXLMClient,
        voice_router: VoiceRouter,
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._router = voice_router

    async def dispatch(self, ws: WebSocket, raw: str, session: Session) -> None:
        """Route an inbound WebSocket message by its ``type`` field."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await _send(ws, {"type": "status", "state": "error", "message": "Invalid JSON"})
            return

        msg_type = msg.get("type")
        if msg_type == "message":
            await self._handle_message(ws, msg, session)
        elif msg_type == "start_story":
            await self._handle_start_story(ws, msg, session)
        elif msg_type == "story_action":
            await _send(ws, {"type": "status", "state": "error", "message": "Story actions not yet implemented"})
        elif msg_type == "voice_config":
            await self._handle_voice_config(ws, msg)
        elif msg_type == "voice_preview":
            await self._handle_voice_preview(ws, msg)
        elif msg_type == "control":
            await _send(ws, {"type": "status", "state": "error", "message": "Control actions not yet implemented"})
        else:
            await _send(ws, {"type": "status", "state": "error", "message": f"Unknown message type: {msg_type}"})

    # ------------------------------------------------------------------
    # Turn cycle
    # ------------------------------------------------------------------

    async def _handle_message(
        self, ws: WebSocket, msg: dict, session: Session
    ) -> None:
        user_text = msg.get("text", "")
        if not user_text:
            return

        if self._llm is None:
            await _send(ws, {"type": "status", "state": "error", "message": "LLM backend is not available"})
            return

        # 1. Send status
        await _send(ws, {"type": "status", "state": "generating", "message": "Generating response…"})

        # 2. Assemble context
        messages = assemble_context(
            system_prompt=session.system_prompt,
            act_title=session.act_title,
            beat_id=session.beat_id,
            objectives=session.objectives,
            completed_beats=session.completed_beats,
            current_location=session.current_location,
            recent_turns=session.recent_turns[-self._config.memory.recent_turns:],
            user_input=user_text,
        )

        # 3. Stream LLM and parse
        parser = StreamingParser()
        full_response = ""
        prev_role: str | None = None

        try:
            await _send(ws, {"type": "status", "state": "generating", "message": "Streaming…"})
            async for token in self._llm.stream_chat(messages):
                full_response += token
                elements = parser.feed(token)
                for elem in elements:
                    prev_role = await self._emit_element(ws, elem, prev_role)
        except LLMUnavailableError as exc:
            await _send(ws, {"type": "status", "state": "error", "message": str(exc)})
            return

        # Flush remaining buffered text
        for elem in parser.flush():
            prev_role = await self._emit_element(ws, elem, prev_role)

        # 4. Update session history
        session.recent_turns.append({"role": "user", "content": user_text})
        session.recent_turns.append({"role": "assistant", "content": full_response})
        session.turn_count += 1

        # 5. Ready
        await _send(ws, {"type": "status", "state": "ready", "message": ""})

    async def _emit_element(
        self, ws: WebSocket, elem: Segment | Choices | SceneChange, prev_role: str | None
    ) -> str | None:
        """Send a parsed element over the WebSocket. Return the current role."""
        if isinstance(elem, Segment):
            await self._emit_speech(ws, elem, prev_role)
            return elem.role
        elif isinstance(elem, Choices):
            await _send(ws, {"type": "choices", "options": elem.options})
            return prev_role
        elif isinstance(elem, SceneChange):
            await _send(ws, {
                "type": "scene_change",
                "location_id": elem.location_id,
                "background": "",
                "ambient": None,
                "lighting": None,
            })
            return prev_role
        return prev_role

    async def _emit_speech(
        self, ws: WebSocket, seg: Segment, prev_role: str | None
    ) -> None:
        if not seg.text.strip():
            return

        await _send(ws, {"type": "status", "state": "synthesizing", "message": f"Synthesizing {seg.role}…"})

        # TTS
        engine, kwargs = await self._router.get_engine(seg.role)
        pcm, sr = await engine.synthesize(seg.text, **kwargs)

        # Audio pipeline
        audio_bytes = process_segment(
            pcm, sr, seg.role, prev_role,
            inter_silence_ms=self._config.audio.inter_speaker_silence_ms,
            intra_silence_ms=self._config.audio.intra_speaker_silence_ms,
        )

        # Word timestamps
        audio_duration_ms = (len(audio_bytes) / 2) / sr * 1000  # int16 = 2 bytes/sample
        words, wtimes, wdurations = estimate_word_timestamps(seg.text, audio_duration_ms)

        # Encode and send
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        await _send(ws, {
            "type": "speech",
            "role": seg.role,
            "text": seg.text,
            "mood": seg.mood,
            "gesture": seg.gesture,
            "audio": audio_b64,
            "words": words,
            "wtimes": wtimes,
            "wdurations": wdurations,
        })

    # ------------------------------------------------------------------
    # Other handlers
    # ------------------------------------------------------------------

    async def _handle_start_story(
        self, ws: WebSocket, msg: dict, session: Session
    ) -> None:
        session.story_id = msg.get("story_id")
        session.player_name = msg.get("player_name", "Player")
        session.recent_turns.clear()
        session.turn_count = 0
        await _send(ws, {"type": "status", "state": "ready", "message": f"Story '{session.story_id}' started"})

    async def _handle_voice_config(self, ws: WebSocket, msg: dict) -> None:
        char = msg.get("character", "")
        if char:
            self._router.register_character(char, engine_name=self._config.tts.narrator_engine)
        await _send(ws, {"type": "status", "state": "ready", "message": f"Voice configured for {char}"})

    async def _handle_voice_preview(self, ws: WebSocket, msg: dict) -> None:
        char_id = msg.get("character_id", "narrator")
        sample = msg.get("sample_text", "This is a voice preview.")
        engine, kwargs = await self._router.get_engine(char_id)
        pcm, sr = await engine.synthesize(sample, **kwargs)
        audio_bytes = process_segment(pcm, sr, char_id, None)
        audio_duration_ms = (len(audio_bytes) / 2) / sr * 1000
        words, wtimes, wdurations = estimate_word_timestamps(sample, audio_duration_ms)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        await _send(ws, {
            "type": "speech",
            "role": char_id,
            "text": sample,
            "mood": "neutral",
            "gesture": None,
            "audio": audio_b64,
            "words": words,
            "wtimes": wtimes,
            "wdurations": wdurations,
        })


async def _send(ws: WebSocket, data: dict) -> None:
    await ws.send_json(data)
