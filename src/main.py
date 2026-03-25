"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import get_config
from src.llm import create_llm_client
from src.llm.client import LLMUnavailableError
from src.tts.router import VoiceRouter
from src.ws.handler import Session, TurnHandler

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    voice_router = VoiceRouter(config)

    # Create handler with no LLM initially — model loads in background
    handler = TurnHandler(config, None, voice_router)  # type: ignore[arg-type]
    app.state.config = config
    app.state.turn_handler = handler

    logger.info("RP-TTS Engine ready on %s:%d", config.server.host, config.server.port)

    # Load LLM in a background task so the server accepts connections immediately
    async def _load_llm() -> None:
        try:
            loop = asyncio.get_event_loop()
            client = await loop.run_in_executor(None, create_llm_client, config)
            handler._llm = client
            handler._llm_loading = False
            logger.info("LLM model loaded and ready")
        except LLMUnavailableError:
            handler._llm_loading = False
            logger.warning("MLX LLM not available — generation is disabled")

    asyncio.create_task(_load_llm())
    yield


app = FastAPI(title="RP-TTS Engine", lifespan=lifespan)

# Mount static web files
if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR), html=True), name="static")


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "rp-tts"}


@app.get("/")
async def root():
    return FileResponse(WEB_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    handler: TurnHandler = ws.app.state.turn_handler
    try:
        while True:
            raw = await ws.receive_text()
            await handler.dispatch(ws, raw, session)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


def main() -> None:
    import uvicorn

    config = get_config()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "src.main:app",
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
