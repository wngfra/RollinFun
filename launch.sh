#!/usr/bin/env bash
# RP-TTS Engine — one-command setup and launch via uv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_VERSION="3.11"

# ------------------------------------------------------------------
# Ensure uv is installed
# ------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$SCRIPT_DIR"

# ------------------------------------------------------------------
# Create / sync virtual environment
# ------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment (.venv, Python $PYTHON_VERSION)..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

echo "Installing / syncing dependencies..."
uv pip install -e ".[dev]" --python "$VENV_DIR/bin/python"

# ------------------------------------------------------------------
# Launch the server
# ------------------------------------------------------------------
echo ""
echo "Starting RP-TTS Engine..."
echo "  Web UI:    http://127.0.0.1:8000"
echo "  WebSocket: ws://127.0.0.1:8000/ws"
echo ""
exec "$VENV_DIR/bin/python" -m src.main "$@"
