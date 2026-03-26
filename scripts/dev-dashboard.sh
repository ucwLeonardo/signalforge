#!/usr/bin/env bash
# Dev dashboard server with hot-reload on code changes.
# Restarts the paper-trading server whenever Python or HTML files change.
#
# Usage:
#   ./scripts/dev-dashboard.sh              # default port 8787, dev mode
#   ./scripts/dev-dashboard.sh --port 9000  # custom port
#
# For production (continuous tracking, no offline reconciliation):
#   python3 src/signalforge/paper/server.py --mode live --port 8787

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${1:-8787}"
if [[ "$PORT" == "--port" ]]; then
    PORT="${2:-8787}"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
WATCH_DIR="$PROJECT_ROOT/src/signalforge"

echo "=== SignalForge Paper Dashboard (dev mode) ==="
echo "  URL:  http://localhost:$PORT/"
echo "  Hot-reload: ON (polling every 2s)"
echo "  Press Ctrl+C to stop."
echo ""

get_fingerprint() {
    find "$WATCH_DIR" \( -name "*.py" -o -name "*.html" -o -name "*.css" -o -name "*.js" \) \
        -exec stat -c '%Y %n' {} + 2>/dev/null | sort | md5sum | cut -d' ' -f1
}

SERVER_PID=""

start_server() {
    python3 "$PROJECT_ROOT/src/signalforge/paper/server.py" --port "$PORT" &
    SERVER_PID=$!
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    SERVER_PID=""
}

cleanup() {
    stop_server
    echo ""
    echo "Server stopped."
    exit 0
}
trap cleanup INT TERM

LAST_FP=""
while true; do
    FP="$(get_fingerprint)"
    if [[ "$FP" != "$LAST_FP" ]]; then
        if [[ -n "$SERVER_PID" ]]; then
            echo ""
            echo "[dev] File change detected — restarting server..."
            stop_server
            sleep 0.5
        fi
        start_server
        LAST_FP="$FP"
    fi

    # Check if server died unexpectedly
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[dev] Server crashed — restarting..."
        sleep 1
        start_server
    fi

    sleep 2
done
