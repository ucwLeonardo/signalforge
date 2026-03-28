#!/usr/bin/env bash
#
# SignalForge Deploy Script
#
# Installs dependencies, downloads market data for all configured assets,
# trains models, and starts the paper trading dashboard.
#
# Usage:
#   ./deploy.sh              # Full deploy (install + data + train + serve)
#   ./deploy.sh --skip-data  # Skip data download (use cached data)
#   ./deploy.sh --data-only  # Only download data, don't start server
#   ./deploy.sh --port 9090  # Custom server port
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
DATA_DIR="${HOME}/.signalforge/data"
MODEL_DIR="${HOME}/.signalforge/models"
DEFAULT_PORT=8787
SKIP_DATA=false
DATA_ONLY=false
PORT="${DEFAULT_PORT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-data)  SKIP_DATA=true; shift ;;
        --data-only)  DATA_ONLY=true; shift ;;
        --port)       PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--skip-data] [--data-only] [--port PORT]"
            echo ""
            echo "  --skip-data   Skip data download, use existing cache"
            echo "  --data-only   Download data only, don't start server"
            echo "  --port PORT   Server port (default: ${DEFAULT_PORT})"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}═══ $* ═══${NC}\n"; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
header "SignalForge Deployment"

info "Project directory: ${SCRIPT_DIR}"
info "Data directory:    ${DATA_DIR}"
info "Model directory:   ${MODEL_DIR}"
info "Server port:       ${PORT}"
echo ""

# Check Python
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    info "Python: ${PY_VERSION}"
else
    error "Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

# Check MASSIVE_API_KEY
if [[ -z "${MASSIVE_API_KEY:-}" ]]; then
    # Try sourcing bashrc
    if [[ -f "${HOME}/.bashrc" ]]; then
        source "${HOME}/.bashrc" 2>/dev/null || true
    fi
fi

if [[ -z "${MASSIVE_API_KEY:-}" ]]; then
    error "MASSIVE_API_KEY not set."
    echo "  Get a free API key at https://massive.com (Polygon.io)"
    echo "  Then: export MASSIVE_API_KEY=\"your_key_here\""
    exit 1
fi
success "MASSIVE_API_KEY is set (${MASSIVE_API_KEY:0:8}...)"

# Check GEMINI_API_KEY (optional)
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    success "GEMINI_API_KEY is set (enables TradingAgents LLM engine)"
else
    warn "GEMINI_API_KEY not set — TradingAgents will use rule-based fallback"
fi

# ---------------------------------------------------------------------------
# Step 1: Virtual environment & dependencies
# ---------------------------------------------------------------------------
header "Step 1/5: Installing Dependencies"

cd "${SCRIPT_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
    info "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    success "Virtual environment created at ${VENV_DIR}"
else
    success "Virtual environment exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

info "Installing core dependencies..."
pip install -e . --quiet 2>&1 | tail -1
success "Core dependencies installed"

# Check for optional deps
info "Checking optional engine dependencies..."
ENGINES_STATUS=""

if python3 -c "import lightgbm" 2>/dev/null; then
    ENGINES_STATUS="${ENGINES_STATUS}  LightGBM:      ${GREEN}installed${NC}\n"
else
    ENGINES_STATUS="${ENGINES_STATUS}  LightGBM:      ${YELLOW}not installed (sklearn fallback)${NC}\n"
fi

if python3 -c "import google.genai" 2>/dev/null; then
    ENGINES_STATUS="${ENGINES_STATUS}  Google GenAI:  ${GREEN}installed${NC}\n"
else
    ENGINES_STATUS="${ENGINES_STATUS}  Google GenAI:  ${YELLOW}not installed (rule-based fallback)${NC}\n"
fi

if python3 -c "import torch" 2>/dev/null; then
    TORCH_DEV=$(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')")
    ENGINES_STATUS="${ENGINES_STATUS}  PyTorch:       ${GREEN}installed (${TORCH_DEV})${NC}\n"
else
    ENGINES_STATUS="${ENGINES_STATUS}  PyTorch:       ${RED}not installed (LSTM engine will fail)${NC}\n"
fi

echo -e "${ENGINES_STATUS}"

# ---------------------------------------------------------------------------
# Step 2: Create directories
# ---------------------------------------------------------------------------
header "Step 2/5: Setting Up Directories"

for dir in "${DATA_DIR}/stock" "${DATA_DIR}/crypto" "${DATA_DIR}/futures" \
           "${MODEL_DIR}/lstm" "${MODEL_DIR}/gbm" \
           "${HOME}/.signalforge/cache" "${HOME}/.signalforge/results" \
           "${HOME}/.signalforge/accounts"; do
    mkdir -p "${dir}"
done
success "Data directories created under ~/.signalforge/"

# Show existing data
STOCK_COUNT=$(ls "${DATA_DIR}/stock/"*.parquet 2>/dev/null | wc -l || echo 0)
CRYPTO_COUNT=$(ls "${DATA_DIR}/crypto/"*.parquet 2>/dev/null | wc -l || echo 0)
FUTURES_COUNT=$(ls "${DATA_DIR}/futures/"*.parquet 2>/dev/null | wc -l || echo 0)
info "Existing cached data: ${STOCK_COUNT} stocks, ${CRYPTO_COUNT} crypto, ${FUTURES_COUNT} futures"

# ---------------------------------------------------------------------------
# Step 3: Download market data
# ---------------------------------------------------------------------------
if [[ "${SKIP_DATA}" == "true" ]]; then
    header "Step 3/5: Data Download (SKIPPED)"
    warn "Using existing cached data. Run without --skip-data to update."
else
    header "Step 3/5: Downloading Market Data"
    info "This downloads OHLCV data for all configured assets via Massive API."
    info "Free tier rate limit: 5 requests/minute — this will take a while on first run."
    echo ""

    # Get asset count
    ASSET_INFO=$(python3 -c "
from signalforge.config import load_config
cfg = load_config()
stocks = len(cfg.us_stocks)
crypto = len(cfg.crypto)
futures = len(cfg.futures)
total = stocks + crypto + futures
print(f'{total} {stocks} {crypto} {futures}')
" 2>/dev/null || echo "38 10 24 4")
    read -r TOTAL N_STOCKS N_CRYPTO N_FUTURES <<< "${ASSET_INFO}"

    info "Config assets: ${N_STOCKS} stocks, ${N_CRYPTO} crypto, ${N_FUTURES} futures = ${TOTAL} total"

    # Discovery expands this
    info "Dynamic discovery will find additional assets (S&P 500 stocks, crypto pairs)."
    echo ""

    ESTIMATE_MIN=$(( (TOTAL * 12 + 59) / 60 ))  # 12s per symbol at 5 req/min
    info "Estimated time for config assets: ~${ESTIMATE_MIN} minutes (first run)"
    info "Note: discovered assets will add more time. Progress shown below."
    echo ""

    # Run fetch for all categories
    info "Starting data download..."
    START_TIME=$(date +%s)

    python3 -c "
import sys, os, time
os.environ.setdefault('MASSIVE_API_KEY', '${MASSIVE_API_KEY}')

from signalforge.config import load_config
from signalforge.data.incremental import IncrementalFetcher
from signalforge.data.store import DataStore
from signalforge.data.discovery import discover_all
from signalforge.pipeline import _classify_symbol, _get_lookback_days

cfg = load_config()
store = DataStore(cfg.data_dir)
fetcher = IncrementalFetcher(store)

# Discover all assets
print('[DISCOVERY] Discovering assets...', flush=True)
categories = ['us_stocks', 'crypto', 'futures']
symbols = discover_all(categories, cfg)
print(f'[DISCOVERY] Found {len(symbols)} assets to download', flush=True)
print(flush=True)

# Download data for each symbol
success = 0
skipped = 0
failed = 0

for i, symbol in enumerate(symbols, 1):
    sym_type = _classify_symbol(symbol)
    lookback = _get_lookback_days(sym_type, cfg)

    # Check if already cached and fresh (< 1 day old)
    cached = store.load(symbol, '1d')
    if not cached.empty and len(cached) >= 30:
        import pandas as _pd
        last_ts = _pd.Timestamp(cached['timestamp'].max())
        age_hours = (time.time() - last_ts.timestamp()) / 3600
        if age_hours < 24:
            skipped += 1
            if skipped <= 5 or skipped % 50 == 0:
                print(f'  [{i}/{len(symbols)}] {symbol}: fresh cache ({len(cached)} bars, {age_hours:.0f}h old)', flush=True)
            continue

    try:
        df = fetcher.fetch(symbol, '1d', lookback)
        if not df.empty and len(df) >= 30:
            success += 1
            print(f'  [{i}/{len(symbols)}] {symbol}: downloaded {len(df)} bars', flush=True)
        else:
            failed += 1
            bars = len(df) if not df.empty else 0
            print(f'  [{i}/{len(symbols)}] {symbol}: insufficient data ({bars} bars)', flush=True)
    except Exception as e:
        failed += 1
        print(f'  [{i}/{len(symbols)}] {symbol}: FAILED ({e})', flush=True)

print(flush=True)
print(f'[DONE] Downloaded: {success}, Cached: {skipped}, Failed: {failed}, Total: {len(symbols)}', flush=True)
" 2>&1

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    ELAPSED_MIN=$(( ELAPSED / 60 ))
    ELAPSED_SEC=$(( ELAPSED % 60 ))
    success "Data download complete in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo ""

    # Show updated counts
    STOCK_COUNT=$(ls "${DATA_DIR}/stock/"*.parquet 2>/dev/null | wc -l || echo 0)
    CRYPTO_COUNT=$(ls "${DATA_DIR}/crypto/"*.parquet 2>/dev/null | wc -l || echo 0)
    FUTURES_COUNT=$(ls "${DATA_DIR}/futures/"*.parquet 2>/dev/null | wc -l || echo 0)
    TOTAL_FILES=$(( STOCK_COUNT + CRYPTO_COUNT + FUTURES_COUNT ))
    info "Cached data: ${STOCK_COUNT} stocks, ${CRYPTO_COUNT} crypto, ${FUTURES_COUNT} futures (${TOTAL_FILES} total)"
fi

# ---------------------------------------------------------------------------
# Step 4: Train models
# ---------------------------------------------------------------------------
header "Step 4/5: Training Models"

LSTM_COUNT=$(ls "${MODEL_DIR}/lstm/"*.pt 2>/dev/null | wc -l || echo 0)
GBM_COUNT=$(ls "${MODEL_DIR}/gbm/"*.pkl 2>/dev/null | wc -l || echo 0)

if [[ "${LSTM_COUNT}" -gt 0 ]] && [[ "${GBM_COUNT}" -gt 0 ]]; then
    info "Existing models: ${LSTM_COUNT} LSTM, ${GBM_COUNT} GBM"
    info "Models will be incrementally updated during the first scan."
    success "Skipping full re-training (use 'signalforge train' to force)"
else
    info "No trained models found. Models will be trained during the first scan."
    info "This adds ~2-5 seconds per asset to the first scan."
fi

# ---------------------------------------------------------------------------
# Step 5: Start server
# ---------------------------------------------------------------------------
if [[ "${DATA_ONLY}" == "true" ]]; then
    header "Step 5/5: Server Start (SKIPPED)"
    warn "Data-only mode. Run 'signalforge paper dashboard --port ${PORT}' to start the server."
else
    header "Step 5/5: Starting Paper Trading Dashboard"

    info "Server: http://localhost:${PORT}"
    info "Press Ctrl+C to stop"
    echo ""

    # Print summary
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║${NC}  ${BOLD}SignalForge Paper Trading Dashboard${NC}             ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}╠══════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}${CYAN}║${NC}                                                  ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}║${NC}  URL:      ${GREEN}http://localhost:${PORT}${NC}                ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}║${NC}  Data:     ${STOCK_COUNT} stocks, ${CRYPTO_COUNT} crypto, ${FUTURES_COUNT} futures     ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}║${NC}  Models:   ${LSTM_COUNT} LSTM, ${GBM_COUNT} GBM                        ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}║${NC}  Engines:  LSTM(35%) GBM(35%) Tech(20%) LLM(10%) ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}║${NC}                                                  ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}║${NC}  ${YELLOW}Tip: Click 'Scan' in the dashboard to generate${NC}  ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}║${NC}  ${YELLOW}signals using your account's asset categories.${NC}  ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}║${NC}                                                  ${BOLD}${CYAN}║${NC}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""

    exec python3 -m signalforge.paper.server --port "${PORT}"
fi
