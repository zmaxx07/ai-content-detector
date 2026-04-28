#!/bin/bash
# ═══════════════════════════════════════════════════════
#  AI Content Detection System — Start Script (Linux/Mac)
#  Starts backend (port 8000) + frontend (port 3000)
# ═══════════════════════════════════════════════════════

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   AI Content Detection System  v3.0.0           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Check .env ─────────────────────────────────────────
if [ ! -f "$BACKEND/.env" ]; then
  echo "⚠  No .env found. Creating from .env.example..."
  cp "$BACKEND/.env.example" "$BACKEND/.env"
  echo "   Edit backend/.env and add your HUGGINGFACE_TOKEN, then re-run."
  exit 1
fi

# ── Check Python venv ──────────────────────────────────
if [ ! -d "$BACKEND/venv" ]; then
  echo "📦 Creating Python virtual environment..."
  cd "$BACKEND" && python3 -m venv venv
  source "$BACKEND/venv/bin/activate"
  echo "📦 Installing backend dependencies..."
  pip install -r "$BACKEND/requirements.txt" -q
fi

# ── Check node_modules ────────────────────────────────
if [ ! -d "$FRONTEND/node_modules" ]; then
  echo "📦 Installing frontend dependencies..."
  cd "$FRONTEND" && npm install --silent
fi

echo ""
echo "🚀 Starting Backend  → http://localhost:8000"
echo "🚀 Starting Frontend → http://localhost:3000"
echo ""
echo "  Press Ctrl+C to stop both servers"
echo ""

# ── Start backend in background ───────────────────────
source "$BACKEND/venv/bin/activate"
cd "$BACKEND"
python run.py --mode api &
BACKEND_PID=$!

# ── Start frontend ────────────────────────────────────
cd "$FRONTEND"
BROWSER=none npm start &
FRONTEND_PID=$!

# ── Trap Ctrl+C ───────────────────────────────────────
cleanup() {
  echo ""
  echo "🛑 Stopping servers..."
  kill $BACKEND_PID 2>/dev/null
  kill $FRONTEND_PID 2>/dev/null
  exit 0
}
trap cleanup INT TERM

wait
